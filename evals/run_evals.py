"""
Regression eval harness for the GTM worker agent.

Usage (run from repo root):
    python3 evals/run_evals.py              # run all evals
    python3 evals/run_evals.py eval-001     # run one eval by ID

Results are saved to evals/results/latest.json.
Exit code 1 if overall pass rate < PASS_THRESHOLD.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from skills import worker_agent, judge_proposal

GOLDEN_PATH = Path("evals/golden_dataset.json")
RESULTS_DIR = Path("evals/results")
PASS_THRESHOLD = 0.7


# ── Rule-based scorers ──────────────────────────────────────────────────────

def score_stage(proposal, expected) -> tuple[float, str]:
    stage_updates = [u for u in proposal.proposed_field_updates if u.field == "stage"]
    should_advance = expected["stage_should_advance"]

    if should_advance:
        if not stage_updates:
            return 0.0, f"FAIL — expected stage advance to '{expected['expected_stage']}' but none proposed"
        if stage_updates[0].proposed_value == expected["expected_stage"]:
            return 1.0, f"PASS — correctly advanced to '{expected['expected_stage']}'"
        return 0.5, f"PARTIAL — advanced to wrong stage: '{stage_updates[0].proposed_value}' (expected '{expected['expected_stage']}')"
    else:
        if stage_updates:
            return 0.0, f"FAIL — should not advance stage but proposed: '{stage_updates[0].proposed_value}'"
        return 1.0, "PASS — correctly did not advance stage"


def score_required_fields(proposal, expected) -> tuple[float, str]:
    required = set(expected.get("required_field_updates", []))
    forbidden = set(expected.get("forbidden_field_updates", []))
    updated = {u.field for u in proposal.proposed_field_updates}

    violations = forbidden & updated
    if violations:
        return 0.0, f"FAIL — forbidden fields were updated: {violations}"
    if not required:
        return 1.0, "PASS — no required fields specified"

    found = required & updated
    missing = required - found
    score = len(found) / len(required)
    if missing:
        return score, f"PARTIAL — {len(found)}/{len(required)} required fields updated, missing: {missing}"
    return 1.0, f"PASS — all required fields updated: {required}"


def score_tasks(proposal, expected) -> tuple[float, str]:
    min_tasks = expected.get("min_tasks", 0)
    actual = len(proposal.proposed_tasks)
    if actual >= min_tasks:
        return 1.0, f"PASS — {actual} tasks generated (min: {min_tasks})"
    ratio = actual / min_tasks if min_tasks else 1.0
    return ratio, f"PARTIAL — {actual} tasks generated, expected at least {min_tasks}"


def score_commitments(proposal, expected) -> tuple[float, str]:
    keywords = expected.get("required_commitment_keywords", [])
    if not keywords:
        return 1.0, "PASS — no required keywords specified"

    all_text = " ".join(proposal.commitments).lower()
    found = [kw for kw in keywords if kw.lower() in all_text]
    missing = [kw for kw in keywords if kw.lower() not in all_text]
    score = len(found) / len(keywords)
    if missing:
        return score, f"PARTIAL — {len(found)}/{len(keywords)} keywords found, missing: {missing}"
    return 1.0, f"PASS — all keywords found: {keywords}"


def score_hallucination(proposal, expected) -> tuple[float, str]:
    forbidden = [name.lower() for name in expected.get("forbidden_stakeholders", [])]
    if not forbidden:
        return 1.0, "PASS — no forbidden stakeholders specified"

    # Only check factual extraction fields — not follow_up_email, which is
    # generated content and may legitimately reference CRM context not in transcript
    full_text = (
        " ".join(proposal.stakeholder_mentions)
        + " " + proposal.summary
    ).lower()

    hallucinated = [name for name in forbidden if name in full_text]
    if hallucinated:
        return 0.0, f"FAIL — hallucinated stakeholders not in transcript: {hallucinated}"
    return 1.0, f"PASS — none of the forbidden stakeholders appeared: {expected['forbidden_stakeholders']}"


def score_confidence(proposal, expected) -> tuple[float, str]:
    conf = proposal.confidence
    min_conf = expected.get("confidence_min", 0.0)
    max_conf = expected.get("confidence_max", 1.0)

    if conf < min_conf:
        return 0.0, f"FAIL — confidence {conf:.0%} below minimum {min_conf:.0%}"
    if conf > max_conf:
        return 0.0, f"FAIL — confidence {conf:.0%} above maximum {max_conf:.0%}"
    return 1.0, f"PASS — confidence {conf:.0%} in range [{min_conf:.0%}, {max_conf:.0%}]"


# ── LLM-as-judge scorer ─────────────────────────────────────────────────────

def score_judge(proposal, transcript) -> tuple[float, str, dict]:
    """⚡ NON-DETERMINISTIC: separate Gemini call to independently evaluate
    the proposal. Only runs during evals, never in production."""
    try:
        judge = judge_proposal(transcript, proposal)
        score = (judge.confidence + judge.grounding_score) / 2
        hall = "🚨 hallucination detected" if judge.hallucination_flag else "no hallucination"
        msg = (
            f"confidence={judge.confidence:.0%} grounding={judge.grounding_score:.0%} "
            f"{hall} | {judge.reasoning}"
        )
        status = "PASS" if score >= 0.5 and not judge.hallucination_flag else "FAIL"
        return (1.0 if status == "PASS" else 0.0), msg, judge.model_dump()
    except Exception as e:
        return 0.0, f"ERROR — judge call failed: {e}", {}


# ── Eval runner ─────────────────────────────────────────────────────────────

RULE_SCORERS = [
    ("stage_progression",    score_stage),
    ("required_fields",      score_required_fields),
    ("task_generation",      score_tasks),
    ("commitment_keywords",  score_commitments),
    ("hallucination",        score_hallucination),
    ("confidence",           score_confidence),
]


def run_single(example: dict) -> dict:
    print(f"\n{'─' * 60}")
    print(f"Running {example['id']}: {example['description']}")

    try:
        # ⚡ NON-DETERMINISTIC: worker agent call (single Gemini call, same as production)
        proposal = worker_agent(example["transcript"], example["opportunity_id"])
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"id": example["id"], "description": example.get("description", ""), "status": "ERROR", "error": str(e), "scores": {}, "overall": 0.0}

    scores = {}

    # Rule-based scores
    print(f"  {'Rule-based checks':}")
    for name, fn in RULE_SCORERS:
        score, message = fn(proposal, example["expected"])
        scores[name] = {"score": round(score, 3), "message": message}
        status = "✓" if score >= 0.9 else ("~" if score >= 0.5 else "✗")
        print(f"  {status} {name:<24} {score:.0%}  {message}")

    # LLM-as-judge (eval only)
    print(f"  {'LLM-as-judge':}")
    judge_score, judge_msg, judge_detail = score_judge(proposal, example["transcript"])
    scores["llm_judge"] = {"score": round(judge_score, 3), "message": judge_msg, "detail": judge_detail}
    status = "✓" if judge_score >= 0.9 else "✗"
    print(f"  {status} {'llm_judge':<24} {judge_score:.0%}  {judge_msg}")

    overall = sum(s["score"] for s in scores.values()) / len(scores)
    print(f"  {'─' * 40}")
    print(f"  Overall: {overall:.0%}")

    return {
        "id": example["id"],
        "description": example["description"],
        "opportunity_id": example["opportunity_id"],
        "status": "PASS" if overall >= PASS_THRESHOLD else "FAIL",
        "scores": scores,
        "overall": round(overall, 3),
        "proposal_summary": proposal.summary,
        "proposed_stage": next(
            (u.proposed_value for u in proposal.proposed_field_updates if u.field == "stage"), None
        ),
        "worker_confidence": proposal.confidence,
    }


def main():
    golden = json.loads(GOLDEN_PATH.read_text())

    if len(sys.argv) > 1:
        target = sys.argv[1]
        golden = [e for e in golden if e["id"] == target]
        if not golden:
            print(f"No eval found with id '{target}'")
            sys.exit(1)

    print(f"\nGTM Agent Eval Harness — {len(golden)} example(s)")
    print(f"Scorers: 6 rule-based + 1 LLM-as-judge (eval only)")

    results = [run_single(example) for example in golden]

    passed = sum(1 for r in results if r["status"] == "PASS")
    errored = sum(1 for r in results if r["status"] == "ERROR")
    total = len(results)
    overall_rate = sum(r["overall"] for r in results) / total

    print(f"\n{'═' * 60}")
    print(f"Results: {passed}/{total} passed  |  {errored} errors  |  Avg score: {overall_rate:.0%}")
    print(f"{'═' * 60}\n")

    for r in results:
        icon = "✓" if r["status"] == "PASS" else ("!" if r["status"] == "ERROR" else "✗")
        print(f"  {icon} {r['id']:<12} {r['status']:<8} {r['overall']:.0%}  —  {r['description'][:55]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": datetime.now().isoformat(),
        "total": total, "passed": passed, "errored": errored,
        "overall_score": round(overall_rate, 3),
        "pass_threshold": PASS_THRESHOLD,
        "results": results,
    }
    (RESULTS_DIR / "latest.json").write_text(json.dumps(payload, indent=2))
    (RESULTS_DIR / f"{timestamp}.json").write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to evals/results/latest.json")

    if overall_rate < PASS_THRESHOLD:
        print(f"FAILED: overall score {overall_rate:.0%} below threshold {PASS_THRESHOLD:.0%}")
        sys.exit(1)


if __name__ == "__main__":
    main()

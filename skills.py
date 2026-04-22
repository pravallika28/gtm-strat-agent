import json
from pathlib import Path

from models import (
    Opportunity, Account, Activity, Playbook,
    OpportunityBrief, WorkerProposal, ResearchResponse, GeneratedTranscript, JudgeEval,
)
from llm import call_llm
import tracing

DATA_DIR = Path("data")


def _load_data():
    # DETERMINISTIC: reading static JSON files from disk
    opportunities = [Opportunity(**o) for o in json.loads((DATA_DIR / "opportunities.json").read_text())]
    accounts = [Account(**a) for a in json.loads((DATA_DIR / "accounts.json").read_text())]
    activities = [Activity(**a) for a in json.loads((DATA_DIR / "activities.json").read_text())]
    playbooks = [Playbook(**p) for p in json.loads((DATA_DIR / "playbooks.json").read_text())]
    return opportunities, accounts, activities, playbooks


def crm_research(query: str) -> ResearchResponse:
    tracing.start_trace("crm_research", metadata={"query_chars": len(query)})
    try:
        opportunities, accounts, activities, playbooks = _load_data()
        tasks = json.loads((DATA_DIR / "tasks.json").read_text())

        prompt = f"""
You are a CRM research agent with access to all sales data. Answer the user's question using only the data provided below.

--- CRM DATA ---

Opportunities:
{json.dumps([o.model_dump() for o in opportunities], indent=2)}

Accounts:
{json.dumps([a.model_dump() for a in accounts], indent=2)}

Activities:
{json.dumps([a.model_dump() for a in activities], indent=2)}

Playbooks:
{json.dumps([p.model_dump() for p in playbooks], indent=2)}

Open Tasks:
{json.dumps(tasks, indent=2)}

--- USER QUESTION ---
{query}

Return:
- answer: a direct, concise answer to the question
- key_findings: bullet-point facts from the data that support the answer
- recommended_actions: any actions the seller should take based on the findings
- risks: any risks or concerns surfaced by the data relevant to the question
"""
        # ⚡ NON-DETERMINISTIC: Gemini API call
        result = call_llm(prompt, ResearchResponse, span_name="crm_research.gemini")
        response = ResearchResponse(**result)
        tracing.end_trace(metadata={"findings": len(response.key_findings)})
        return response
    except Exception as e:
        tracing.end_trace(metadata={"error": str(e)})
        raise


def research_agent(opportunity_id: str) -> OpportunityBrief:
    tracing.start_trace("research_agent", metadata={"opportunity_id": opportunity_id})
    try:
        # DETERMINISTIC: loading and filtering data from local JSON files
        opportunities, accounts, activities, playbooks = _load_data()

        opp = next((o for o in opportunities if o.opportunity_id == opportunity_id), None)
        if not opp:
            raise ValueError(f"Opportunity {opportunity_id} not found")

        account = next((a for a in accounts if a.account_id == opp.account_id), None)
        opp_activities = [a for a in activities if a.opportunity_id == opportunity_id]
        playbook = next((p for p in playbooks if p.stage == opp.stage), None)

        prompt = f"""
You are a GTM research agent. Analyze this opportunity and return a structured brief.

Opportunity:
{opp.model_dump_json(indent=2)}

Account:
{account.model_dump_json(indent=2) if account else "Not found"}

Recent Activities:
{json.dumps([a.model_dump() for a in opp_activities], indent=2)}

Playbook for stage "{opp.stage}":
{playbook.model_dump_json(indent=2) if playbook else "No playbook found"}

Return a structured opportunity brief with:
- summary: 2-3 sentence deal summary
- health_signals: positive or negative account signals
- recommended_actions: next best actions based on playbook and activities
- risks: deal risks or blockers
- whitespace: upsell/cross-sell products not yet in the deal
"""
        # ⚡ NON-DETERMINISTIC: delegates to LLM via call_llm().
        result = call_llm(prompt, OpportunityBrief, span_name="research_agent.gemini")

        result["opportunity_id"] = opportunity_id
        result["account_name"] = account.account_name if account else ""
        brief = OpportunityBrief(**result)
        tracing.end_trace(metadata={"stage": opp.stage, "account": brief.account_name})
        return brief
    except Exception as e:
        tracing.end_trace(metadata={"error": str(e)})
        raise


def generate_transcript(opportunity_id: str) -> GeneratedTranscript:
    tracing.start_trace("generate_transcript", metadata={"opportunity_id": opportunity_id})
    try:
        opportunities, accounts, activities, _ = _load_data()

        opp = next((o for o in opportunities if o.opportunity_id == opportunity_id), None)
        if not opp:
            raise ValueError(f"Opportunity {opportunity_id} not found")

        account = next((a for a in accounts if a.account_id == opp.account_id), None)
        opp_activities = [a for a in activities if a.opportunity_id == opportunity_id]

        prompt = f"""
You are simulating a realistic B2B sales call between a seller and a customer.

Use the context below to generate a natural, realistic conversation transcript.
The call should advance the deal in some meaningful way — surface a new pain point,
handle an objection, make a commitment, or move toward next steps.
Include realistic dialogue from both sides. Format as:

Date: <today>
Participants: <seller name> (Autodesk), <stakeholder names and roles>

---

<Seller>: ...
<Customer>: ...
(continue for 15-20 exchanges)

Opportunity:
{opp.model_dump_json(indent=2)}

Account:
{account.model_dump_json(indent=2) if account else "Not found"}

Recent Activity History:
{json.dumps([a.model_dump() for a in opp_activities], indent=2)}
"""
        # ⚡ NON-DETERMINISTIC: Gemini generates a simulated sales conversation
        result = call_llm(prompt, GeneratedTranscript, span_name="generate_transcript.gemini")
        transcript = GeneratedTranscript(**result)
        tracing.end_trace(metadata={"transcript_chars": len(transcript.transcript)})
        return transcript
    except Exception as e:
        tracing.end_trace(metadata={"error": str(e)})
        raise


def judge_proposal(transcript: str, proposal: WorkerProposal) -> JudgeEval:
    """⚡ NON-DETERMINISTIC: independent Gemini call that evaluates the worker's output.
    Separate context window = less self-serving bias than asking the worker to self-score."""
    prompt = f"""
You are an independent quality evaluator for a GTM AI agent. You are NOT the agent that produced the output below — your job is to critically assess it.

--- ORIGINAL TRANSCRIPT ---
{transcript}

--- AGENT OUTPUT TO EVALUATE ---
You are evaluating only the factual extraction below — what the agent pulled from the transcript
and what CRM changes it proposed. Do NOT evaluate any generated content (e.g. follow-up emails)
since those are allowed to draw on broader CRM context beyond the transcript.

Summary:
{proposal.summary}

Commitments extracted (must be grounded in transcript):
{json.dumps(proposal.commitments, indent=2)}

Proposed CRM field updates (must be supported by transcript):
{json.dumps([u.model_dump() for u in proposal.proposed_field_updates], indent=2)}

Proposed next actions / tasks (must be grounded in transcript):
{json.dumps([t.model_dump() for t in proposal.proposed_tasks], indent=2)}

Stakeholders mentioned (must appear in transcript):
{json.dumps(proposal.stakeholder_mentions, indent=2)}

--- YOUR EVALUATION ---

Score the following:

1. confidence (0.0–1.0): How well does the transcript support the proposals overall?
   Calibration guide — be strict:
   0.0–0.3: Vague or very short call, no clear decisions, no explicit commitments
   0.3–0.5: Some interest expressed but no concrete outcomes or dates
   0.5–0.7: Clear next steps and at least one named commitment, some ambiguity remains
   0.7–0.9: Multiple explicit commitments with dates, CRM updates clearly supported
   0.9–1.0: Every single proposal is directly and unambiguously stated in the transcript

2. grounding_score (0.0–1.0): What fraction of the commitments and field updates are directly traceable to specific words or sentences in the transcript? 1.0 = all of them, 0.0 = none.

3. hallucination_flag (true/false): Did the agent include any stakeholder names, facts, decisions, or claims that do NOT appear in the transcript? Be strict — if a name appears in the output but not in the transcript text, flag it.

4. reasoning: 2-3 sentences explaining your scores, citing specific evidence (or lack of it) from the transcript.
"""
    tracing.start_trace("judge_proposal", metadata={"transcript_chars": len(transcript)})
    try:
        # ⚡ NON-DETERMINISTIC: separate Gemini call for independent evaluation
        result = call_llm(prompt, JudgeEval, span_name="judge_proposal.gemini")
        eval_result = JudgeEval(**result)
        tracing.end_trace(metadata={
            "confidence": eval_result.confidence,
            "grounding": eval_result.grounding_score,
            "hallucination_flag": eval_result.hallucination_flag,
        })
        return eval_result
    except Exception as e:
        tracing.end_trace(metadata={"error": str(e)})
        raise


def worker_agent(transcript: str, opportunity_id: str) -> WorkerProposal:
    tracing.start_trace("worker_agent", metadata={"opportunity_id": opportunity_id, "transcript_chars": len(transcript)})
    try:
        # DETERMINISTIC: loading context data from local JSON files
        opportunities, accounts, _, _ = _load_data()
        progression_rules = json.loads((DATA_DIR / "progression_rules.json").read_text())

        opp = next((o for o in opportunities if o.opportunity_id == opportunity_id), None)
        account = next((a for a in accounts if a.account_id == opp.account_id), None) if opp else None

        prompt = f"""
You are a GTM worker agent. Analyze the sales call transcript and return structured proposals.

Current Opportunity:
{opp.model_dump_json(indent=2) if opp else "Not found"}

Account:
{account.model_dump_json(indent=2) if account else "Not found"}

Transcript:
{transcript}

Opportunity Progression Rules:
{json.dumps(progression_rules, indent=2)}

Instructions:
- Extract a summary, commitments, risks, and stakeholder mentions from the transcript.
- For proposed_field_updates: only propose updates to stage, next_step, amount, close_date, or forecast_category, and only when clearly supported by the transcript.
- For stage progression: check the progression rules above. Only propose a stage advance if the transcript provides evidence that the requirements for the next stage are met. State which requirements were met in the reason field.
- For proposed_tasks: identify concrete follow-up actions with due dates where mentioned.
- Write a professional follow-up email.
- Set confidence (0.0–1.0) using this calibration rubric:
    0.0–0.3: Vague or short call, no clear decisions, no explicit commitments, no next steps agreed
    0.3–0.5: Some signals but ambiguous — interest expressed but no concrete outcomes
    0.5–0.7: Clear next steps and at least one commitment, but some ambiguity remains
    0.7–0.9: Multiple explicit commitments, concrete dates, and clear CRM updates supported
    0.9–1.0: Every proposal is directly and unambiguously stated in the transcript
"""
        # ⚡ NON-DETERMINISTIC: worker generates proposals (single call in production)
        result = call_llm(prompt, WorkerProposal, span_name="worker_agent.gemini")
        proposal = WorkerProposal(**result)
        tracing.end_trace(metadata={
            "confidence": proposal.confidence,
            "field_updates": len(proposal.proposed_field_updates),
            "tasks": len(proposal.proposed_tasks),
        })
        return proposal
    except Exception as e:
        tracing.end_trace(metadata={"error": str(e)})
        raise

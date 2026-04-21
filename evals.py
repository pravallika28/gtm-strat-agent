from models import WorkerProposal

VALID_STAGES = [
    "Prospecting", "Discovery", "Technical Validation",
    "Proposal", "Negotiation", "Closed Won", "Closed Lost",
]


def validate_confidence(proposal: WorkerProposal) -> tuple[bool, str]:
    # DETERMINISTIC: pure range check on a number
    # NOTE: the confidence value itself came from the model (non-deterministic upstream)
    # and is not statistically calibrated — treat it as a rough signal, not a probability
    if not 0.0 <= proposal.confidence <= 1.0:
        return False, f"Confidence {proposal.confidence} is out of range [0.0, 1.0]"
    if proposal.confidence < 0.4:
        return False, f"Low confidence ({proposal.confidence:.0%}) — review proposals carefully before committing"
    return True, f"Confidence {proposal.confidence:.0%}"


def validate_stage(proposal: WorkerProposal) -> tuple[bool, str]:
    # DETERMINISTIC: checks proposed stage value against a fixed allowlist
    # NOTE: the proposed_value came from the model — this eval catches hallucinated stage names
    for update in proposal.proposed_field_updates:
        if update.field == "stage" and update.proposed_value not in VALID_STAGES:
            return False, f"Invalid stage '{update.proposed_value}'. Valid: {VALID_STAGES}"
    return True, "OK"


def grounding_check(proposal: WorkerProposal, transcript: str) -> tuple[bool, str]:
    # DETERMINISTIC: keyword overlap check between model output and source transcript
    # Weak signal only — a commitment can share no 5-letter words with the transcript
    # and still be valid, or share words and still be hallucinated.
    # Does NOT catch fabricated facts that happen to reuse words from the transcript.
    transcript_lower = transcript.lower()
    ungrounded = []
    for commitment in proposal.commitments:
        keywords = [w for w in commitment.lower().split() if len(w) > 4]
        if keywords and not any(w in transcript_lower for w in keywords):
            ungrounded.append(commitment)
    if ungrounded:
        return False, f"Commitments may not be grounded in transcript: {ungrounded}"
    return True, "OK"


def run_evals(proposal: WorkerProposal, transcript: str) -> list[dict]:
    # DETERMINISTIC: runs all checks, all purely rule-based
    checks = [
        ("confidence", validate_confidence(proposal)),
        ("stage_validation", validate_stage(proposal)),
        ("grounding", grounding_check(proposal, transcript)),
    ]
    return [
        {"check": name, "passed": passed, "message": msg}
        for name, (passed, msg) in checks
    ]

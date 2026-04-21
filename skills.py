import json
from pathlib import Path

from models import (
    Opportunity, Account, Activity, Playbook,
    OpportunityBrief, WorkerProposal, ResearchResponse, GeneratedTranscript,
)
from llm import call_gemini

DATA_DIR = Path("data")


def _load_data():
    # DETERMINISTIC: reading static JSON files from disk
    opportunities = [Opportunity(**o) for o in json.loads((DATA_DIR / "opportunities.json").read_text())]
    accounts = [Account(**a) for a in json.loads((DATA_DIR / "accounts.json").read_text())]
    activities = [Activity(**a) for a in json.loads((DATA_DIR / "activities.json").read_text())]
    playbooks = [Playbook(**p) for p in json.loads((DATA_DIR / "playbooks.json").read_text())]
    return opportunities, accounts, activities, playbooks


def crm_research(query: str) -> ResearchResponse:
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
    result = call_gemini(prompt, ResearchResponse)
    return ResearchResponse(**result)


def research_agent(opportunity_id: str) -> OpportunityBrief:
    # DETERMINISTIC: loading and filtering data from local JSON files
    opportunities, accounts, activities, playbooks = _load_data()

    opp = next((o for o in opportunities if o.opportunity_id == opportunity_id), None)
    if not opp:
        raise ValueError(f"Opportunity {opportunity_id} not found")

    account = next((a for a in accounts if a.account_id == opp.account_id), None)
    opp_activities = [a for a in activities if a.opportunity_id == opportunity_id]
    playbook = next((p for p in playbooks if p.stage == opp.stage), None)

    # DETERMINISTIC: building a prompt string from structured data
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
    # ⚡ NON-DETERMINISTIC: delegates to Gemini via call_gemini().
    # The summary, health_signals, recommended_actions, risks, and whitespace
    # are all model-generated and will vary between runs.
    result = call_gemini(prompt, OpportunityBrief)

    # DETERMINISTIC: overwriting IDs with ground-truth values from local data
    # (prevents the model from hallucinating these)
    result["opportunity_id"] = opportunity_id
    result["account_name"] = account.account_name if account else ""
    return OpportunityBrief(**result)


def generate_transcript(opportunity_id: str) -> GeneratedTranscript:
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
    result = call_gemini(prompt, GeneratedTranscript)
    return GeneratedTranscript(**result)


def worker_agent(transcript: str, opportunity_id: str) -> WorkerProposal:
    # DETERMINISTIC: loading context data from local JSON files
    opportunities, accounts, _, _ = _load_data()
    progression_rules = json.loads((DATA_DIR / "progression_rules.json").read_text())

    opp = next((o for o in opportunities if o.opportunity_id == opportunity_id), None)
    account = next((a for a in accounts if a.account_id == opp.account_id), None) if opp else None

    # DETERMINISTIC: building prompt from transcript + CRM context + progression rules
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
- Set confidence (0.0–1.0) based on how clearly the transcript supports the proposals.
"""
    # ⚡ NON-DETERMINISTIC: delegates to Gemini via call_gemini().
    # proposed_field_updates (including stage) are now gated by progression_rules.
    result = call_gemini(prompt, WorkerProposal)
    return WorkerProposal(**result)

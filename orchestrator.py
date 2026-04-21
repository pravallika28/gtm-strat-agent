import json
from datetime import datetime
from pathlib import Path

from models import WorkerProposal, CommitLogEntry
from skills import crm_research, generate_transcript, research_agent, worker_agent

DATA_DIR = Path("data")


def run_crm_research(query: str):
    # ⚡ NON-DETERMINISTIC: calls crm_research → call_gemini → Gemini API
    return crm_research(query)


def run_research(opportunity_id: str):
    # ⚡ NON-DETERMINISTIC: calls research_agent → call_gemini → Gemini API
    return research_agent(opportunity_id)


def run_generate_transcript(opportunity_id: str):
    # ⚡ NON-DETERMINISTIC: calls generate_transcript → call_gemini → Gemini API
    return generate_transcript(opportunity_id)


def run_worker(transcript: str, opportunity_id: str):
    # ⚡ NON-DETERMINISTIC: calls worker_agent → call_gemini → Gemini API
    return worker_agent(transcript, opportunity_id)


def commit_changes(opportunity_id: str, proposal: WorkerProposal) -> CommitLogEntry:
    # DETERMINISTIC: all writes below use values already extracted by the model
    # (the non-determinism happened earlier in run_worker; this is just persistence)

    # Update opportunity fields
    opps = json.loads((DATA_DIR / "opportunities.json").read_text())
    for opp in opps:
        if opp["opportunity_id"] == opportunity_id:
            for update in proposal.proposed_field_updates:
                if update.field in opp:
                    opp[update.field] = update.proposed_value
    (DATA_DIR / "opportunities.json").write_text(json.dumps(opps, indent=2))

    # Append new tasks
    tasks = json.loads((DATA_DIR / "tasks.json").read_text())
    new_task_ids = []
    for i, pt in enumerate(proposal.proposed_tasks):
        task_id = f"TASK-{len(tasks) + i + 1:03d}"
        tasks.append({
            "task_id": task_id,
            "opportunity_id": opportunity_id,
            "description": pt.description,
            "due_date": pt.due_date,
            "status": "open",
        })
        new_task_ids.append(task_id)
    (DATA_DIR / "tasks.json").write_text(json.dumps(tasks, indent=2))

    # Log transcript analysis as activity
    activities = json.loads((DATA_DIR / "activities.json").read_text())
    activities.append({
        "opportunity_id": opportunity_id,
        "type": "transcript_analysis",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "summary": proposal.summary,
    })
    (DATA_DIR / "activities.json").write_text(json.dumps(activities, indent=2))

    # Write to commit log
    log = json.loads((DATA_DIR / "commit_log.json").read_text())
    entry_data = {
        "timestamp": datetime.now().isoformat(),
        "opportunity_id": opportunity_id,
        "changes": {u.field: u.proposed_value for u in proposal.proposed_field_updates},
        "tasks_added": new_task_ids,
    }
    log.append(entry_data)
    (DATA_DIR / "commit_log.json").write_text(json.dumps(log, indent=2))

    return CommitLogEntry(**entry_data)

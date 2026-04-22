from pydantic import BaseModel
from typing import Optional


class Stakeholder(BaseModel):
    name: str
    role: str


class Opportunity(BaseModel):
    opportunity_id: str
    account_id: str
    name: str
    stage: str
    amount: float
    close_date: str
    next_step: str
    owner: str
    products: list[str]
    forecast_category: str


class Account(BaseModel):
    account_id: str
    account_name: str
    industry: str
    installed_base: list[str]
    health_score: int
    renewal_date: str
    whitespace_products: list[str]
    stakeholders: list[Stakeholder]


class Activity(BaseModel):
    opportunity_id: str
    type: str
    date: str
    summary: str


class Playbook(BaseModel):
    motion: str
    stage: str
    recommended_actions: list[str]


class Task(BaseModel):
    task_id: str
    opportunity_id: str
    description: str
    due_date: Optional[str] = None
    status: str = "open"


class CommitLogEntry(BaseModel):
    timestamp: str
    opportunity_id: str
    changes: dict
    tasks_added: list[str]


# Agent output models

class ResearchResponse(BaseModel):
    answer: str
    key_findings: list[str]
    recommended_actions: list[str]
    risks: list[str]


class OpportunityBrief(BaseModel):
    opportunity_id: str
    account_name: str
    summary: str
    health_signals: list[str]
    recommended_actions: list[str]
    risks: list[str]
    whitespace: list[str]


class FieldUpdate(BaseModel):
    field: str
    current_value: str
    proposed_value: str
    reason: str


class ProposedTask(BaseModel):
    description: str
    due_date: Optional[str] = None


class JudgeEval(BaseModel):
    confidence: float        # calibrated 0.0–1.0 based on transcript evidence
    grounding_score: float   # fraction of proposals directly supported by transcript text
    hallucination_flag: bool # true if agent included facts/people not present in transcript
    reasoning: str           # brief explanation of the scores


class WorkerProposal(BaseModel):
    summary: str
    commitments: list[str]
    risks: list[str]
    stakeholder_mentions: list[str]
    proposed_field_updates: list[FieldUpdate]
    proposed_tasks: list[ProposedTask]
    follow_up_email: str
    confidence: float


class GeneratedTranscript(BaseModel):
    transcript: str

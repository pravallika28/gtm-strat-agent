"""
Lightweight structured tracing for the GTM agent.

Every agent invocation is a trace. Every LLM call inside it is a span.
All output goes to traces/traces.jsonl (one JSON object per line).

Designed to be easy to replace with LangSmith, Arize Phoenix, or
OpenTelemetry later — the call sites in skills.py and llm.py stay the same,
only this module needs to change.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

TRACES_DIR = Path("traces")
TRACES_FILE = TRACES_DIR / "traces.jsonl"

# Module-level trace context — one active trace at a time per process.
# For a multi-user server you'd swap this for a ContextVar.
_active_trace_id: Optional[str] = None
_active_trace_name: Optional[str] = None


def start_trace(name: str, metadata: dict = None) -> str:
    """Open a new trace. Returns trace_id. Call once per agent invocation."""
    global _active_trace_id, _active_trace_name
    _active_trace_id = str(uuid.uuid4())[:8]
    _active_trace_name = name
    _append({
        "type": "trace_start",
        "trace_id": _active_trace_id,
        "trace_name": name,
        "timestamp": _now(),
        "metadata": metadata or {},
    })
    return _active_trace_id


def end_trace(metadata: dict = None):
    """Close the active trace."""
    global _active_trace_id, _active_trace_name
    _append({
        "type": "trace_end",
        "trace_id": _active_trace_id,
        "trace_name": _active_trace_name,
        "timestamp": _now(),
        "metadata": metadata or {},
    })
    _active_trace_id = None
    _active_trace_name = None


def log_span(
    span_name: str,
    inputs: dict,
    outputs: dict,
    latency_ms: float,
    metadata: dict = None,
    error: str = None,
):
    """Record a single LLM call span inside the active trace."""
    entry = {
        "type": "span",
        "trace_id": _active_trace_id,
        "trace_name": _active_trace_name,
        "span_name": span_name,
        "timestamp": _now(),
        "latency_ms": round(latency_ms),
        "inputs": inputs,
        "outputs": outputs,
        "metadata": metadata or {},
    }
    if error:
        entry["error"] = error
    _append(entry)


def _now() -> str:
    return datetime.now().isoformat()


def _append(entry: dict):
    TRACES_DIR.mkdir(exist_ok=True)
    with TRACES_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")

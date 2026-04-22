"""
CLI viewer for traces/traces.jsonl

Usage (run from repo root):
    python3 traces/view_traces.py              # show all traces, most recent last
    python3 traces/view_traces.py --last 5     # show last N traces
    python3 traces/view_traces.py --id abc123  # show one trace by ID
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

TRACES_FILE = Path("traces/traces.jsonl")


def load_traces() -> list[dict]:
    if not TRACES_FILE.exists():
        print("No traces file found. Run the app first.")
        sys.exit(0)
    entries = []
    with TRACES_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def group_by_trace(entries: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for e in entries:
        groups[e.get("trace_id", "unknown")].append(e)
    return groups


def print_trace(trace_id: str, events: list[dict]):
    start = next((e for e in events if e["type"] == "trace_start"), None)
    end = next((e for e in events if e["type"] == "trace_end"), None)
    spans = [e for e in events if e["type"] == "span"]

    name = start["trace_name"] if start else "?"
    ts = start["timestamp"][:19].replace("T", " ") if start else "?"
    meta_start = start.get("metadata", {}) if start else {}
    meta_end = end.get("metadata", {}) if end else {}

    total_latency = sum(s.get("latency_ms", 0) for s in spans)
    total_tokens = sum(
        (s.get("metadata", {}).get("total_tokens") or 0) for s in spans
    )

    print(f"\n{'━' * 64}")
    print(f"  Trace  {trace_id}  |  {name}  |  {ts}")
    print(f"  Spans: {len(spans)}  |  Total latency: {total_latency:,.0f} ms  |  Tokens: {total_tokens or '—'}")

    if meta_start:
        print(f"  Input:  {meta_start}")
    if meta_end:
        label = "Error:" if "error" in meta_end else "Output:"
        print(f"  {label} {meta_end}")

    if spans:
        print(f"\n  {'Span':<32} {'Latency':>9}  {'In tok':>7}  {'Out tok':>8}  {'Status'}")
        print(f"  {'─' * 62}")
        for s in spans:
            m = s.get("metadata", {})
            in_tok = m.get("input_tokens") or "—"
            out_tok = m.get("output_tokens") or "—"
            status = "ERROR" if s.get("error") else "ok"
            print(f"  {s['span_name']:<32} {s['latency_ms']:>8.0f}ms  {str(in_tok):>7}  {str(out_tok):>8}  {status}")


def main():
    args = sys.argv[1:]
    entries = load_traces()
    groups = group_by_trace(entries)

    # Preserve insertion order (trace_start timestamp)
    ordered_ids = list(dict.fromkeys(
        e["trace_id"] for e in entries if "trace_id" in e
    ))

    target_id = None
    last_n = None

    i = 0
    while i < len(args):
        if args[i] == "--id" and i + 1 < len(args):
            target_id = args[i + 1]
            i += 2
        elif args[i] == "--last" and i + 1 < len(args):
            last_n = int(args[i + 1])
            i += 2
        else:
            i += 1

    if target_id:
        if target_id not in groups:
            print(f"Trace '{target_id}' not found.")
            sys.exit(1)
        print_trace(target_id, groups[target_id])
    else:
        ids = ordered_ids[-last_n:] if last_n else ordered_ids
        print(f"\nShowing {len(ids)} of {len(ordered_ids)} trace(s) in {TRACES_FILE}")
        for tid in ids:
            print_trace(tid, groups[tid])

    print()


if __name__ == "__main__":
    main()

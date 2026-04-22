import json
import streamlit as st
import pandas as pd
from pathlib import Path
from collections import defaultdict

from orchestrator import run_crm_research, run_generate_transcript, run_research, run_worker, commit_changes
from evals import run_evals

TRACES_FILE = Path("traces/traces.jsonl")
RESULTS_DIR = Path("evals/results")
DATA_DIR    = Path("data")

SCORERS = [
    "stage_progression", "required_fields", "task_generation",
    "commitment_keywords", "hallucination", "confidence", "llm_judge",
]

st.set_page_config(page_title="GTM Agent Dashboard", layout="wide")
st.title("GTM Agent — Testing Dashboard")


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_trace_entries():
    if not TRACES_FILE.exists():
        return []
    entries = []
    with TRACES_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries


def build_trace_df(entries):
    groups = defaultdict(list)
    for e in entries:
        tid = e.get("trace_id")
        if tid:
            groups[tid].append(e)

    rows = []
    for tid, events in groups.items():
        start = next((e for e in events if e["type"] == "trace_start"), None)
        end   = next((e for e in events if e["type"] == "trace_end"), None)
        spans = [e for e in events if e["type"] == "span"]

        total_latency = sum(s.get("latency_ms", 0) for s in spans)
        total_tokens  = sum((s.get("metadata", {}).get("total_tokens") or 0) for s in spans)
        has_error     = any(s.get("error") for s in spans)

        rows.append({
            "trace_id":    tid,
            "agent":       start["trace_name"] if start else "unknown",
            "timestamp":   start["timestamp"][:19].replace("T", " ") if start else "",
            "latency_ms":  round(total_latency),
            "total_tokens": total_tokens or None,
            "span_count":  len(spans),
            "status":      "ERROR" if has_error else "ok",
            "output_meta": end.get("metadata", {}) if end else {},
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df["timestamp"] != ""].sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


def load_eval_runs():
    if not RESULTS_DIR.exists():
        return []
    runs = []
    for f in sorted(RESULTS_DIR.glob("2*.json")):
        try:
            runs.append(json.loads(f.read_text()))
        except Exception:
            pass
    if not runs:
        latest = RESULTS_DIR / "latest.json"
        if latest.exists():
            try:
                runs.append(json.loads(latest.read_text()))
            except Exception:
                pass
    return runs


def color_score(val):
    if pd.isna(val):
        return "background-color: #f0f0f0; color: #999"
    if val >= 0.9:
        return "background-color: #c6efce; color: #276221"
    if val >= 0.5:
        return "background-color: #ffeb9c; color: #9c5700"
    return "background-color: #ffc7ce; color: #9c0006"


# ── Load data once ────────────────────────────────────────────────────────────

runs     = load_eval_runs()
entries  = load_trace_entries()
trace_df = build_trace_df(entries) if entries else pd.DataFrame()

tab1, tab2, tab3, tab4 = st.tabs(["📋  Eval Scorecard", "🔍  Trace Explorer", "⚠️  Improvement Signals", "🤖  GTM Agent"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Eval Scorecard
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    if not runs:
        st.info("No eval results yet. Run `python3 evals/run_evals.py` to generate them.")
    else:
        latest = runs[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Score",  f"{latest['overall_score']:.0%}")
        c2.metric("Passed",         f"{latest['passed']}/{latest['total']}")
        c3.metric("Pass Threshold", f"{latest['pass_threshold']:.0%}")
        c4.metric("Runs Recorded",  len(runs))

        st.divider()

        # ── Heatmap ───────────────────────────────────────────────────────
        st.subheader("Latest Run — Scores by Dimension")

        score_rows = []
        for r in latest["results"]:
            row = {
                "eval":    r["id"],
                "status":  r["status"],
                "overall": r["overall"],
            }
            for s in SCORERS:
                row[s] = r["scores"].get(s, {}).get("score")
            score_rows.append(row)

        score_df = pd.DataFrame(score_rows).set_index("eval")
        num_cols = ["overall"] + SCORERS

        styled = (
            score_df[["status"] + num_cols]
            .style
            .map(color_score, subset=num_cols)
            .format({c: "{:.0%}" for c in num_cols}, na_rep="—")
        )
        st.dataframe(styled, use_container_width=True)

        # ── Scorer messages ───────────────────────────────────────────────
        with st.expander("Scorer messages"):
            for r in latest["results"]:
                st.markdown(f"**{r['id']}** — _{r.get('description', '')[:70]}_")
                for scorer in SCORERS:
                    s = r["scores"].get(scorer, {})
                    score = s.get("score", 0)
                    icon = "✓" if score >= 0.9 else ("~" if score >= 0.5 else "✗")
                    st.markdown(f"&nbsp;&nbsp;{icon} `{scorer}` {score:.0%} — {s.get('message', '—')}")
                st.markdown("---")

        # ── Trend ─────────────────────────────────────────────────────────
        if len(runs) > 1:
            st.subheader("Score Trend Across Runs")
            trend_rows = []
            for run in runs:
                ts = run["timestamp"][:16].replace("T", " ")
                for r in run["results"]:
                    trend_rows.append({"run": ts, "eval": r["id"], "score": r["overall"]})
            pivot = (
                pd.DataFrame(trend_rows)
                .pivot(index="run", columns="eval", values="score")
            )
            st.line_chart(pivot)
        else:
            st.caption("Run evals more than once to see trend charts.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Trace Explorer
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    if trace_df.empty:
        st.info("No traces yet. Use the app or run evals, then refresh.")
    else:
        n_errors = int((trace_df["status"] == "ERROR").sum())
        ok_df    = trace_df[trace_df["status"] == "ok"]
        avg_lat  = ok_df["latency_ms"].mean() if not ok_df.empty else None
        total_tok = trace_df["total_tokens"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Traces", len(trace_df))
        c2.metric("Errors",       n_errors)
        c3.metric("Avg Latency",  f"{avg_lat/1000:.1f}s" if avg_lat else "—")
        c4.metric("Total Tokens", f"{int(total_tok):,}" if total_tok else "—")

        st.divider()

        # ── Per-agent stats ───────────────────────────────────────────────
        st.subheader("Per-Agent Performance")
        if not ok_df.empty:
            agent_stats = (
                ok_df.groupby("agent")
                .agg(
                    calls        = ("trace_id",    "count"),
                    avg_lat_s    = ("latency_ms",  lambda x: round(x.mean() / 1000, 1)),
                    p95_lat_s    = ("latency_ms",  lambda x: round(x.quantile(0.95) / 1000, 1)),
                    avg_tokens   = ("total_tokens", lambda x: int(x.mean()) if x.notna().any() else None),
                )
                .reset_index()
            )
            st.dataframe(agent_stats, use_container_width=True, hide_index=True)

            st.subheader("Avg Latency by Agent (seconds)")
            lat_chart = ok_df.groupby("agent")["latency_ms"].mean() / 1000
            st.bar_chart(lat_chart)

        # ── Recent traces table ───────────────────────────────────────────
        st.subheader("Recent Traces")
        disp = trace_df[["trace_id", "agent", "timestamp", "latency_ms", "total_tokens", "status"]].copy()
        disp["latency_s"] = (disp["latency_ms"] / 1000).round(1)
        disp = disp.drop(columns=["latency_ms"])

        def row_style(row):
            if row["status"] == "ERROR":
                return ["background-color: #ffc7ce"] * len(row)
            return [""] * len(row)

        st.dataframe(
            disp.style.apply(row_style, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # ── Error details ─────────────────────────────────────────────────
        error_spans = [e for e in entries if e.get("type") == "span" and e.get("error")]
        if error_spans:
            with st.expander(f"Error details — {len(error_spans)} error(s)"):
                for s in error_spans[-10:]:
                    st.markdown(f"**{s['span_name']}** @ {s['timestamp'][:19]}")
                    err = s["error"][:300] + ("..." if len(s["error"]) > 300 else "")
                    st.code(err)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Improvement Signals
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    if not runs:
        st.info("No eval results yet.")
    else:
        latest = runs[-1]

        # ── Confidence calibration ────────────────────────────────────────
        st.subheader("Confidence Calibration")
        st.caption("Worker self-reported confidence vs. judge score — gap > 30% signals a calibration issue.")

        cal_rows = []
        for r in latest["results"]:
            worker_conf = r.get("worker_confidence")
            judge_score = r["scores"].get("llm_judge", {}).get("score")
            if worker_conf is not None and judge_score is not None:
                gap  = worker_conf - judge_score
                flag = "Overconfident 🔴" if gap > 0.3 else ("Underconfident 🟡" if gap < -0.3 else "OK 🟢")
                cal_rows.append({
                    "eval":       r["id"],
                    "worker":     worker_conf,
                    "judge":      judge_score,
                    "gap":        round(gap, 2),
                    "signal":     flag,
                })

        if cal_rows:
            cal_df = pd.DataFrame(cal_rows)
            st.dataframe(
                cal_df.style.format({"worker": "{:.0%}", "judge": "{:.0%}", "gap": "{:+.0%}"}),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        # ── Scorer failure frequency ──────────────────────────────────────
        st.subheader("Scorer Failure Rate (across all runs)")
        st.caption("Which dimensions fail most — these are your highest-leverage prompt fixes.")

        fail_counts  = defaultdict(int)
        total_counts = defaultdict(int)
        for run in runs:
            for r in run["results"]:
                for scorer, s in r["scores"].items():
                    total_counts[scorer] += 1
                    if s.get("score", 1.0) < 0.9:
                        fail_counts[scorer] += 1

        if total_counts:
            fail_rows = []
            for scorer in SCORERS:
                total = total_counts.get(scorer, 0)
                fails = fail_counts.get(scorer, 0)
                fail_rows.append({
                    "scorer":    scorer,
                    "fail_rate": fails / total if total else 0,
                    "failures":  fails,
                    "total":     total,
                })
            fail_df = pd.DataFrame(fail_rows).sort_values("fail_rate", ascending=False)
            st.bar_chart(fail_df.set_index("scorer")["fail_rate"])
            st.dataframe(
                fail_df.style.format({"fail_rate": "{:.0%}"}),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        # ── Latency hotspots ──────────────────────────────────────────────
        if not trace_df.empty:
            st.subheader("Latency Hotspots")
            ok_df = trace_df[trace_df["status"] == "ok"]
            if not ok_df.empty:
                slow = ok_df.nlargest(5, "latency_ms")[["agent", "timestamp", "latency_ms"]].copy()
                slow["latency_s"] = (slow["latency_ms"] / 1000).round(1)
                slow = slow.drop(columns=["latency_ms"])
                st.dataframe(slow, use_container_width=True, hide_index=True)

        st.divider()

        # ── Judge reasoning on failures ───────────────────────────────────
        st.subheader("Judge Reasoning on Failures")
        any_shown = False
        for r in latest["results"]:
            judge_score  = r["scores"].get("llm_judge", {}).get("score", 1.0)
            judge_detail = r["scores"].get("llm_judge", {}).get("detail", {})
            if judge_score < 1.0 and judge_detail.get("reasoning"):
                any_shown = True
                with st.expander(f"{r['id']} — judge {judge_score:.0%}  |  worker conf {r.get('worker_confidence', 0):.0%}"):
                    cols = st.columns(3)
                    cols[0].metric("Confidence",   f"{judge_detail.get('confidence', 0):.0%}")
                    cols[1].metric("Grounding",    f"{judge_detail.get('grounding_score', 0):.0%}")
                    cols[2].metric("Hallucination",str(judge_detail.get('hallucination_flag', False)))
                    st.markdown(f"**Reasoning:** {judge_detail.get('reasoning', '—')}")
        if not any_shown:
            st.success("All evals passed the judge — nothing to flag.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — GTM Agent
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    opportunities = json.loads((DATA_DIR / "opportunities.json").read_text())
    opp_map = {o["opportunity_id"]: f"{o['name']} ({o['stage']})" for o in opportunities}

    selected_id = st.selectbox(
        "Opportunity",
        options=list(opp_map.keys()),
        format_func=lambda x: f"{x} — {opp_map[x]}",
        key="agent_opp_select",
    )

    st.divider()

    # ── CRM Research Agent ────────────────────────────────────────────────────
    st.subheader("CRM Research Agent")

    crm_query = st.text_input(
        "Ask a question across all CRM data",
        placeholder="e.g. Which accounts have low health scores? What are all open tasks?",
        key="agent_crm_query",
    )

    if st.button("Ask CRM Agent", key="agent_crm_btn") and crm_query.strip():
        with st.spinner("Researching CRM data..."):
            try:
                st.session_state["agent_crm_response"] = run_crm_research(crm_query)
            except Exception as e:
                st.error(str(e))

    if "agent_crm_response" in st.session_state:
        res = st.session_state["agent_crm_response"]
        st.markdown(f"**Answer:** {res.answer}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Key Findings**")
            for f in res.key_findings:
                st.markdown(f"- {f}")
            st.markdown("**Recommended Actions**")
            for a in res.recommended_actions:
                st.markdown(f"- {a}")
        with col2:
            st.markdown("**Risks**")
            for r in res.risks:
                st.markdown(f"- {r}")

    st.divider()

    # ── Opportunity Brief ─────────────────────────────────────────────────────
    st.subheader("Opportunity Brief")

    if st.button("Generate Opportunity Brief", key="agent_brief_btn"):
        with st.spinner("Running research agent..."):
            try:
                st.session_state["agent_brief"] = run_research(selected_id)
            except Exception as e:
                st.error(str(e))

    if "agent_brief" in st.session_state:
        brief = st.session_state["agent_brief"]
        st.markdown(f"**Summary:** {brief.summary}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Health Signals**")
            for s in brief.health_signals:
                st.markdown(f"- {s}")
            st.markdown("**Recommended Actions**")
            for a in brief.recommended_actions:
                st.markdown(f"- {a}")
        with col2:
            st.markdown("**Risks**")
            for r in brief.risks:
                st.markdown(f"- {r}")
            st.markdown("**Whitespace**")
            for w in brief.whitespace:
                st.markdown(f"- {w}")

    st.divider()

    # ── Post-Call Activities ──────────────────────────────────────────────────
    st.subheader("Post-Call Activities")

    if st.button("Generate Sample Transcript", key="agent_transcript_btn"):
        with st.spinner("Simulating call..."):
            try:
                result = run_generate_transcript(selected_id)
                st.session_state["agent_generated_transcript"] = result.transcript
            except Exception as e:
                st.error(str(e))

    if "agent_generated_transcript" in st.session_state:
        st.caption("Generated transcript — copy and paste into the box below, edit as needed.")
        st.text_area(
            "Generated transcript",
            value=st.session_state["agent_generated_transcript"],
            height=300,
            key="agent_gen_preview",
        )

    transcript = st.text_area("Paste call transcript here", height=220, key="agent_transcript_input")

    if st.button("Process Post-Call", type="primary", key="agent_process_btn") and transcript.strip():
        with st.spinner("Analyzing call..."):
            try:
                proposal = run_worker(transcript, selected_id)
                st.session_state["agent_proposal"]    = proposal
                st.session_state["agent_transcript"]  = transcript
                st.session_state["agent_evals"]       = run_evals(proposal, transcript)
            except Exception as e:
                st.error(str(e))

    if "agent_proposal" in st.session_state:
        proposal = st.session_state["agent_proposal"]
        evals    = st.session_state.get("agent_evals", [])

        for ev in evals:
            (st.success if ev["passed"] else st.warning)(
                f"{'✓' if ev['passed'] else '⚠'} {ev['check']}: {ev['message']}"
            )

        st.markdown(f"**Call Summary:** {proposal.summary}")
        st.markdown(f"**Confidence:** {proposal.confidence:.0%}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Commitments Made**")
            for c in proposal.commitments:
                st.markdown(f"- {c}")
            st.markdown("**Stakeholders Mentioned**")
            for s in proposal.stakeholder_mentions:
                st.markdown(f"- {s}")
        with col2:
            st.markdown("**Risks Identified**")
            for r in proposal.risks:
                st.markdown(f"- {r}")

        with st.expander("Draft Follow-Up Email"):
            st.text(proposal.follow_up_email)

        st.divider()
        st.markdown("### Review & Confirm Updates")
        st.caption("Check the items you want to apply, then click Apply Selected Changes.")

        st.markdown("**CRM Field Updates**")
        field_checks = []
        if proposal.proposed_field_updates:
            for i, u in enumerate(proposal.proposed_field_updates):
                checked = st.checkbox(
                    f"`{u.field}`: ~~{u.current_value}~~ → **{u.proposed_value}**  \n_{u.reason}_",
                    value=True,
                    key=f"agent_field_{i}",
                )
                field_checks.append(checked)
        else:
            st.markdown("_No field updates proposed._")

        st.markdown("**Proposed Tasks**")
        task_checks = []
        if proposal.proposed_tasks:
            for i, t in enumerate(proposal.proposed_tasks):
                due = f" (due {t.due_date})" if t.due_date else ""
                checked = st.checkbox(
                    f"{t.description}{due}",
                    value=True,
                    key=f"agent_task_{i}",
                )
                task_checks.append(checked)
        else:
            st.markdown("_No tasks proposed._")

        st.divider()
        if st.button("Apply Selected Changes", type="primary", key="agent_apply_btn"):
            selected_proposal = proposal.model_copy(update={
                "proposed_field_updates": [
                    u for u, checked in zip(proposal.proposed_field_updates, field_checks) if checked
                ],
                "proposed_tasks": [
                    t for t, checked in zip(proposal.proposed_tasks, task_checks) if checked
                ],
            })

            if not selected_proposal.proposed_field_updates and not selected_proposal.proposed_tasks:
                st.warning("No items selected — nothing to apply.")
            else:
                with st.spinner("Applying changes..."):
                    try:
                        entry = commit_changes(selected_id, selected_proposal)
                        st.success(f"Changes applied at {entry.timestamp}")
                        st.json(entry.model_dump())
                        del st.session_state["agent_proposal"]
                    except Exception as e:
                        st.error(str(e))

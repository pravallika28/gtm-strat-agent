import json
import streamlit as st
from pathlib import Path

from orchestrator import run_crm_research, run_generate_transcript, run_research, run_worker, commit_changes
from evals import run_evals

DATA_DIR = Path("data")

st.set_page_config(page_title="GTM Strat Agent", layout="wide")
st.title("GTM Strat Agent V1")

# DETERMINISTIC: reading static JSON from disk on every page load
opportunities = json.loads((DATA_DIR / "opportunities.json").read_text())
opp_map = {o["opportunity_id"]: f"{o['name']} ({o['stage']})" for o in opportunities}

selected_id = st.selectbox(
    "Opportunity",
    options=list(opp_map.keys()),
    format_func=lambda x: f"{x} — {opp_map[x]}",
)

st.divider()

# ── CRM Research Agent ──────────────────────────────────────────────────────
st.subheader("CRM Research Agent")

crm_query = st.text_input(
    "Ask a question across all CRM data",
    placeholder="e.g. Which accounts have low health scores? What are all open tasks?",
)

if st.button("Ask CRM Agent") and crm_query.strip():
    with st.spinner("Researching CRM data..."):
        try:
            # ⚡ NON-DETERMINISTIC: Gemini API call across all CRM data
            st.session_state["crm_response"] = run_crm_research(crm_query)
        except Exception as e:
            st.error(str(e))

if "crm_response" in st.session_state:
    res = st.session_state["crm_response"]
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

# ── Opportunity Brief ───────────────────────────────────────────────────────
st.subheader("Opportunity Brief")

if st.button("Generate Opportunity Brief"):
    with st.spinner("Running research agent..."):
        try:
            # ⚡ NON-DETERMINISTIC: Gemini API call
            st.session_state["brief"] = run_research(selected_id)
        except Exception as e:
            st.error(str(e))

if "brief" in st.session_state:
    brief = st.session_state["brief"]
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

# ── Post-Call Activities ────────────────────────────────────────────────────
st.subheader("Post-Call Activities")

if st.button("Generate Sample Transcript"):
    with st.spinner("Simulating call..."):
        try:
            # ⚡ NON-DETERMINISTIC: Gemini generates a simulated sales conversation
            result = run_generate_transcript(selected_id)
            st.session_state["generated_transcript"] = result.transcript
        except Exception as e:
            st.error(str(e))

if "generated_transcript" in st.session_state:
    st.caption("Generated transcript — copy and paste into the box below, edit as needed.")
    st.text_area("Generated transcript", value=st.session_state["generated_transcript"], height=300, key="gen_preview")

transcript = st.text_area("Paste call transcript here", height=220, key="transcript_input")

if st.button("Process Post-Call", type="primary") and transcript.strip():
    with st.spinner("Analyzing call..."):
        try:
            # ⚡ NON-DETERMINISTIC: Gemini API call
            proposal = run_worker(transcript, selected_id)
            st.session_state["proposal"] = proposal
            st.session_state["transcript"] = transcript
            st.session_state["evals"] = run_evals(proposal, transcript)
        except Exception as e:
            st.error(str(e))

if "proposal" in st.session_state:
    proposal = st.session_state["proposal"]
    evals = st.session_state.get("evals", [])

    # Rule-based eval badges
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

    # ── Per-item checkboxes for field updates ───────────────────────────────
    st.markdown("**CRM Field Updates**")
    field_checks = []
    if proposal.proposed_field_updates:
        for i, u in enumerate(proposal.proposed_field_updates):
            checked = st.checkbox(
                f"`{u.field}`: ~~{u.current_value}~~ → **{u.proposed_value}**  \n_{u.reason}_",
                value=True,
                key=f"field_{i}",
            )
            field_checks.append(checked)
    else:
        st.markdown("_No field updates proposed._")

    # ── Per-item checkboxes for tasks ───────────────────────────────────────
    st.markdown("**Proposed Tasks**")
    task_checks = []
    if proposal.proposed_tasks:
        for i, t in enumerate(proposal.proposed_tasks):
            due = f" (due {t.due_date})" if t.due_date else ""
            checked = st.checkbox(
                f"{t.description}{due}",
                value=True,
                key=f"task_{i}",
            )
            task_checks.append(checked)
    else:
        st.markdown("_No tasks proposed._")

    st.divider()
    if st.button("Apply Selected Changes", type="primary"):
        # Build a filtered proposal from only the checked items
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
                    # DETERMINISTIC: writes only selected items to JSON files
                    entry = commit_changes(selected_id, selected_proposal)
                    st.success(f"Changes applied at {entry.timestamp}")
                    st.json(entry.model_dump())
                    del st.session_state["proposal"]
                except Exception as e:
                    st.error(str(e))

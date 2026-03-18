"""
app.py — Guardian AI: Agentic Accountant
Main Streamlit application — Streamlit Cloud deployment.
"""

import json
import base64
import datetime
import uuid

import pandas as pd
import streamlit as st
import plotly.io as pio

from state import (
    AgentState, ConversationMessage, HITLQuestion,
    ConfidenceLevel, FinancialDocument, DocumentType,
)
from extractor   import extract_document
from validator   import validate_document, get_confidence_level, summarise_flags
from analyst import run_analysis, chat_with_analyst, documents_to_dataframe
from amazon_recon import reconcile_mtr_settlement, leakage_waterfall_json, fee_pie_json
from ppt_gen     import create_presentation

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Guardian AI — Agentic Accountant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

/* Dark base */
.stApp { background-color: #0D0D1A; color: #E0E0E0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122A 0%, #0D0D1A 100%);
    border-right: 1px solid #1E1E3A;
}

/* Metric cards */
.metric-card {
    background: #1A1A30;
    border: 1px solid #2A2A4A;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #00B48A; }
.metric-label { font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 0.08em; }

/* Alert boxes */
.alert-green {
    background: rgba(0,180,138,0.1);
    border-left: 4px solid #00B48A;
    padding: 0.8rem 1rem;
    border-radius: 6px;
    margin: 0.5rem 0;
}
.alert-orange {
    background: rgba(255,107,53,0.1);
    border-left: 4px solid #FF6B35;
    padding: 0.8rem 1rem;
    border-radius: 6px;
    margin: 0.5rem 0;
}
.alert-blue {
    background: rgba(75,156,211,0.1);
    border-left: 4px solid #4B9CD3;
    padding: 0.8rem 1rem;
    border-radius: 6px;
    margin: 0.5rem 0;
}

/* Chat bubbles */
.chat-user {
    background: #1E1E38;
    border-radius: 12px 12px 4px 12px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    max-width: 80%;
    float: right;
    clear: both;
}
.chat-ai {
    background: rgba(0,180,138,0.08);
    border: 1px solid rgba(0,180,138,0.2);
    border-radius: 12px 12px 12px 4px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    max-width: 85%;
    float: left;
    clear: both;
}
.chat-wrap { overflow: hidden; }

/* Pill badges */
.badge-green {
    background: #00B48A; color: #000; border-radius: 20px;
    padding: 2px 12px; font-size: 0.78rem; font-weight: 600;
}
.badge-orange {
    background: #FF6B35; color: #fff; border-radius: 20px;
    padding: 2px 12px; font-size: 0.78rem; font-weight: 600;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00B48A, #007A5E);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover { filter: brightness(1.15); transform: translateY(-1px); }

/* Tab styling */
.stTabs [data-baseweb="tab"] { font-weight: 600; }
.stTabs [aria-selected="true"] { color: #00B48A !important; border-bottom-color: #00B48A !important; }

/* Upload zone */
[data-testid="stFileUploader"] {
    border: 2px dashed #2A2A4A;
    border-radius: 12px;
    padding: 1rem;
}

/* Code / mono */
code, .mono { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ── Session state bootstrap ───────────────────────────────────────────────────

def _init_state():
    if "agent" not in st.session_state:
        st.session_state.agent = AgentState(
            session_id=str(uuid.uuid4())[:8]
        )
    if "chat_input_key" not in st.session_state:
        st.session_state.chat_input_key = 0


_init_state()
agent: AgentState = st.session_state.agent


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ Guardian AI")
    st.caption(f"Session `{agent.session_id}`")
    st.divider()

    st.markdown("### Pipeline Status")
    steps = [
        ("1 · Upload",    "upload"),
        ("2 · Extract",   "extracting"),
        ("3 · Validate",  "validating"),
        ("4 · Analyse",   "analysing"),
        ("5 · Report",    "done"),
    ]
    step_icons = {"upload": "⬜", "extracting": "⬜", "validating": "⬜",
                  "analysing": "⬜", "done": "⬜"}
    # highlight current + done
    order = [s[1] for s in steps]
    current_idx = order.index(agent.current_step) if agent.current_step in order else 0
    for i, (label, key) in enumerate(steps):
        icon = "✅" if i < current_idx else ("🔵" if i == current_idx else "⬜")
        st.markdown(f"{icon} {label}")

    st.divider()

    # Documents loaded
    if agent.documents:
        st.markdown(f"### 📂 Loaded ({len(agent.documents)})")
        for doc in agent.documents:
            flag_summary = summarise_flags(doc)
            conf_color = "🟢" if doc.overall_confidence > 0.9 else "🟡" if doc.overall_confidence > 0.6 else "🔴"
            st.markdown(
                f"{conf_color} `{doc.filename or doc.doc_type.value}` "
                f"— conf: **{doc.overall_confidence:.2f}**"
            )

    st.divider()
    if st.button("🗑️ Clear Session"):
        st.session_state.agent = AgentState(session_id=str(uuid.uuid4())[:8])
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:1.2rem 0 0.5rem 0'>
  <h1 style='margin:0;color:#00B48A;font-size:2.2rem;font-weight:700'>
    🛡️ Guardian AI <span style='font-size:1rem;color:#666;font-weight:400'>— Agentic Accountant</span>
  </h1>
  <p style='margin:0;color:#888;font-size:0.95rem'>
    Universal input · Forensic extraction · LangGraph confidence gate · Auto-charts · PPT export
  </p>
</div>
""", unsafe_allow_html=True)
st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_upload, tab_validate, tab_analyse, tab_amazon, tab_report, tab_chat = st.tabs([
    "📤 Upload & Extract",
    "🔍 Validate",
    "📊 Analyse",
    "🛒 Amazon Recon",
    "📑 Report & Export",
    "💬 AI Chat",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Extract
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader("Universal Document Input")
    st.markdown(
        "Upload any financial document — CSV, Excel, PDF, receipts, "
        "WhatsApp photos, bank statements."
    )

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["csv", "xlsx", "xls", "pdf", "jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        extract_btn = st.button("⚡ Extract All Documents", use_container_width=True)
    with col2:
        if st.button("Clear Docs"):
            agent.documents = []
            agent.current_step = "upload"
            st.rerun()

    if extract_btn and uploaded_files:
        agent.current_step = "extracting"
        progress = st.progress(0, text="Starting extraction…")
        new_docs = []

        for i, f in enumerate(uploaded_files):
            progress.progress((i) / len(uploaded_files), text=f"Extracting {f.name}…")
            doc = extract_document(f.read(), f.name)
            new_docs.append(doc)

        # Deduplicate by filename
        existing_names = {d.filename for d in agent.documents}
        added = [d for d in new_docs if d.filename not in existing_names]
        agent.documents.extend(added)

        progress.progress(1.0, text="Extraction complete ✅")
        agent.current_step = "validating"
        st.rerun()

    # Show extraction results
    if agent.documents:
        st.markdown("---")
        st.markdown(f"### Extracted Documents ({len(agent.documents)})")
        for doc in agent.documents:
            with st.expander(
                f"{'✅' if not doc.extraction_errors else '❌'}  "
                f"{doc.filename or doc.doc_type.value}  "
                f"— {len(doc.line_items)} items  "
                f"| confidence {doc.overall_confidence:.0%}",
                expanded=False,
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Line Items",  len(doc.line_items))
                c2.metric("Confidence",  f"{doc.overall_confidence:.0%}")
                c3.metric("Total",       f"₹{doc.total_amount:,.0f}" if doc.total_amount else "—")

                if doc.nulls_flagged:
                    st.markdown(
                        f'<div class="alert-orange">⚠️ Missing fields: {", ".join(doc.nulls_flagged)}</div>',
                        unsafe_allow_html=True,
                    )
                if doc.extraction_errors:
                    st.error(f"Extraction errors: {'; '.join(doc.extraction_errors)}")

                if doc.line_items:
                    df_preview = pd.DataFrame([it.model_dump() for it in doc.line_items[:10]])
                    st.dataframe(df_preview, use_container_width=True)
    else:
        st.markdown(
            '<div class="alert-blue">ℹ️ No documents loaded yet. Upload files above to begin.</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Validate (LangGraph confidence gate)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_validate:
    st.subheader("LangGraph Confidence Gate")
    st.markdown(
        "Each document is scored 0–1. "
        "Score > 0.9 → **straight to analysis**. "
        "Score ≤ 0.9 → **HITL loop** (one targeted question)."
    )

    if not agent.documents:
        st.info("Upload documents first.")
    else:
        if st.button("🔍 Run Validation", use_container_width=True):
            agent.pending_hitl = []
            for i, doc in enumerate(agent.documents):
                validated_doc, questions = validate_document(doc)
                agent.documents[i] = validated_doc
                agent.pending_hitl.extend(questions)
            agent.current_step = "validating"
            st.rerun()

        # Overall confidence summary
        if agent.documents:
            avg_conf = sum(d.overall_confidence for d in agent.documents) / len(agent.documents)
            gate = "✅ HIGH CONFIDENCE — proceed to analysis" if avg_conf > 0.9 else "⚠️ LOW CONFIDENCE — HITL required"
            badge_class = "alert-green" if avg_conf > 0.9 else "alert-orange"
            st.markdown(
                f'<div class="{badge_class}"><b>{gate}</b>  |  Average confidence: {avg_conf:.0%}</div>',
                unsafe_allow_html=True,
            )

        # HITL questions
        if agent.pending_hitl:
            st.markdown("### 🙋 Human-in-the-Loop Questions")
            st.caption("UX rule: max 1 question per turn — framed as intelligence, never as weakness.")

            for idx, q in enumerate(agent.pending_hitl):
                if not q.answered:
                    with st.form(key=f"hitl_{idx}"):
                        st.markdown(
                            f'<div class="alert-blue">🤔  <b>{q.question}</b></div>',
                            unsafe_allow_html=True,
                        )
                        answer = st.text_input("Your answer:", key=f"ans_{idx}")
                        if st.form_submit_button("Submit"):
                            q.answered = True
                            q.answer = answer
                            # Append to conversation history
                            agent.conversation_history.append(
                                ConversationMessage(role="assistant", content=q.question)
                            )
                            agent.conversation_history.append(
                                ConversationMessage(role="user", content=answer)
                            )
                            st.rerun()
                else:
                    st.markdown(
                        f'<div class="alert-green">✅  {q.question}<br/>'
                        f'<i style="color:#aaa">→ {q.answer}</i></div>',
                        unsafe_allow_html=True,
                    )
        else:
            if agent.documents:
                st.markdown(
                    '<div class="alert-green">✅ No HITL questions — all documents passed validation.</div>',
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Analyse
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analyse:
    st.subheader("Intelligent Analysis Engine")
    st.markdown(
        "Pandas stats + Claude narrative insights + auto-selected charts."
    )

    if not agent.documents:
        st.info("Upload and extract documents first.")
    else:

        df_all = documents_to_dataframe(agent.documents)

        col_run, col_info = st.columns([2, 3])
        with col_run:
            run_btn = st.button("🧠 Run Full Analysis", use_container_width=True)
        with col_info:
            if df_all is not None:
                st.markdown(
                    f'<div class="alert-green">'
                    f'📋 {len(df_all)} total transactions across {len(agent.documents)} document(s)'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if run_btn:
            with st.spinner("Running forensic analysis…"):
                agent.analysis_result = run_analysis(
                    agent.documents,
                    agent.conversation_history,
                    agent.memory_context,
                )
                agent.current_step = "analysing"
            st.rerun()

        if agent.analysis_result:
            ar = agent.analysis_result

            # Summary banner
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Analysis Summary</div>'
                f'<div style="font-size:1.2rem;color:#fff;margin-top:0.3rem">{ar.summary}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            col_obs, col_act = st.columns(2)

            with col_obs:
                st.markdown("#### 🔭 Observations")
                for obs in ar.observations:
                    st.markdown(
                        f'<div class="alert-blue">▸ {obs}</div>',
                        unsafe_allow_html=True,
                    )

            with col_act:
                st.markdown("#### ⚡ Action Items")
                for act in ar.action_items:
                    st.markdown(
                        f'<div class="alert-green">☐ {act}</div>',
                        unsafe_allow_html=True,
                    )

            if ar.focused_question:
                st.markdown(
                    f'<div class="alert-orange">'
                    f'💡 <b>Focused Question:</b> {ar.focused_question}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Charts
            if ar.chart_jsons:
                st.markdown("#### 📈 Auto-Generated Charts")
                chart_cols = st.columns(min(len(ar.chart_jsons), 2))
                for i, chart_json in enumerate(ar.chart_jsons):
                    with chart_cols[i % 2]:
                        fig = pio.from_json(chart_json)
                        st.plotly_chart(fig, use_container_width=True)

            # Raw DataFrame
            if df_all is not None:
                with st.expander("📋 Full Transaction Table"):
                    st.dataframe(df_all, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Amazon Reconciliation
# ═══════════════════════════════════════════════════════════════════════════════
with tab_amazon:
    st.subheader("Amazon India Profit Guardian")
    st.markdown(
        "MTR vs Settlement leakage · GST reconciliation · FBA fee analysis · Ad waste map"
    )

    col_mtr, col_sett, col_ad = st.columns(3)

    with col_mtr:
        st.markdown("**MTR (Merchant Tax Report)**")
        mtr_file = st.file_uploader("Upload MTR", type=["csv", "xlsx"],
                                     key="mtr_upload", label_visibility="collapsed")

    with col_sett:
        st.markdown("**Settlement Report**")
        sett_file = st.file_uploader("Upload Settlement", type=["csv", "xlsx"],
                                      key="sett_upload", label_visibility="collapsed")

    with col_ad:
        st.markdown("**Advertising Report (optional)**")
        ad_file = st.file_uploader("Upload Ads", type=["csv", "xlsx"],
                                    key="ad_upload", label_visibility="collapsed")

    if st.button("🔍 Run Reconciliation", use_container_width=True):
        mtr_doc  = extract_document(mtr_file.read(),  mtr_file.name)  if mtr_file  else None
        sett_doc = extract_document(sett_file.read(), sett_file.name) if sett_file else None
        ad_doc   = extract_document(ad_file.read(),   ad_file.name)   if ad_file   else None

        with st.spinner("Reconciling reports…"):
            agent.amazon_recon = reconcile_mtr_settlement(mtr_doc, sett_doc, ad_doc)
        st.rerun()

    if agent.amazon_recon:
        recon = agent.amazon_recon

        # KPI row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MTR Total",     f"₹{recon.mtr_total:,.0f}"       if recon.mtr_total        else "N/A")
        m2.metric("Settlement",    f"₹{recon.settlement_total:,.0f}" if recon.settlement_total else "N/A")
        m3.metric("Leakage",       f"₹{recon.leakage_amount:,.0f}"   if recon.leakage_amount   else "N/A",
                  delta=f"-{recon.leakage_percentage:.1f}%" if recon.leakage_percentage else None,
                  delta_color="inverse")
        m4.metric("ACOS",          f"{recon.acos:.1f}%"              if recon.acos             else "N/A")

        col_charts = st.columns(2)

        # Waterfall
        wf_json = leakage_waterfall_json(recon)
        if wf_json:
            with col_charts[0]:
                st.plotly_chart(pio.from_json(wf_json), use_container_width=True)

        # Fee pie
        fp_json = fee_pie_json(recon)
        if fp_json:
            with col_charts[1]:
                st.plotly_chart(pio.from_json(fp_json), use_container_width=True)

        # GST breakdown table
        if recon.gst_reconciliation:
            st.markdown("#### GST Reconciliation")
            gst = recon.gst_reconciliation
            gst_df = pd.DataFrame([
                {"Component": k.replace("_", " ").title(), "Amount (₹)": v}
                for k, v in gst.items()
            ])
            st.dataframe(gst_df, use_container_width=True, hide_index=True)

        # Recommendations
        if recon.recommendations:
            st.markdown("#### 💡 Recommendations")
            for rec in recon.recommendations:
                st.markdown(
                    f'<div class="alert-green">✓ {rec}</div>',
                    unsafe_allow_html=True,
                )

        # Ad waste section
        if recon.ad_waste:
            st.markdown("#### 📣 Ad Waste Analysis")
            ad = recon.ad_waste
            ac1, ac2, ac3 = st.columns(3)
            ac1.metric("Total Ad Spend",   f"₹{ad.get('total_ad_spend', 0):,.0f}")
            ac2.metric("Total Ad Revenue", f"₹{ad.get('total_ad_revenue', 0):,.0f}")
            ac3.metric("Wasted Spend",     f"₹{ad.get('wasted_spend', 0):,.0f}")
            if ad.get("high_acos_campaigns"):
                st.warning(f"High-ACOS campaigns: {', '.join(ad['high_acos_campaigns'])}")
    else:
        st.markdown(
            '<div class="alert-blue">ℹ️ Upload MTR and Settlement files to run reconciliation.</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Report & Export
# ═══════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.subheader("Report Generation & Export")

    company_name = st.text_input(
        "Company / Seller name (for PPT cover)", value="My Business"
    )

    col_gen, col_dl = st.columns(2)

    with col_gen:
        gen_btn = st.button("🎨 Generate PPT Report", use_container_width=True)

    if gen_btn:
        if not agent.analysis_result:
            st.warning("Run the analysis first (tab: Analyse).")
        else:
            with st.spinner("Building PowerPoint presentation…"):
                ppt_bytes = create_presentation(
                    analysis=agent.analysis_result,
                    amazon_recon=agent.amazon_recon,
                    company_name=company_name,
                )
            st.success("Presentation ready! ✅")

            st.download_button(
                label="⬇️ Download .pptx",
                data=ppt_bytes,
                file_name=f"guardian_ai_report_{datetime.date.today()}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=True,
            )

    # CSV export
    if agent.documents:
        df_export = documents_to_dataframe(agent.documents)
        if df_export is not None:
            st.markdown("---")
            st.markdown("### 📋 Download Consolidated Data")
            csv_bytes = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                data=csv_bytes,
                file_name=f"guardian_ai_data_{datetime.date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # Slide preview info
    st.markdown("---")
    st.markdown("### Slide Structure")
    slides = [
        ("Slide 1", "Title + Date",                   "Cover with company name"),
        ("Slide 2", "Executive Summary",               "Summary + Observations + Focused Q"),
        ("Slide 3", "Revenue Leakage Map",             "MTR vs Settlement + GST + Fee breakdown"),
        ("Slide 4", "AI Action Plan",                  "Insights + Action Items + Next Question"),
    ]
    slides_df = pd.DataFrame(slides, columns=["Slide", "Title", "Content"])
    st.dataframe(slides_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — AI Chat
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("💬 Ask Guardian AI")
    st.markdown("Chat with your financial data. Ask anything — trends, anomalies, comparisons.")

    # Chat history display
    chat_container = st.container()
    with chat_container:
        if agent.conversation_history:
            for msg in agent.conversation_history[-20:]:
                bubble_class = "chat-user" if msg.role == "user" else "chat-ai"
                icon = "👤" if msg.role == "user" else "🛡️"
                st.markdown(
                    f'<div class="chat-wrap">'
                    f'<div class="{bubble_class}">{icon} {msg.content}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="alert-blue">'
                'ℹ️ No conversation yet. Ask a question about your data below.'
                '</div>',
                unsafe_allow_html=True,
            )

    # Input
    st.divider()
    user_input = st.chat_input("Ask Guardian AI about your finances…")

    if user_input:
        agent.conversation_history.append(
            ConversationMessage(role="user", content=user_input)
        )

        if not agent.documents:
            reply = "No documents loaded yet. Please upload financial documents first."
        else:
            with st.spinner("Thinking…"):
                reply = chat_with_analyst(
                    user_input,
                    agent.documents,
                    agent.conversation_history,
                )

        agent.conversation_history.append(
            ConversationMessage(role="assistant", content=reply)
        )
        st.rerun()

    # Suggested questions
    if not agent.conversation_history:
        st.markdown("**Suggested questions:**")
        suggestions = [
            "What is my total revenue this month?",
            "Which category has the highest spend?",
            "Are there any GST mismatches?",
            "Show me the top 5 transactions by amount.",
            "What was the revenue trend over the last 3 months?",
        ]
        for suggestion in suggestions:
            if st.button(suggestion, key=f"sug_{suggestion[:20]}"):
                agent.conversation_history.append(
                    ConversationMessage(role="user", content=suggestion)
                )
                if agent.documents:
                    with st.spinner("Thinking…"):
                        reply = chat_with_analyst(
                            suggestion, agent.documents, agent.conversation_history
                        )
                    agent.conversation_history.append(
                        ConversationMessage(role="assistant", content=reply)
                    )
                st.rerun()

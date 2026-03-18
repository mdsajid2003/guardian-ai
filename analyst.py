"""
analyst.py — Guardian AI (Gemini Edition): Financial analysis engine
Pandas stats + auto-chart selection + Gemini narrative insights.
Free tier: gemini-1.5-flash
"""

import json
import os
from typing import List, Optional, Dict, Any

import google.generativeai as genai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from state import (
    FinancialDocument,
    AnalysisResult,
    ConversationMessage,
)


# ── Gemini client helper ───────────────────────────────────────────────────────
def _get_model(system_instruction: str) -> genai.GenerativeModel:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction,
    )


_GEN_CFG_FULL = genai.GenerationConfig(max_output_tokens=2048, temperature=0.2)
_GEN_CFG_CHAT = genai.GenerationConfig(max_output_tokens=1024, temperature=0.3)


# ── System prompts ─────────────────────────────────────────────────────────────
ANALYST_SYSTEM = """You are Guardian AI, an expert Indian financial analyst.
Analyse the supplied data and respond ONLY with valid JSON (no fences):
{
  "summary": "one confident sentence (e.g. 'Found 847 rows, 6 months of data')",
  "observations": ["up to 5 sharp observations, e.g. 'Revenue dipped 23% in March'"],
  "focused_question": "ONE strategic question to drive the next decision",
  "insights": ["3-5 actionable insights"],
  "action_items": ["3-5 concrete next steps"]
}
Use ₹ for Indian rupees. Be specific — cite figures where possible."""

CHAT_SYSTEM = """You are Guardian AI, a smart Indian financial analyst.
Answer questions about the user's financial data concisely and helpfully.
Use ₹ for amounts. Be direct and data-driven. Never make up numbers."""


# ── DataFrame helpers ──────────────────────────────────────────────────────────

def documents_to_dataframe(documents: List[FinancialDocument]) -> Optional[pd.DataFrame]:
    rows = []
    for doc in documents:
        for item in doc.line_items:
            rows.append({
                "date":        item.date,
                "description": item.description,
                "amount":      item.amount,
                "gst_amount":  item.gst_amount,
                "gstin":       item.gstin,
                "invoice_no":  item.invoice_no,
                "category":    item.category,
                "source":      doc.filename or doc.doc_type.value,
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


def _data_stats(df: pd.DataFrame) -> dict:
    stats: Dict[str, Any] = {
        "total_rows":   len(df),
        "total_amount": None,
        "avg_amount":   None,
        "date_range":   None,
        "categories":   {},
        "null_counts":  df.isnull().sum().to_dict(),
    }
    if "amount" in df.columns:
        stats["total_amount"] = round(float(df["amount"].sum()), 2)
        stats["avg_amount"]   = round(float(df["amount"].mean()), 2)
    if "date" in df.columns:
        valid_dates = df["date"].dropna()
        if not valid_dates.empty:
            stats["date_range"] = (
                f"{valid_dates.min().date()} to {valid_dates.max().date()}"
            )
    if "category" in df.columns:
        stats["categories"] = df["category"].value_counts().to_dict()
    return stats


# ── Chart generation ───────────────────────────────────────────────────────────

CHART_THEME = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E0E0E0"),
)


def _select_chart_type(df: pd.DataFrame) -> str:
    has_dates = "date" in df.columns and df["date"].notna().sum() > 3
    has_cats  = "category" in df.columns and df["category"].notna().sum() > 0
    n = len(df)
    if has_dates and n > 5:
        return "line"
    if has_cats:
        return "pie" if df["category"].nunique() <= 6 else "bar"
    if n <= 20:
        return "waterfall"
    return "bar"


def generate_charts(df: pd.DataFrame) -> List[str]:
    charts = []
    if df is None or df.empty:
        return charts

    ct = _select_chart_type(df)

    if ct == "line" and "date" in df.columns and "amount" in df.columns:
        try:
            monthly = (
                df.dropna(subset=["date", "amount"])
                .groupby(df["date"].dt.to_period("M"))["amount"]
                .sum()
                .reset_index()
            )
            monthly["date"] = monthly["date"].astype(str)
            fig = px.line(
                monthly, x="date", y="amount",
                title="Monthly Revenue / Expense Trend",
                labels={"amount": "Amount (₹)", "date": "Month"},
                color_discrete_sequence=["#00B48A"],
            )
            fig.update_layout(**CHART_THEME)
            charts.append(fig.to_json())
        except Exception:
            pass

    if "category" in df.columns and df["category"].notna().sum() > 0 and "amount" in df.columns:
        try:
            cat_df = (
                df.dropna(subset=["category", "amount"])
                .groupby("category")["amount"]
                .sum()
                .reset_index()
                .sort_values("amount", ascending=False)
            )
            if len(cat_df) <= 6:
                fig = px.pie(
                    cat_df, values="amount", names="category",
                    title="Spend by Category",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
            else:
                fig = px.bar(
                    cat_df, x="category", y="amount",
                    title="Spend by Category",
                    labels={"amount": "Amount (₹)"},
                    color_discrete_sequence=["#FF6B35"],
                )
            fig.update_layout(**CHART_THEME)
            charts.append(fig.to_json())
        except Exception:
            pass

    if ct == "waterfall" and "amount" in df.columns:
        try:
            sample = df.dropna(subset=["amount"]).head(15)
            labels = (
                sample["description"].fillna("?").tolist()
                if "description" in sample.columns
                else [f"Item {i}" for i in range(len(sample))]
            )
            fig = go.Figure(go.Waterfall(
                orientation="v",
                x=labels,
                y=sample["amount"].tolist(),
                connector={"line": {"color": "#888"}},
                increasing={"marker": {"color": "#00B48A"}},
                decreasing={"marker": {"color": "#FF6B35"}},
            ))
            fig.update_layout(title="Transaction Waterfall", **CHART_THEME)
            charts.append(fig.to_json())
        except Exception:
            pass

    if "gst_amount" in df.columns and df["gst_amount"].notna().sum() > 3 and "date" in df.columns:
        try:
            heat_df = (
                df.dropna(subset=["date", "gst_amount"])
                .assign(month=df["date"].dt.to_period("M").astype(str))
                .groupby("month")["gst_amount"]
                .sum()
                .reset_index()
            )
            fig = px.bar(
                heat_df, x="month", y="gst_amount",
                title="Monthly GST Collected",
                labels={"gst_amount": "GST (₹)", "month": "Month"},
                color="gst_amount",
                color_continuous_scale="Teal",
            )
            fig.update_layout(**CHART_THEME)
            charts.append(fig.to_json())
        except Exception:
            pass

    return charts


# ── Main analysis function ─────────────────────────────────────────────────────

def run_analysis(
    documents: List[FinancialDocument],
    conversation_history: List[ConversationMessage],
    memory_context: List[str],
) -> AnalysisResult:
    df = documents_to_dataframe(documents)

    if df is not None and not df.empty:
        stats = _data_stats(df)
        data_summary = json.dumps(stats, default=str)
    else:
        data_summary = "No structured line-item data available."

    history_str = "\n".join(f"{m.role}: {m.content}" for m in conversation_history[-5:])
    memory_str  = "\n".join(memory_context[-3:]) if memory_context else ""

    prompt = (
        f"Financial data summary:\n{data_summary}\n\n"
        f"Past session context:\n{memory_str}\n\n"
        f"Recent conversation:\n{history_str}\n\n"
        "Provide your analysis."
    )

    try:
        model = _get_model(ANALYST_SYSTEM)
        response = model.generate_content(prompt, generation_config=_GEN_CFG_FULL)
        raw = response.text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
    except Exception as e:
        data = {
            "summary": f"Analysis encountered an error: {e}",
            "observations": [],
            "focused_question": None,
            "insights": [],
            "action_items": [],
        }

    chart_jsons = generate_charts(df) if df is not None else []

    return AnalysisResult(
        summary=data.get("summary", ""),
        observations=data.get("observations", []),
        focused_question=data.get("focused_question"),
        insights=data.get("insights", []),
        action_items=data.get("action_items", []),
        chart_jsons=chart_jsons,
    )


def chat_with_analyst(
    user_message: str,
    documents: List[FinancialDocument],
    conversation_history: List[ConversationMessage],
) -> str:
    df = documents_to_dataframe(documents)

    ctx = "No data loaded yet."
    if df is not None and not df.empty:
        total = df["amount"].sum() if "amount" in df.columns else 0
        ctx = f"Loaded {len(df)} transactions. Total: ₹{total:,.2f}."

    # Build multi-turn history for Gemini
    history = []
    for m in conversation_history[-10:]:
        role = "user" if m.role == "user" else "model"
        history.append({"role": role, "parts": [m.content]})

    model = _get_model(CHAT_SYSTEM)
    chat   = model.start_chat(history=history[:-1] if history else [])

    final_msg = f"Context: {ctx}\n\nQuestion: {user_message}"
    response = chat.send_message(final_msg, generation_config=_GEN_CFG_CHAT)
    return response.text

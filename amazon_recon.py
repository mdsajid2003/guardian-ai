"""
amazon_recon.py — Guardian AI (Gemini Edition): Amazon India Profit Guardian
MTR vs Settlement reconciliation, GST gap analysis, Ad waste detection.
Free tier: gemini-1.5-flash
"""

import json
import os
from typing import List, Optional, Dict, Any

import google.generativeai as genai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from state import FinancialDocument, AmazonReconResult


# ── Gemini client helper ───────────────────────────────────────────────────────
def _get_model(system_instruction: str) -> genai.GenerativeModel:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get(
                "GOOGLE_API_KEY"
            )
        except Exception:
            pass
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction,
    )


_GEN_CFG = genai.GenerationConfig(max_output_tokens=2048, temperature=0.1)


# ── System prompt ──────────────────────────────────────────────────────────────
RECON_SYSTEM = """You are an Amazon India seller account expert specialising in:
- MTR (Merchant Tax Report) vs Settlement reconciliation
- GST leakage detection (IGST/CGST/SGST mismatches)
- FBA fee anomaly identification
- Advertising ACOS analysis and waste detection

Respond ONLY with valid JSON (no markdown fences):
{
  "mtr_total": number or null,
  "settlement_total": number or null,
  "leakage_amount": number or null,
  "leakage_percentage": number or null,
  "gst_reconciliation": {
    "igst_collected": number,
    "cgst_collected": number,
    "sgst_collected": number,
    "total_gst": number,
    "gst_in_settlement": number,
    "gst_gap": number
  },
  "fee_breakdown": {
    "referral_fees": number,
    "fba_fees": number,
    "closing_fees": number,
    "other_fees": number
  },
  "ad_waste": {
    "total_ad_spend": number,
    "total_ad_revenue": number,
    "acos": number,
    "wasted_spend": number,
    "high_acos_campaigns": ["campaign names"]
  },
  "recommendations": ["up to 5 specific, actionable recommendations"]
}
All monetary values in INR (₹). ACOS as a percentage (e.g. 34.5 not 0.345)."""


# ── Core reconciliation ────────────────────────────────────────────────────────

def reconcile_mtr_settlement(
    mtr_doc: Optional[FinancialDocument],
    settlement_doc: Optional[FinancialDocument],
    ad_doc: Optional[FinancialDocument] = None,
) -> AmazonReconResult:
    if not mtr_doc and not settlement_doc:
        return AmazonReconResult(
            recommendations=["Upload MTR and Settlement reports to enable reconciliation."]
        )

    def _doc_summary(doc: Optional[FinancialDocument], label: str) -> str:
        if not doc:
            return f"{label}: not provided"
        total = sum(it.amount or 0 for it in doc.line_items)
        gst_total = sum(it.gst_amount or 0 for it in doc.line_items)
        return (
            f"{label}: {len(doc.line_items)} line items, "
            f"total ₹{total:,.2f}, GST ₹{gst_total:,.2f}"
        )

    mtr_summary  = _doc_summary(mtr_doc, "MTR (Merchant Tax Report)")
    sett_summary = _doc_summary(settlement_doc, "Settlement Report")
    ad_summary   = _doc_summary(ad_doc, "Advertising Report") if ad_doc else ""

    prompt = (
        f"Reconcile these Amazon India seller reports:\n\n"
        f"{mtr_summary}\n{sett_summary}\n{ad_summary}\n\n"
        "Identify revenue leakage, GST gaps, fee anomalies, "
        "and advertising waste. Return the JSON analysis."
    )

    try:
        model = _get_model(RECON_SYSTEM)
        response = model.generate_content(prompt, generation_config=_GEN_CFG)
        raw = response.text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
    except Exception as e:
        return AmazonReconResult(recommendations=[f"Reconciliation failed: {e}"])

    return AmazonReconResult(
        mtr_total=data.get("mtr_total"),
        settlement_total=data.get("settlement_total"),
        leakage_amount=data.get("leakage_amount"),
        leakage_percentage=data.get("leakage_percentage"),
        gst_reconciliation=data.get("gst_reconciliation", {}),
        fee_breakdown=data.get("fee_breakdown", {}),
        ad_waste=data.get("ad_waste", {}),
        acos=data.get("ad_waste", {}).get("acos"),
        recommendations=data.get("recommendations", []),
    )


# ── Ad performance analysis ────────────────────────────────────────────────────

def analyse_advertising_csv(df: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    df.columns = [c.lower().strip() for c in df.columns]
    spend_col = next((c for c in df.columns if "spend" in c or "cost" in c), None)
    rev_col   = next((c for c in df.columns if "revenue" in c or "sales" in c), None)
    camp_col  = next((c for c in df.columns if "campaign" in c), None)

    if not spend_col or not rev_col:
        return {"error": "Could not identify spend and revenue columns"}

    df[spend_col] = pd.to_numeric(df[spend_col], errors="coerce").fillna(0)
    df[rev_col]   = pd.to_numeric(df[rev_col],   errors="coerce").fillna(0)

    total_spend   = float(df[spend_col].sum())
    total_revenue = float(df[rev_col].sum())
    acos          = (total_spend / total_revenue * 100) if total_revenue > 0 else 0.0

    result["total_spend"]   = round(total_spend, 2)
    result["total_revenue"] = round(total_revenue, 2)
    result["acos"]          = round(acos, 2)

    if camp_col:
        camp_df = (
            df.groupby(camp_col)[[spend_col, rev_col]]
            .sum()
            .assign(
                acos=lambda x: (x[spend_col] / x[rev_col].replace(0, float("nan"))) * 100
            )
            .reset_index()
        )
        wasted = camp_df[camp_df["acos"] > 50]
        result["wasted_spend"]        = round(float(wasted[spend_col].sum()), 2)
        result["high_acos_campaigns"] = wasted[camp_col].tolist()

    return result


# ── Plotly visualisations ──────────────────────────────────────────────────────

CHART_THEME = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E0E0E0"),
)


def leakage_waterfall_json(recon: AmazonReconResult) -> Optional[str]:
    if not recon.mtr_total:
        return None
    try:
        labels  = ["MTR Total"]
        values  = [recon.mtr_total]
        measure = ["absolute"]

        if recon.fee_breakdown:
            for k, v in recon.fee_breakdown.items():
                labels.append(k.replace("_", " ").title())
                values.append(-abs(v))
                measure.append("relative")

        if recon.gst_reconciliation.get("gst_gap"):
            labels.append("GST Gap")
            values.append(-abs(recon.gst_reconciliation["gst_gap"]))
            measure.append("relative")

        if recon.settlement_total:
            labels.append("Net Settlement")
            values.append(recon.settlement_total)
            measure.append("total")

        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=measure,
            x=labels,
            y=values,
            connector={"line": {"color": "#555"}},
            increasing={"marker": {"color": "#00B48A"}},
            decreasing={"marker": {"color": "#FF6B35"}},
            totals={"marker": {"color": "#4B9CD3"}},
        ))
        fig.update_layout(title="Revenue Leakage Waterfall", **CHART_THEME)
        return fig.to_json()
    except Exception:
        return None


def fee_pie_json(recon: AmazonReconResult) -> Optional[str]:
    if not recon.fee_breakdown:
        return None
    try:
        df = pd.DataFrame([
            {"Fee Type": k.replace("_", " ").title(), "Amount": v}
            for k, v in recon.fee_breakdown.items()
            if v > 0
        ])
        fig = px.pie(
            df, values="Amount", names="Fee Type",
            title="FBA Fee Breakdown",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(**CHART_THEME)
        return fig.to_json()
    except Exception:
        return None

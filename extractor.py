"""
extractor.py — Guardian AI (Gemini Edition): Universal document extraction
Supports CSV, Excel, PDF (pdfplumber), and Images (Gemini Vision).
Free tier: gemini-1.5-flash
"""

import io
import json
import base64
import os
from pathlib import Path
from typing import Optional

import google.generativeai as genai
import pdfplumber
import pandas as pd
from PIL import Image

from state import DocumentType, FinancialDocument, LineItem
from image_proc import preprocess_image


# ── Gemini client ──────────────────────────────────────────────────────────────
def _get_model(system_instruction: str) -> genai.GenerativeModel:
    """Create a Gemini model configured with a system instruction."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
        "GOOGLE_API_KEY"
    )
    # Also check Streamlit secrets
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


_GEN_CFG = genai.GenerationConfig(max_output_tokens=4096, temperature=0.1)


# ── System prompt ──────────────────────────────────────────────────────────────
EXTRACTION_SYSTEM = """You are a forensic financial document extractor.
Extract ALL financial data and return ONLY valid JSON — no preamble, no markdown fences.

Required schema:
{
  "line_items": [
    {
      "date": "YYYY-MM-DD or null",
      "description": "string or null",
      "amount": number or null,
      "gst_amount": number or null,
      "gstin": "15-char GSTIN or null",
      "invoice_no": "string or null",
      "category": "one of: Sales/Purchase/Expense/Tax/Refund/Fee/Other or null",
      "ambiguity_score": 0.0 to 1.0
    }
  ],
  "total_amount": number or null,
  "period_start": "YYYY-MM-DD or null",
  "period_end": "YYYY-MM-DD or null",
  "gstin": "primary document GSTIN or null",
  "nulls_flagged": ["field names that were missing/null"],
  "gstin_gaps": ["row descriptions missing a GSTIN where expected"],
  "date_mismatches": ["any date inconsistencies detected"],
  "overall_confidence": 0.0 to 1.0
}

Rules:
- ambiguity_score: 0.0 = crystal clear, 1.0 = completely unclear
- overall_confidence = average clarity across the whole document
- Flag ALL missing required fields in nulls_flagged
- Negative amounts are valid (credits/refunds)
- Amounts should be in INR (₹) unless clearly otherwise
"""


def _parse_response(text: str) -> dict:
    """Strip accidental markdown fences and parse JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def _build_document(
    data: dict,
    doc_type: DocumentType,
    raw_text: str = "",
    filename: str = "",
) -> FinancialDocument:
    line_items = []
    for item in data.get("line_items", []):
        try:
            line_items.append(LineItem(**item))
        except Exception:
            pass

    return FinancialDocument(
        doc_type=doc_type,
        filename=filename,
        raw_text=raw_text[:3000],
        line_items=line_items,
        total_amount=data.get("total_amount"),
        period_start=data.get("period_start"),
        period_end=data.get("period_end"),
        gstin=data.get("gstin"),
        nulls_flagged=data.get("nulls_flagged", []),
        gstin_gaps=data.get("gstin_gaps", []),
        date_mismatches=data.get("date_mismatches", []),
        overall_confidence=float(data.get("overall_confidence", 0.5)),
    )


# ── Per-format extractors ──────────────────────────────────────────────────────

def extract_from_csv(file_bytes: bytes, filename: str) -> FinancialDocument:
    """Extract from CSV or Excel using pandas + Gemini."""
    try:
        ext = Path(filename).suffix.lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(file_bytes))
            doc_type = DocumentType.EXCEL
        else:
            df = None
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                df = pd.read_csv(
                    io.BytesIO(file_bytes), encoding="utf-8", errors="replace"
                )
            doc_type = DocumentType.CSV

        preview = df.head(30).to_string(index=False)
        schema_hint = (
            f"Columns: {list(df.columns)}\nRows: {len(df)}\n\nPreview:\n{preview}"
        )

        model = _get_model(EXTRACTION_SYSTEM)
        prompt = f"Extract financial data from this tabular data:\n\n{schema_hint[:8000]}"
        response = model.generate_content(prompt, generation_config=_GEN_CFG)
        data = _parse_response(response.text)
        return _build_document(data, doc_type, raw_text=preview, filename=filename)

    except Exception as e:
        return FinancialDocument(
            doc_type=DocumentType.CSV,
            filename=filename,
            extraction_errors=[f"CSV extraction failed: {str(e)}"],
            overall_confidence=0.0,
        )


def extract_from_pdf(
    file_bytes: bytes, filename: str = "document.pdf"
) -> FinancialDocument:
    """Extract from PDF via pdfplumber text + tables, then Gemini."""
    try:
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages[:20]):
                page_text = page.extract_text() or ""
                if page_text:
                    text_parts.append(f"--- Page {i+1} ---\n{page_text}")
                for table in page.extract_tables() or []:
                    rows = [
                        " | ".join(str(c or "") for c in row)
                        for row in table
                        if row
                    ]
                    text_parts.append("\n".join(rows))

        full_text = "\n".join(text_parts)

        model = _get_model(EXTRACTION_SYSTEM)
        prompt = f"Extract financial data from this PDF text:\n\n{full_text[:10000]}"
        response = model.generate_content(prompt, generation_config=_GEN_CFG)
        data = _parse_response(response.text)
        return _build_document(
            data, DocumentType.PDF, raw_text=full_text[:3000], filename=filename
        )

    except Exception as e:
        return FinancialDocument(
            doc_type=DocumentType.PDF,
            filename=filename,
            extraction_errors=[f"PDF extraction failed: {str(e)}"],
            overall_confidence=0.0,
        )


def extract_from_image(
    file_bytes: bytes,
    filename: str = "image.jpg",
    doc_type: DocumentType = DocumentType.IMAGE,
) -> FinancialDocument:
    """Extract from image via Gemini Vision (natively multimodal)."""
    try:
        processed = preprocess_image(file_bytes)
        pil_image = Image.open(io.BytesIO(processed)).convert("RGB")

        model = _get_model(EXTRACTION_SYSTEM)
        response = model.generate_content(
            [pil_image, "Extract all financial data from this image."],
            generation_config=_GEN_CFG,
        )
        data = _parse_response(response.text)
        return _build_document(data, doc_type, filename=filename)

    except Exception as e:
        return FinancialDocument(
            doc_type=doc_type,
            filename=filename,
            extraction_errors=[f"Image extraction failed: {str(e)}"],
            overall_confidence=0.0,
        )


# ── Public router ──────────────────────────────────────────────────────────────

def extract_document(file_bytes: bytes, filename: str) -> FinancialDocument:
    """Route to the correct extractor based on file extension."""
    ext = Path(filename).suffix.lower()

    if ext in (".csv", ".xlsx", ".xls"):
        return extract_from_csv(file_bytes, filename)
    elif ext == ".pdf":
        return extract_from_pdf(file_bytes, filename)
    elif ext in (".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp"):
        lower = filename.lower()
        if "whatsapp" in lower or "wa_" in lower:
            return extract_from_image(file_bytes, filename, DocumentType.WHATSAPP)
        elif "bank" in lower or "statement" in lower:
            return extract_from_image(
                file_bytes, filename, DocumentType.BANK_STATEMENT
            )
        else:
            return extract_from_image(file_bytes, filename, DocumentType.IMAGE)
    else:
        return FinancialDocument(
            doc_type=DocumentType.IMAGE,
            filename=filename,
            extraction_errors=[f"Unsupported file type: {ext}"],
            overall_confidence=0.0,
        )

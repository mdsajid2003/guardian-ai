"""
extractor.py — Guardian AI (Upgraded)
Reliable extraction with rule-based + AI fallback.

Architecture:
  extract_document()          ← public router
    ├── extract_from_csv()    ← rule-based fast path → AI fallback
    ├── extract_from_pdf()    ← pdfplumber text → AI
    └── extract_from_image()  ← Gemini Vision
"""

import io
import json
import os
from pathlib import Path
from typing import Optional

import google.generativeai as genai
import pandas as pd
import pdfplumber
from PIL import Image

from image_proc import preprocess_image
from state import DocumentType, FinancialDocument, LineItem


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "gemini-2.5-flash"
MAX_PDF_PAGES = 20
CSV_PREVIEW_ROWS = 30
PDF_CHAR_LIMIT = 10_000
CSV_CHAR_LIMIT = 8_000
RAW_TEXT_LIMIT = 3_000

EXTRACTION_SYSTEM = """You are a forensic financial document extractor.
Extract ALL financial data and return ONLY valid JSON — no preamble, no markdown.

Required schema:
{
  "line_items": [
    {
      "date":            "YYYY-MM-DD or null",
      "description":     "string or null",
      "amount":          number or null,
      "gst_amount":      number or null,
      "gstin":           "string or null",
      "invoice_no":      "string or null",
      "category":        "string or null",
      "ambiguity_score": 0.0 to 1.0
    }
  ],
  "total_amount":       number or null,
  "overall_confidence": 0.0 to 1.0
}
"""

# Candidate column names for the "amount" field, in priority order.
# Amazon CSVs use names like "total", "net proceeds", "item price", etc.
_AMOUNT_CANDIDATES = (
    "amount",
    "total",
    "net proceeds",
    "item price",
    "selling price",
    "settlement amount",
    "net amount",
    "price",
    "value",
    "debit",
    "credit",
    "transaction amount",
)

# Candidate column names for the "date" and "description" fields.
_DATE_CANDIDATES = ("date", "order date", "transaction date", "posting date", "ship date")
_DESC_CANDIDATES = ("description", "item description", "product name", "title", "narration", "particulars")


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_api_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


def _get_model(system_instruction: str) -> genai.GenerativeModel:
    genai.configure(api_key=_resolve_api_key())
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_instruction,
    )


_GEN_CFG = genai.GenerationConfig(max_output_tokens=4096, temperature=0.1)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_response(text: str) -> dict:
    """Strip optional markdown fences and parse JSON."""
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
    *,
    raw_text: str = "",
    filename: str = "",
) -> FinancialDocument:
    """Convert a parsed AI response dict into a FinancialDocument."""
    line_items: list[LineItem] = []
    for item in data.get("line_items", []):
        try:
            line_items.append(LineItem(**item))
        except Exception:
            continue  # skip malformed rows silently

    return FinancialDocument(
        doc_type=doc_type,
        filename=filename,
        raw_text=raw_text[:RAW_TEXT_LIMIT],
        line_items=line_items,
        total_amount=data.get("total_amount"),
        overall_confidence=float(data.get("overall_confidence", 0.5)),
    )


def _error_document(
    doc_type: DocumentType,
    filename: str,
    error: Exception,
    stage: str,
) -> FinancialDocument:
    """Return a zero-confidence document that records the extraction failure."""
    return FinancialDocument(
        doc_type=doc_type,
        filename=filename,
        extraction_errors=[f"{stage} failed: {error}"],
        overall_confidence=0.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CSV / EXCEL EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _load_dataframe(file_bytes: bytes, ext: str) -> tuple[pd.DataFrame, DocumentType]:
    """Return a normalised DataFrame and its DocumentType."""
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(file_bytes))
        return df, DocumentType.EXCEL

    # CSV — try encodings in order
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            return df, DocumentType.CSV
        except UnicodeDecodeError:
            continue

    # Last resort — replace bad bytes
    df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8", errors="replace")
    return df, DocumentType.CSV


def _resolve_column(df_columns: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    """
    Return the first candidate name that exists in df_columns (case-insensitive).
    Also matches columns that *contain* the candidate as a substring, so
    'net proceeds' matches 'net proceeds (inr)' etc.
    """
    col_set = set(df_columns)
    for candidate in candidates:
        # Exact match first
        if candidate in col_set:
            return candidate
        # Substring match fallback
        for col in df_columns:
            if candidate in col:
                return col
    return None


def _rule_based_extraction(
    df: pd.DataFrame,
    doc_type: DocumentType,
    filename: str,
) -> Optional[FinancialDocument]:
    """
    Fast path: map columns directly without calling the AI.
    Handles non-standard column names (e.g. Amazon 'total', 'net proceeds').
    Returns None only when no recognisable amount column can be found at all.
    """
    cols = list(df.columns)

    amount_col = _resolve_column(cols, _AMOUNT_CANDIDATES)
    if amount_col is None:
        return None  # hand off to AI fallback

    date_col  = _resolve_column(cols, _DATE_CANDIDATES)
    desc_col  = _resolve_column(cols, _DESC_CANDIDATES)

    line_items: list[LineItem] = []
    for _, row in df.iterrows():
        try:
            raw_amount = row.get(amount_col)
            line_items.append(LineItem(
                date=str(row[date_col]) if date_col else None,
                description=str(row[desc_col]) if desc_col else None,
                amount=float(raw_amount) if pd.notna(raw_amount) else None,
                gst_amount=(
                    float(row["gst_amount"])
                    if "gst_amount" in cols and pd.notna(row.get("gst_amount"))
                    else None
                ),
                category=row.get("category") if "category" in cols else None,
                ambiguity_score=0.1,
            ))
        except Exception:
            continue

    return FinancialDocument(
        doc_type=doc_type,
        filename=filename,
        line_items=line_items,
        total_amount=float(df[amount_col].sum()),
        overall_confidence=0.9,
    )


def _ai_csv_extraction(df: pd.DataFrame, doc_type: DocumentType, filename: str) -> FinancialDocument:
    """AI fallback for CSVs without a clear 'amount' column."""
    preview = df.head(CSV_PREVIEW_ROWS).to_string(index=False)
    prompt = (
        f"Columns: {list(df.columns)}\n"
        f"Rows: {len(df)}\n\n"
        f"Preview:\n{preview}"
    )

    model = _get_model(EXTRACTION_SYSTEM)
    response = model.generate_content(
        f"Extract financial data from this table:\n\n{prompt[:CSV_CHAR_LIMIT]}",
        generation_config=_GEN_CFG,
    )
    data = _parse_response(response.text)
    return _build_document(data, doc_type, raw_text=preview, filename=filename)


def extract_from_csv(file_bytes: bytes, filename: str) -> FinancialDocument:
    """
    Extract structured data from a CSV or Excel file.

    Strategy:
      1. Rule-based — O(n), no API call, confidence 0.9
      2. AI fallback — when 'amount' column is absent
    """
    try:
        ext = Path(filename).suffix.lower()
        df, doc_type = _load_dataframe(file_bytes, ext)
        df.columns = [c.lower().strip() for c in df.columns]

        result = _rule_based_extraction(df, doc_type, filename)
        return result if result is not None else _ai_csv_extraction(df, doc_type, filename)

    except Exception as e:
        return _error_document(DocumentType.CSV, filename, e, "CSV extraction")


# ══════════════════════════════════════════════════════════════════════════════
# PDF EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_pdf_text(file_bytes: bytes) -> str:
    """Pull plain text from up to MAX_PDF_PAGES pages via pdfplumber."""
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages[:MAX_PDF_PAGES]:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def extract_from_pdf(file_bytes: bytes, filename: str = "document.pdf") -> FinancialDocument:
    """Extract financial data from a PDF via pdfplumber + Gemini."""
    try:
        full_text = _extract_pdf_text(file_bytes)

        model = _get_model(EXTRACTION_SYSTEM)
        response = model.generate_content(
            f"Extract financial data from this PDF:\n\n{full_text[:PDF_CHAR_LIMIT]}",
            generation_config=_GEN_CFG,
        )
        data = _parse_response(response.text)
        return _build_document(data, DocumentType.PDF, raw_text=full_text, filename=filename)

    except Exception as e:
        return _error_document(DocumentType.PDF, filename, e, "PDF extraction")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_from_image(
    file_bytes: bytes,
    filename: str,
    doc_type: DocumentType = DocumentType.IMAGE,
) -> FinancialDocument:
    """Extract financial data from an image via Gemini Vision."""
    try:
        processed = preprocess_image(file_bytes)
        image = Image.open(io.BytesIO(processed)).convert("RGB")

        model = _get_model(EXTRACTION_SYSTEM)
        response = model.generate_content(
            [image, "Extract all financial data from this image."],
            generation_config=_GEN_CFG,
        )
        data = _parse_response(response.text)
        return _build_document(data, doc_type, filename=filename)

    except Exception as e:
        return _error_document(doc_type, filename, e, "Image extraction")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ROUTER
# ══════════════════════════════════════════════════════════════════════════════

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def extract_document(file_bytes: bytes, filename: str) -> FinancialDocument:
    """
    Route a file to the correct extractor based on its extension.

    Supported formats:
      CSV / Excel  → rule-based then AI fallback
      PDF          → pdfplumber text → Gemini
      Image        → Gemini Vision
    """
    ext = Path(filename).suffix.lower()

    if ext in (".csv", ".xlsx", ".xls"):
        return extract_from_csv(file_bytes, filename)

    if ext == ".pdf":
        return extract_from_pdf(file_bytes, filename)

    if ext in _IMAGE_EXTENSIONS:
        return extract_from_image(file_bytes, filename, DocumentType.IMAGE)

    return FinancialDocument(
        doc_type=DocumentType.IMAGE,
        filename=filename,
        extraction_errors=[f"Unsupported file type: '{ext}'"],
        overall_confidence=0.0,
    )

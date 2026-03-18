"""
validator.py — Guardian AI: LangGraph confidence gate + HITL question generator
"""

import re
from datetime import datetime
from typing import List, Tuple

from state import (
    FinancialDocument,
    LineItem,
    HITLQuestion,
    ConfidenceLevel,
)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.9          # above this → skip HITL
LARGE_AMOUNT_THRESHOLD = 10_000     # flag un-described amounts above this (₹)
GSTIN_RE = re.compile(
    r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"
)


# ── GSTIN helpers ─────────────────────────────────────────────────────────────

def validate_gstin(gstin: str) -> bool:
    """Return True if the GSTIN matches the official 15-char pattern."""
    if not gstin:
        return False
    return bool(GSTIN_RE.match(gstin.upper().strip()))


# ── Per-item scoring ──────────────────────────────────────────────────────────

def compute_item_ambiguity(item: LineItem) -> float:
    """
    Score 0.0 (clear) → 1.0 (ambiguous) for a single line item.
    Penalties:
      - Missing required fields: date, description, amount  (+0.33 each)
      - Invalid GSTIN format                               (+0.20)
      - Suspiciously negative amount                       (+0.10)
    """
    required = ["date", "description", "amount"]
    missing = sum(1 for f in required if getattr(item, f) is None)
    score = missing / len(required)

    if item.gstin and not validate_gstin(item.gstin):
        score += 0.20

    if item.amount is not None and item.amount < 0:
        score += 0.10

    return min(score, 1.0)


# ── Document-level validation ─────────────────────────────────────────────────

def validate_document(
    doc: FinancialDocument,
) -> Tuple[FinancialDocument, List[HITLQuestion]]:
    """
    1. Re-score each line item's ambiguity.
    2. Validate primary GSTIN.
    3. Check date range consistency.
    4. Flag large unexplained transactions.
    5. Update doc.overall_confidence.
    6. Return at most 1 HITL question (UX rule).
    """
    questions: List[HITLQuestion] = []

    # — Re-score items —
    for item in doc.line_items:
        item.ambiguity_score = compute_item_ambiguity(item)

    # — Validate primary GSTIN —
    if doc.gstin:
        if not validate_gstin(doc.gstin):
            doc.nulls_flagged.append(f"Invalid GSTIN format: {doc.gstin}")
            questions.append(HITLQuestion(
                question=(
                    f"The GSTIN '{doc.gstin}' doesn't match the standard format. "
                    "Could you confirm or correct it?"
                ),
                context="Primary GSTIN validation failed",
                field_reference="gstin",
            ))

    # — Date range sanity —
    if doc.period_start and doc.period_end and not questions:
        try:
            start = datetime.strptime(doc.period_start, "%Y-%m-%d")
            end = datetime.strptime(doc.period_end, "%Y-%m-%d")
            if end < start:
                doc.date_mismatches.append(
                    f"Period end ({doc.period_end}) is before period start ({doc.period_start})"
                )
                amt_str = (
                    f"₹{doc.total_amount:,.0f}" if doc.total_amount else "unknown amount"
                )
                questions.append(HITLQuestion(
                    question=(
                        f"The date range looks reversed "
                        f"({doc.period_start} → {doc.period_end}). "
                        f"Is the {amt_str} total for this period correct?"
                    ),
                    context="Date range mismatch detected",
                    field_reference="period",
                ))
        except ValueError:
            pass

    # — Large unexplained amounts —
    if not questions:
        for i, item in enumerate(doc.line_items):
            if (
                item.amount is not None
                and abs(item.amount) > LARGE_AMOUNT_THRESHOLD
                and not item.description
            ):
                date_str = item.date or "an unknown date"
                questions.append(HITLQuestion(
                    question=(
                        f"There's a ₹{abs(item.amount):,.2f} transaction on {date_str} "
                        "with no description — what was this for?"
                    ),
                    context=f"Line item {i + 1}: amount={item.amount}, date={item.date}",
                    field_reference=f"line_items[{i}].description",
                ))
                break  # only one question per turn

    # — Recompute overall confidence —
    if doc.line_items:
        avg_ambiguity = sum(it.ambiguity_score for it in doc.line_items) / len(doc.line_items)
        doc.overall_confidence = round(1.0 - avg_ambiguity, 4)
    # If no items, keep the extraction-time confidence estimate

    # UX rule: max 1 HITL question per turn
    return doc, questions[:1]


def get_confidence_level(doc: FinancialDocument) -> ConfidenceLevel:
    """LangGraph gate: HIGH → proceed, LOW → trigger HITL loop."""
    return (
        ConfidenceLevel.HIGH
        if doc.overall_confidence > CONFIDENCE_THRESHOLD
        else ConfidenceLevel.LOW
    )


def summarise_flags(doc: FinancialDocument) -> str:
    """Human-readable summary of all flags found in a document."""
    parts = []
    if doc.nulls_flagged:
        parts.append(f"Missing fields: {', '.join(doc.nulls_flagged)}")
    if doc.gstin_gaps:
        parts.append(f"GSTIN gaps ({len(doc.gstin_gaps)} items)")
    if doc.date_mismatches:
        parts.append(f"Date issues: {'; '.join(doc.date_mismatches)}")
    if doc.extraction_errors:
        parts.append(f"Errors: {'; '.join(doc.extraction_errors)}")
    return " | ".join(parts) if parts else "No issues detected"

"""
state.py — Guardian AI: All Pydantic models and shared state definitions
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    CSV = "csv"
    PDF = "pdf"
    IMAGE = "image"
    EXCEL = "excel"
    WHATSAPP = "whatsapp"
    BANK_STATEMENT = "bank_statement"


class ConfidenceLevel(str, Enum):
    HIGH = "high"   # overall_confidence > 0.9 → straight to analysis
    LOW = "low"     # overall_confidence ≤ 0.9 → HITL loop


class LineItem(BaseModel):
    date: Optional[str] = None
    description: Optional[str] = None
    amount: Optional[float] = None
    gst_amount: Optional[float] = None
    gstin: Optional[str] = None
    invoice_no: Optional[str] = None
    category: Optional[str] = None
    ambiguity_score: float = Field(default=0.0, ge=0.0, le=1.0)


class FinancialDocument(BaseModel):
    doc_type: DocumentType = DocumentType.CSV
    filename: str = ""
    raw_text: Optional[str] = None
    line_items: List[LineItem] = []
    total_amount: Optional[float] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    gstin: Optional[str] = None
    nulls_flagged: List[str] = []
    gstin_gaps: List[str] = []
    date_mismatches: List[str] = []
    overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    extraction_errors: List[str] = []


class HITLQuestion(BaseModel):
    question: str
    context: str = ""
    field_reference: Optional[str] = None
    answered: bool = False
    answer: Optional[str] = None


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AnalysisResult(BaseModel):
    summary: str = ""
    observations: List[str] = []
    focused_question: Optional[str] = None
    insights: List[str] = []
    action_items: List[str] = []
    # Charts stored as JSON-serialisable dicts (Plotly figures via .to_json())
    chart_jsons: List[str] = []


class AmazonReconResult(BaseModel):
    mtr_total: Optional[float] = None
    settlement_total: Optional[float] = None
    leakage_amount: Optional[float] = None
    leakage_percentage: Optional[float] = None
    gst_reconciliation: Dict[str, Any] = {}
    ad_waste: Dict[str, Any] = {}
    acos: Optional[float] = None
    fee_breakdown: Dict[str, float] = {}
    recommendations: List[str] = []


class AgentState(BaseModel):
    session_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    documents: List[FinancialDocument] = []
    conversation_history: List[ConversationMessage] = []
    pending_hitl: List[HITLQuestion] = []
    analysis_result: Optional[AnalysisResult] = None
    amazon_recon: Optional[AmazonReconResult] = None
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    current_step: str = "upload"   # upload | validating | hitl | analysing | done
    memory_context: List[str] = []
    dataframe_json: Optional[str] = None   # serialised pandas DataFrame

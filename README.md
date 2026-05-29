# Guardian AI — Agentic Accountant

Guardian AI is a document-native financial intelligence platform for Indian e-commerce sellers. It ingests unstructured financial documents across any format, extracts structured records using Google Gemini 1.5 Flash, validates each record through a confidence-gated pipeline with human-in-the-loop escalation, and delivers forensic analysis, Amazon marketplace reconciliation, and exportable PowerPoint reports — all within a single Streamlit application.

> This repository is the free-tier Gemini edition. The original was built on Anthropic Claude. All application logic, UI, and pipeline behaviour are identical; only the LLM backend differs.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
- [Application Walkthrough](#application-walkthrough)
- [Upgrading the Model](#upgrading-the-model)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)

---

## Problem Statement

Indian marketplace sellers receive financial data across incompatible formats: Amazon MTR reports in CSV, GST invoices as PDFs, purchase records in Excel, and payment confirmations as WhatsApp screenshots. Reconciling these manually is error-prone and time-consuming. Guardian AI unifies all inputs through a single extraction layer, applies rule-based validation with confidence scoring, and surfaces actionable insights including revenue leakage, GST gaps, and advertising cost analysis — without requiring any manual data preparation.

---

## Features

### Universal Document Input
Accepts CSV, Excel (`.xls`, `.xlsx`), PDF, JPEG, PNG, and WebP files in a single upload session. Gemini's multimodal capabilities handle both text-based and image-based documents, including handwritten receipts and WhatsApp payment screenshots.

### Confidence-Gated Validation Pipeline
Each extracted document is assigned a confidence score between 0 and 1. Documents scoring above 0.9 proceed directly to analysis. Documents below the threshold enter a Human-in-the-Loop (HITL) loop that surfaces one targeted clarifying question per turn — framed as intelligence gathering, not as an error condition. GSTIN format validation and missing-field detection run at this stage regardless of confidence level.

### Forensic Analysis Engine
Pandas-powered statistical analysis is paired with Gemini-generated narrative summaries. Charts are selected and rendered automatically using Plotly based on the data type present — line charts for time-series, bar for categorical comparisons, waterfall for leakage attribution, pie for fee composition, and heatmap for correlation. Each analysis session produces a ranked list of observations and a prioritised action plan.

### Amazon India Reconciliation Module
A dedicated module for marketplace-specific analysis. Accepts separate upload of MTR (Merchant Tax Report), Settlement Report, and Advertising Report. Produces:

- MTR vs. Settlement variance with rupee and percentage leakage
- GST collection gap breakdown (CGST, SGST, IGST)
- FBA and referral fee composition
- Advertising ACOS and wasted-spend identification by campaign

### PowerPoint Report Generation
A four-slide presentation is generated using `python-pptx` with no AI involvement at the layout stage. Slides are structured as: Title, Executive Summary, Revenue Leakage Map, and AI Action Plan. The report is available for immediate download on generation.

### Conversational Interface
A persistent chat interface allows natural-language queries against all loaded documents within a session. Conversation history is maintained across turns and passed to Gemini as context. Suggested starter questions are surfaced when no history exists.

### Session State Management
Application state — documents, validation results, analysis output, Amazon reconciliation, and conversation history — is maintained in Streamlit session state across tab navigation within a session. A session identifier is displayed in the sidebar for reference.

---

## Architecture

```
Universal Input
(CSV / Excel / PDF / Image / WhatsApp)
          |
          v
  image_proc.py
  PIL preprocessing for image inputs
          |
          v
  extractor.py
  Gemini 1.5 Flash → structured extraction
  Output validated against Pydantic schema
          |
          v
  validator.py                         [No AI — pure Python]
  Confidence scoring (0.0 – 1.0)
  GSTIN format check
  Missing field detection
  Score > 0.9  ──────────────────────> analyst.py
  Score ≤ 0.9  ──> HITL question loop ──> analyst.py (after answer)
          |
          v
  analyst.py
  Pandas calculations
  Auto-selected Plotly charts
  Gemini narrative + action items
          |
          v
  amazon_recon.py                      [Optional — separate upload]
  MTR vs Settlement matching
  GST gap detection
  FBA fee audit
  Advertising ACOS + waste analysis
          |
          v
  ppt_gen.py                           [No AI — pure python-pptx]
  Slide 1: Title
  Slide 2: Executive Summary
  Slide 3: Revenue Leakage Map
  Slide 4: AI Action Plan
          |
          v
  app.py — Streamlit (6 tabs)
  Upload & Extract  |  Validate  |  Analyse
  Amazon Recon      |  Report    |  AI Chat
```

---

## Tech Stack

| Component | Library / Service | Version |
|---|---|---|
| LLM Backend | Google Gemini 1.5 Flash (`google-generativeai`) | >= 0.7.0 |
| Web Framework | Streamlit | >= 1.32.0 |
| Data Validation | Pydantic | >= 2.0.0 |
| Data Processing | Pandas | >= 2.0.0 |
| PDF Parsing | pdfplumber | >= 0.10.0 |
| Image Processing | Pillow | >= 10.0.0 |
| Visualization | Plotly | >= 5.18.0 |
| Report Generation | python-pptx | >= 0.6.21 |
| Spreadsheet I/O | openpyxl | >= 3.1.0 |

---

## Repository Structure

```
guardian-ai/
├── app.py              # Streamlit entry point — 6-tab UI, session state, CSS
├── state.py            # Pydantic models: AgentState, FinancialDocument,
│                       #   ConversationMessage, HITLQuestion, AnalysisResult,
│                       #   AmazonRecon, ConfidenceLevel, DocumentType
├── extractor.py        # Gemini multimodal extraction → Pydantic-validated JSON
├── validator.py        # Confidence scoring, GSTIN validation, HITL generation
├── analyst.py          # Pandas stats, Plotly auto-charts, Gemini narrative
├── amazon_recon.py     # MTR/Settlement reconciliation, leakage waterfall,
│                       #   fee pie chart, ad waste analysis
├── image_proc.py       # PIL preprocessing pipeline for image inputs
├── ppt_gen.py          # python-pptx report builder (4-slide structure)
├── requirements.txt    # Pinned dependencies
└── .devcontainer/      # Dev container configuration for Codespaces
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A Google Gemini API key (free tier is sufficient — see below)

### Obtaining a Gemini API Key

1. Visit [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with a Google account and click **Create API Key**
3. Copy the key — it begins with `AIza`

Free tier limits as of 2025: 1,500 requests per day, 1 million tokens per minute. This is sufficient for development and moderate production use.

### Installation

```bash
git clone https://github.com/mdsajid2003/guardian-ai.git
cd guardian-ai
pip install -r requirements.txt
```

### Configuration

Set the API key as an environment variable before running:

```bash
# macOS / Linux
export GEMINI_API_KEY="AIza..."

# Windows (Command Prompt)
set GEMINI_API_KEY=AIza...

# Windows (PowerShell)
$env:GEMINI_API_KEY = "AIza..."
```

### Running Locally

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`.

---

## Streamlit Cloud Deployment

### Repository Setup

Ensure `requirements.txt` is present at the root of the repository. Do not commit `.streamlit/secrets.toml`.

### Secrets Configuration

Create `.streamlit/secrets.toml` for local testing only:

```toml
GEMINI_API_KEY = "AIza..."
```

Add this file to `.gitignore`.

### Deployment Steps

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repository
3. Set the main file path to `app.py`
4. Open **App Settings → Secrets** and paste:

```toml
GEMINI_API_KEY = "AIza..."
```

5. Click **Deploy**

The application will be available at a public `*.streamlit.app` URL within a few minutes.

---

## Application Walkthrough

### Tab 1 — Upload and Extract

Upload one or more financial documents. Supported formats: CSV, Excel, PDF, JPEG, PNG, WebP. Click **Extract All Documents** to run Gemini extraction across all uploaded files simultaneously. Each document displays its extracted line item count, overall confidence score, total amount, and any flagged missing fields. The first ten line items are shown in a preview table.

### Tab 2 — Validate

Click **Run Validation** to score all loaded documents. The confidence gate is displayed with an overall average. Documents below the 0.9 threshold surface one HITL question at a time; answers are submitted inline and appended to the conversation history. Once all questions are resolved, the session proceeds.

### Tab 3 — Analyse

Click **Run Full Analysis** to execute the Pandas analysis pipeline and generate Gemini narrative insights. The output includes a summary statement, a ranked list of observations, a prioritised action plan, and up to two automatically selected Plotly charts. The full consolidated transaction table is available in an expandable section.

### Tab 4 — Amazon Reconciliation

Upload MTR and Settlement reports (advertising report is optional). Click **Run Reconciliation** to produce the leakage waterfall chart, fee pie chart, GST breakdown table, and advertising waste analysis. Key metrics — MTR total, settlement total, leakage amount, and ACOS — are displayed as headline figures.

### Tab 5 — Report and Export

Enter a company or seller name for the presentation cover slide, then click **Generate PPT Report**. The four-slide deck is available for immediate download. A consolidated CSV export of all loaded transaction data is also available on this tab.

### Tab 6 — AI Chat

Submit natural-language questions about the loaded data. The last 20 messages of conversation history are displayed. Suggested starter questions are shown when no conversation exists. All prior turns are passed to Gemini as context for each new query.

---

## Upgrading the Model

To use Gemini 1.5 Pro for higher extraction quality (at lower free-tier rate limits), update the `model_name` parameter in `extractor.py`, `analyst.py`, and `amazon_recon.py`:

```python
# Current
model_name="gemini-1.5-flash"

# Higher quality
model_name="gemini-1.5-pro"
```

Note that Gemini 1.5 Pro has a lower free-tier request quota. For production workloads, enabling billing on your Google AI Studio account is recommended.

---

## Known Limitations

- **Context window for analysis:** The analyst module operates on all loaded documents, but very large datasets (tens of thousands of line items) may exceed Gemini's context window. Pre-filtering large reports before upload is advisable.
- **HITL is single-threaded:** The HITL loop surfaces one question per turn. Sessions with many low-confidence documents may require multiple validation passes.
- **Amazon reconciliation format dependency:** The `amazon_recon` module expects MTR and Settlement reports in their standard Amazon Seller Central export format. Non-standard column names will require a pre-processing step.
- **WhatsApp image quality:** Extraction accuracy for image-based receipts is dependent on input image resolution and contrast. Blurred or low-resolution screenshots may produce confidence scores below the validation threshold.
- **Session persistence:** Application state is held in Streamlit session state and does not persist across browser sessions or server restarts.

---

## Contributing

Contributions are welcome. To propose a change:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes with a descriptive message: `git commit -m "Describe the change"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request against `main`

Please ensure any new dependencies are added to `requirements.txt` with minimum version constraints.

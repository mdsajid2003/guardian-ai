# Guardian AI (Gemini Edition) — Deployment Guide

> **Free-tier version** — powered by Google Gemini (`gemini-1.5-flash`).  
> Drop-in replacement for the Anthropic/Claude original. All UI and logic is identical.

---
---

## 1 · Get a Free Gemini API Key

1. Go to → **https://aistudio.google.com/app/apikey**
2. Click **Create API Key**
3. Copy the key (starts with `AIza…`)

**Free tier limits** (as of 2025): 1,500 requests/day, 1M tokens/min — plenty for testing.

---

## 2 · Local Development

```bash
# Clone / unzip the project
cd guardian_ai_gemini

# Install dependencies (no anthropic package needed)
pip install -r requirements.txt

# Set your Gemini key
export GEMINI_API_KEY="AIza..."          # macOS / Linux
set GEMINI_API_KEY=AIza...               # Windows CMD

# Run
streamlit run app.py
```

---

## 3 · Streamlit Cloud Deployment

### File structure expected
```
guardian_ai_gemini/
├── app.py
├── state.py
├── extractor.py
├── validator.py
├── analyst.py
├── amazon_recon.py
├── ppt_gen.py
├── image_proc.py
├── requirements.txt
└── .streamlit/
    └── secrets.toml        ← never commit this file
```

### secrets.toml
```toml
GEMINI_API_KEY = "AIza..."
```

### Deploy steps
1. Push to GitHub
2. Go to **https://share.streamlit.io**
3. Connect repo → set main file: `app.py`
4. **App Settings → Secrets** → paste:
   ```
   GEMINI_API_KEY = "AIza..."
   ```
5. Click **Deploy**

---

## Architecture Overview

```
Universal Input (CSV/Excel/PDF/Image/WhatsApp)
        │
        ▼
extractor.py  ←─ image_proc.py (PIL preprocessing)
  Gemini 1.5 Flash → Pydantic JSON schema
        │
        ▼
validator.py  (LangGraph confidence gate — pure Python, no AI)
  Score > 0.9 → analysis | Score ≤ 0.9 → HITL loop
        │
        ▼
analyst.py  (Pandas + Plotly + Gemini insights)
  Auto-chart: line / bar / waterfall / pie / heatmap
        │
        ▼
amazon_recon.py  (optional module)
  MTR vs Settlement · GST gap · FBA fees · Ad ACOS
        │
        ▼
ppt_gen.py  (python-pptx — no AI, pure layout)
  4 slides: Title | Exec Summary | Leakage Map | Action Plan
        │
        ▼
app.py  (Streamlit Cloud)
  6 tabs: Upload | Validate | Analyse | Amazon | Report | Chat
```

---

## Upgrading to Gemini 1.5 Pro (higher quality, still free tier)

In `extractor.py`, `analyst.py`, and `amazon_recon.py`, change:

```python
model_name="gemini-1.5-flash",
```
→
```python
model_name="gemini-1.5-pro",
```

Pro has higher quality but lower free-tier rate limits.

# TruthLens — Fake News Detection & Verification System

> **Hybrid AI system combining LSTM neural network + N8N automation + NewsAPI live evidence for real-time fake news detection.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![N8N](https://img.shields.io/badge/N8N-Workflow-EA4B71?style=flat&logo=n8n&logoColor=white)](https://n8n.io)
[![Tests](https://img.shields.io/badge/Tests-8%2F8%20Passing-brightgreen?style=flat)](#test-results)

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [N8N Setup — Evidence Engine](#n8n-setup--evidence-engine)
7. [Running the System](#running-the-system)
8. [File Execution Sequence](#file-execution-sequence)
9. [API Endpoints](#api-endpoints)
10. [Input Types](#input-types)
11. [How the Verdict Works](#how-the-verdict-works)
12. [Test Results](#test-results)
13. [Known Limitations](#known-limitations)
14. [Future Roadmap](#future-roadmap)

---

## What This Project Does

TruthLens takes a news article, headline, or claim and:

1. **Classifies it** using a trained LSTM neural network (REAL or FAKE, with confidence %)
2. **Retrieves live evidence** via an automated N8N workflow that queries NewsAPI and scores article credibility
3. **Explains the prediction** using LIME — showing which words pushed the model toward FAKE or REAL
4. **Detects language manipulation** — scanning for sensationalism, propaganda, vague claims, emotional manipulation
5. **Synthesises a final verdict** by combining all four signals

```
User Input (text)
      │
      ├──► LSTM Model          → FAKE/REAL + confidence
      ├──► N8N + NewsAPI       → credibility + related articles
      ├──► LIME Explainer      → top 10 influential words
      └──► Language Scanner    → suspicion score
                │
                ▼
         Final Verdict  (🔴 STRONGLY FAKE / 🟢 VERIFIED REAL / ⚠️ CONFLICT / ...)
```

### Current Status

| Input Mode | Status |
|---|---|
| Plain text analysis | ✅ Fully working |
| CLI interactive menu | ✅ Fully working |
| FastAPI REST endpoints | ✅ Fully working |
| URL / webpage scraping | 🚧 Architecture ready, testing in progress |
| PDF / DOCX / TXT upload | 🚧 Architecture ready, testing in progress |
| Live news feed | 🔮 Planned |
| Streamlit web UI | 🔮 Planned |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUT                           │
│              (text / url / document / live news)            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI  (port 8000)                     │
│                      src/api/main.py                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               predict.py  (core pipeline)                   │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  LSTM Model │  │    LIME XAI  │  │  Language Scanner │  │
│  │  689K params│  │  500 perturb │  │  4 pattern types  │  │
│  └──────┬──────┘  └──────┬───────┘  └─────────┬─────────┘  │
│         │                │                     │            │
│         └────────────────┴─────────────────────┘            │
│                          │                                  │
│                          ▼                                  │
│                   combine_results()                         │
└──────────────────────────┬──────────────────────────────────┘
                           │              │
                           │         POST │ {"text": "..."}
                           │              ▼
                           │   ┌──────────────────────────┐
                           │   │   N8N  (port 5678)       │
                           │   │                          │
                           │   │  Webhook                 │
                           │   │    → Text Cleaning       │
                           │   │    → Keyword Extraction  │
                           │   │    → NewsAPI Request     │
                           │   │    → Article Analysis    │
                           │   │    → Respond to Webhook  │
                           │   └──────────┬───────────────┘
                           │              │ credibility, matches,
                           │              │ trusted_sources, articles[]
                           └──────────────┘
                                    │
                                    ▼
                             Final JSON Response
```

---

## Project Structure

```
fake-news-detector/
│
├── models/
│   ├── lstm_model.h5              ← Trained LSTM model (689K params, ~8MB)
│   └── tokenizer.pkl              ← Fitted Keras tokenizer (115K vocab)
│
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                ← FastAPI app (all 4 input endpoints)
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── predict.py             ← Core: LSTM + LIME + N8N + verdict
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py           ← CSV ingestion and label assignment
│   │   └── preprocess.py          ← Text cleaning (lowercase, strip, regex)
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── tokenizer.py           ← Keras tokenizer training and saving
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── lstm_model.py          ← LSTM architecture definition
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── lime_explainer.py      ← LIME wrapper utilities
│   │
│   └── services/
│       ├── __init__.py
│       ├── news_service.py        ← NewsAPI client (fetch headlines, search)
│       ├── url_handler.py         ← BeautifulSoup URL scraper    [🚧]
│       ├── document_handler.py    ← PDF / DOCX / TXT extractor   [🚧]
│       └── evidence_service.py    ← Evidence aggregator
│
├── n8n/
│   └── Fake_News_Evidence_Engine_-_Final_Working.json   ← Workflow JSON
│
├── frontend/
│   └── app.py                     ← Streamlit UI (4 tabs)        [🔮]
│
├── main_interactive.py            ← CLI interactive menu (7 options)
├── test_system.py                 ← Automated 8-test suite
├── config.py                      ← Global constants and paths
├── requirements.txt               ← Python dependencies
└── README.md
```

---

## Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Node.js | 18+ | Required by N8N |
| N8N | Latest | Workflow automation (evidence engine) |
| NewsAPI key | Free tier | Live article retrieval |
| Git | Any | Clone the repo |

Get a free NewsAPI key at: https://newsapi.org/register

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

### Step 2 — Create and activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install fastapi uvicorn tensorflow keras \
    lime scikit-learn requests beautifulsoup4 \
    python-multipart PyPDF2 python-docx \
    pandas numpy aiofiles streamlit
```

### Step 4 — Place model files

Copy the trained model and tokenizer into the `models/` directory:

```
models/
├── lstm_model.h5      ← required
└── tokenizer.pkl      ← required
```

### Step 5 — Create package init files

Run once from the project root:

```bash
# Windows PowerShell
@("src","src/api","src/pipeline","src/data","src/features","src/models","src/explainability","src/services") | ForEach-Object { New-Item -Path "$_/__init__.py" -ItemType File -Force }

# macOS / Linux
for d in src src/api src/pipeline src/data src/features src/models src/explainability src/services; do
    touch $d/__init__.py
done
```

---

## N8N Setup — Evidence Engine

The N8N workflow is the real-time evidence engine. It must be running and **published** before predictions work with full evidence.

### Step 1 — Install N8N

```bash
npm install -g n8n
```

### Step 2 — Start N8N

```bash
n8n start
```

Open in browser: **http://localhost:5678**

Create an account on first launch (local only, no external signup needed).

### Step 3 — Import the workflow

1. In the N8N UI, click **"+"** → **"Import"** → **"Import from File"**
2. Select: `n8n/Fake_News_Evidence_Engine_-_Final_Working.json`
3. The full 6-node workflow loads automatically

### Step 4 — Set your NewsAPI key

1. Open the **"NewsAPI Request"** node (the globe icon)
2. Find the `apiKey` parameter in the query parameters
3. Replace the existing key with your own from https://newsapi.org

### Step 5 — Publish the workflow

> ⚠️ This is the most commonly missed step.

Click **"Publish"** in the top-right corner of the workflow editor.

- **Test mode** (red "Execute workflow" button visible) → uses `/webhook-test/evidence` → Python **cannot** reach it
- **Published mode** (green "Published" badge) → uses `/webhook/evidence` → Python connects ✅

### Step 6 — Verify the webhook

```powershell
# PowerShell
Invoke-RestMethod -Method POST `
  -Uri "http://localhost:5678/webhook/evidence" `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"text":"India central bank inflation rate announcement"}'
```

```bash
# bash / curl
curl -X POST http://localhost:5678/webhook/evidence \
  -H "Content-Type: application/json" \
  -d '{"text":"India central bank inflation rate announcement"}'
```

**Expected response:**

```json
{
  "query_used": "india central bank inflation rate",
  "match_count": 5,
  "trusted_sources": 0,
  "credibility": "MEDIUM",
  "related_articles": [
    "https://...",
    "https://..."
  ]
}
```

If you see `"No Respond to Webhook node found"` → the workflow is not published. Click Publish.

---

## Running the System

Always start in this order:

### Terminal 1 — N8N

```bash
n8n start
```

Confirm: workflow is Published at http://localhost:5678

---

### Terminal 2 — FastAPI backend

```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Confirm: http://127.0.0.1:8000 returns the endpoint list

---

### Terminal 3 — CLI interface (primary user interface)

```bash
python main_interactive.py
```

Presents the 7-option interactive menu:

```
╔══════════════════════════════════════════════════════════╗
║         🚀 FAKE NEWS DETECTION SYSTEM v3.0 🚀           ║
║         Powered by: LSTM + LIME + N8N + NewsAPI          ║
╚══════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────┐
│                      MAIN MENU                          │
├─────────────────────────────────────────────────────────┤
│  1. Analyze Text News Article          ← ✅ working     │
│  2. Analyze News from URL              ← 🚧 testing     │
│  3. Analyze Uploaded Document          ← 🚧 testing     │
│  4. Batch Analysis (Multiple Articles) ← 🚧 testing     │
│  5. Search & Verify News Topic         ← 🚧 testing     │
│  6. About This System                                   │
│  7. Exit                                                │
└─────────────────────────────────────────────────────────┘
```

---

### Run the test suite

```bash
python test_system.py
```

All 8 tests should pass when both N8N (published) and FastAPI are running.

---

## File Execution Sequence

This is the canonical order of execution — from training a new model to running predictions.

### A — If you need to retrain the model from scratch

```
1. data/raw/Fake.csv + True.csv        ← source datasets (Kaggle)
        │
        ▼
2. python -m src.data.load_data        ← loads CSVs, adds labels (FAKE=1, REAL=0), merges
        │
        ▼
3. python -m src.data.preprocess       ← cleans text (lowercase, strip, regex), saves cleaned_data.csv
        │
        ▼
4. python -m src.features.tokenizer    ← fits Keras tokenizer, pads sequences, saves tokenizer.pkl
        │
        ▼
5. python -m src.training.train        ← builds LSTM, trains with class weights, saves lstm_model.h5
        │
        ▼
6. models/lstm_model.h5 + tokenizer.pkl   ← ready for inference
```

### B — If you already have the trained model (normal usage)

```
1. n8n start                           ← start evidence engine (keep running)
        │
        ▼
2. Publish workflow in N8N UI          ← http://localhost:5678 → click Publish
        │
        ▼
3. python -m uvicorn src.api.main:app  ← start REST API (keep running in separate terminal)
        │
        ▼
4. python main_interactive.py          ← use the CLI menu for text analysis
   OR
   python test_system.py               ← verify all 8 components pass
   OR
   streamlit run frontend/app.py       ← web UI (when ready)
```

### C — Single prediction (direct module call)

```bash
python -m src.pipeline.predict
```

Runs the built-in test cases and prints predictions to terminal.

---

## API Endpoints

Base URL: `http://127.0.0.1:8000`

| Method | Endpoint | Input | Description |
|---|---|---|---|
| GET | `/` | — | Lists all endpoints |
| GET | `/health` | — | Model loaded status, N8N URL |
| POST | `/predict/text` | `{"text": "..."}` | Analyze plain text ✅ |
| POST | `/predict/url` | `{"url": "https://..."}` | Scrape and analyze URL 🚧 |
| POST | `/predict/document` | `multipart/form-data file=` | PDF / DOCX / TXT 🚧 |
| GET | `/news` | `?limit=10` | Live headlines batch analysis 🔮 |

### Example — Text prediction

```bash
curl -X POST http://127.0.0.1:8000/predict/text \
  -H "Content-Type: application/json" \
  -d '{"text": "India central bank RBI keeps repo rate unchanged amid inflation"}'
```

**Response:**

```json
{
  "input_type": "text",
  "prediction": "REAL",
  "confidence": 0.9908,
  "raw_probability": 0.0076,
  "final_verdict": "⚠️ CONFLICT: Model says REAL but Low Credibility",
  "evidence": {
    "query_used": "india central bank repo rate inflation",
    "match_count": 2,
    "trusted_sources": 0,
    "credibility": "LOW",
    "related_articles": ["https://..."],
    "n8n_status": "success"
  },
  "explanation": [
    {"word": "bank", "impact": -0.185, "towards": "REAL"},
    {"word": "central", "impact": -0.184, "towards": "REAL"},
    {"word": "new", "impact": 0.099, "towards": "FAKE"}
  ],
  "language_analysis": {
    "suspicion_score": 0.0,
    "language_analysis": "NORMAL"
  },
  "processing_time": 3.87,
  "status": "success"
}
```

---

## Input Types

### ✅ Plain Text (working)

Paste any news article, headline, or claim directly. Minimum 10 characters.

```
python main_interactive.py → option 1
```

### 🚧 URL Analysis (in progress)

Provide any `http://` or `https://` URL. The system fetches the page, strips boilerplate (nav/footer/ads), extracts headline and article body, then runs the full prediction pipeline.

```
python main_interactive.py → option 2
```

### 🚧 Document Upload (in progress)

Upload a PDF, DOCX, or TXT file (max 10 MB). Long documents are automatically chunked into 1500-character segments with 200-character overlap. Each chunk is analysed independently; a summary verdict is computed from the document summary.

```
python main_interactive.py → option 3
POST http://127.0.0.1:8000/predict/document  (multipart/form-data)
```

---

## How the Verdict Works

The `combine_results()` function in `predict.py` maps four inputs to a final verdict:

| ML Prediction | Confidence | N8N Credibility | Final Verdict |
|---|---|---|---|
| FAKE | > 85% | LOW + suspicious language | 🔴 STRONGLY FAKE |
| FAKE | > 85% | LOW | 🔴 LIKELY FAKE (High ML + Low Credibility) |
| FAKE | > 85% | MEDIUM | ⚠️ POTENTIALLY FAKE (Mixed Evidence) |
| FAKE | > 85% | HIGH | ⚠️ CONFLICT: Model FAKE, Evidence Credible |
| REAL | > 85% | HIGH + 2+ trusted sources | 🟢 VERIFIED REAL |
| REAL | > 85% | HIGH | 🟢 LIKELY REAL |
| REAL | > 85% | LOW | ⚠️ CONFLICT: Model REAL, Low Credibility |
| Any | 60–85% | Any | ⚠️ QUESTIONABLE |
| Any | < 60% | Any / N8N down | ⚠️ INCONCLUSIVE — ML-only fallback |

> **Why does "CONFLICT: Model says REAL but Low Credibility" appear so often?**
> The free tier of NewsAPI does not reliably return BBC / Reuters articles. `trusted_sources` is usually 0 even for legitimate news. This is a NewsAPI tier limitation — the ML prediction is still correct. Upgrade to the paid NewsAPI tier or add GNews API as a secondary source to resolve this.

---

## Test Results

Run: `python test_system.py`

```
✅ N8N Connectivity ........................... PASS
✅ N8N Webhook (POST /webhook/evidence) ....... PASS
✅ Model Loading (lstm_model.h5) .............. PASS
✅ Prediction Pipeline ........................ PASS
✅ URL Handler ................................ PASS
✅ Document Handler ........................... PASS
✅ API Endpoints .............................. PASS
✅ Complete Integration (LSTM + N8N + LIME) ... PASS

Total: 8/8 tests passed (100%)
```

> Requires both N8N (published) and FastAPI to be running for all 8 to pass.

---

## Configuration

All constants are in `config.py` and `src/pipeline/predict.py`:

| Parameter | Value | Description |
|---|---|---|
| `MODEL_PATH` | `models/lstm_model.h5` | Trained LSTM model |
| `TOKENIZER_PATH` | `models/tokenizer.pkl` | Fitted tokenizer |
| `MAX_LEN` | `100` | Token sequence length (raise to 300 for better accuracy) |
| `VOCAB_SIZE` | `5000` | Active vocabulary at training (raise to 20K for better accuracy) |
| `TEMPERATURE` | `1.2` | Confidence calibration — T > 1 softens overconfident predictions |
| `N8N_WEBHOOK_URL` | `http://localhost:5678/webhook/evidence` | N8N endpoint |
| `N8N_TIMEOUT` | `15` seconds | Per-request timeout; 3 retries with 1.5s backoff |
| `API_PORT` | `8000` | FastAPI server |
| `N8N_PORT` | `5678` | N8N server |
| `MAX_FILE_SIZE` | `10 MB` | Document upload limit |
| `CHUNK_SIZE` | `1500 chars` | Chunk size for long documents (200-char overlap) |

---

## Known Limitations

| Issue | Impact | Fix |
|---|---|---|
| NewsAPI free tier | `trusted_sources` usually = 0; causes frequent CONFLICT verdicts | Upgrade NewsAPI tier or add GNews API |
| `VOCAB_SIZE = 5000` | Only 4% of 115K vocabulary used; loses rare-word signal | Raise to 20,000–30,000 and retrain |
| `MAX_LEN = 100` | Articles > ~80 words are truncated | Raise to 300 and retrain |
| LIME adds 2–4s latency | Slows each prediction | Reduce `num_samples=500` to `100` for speed |
| Single LSTM layer | Misses deeper patterns in complex articles | Stack LSTM(128) → LSTM(64) |
| Windows GPU warning | TF ≥ 2.11 has no native Windows GPU support | Use WSL2 for GPU training |

---

## Future Roadmap

```
Phase 1 — Model Improvements
  └── VOCAB_SIZE 5K → 20K, MAX_LEN 100 → 300, stacked LSTM, GloVe embeddings

Phase 2 — Transformer Upgrade
  └── RoBERTa / BERT fine-tuned on FakeNewsNet + LIAR datasets

Phase 3 — Evidence Engine Expansion
  └── Validate URL / PDF / DOCX inputs, add GNews + Bing News, AI query refinement

Phase 4 — Multi-Modal Input
  └── OCR for screenshots/memes, Twitter/X API, Reddit API

Phase 5 — Deployment
  └── Docker Compose (FastAPI + N8N + Streamlit), AWS / GCP, CI/CD, Redis caching
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No Respond to Webhook node found` | Workflow in test mode | Click **Publish** in N8N UI |
| `Connection refused :5678` | N8N not running | Run `n8n start` |
| `Connection refused :8000` | FastAPI not running | Run `uvicorn src.api.main:app --reload --port 8000` |
| `Model file not found` | Missing `models/` files | Copy `lstm_model.h5` and `tokenizer.pkl` to `models/` |
| `ModuleNotFoundError: src.*` | Wrong working directory | Run all commands from project root |
| `WARNING: TensorFlow GPU not available` | TF ≥ 2.11 on Windows | Expected — model runs on CPU; use WSL2 for GPU |
| `LIME explanation unavailable` | `lime` package not installed | `pip install lime` |

---

## Quick Start (TL;DR)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Place model files
#    models/lstm_model.h5
#    models/tokenizer.pkl

# 3. Start N8N (Terminal 1)
n8n start
# → open localhost:5678 → import workflow JSON → click Publish

# 4. Start FastAPI (Terminal 2)
python -m uvicorn src.api.main:app --reload --port 8000

# 5. Run (Terminal 3)
python main_interactive.py     # interactive CLI
# OR
python test_system.py          # verify everything passes
```

---

*TruthLens v3.1 — LSTM + LIME + N8N + NewsAPI + FastAPI*
*Core pipeline: Text Analysis fully operational | URL / Document: in progress*

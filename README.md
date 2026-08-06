# 🛡️ Fraud Detection System

> A web application that automatically scans financial transactions,
> scores them for risk, and flags suspicious ones — built with a FastAPI backend
> deployed on **Hugging Face** and an interactive dashboard deployed on **Streamlit Cloud**.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Backend-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Streamlit Cloud](https://img.shields.io/badge/Frontend-Streamlit%20Cloud-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

---

## 🔗 Live Demo

| | URL |
|---|---|
| 🖥️ **Web Dashboard (Frontend)** | `https://fraud-detection-system0.streamlit.app` |
| ⚙️ **API Backend** | `https://geetanjali09-fraud-detection-api.hf.space` |
| 📖 **Interactive API Docs** | `https://geetanjali09-fraud-detection-api.hf.space/docs` |
| ❤️ **API Health Check** | `https://geetanjali09-fraud-detection-api.hf.space/health` |
| 💾 **Source Code** | [github.com/Geetanjali098/Fraud-Detection-System](https://github.com/Geetanjali098/Fraud-Detection-System) |

> 💡 **Quick test:** Open the dashboard → upload `data/sample_data.csv` → click **Analyze** → see results instantly!

---

## 📌 Table of Contents

1. [What Problem Does This Solve?](#-what-problem-does-this-solve)
2. [How Does It Work?](#-how-does-it-work)
3. [Features](#-features)
4. [Tech Stack](#️-tech-stack)
5. [Project Structure](#-project-structure)
6. [How to Use It](#️-how-to-use-it)
7. [API Reference](#-api-reference)
8. [Run Locally](#-run-locally)
9. [Deploy — Hugging Face + Streamlit Cloud](#️-deploy--hugging-face--streamlit-cloud)
10. [What I Learned](#-what-i-learned)

---

## 🤔 What Problem Does This Solve?

### The Real-World Problem

Banks and payment companies process **billions of transactions every single day**.
Hidden inside those transactions are fraudulent ones — stolen cards, fake accounts,
and money laundering attempts.

**Why is this hard to solve?**

- ❌ **Manual review** — impossible at scale, too slow, too expensive
- ❌ **Simple rules** like *"flag anything above $5,000"* — misses clever fraud
  that stays under the threshold, and blocks legitimate large purchases
- ❌ **Single algorithm** — one technique alone produces too many false alarms

### What This System Does

This system acts like an **automated fraud analyst** that:

- 📥 Takes in a list of financial transactions
- 🔬 Analyzes each one using **two different AI techniques simultaneously**
- 📊 Gives every transaction a **risk score from 0 to 100**
- 🚦 Labels each transaction as **LOW**, **MEDIUM**, or **HIGH** risk
- 🚨 Highlights suspicious ones so humans only review what actually matters
- ⚡ Processes hundreds of transactions in **seconds — not hours**

### Who Would Use This?

| User | How They Use It |
|---|---|
| **Fintech Startups** | Plug the API into their payment system for real-time fraud screening |
| **Data Analysts** | Upload transaction CSV files to find anomalies in large datasets |
| **Banks & Finance Teams** | Integrate into existing fraud review workflows via REST API |
| **Developers** | Call the API from any application to add fraud detection |

---

## 💡 How Does It Work?

The system uses **two algorithms working together** — combining methods gives
better accuracy than relying on just one.

---

### 🔢 Algorithm 1 — Z-Score (Statistical Method)

Imagine your customers typically spend between $50 and $300 per transaction.
That is your normal range. Now someone spends $14,000 in one transaction.
That is statistically very far from normal — it stands out immediately.

Z-score measures exactly **how far a transaction is from the average**,
in units called standard deviations.

```
Example:
Average transaction  =  $150
Standard deviation   =  $80

$200 transaction   →  Z-score = 0.6  →  Normal ✅
$14,000 transaction →  Z-score = 4.8  →  Very suspicious 🚨
```

**Rule of thumb:**
- Z-score below 2 → probably normal
- Z-score above 3 → worth investigating
- Z-score above 4 → very likely fraudulent

---

### 🌲 Algorithm 2 — Isolation Forest (Machine Learning)

Imagine a guessing game where you ask yes/no questions to separate one
transaction from all the others. For example:
- *"Is the amount above $1,000?"*
- *"Is it below $500?"*

Normal transactions blend in with the crowd — it takes **many questions**
to isolate them because they look like everyone else.

Fraudulent transactions are rare and unusual — they get **isolated in just
a few questions** because they stand out from the rest.

Isolation Forest automates this across thousands of transactions and scores
each one on how quickly it gets isolated.
The easier to isolate → the more suspicious.

**Why this matters:** It catches patterns that Z-score misses — unusual
combinations that aren't just about the amount alone.

---

### 🧮 Combined Risk Score (0 to 100)

Both signals are blended into one final score:

```
Z-component  =  min(Z-score, 10) / 10         → scaled to 0–1
IF-component =  1 - normalise(IF raw score)   → inverted (lower = more anomalous)

Risk Score   =  (50% × Z-component) + (50% × IF-component) × 100

Result: a number from 0 to 100
```

| Score | Risk Level | Meaning |
|---|---|---|
| 0 – 30 | 🟢 **LOW** | Normal transaction — no action needed |
| 31 – 70 | 🟡 **MEDIUM** | Slightly unusual — worth monitoring |
| 71 – 100 | 🔴 **HIGH** | Likely fraudulent — review immediately |

---

### 🔄 End-to-End System Flow

```
┌──────────────────────────────────────────────────────────────┐
│  USER                                                        │
│  Uploads CSV file  OR  manually enters transaction amounts   │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  STREAMLIT FRONTEND  (Streamlit Cloud)                       │
│  • Accepts file or manual input from user                    │
│  • Sends HTTP POST request to FastAPI backend                │
│  • Displays results, charts, and flagged transactions        │
└───────────────────────────┬──────────────────────────────────┘
                            │  HTTP Request
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  FASTAPI BACKEND  (Hugging Face Space)                       │
│                                                              │
│  Step 1 → Validate all inputs                               │
│  Step 2 → Calculate Z-score for every transaction           │
│  Step 3 → Run Isolation Forest ML model                     │
│  Step 4 → Combine into Risk Score (0–100)                   │
│  Step 5 → Assign Risk Level: LOW / MEDIUM / HIGH            │
│  Step 6 → Return results + summary as JSON                  │
└───────────────────────────┬──────────────────────────────────┘
                            │  JSON Response
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  STREAMLIT FRONTEND  (shows results)                         │
│  • Summary cards: total, flagged, avg risk score            │
│  • 3 interactive Plotly charts                              │
│  • Full results table with colour-coded risk badges         │
│  • Flagged transactions table highlighted in red            │
│  • Download results as CSV                                  │
└──────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Backend (FastAPI on Hugging Face)
- 🔌 **4 REST API endpoints** — batch JSON, CSV upload, single transaction, health check
- ✅ **Input validation** — Pydantic v2 catches bad data before it reaches the ML model
- 🚫 **Helpful error messages** — tells exactly what is wrong and which row
- 📝 **Structured logging** — every request and prediction is recorded
- 📖 **Auto-generated Swagger docs** — available at `/docs` with zero extra work
- 🔓 **CORS enabled** — any frontend can connect to the API

### Frontend (Streamlit on Streamlit Cloud)
- 📂 **CSV file upload** — analyze hundreds of transactions at once
- ✏️ **Manual entry mode** — add and remove individual transactions interactively
- 📊 **3 interactive Plotly charts:**
  - Bar chart — count of LOW / MEDIUM / HIGH risk transactions
  - Scatter plot — Amount vs Risk Score (outliers jump out visually)
  - Histogram — Z-score distribution showing data spread
- 📋 **Summary metrics** — total, flagged count, flag rate %, average risk score
- 🚨 **Flagged-only table** — suspicious transactions shown separately with red highlights
- ⬇️ **CSV export** — download the full analyzed results
- 🔗 **API health checker** — verify backend is reachable from sidebar

---

## 🛠️ Tech Stack

| Layer | Technology | Version | Why This Choice |
|---|---|---|---|
| **Backend framework** | FastAPI | 0.111+ | Fastest Python web framework, auto-generates API docs |
| **API server** | Uvicorn | 0.29+ | Production-grade ASGI server |
| **ML — Statistics** | NumPy | 1.26+ | Fast Z-score computation |
| **ML — Algorithm** | scikit-learn | 1.4+ | Industry-standard Isolation Forest |
| **Data handling** | pandas | 2.2+ | CSV parsing and DataFrame operations |
| **Frontend** | Streamlit | 1.35+ | Full interactive dashboard in pure Python |
| **Charts** | Plotly | 5.22+ | Interactive charts with hover and zoom |
| **Validation** | Pydantic v2 | 2.7+ | Type-safe request and response validation |
| **Backend hosting** | Hugging Face Spaces | — | Free, always on, Docker support, ML-focused |
| **Frontend hosting** | Streamlit Cloud | — | Free forever, purpose-built for Streamlit |

---

## 📂 Project Structure

```
Fraud-Detection-System/
│
├── 📁 backend/
│   ├── __init__.py          ← makes it a Python package (required)
│   ├── main.py              ← API routes, endpoints, middleware
│   ├── fraud_model.py       ← Z-score + Isolation Forest + Risk Score
│   └── utils.py             ← CSV parsing, validation, summary helpers
│
├── 📁 frontend/
│   ├── __init__.py
│   └── app.py               ← Streamlit dashboard: upload, charts, results
│
├── 📁 data/
│   └── sample_data.csv      ← 50 test transactions (normal + suspicious)
│
├── Dockerfile               ← Docker config for Hugging Face deployment
├── requirements.txt         ← All Python dependencies
└── README.md
```

**Why is the code split this way?**

Each file has one job:
- `fraud_model.py` — only does ML logic
- `utils.py` — only does parsing and validation
- `main.py` — only handles API routing
- `app.py` — only handles the UI

This makes the code easy to test, debug, and extend.

---

## 🖥️ How to Use It

### Using the Web Dashboard (No coding needed)

**Step 1** — Open the app:
```
https://geetanjali098-fraud-detection-system.streamlit.app
```

**Step 2 — Connect With API:**

1.) Restart the given hugging face spaces n fill the API base URL:
```
https://huggingface.co/spaces/Geetanjali09/fraud-detection-api
```
2.) connect with your localhostURL or these given backend API
```
https://geetanjali09-fraud-detection-api.hf.space
```
**Step 3 — Upload CSV tab:**

Your CSV must have these two columns:
```csv
transaction_id,amount
TXN001,150.00
TXN002,9500.00
TXN003,75.50
```

- Click **"Browse files"** → select your CSV
- Click **"🔍 Analyze CSV"**
- Results appear below with charts and risk levels

**Step 4 — Manual Entry tab:**

- Type a Transaction ID (e.g. `TXN_TEST`)
- Enter an amount (e.g. `12000`)
- Click **"➕ Add Row"** — add as many as you want
- Click **"🔍 Analyze"**

**Step 5 — Read the results:**

```
📊 Summary Cards
   Total: 50  |  Flagged: 4  |  Flag Rate: 8%  |  Avg Risk: 23.4

📈 Three Charts
   Left:   Risk distribution bar chart
   Middle: Amount vs Risk Score scatter plot
   Right:  Z-score histogram

📋 Full Results Table
   Every transaction with Z-score, risk score, risk level, flag status

🚨 Flagged Transactions
   Only suspicious ones, highlighted in red

⬇️ Download Results as CSV
```

---

## 📡 API Reference

**Base URL:**
```
https://geetanjali09-fraud-detection-system-api.hf.space
```

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Landing page — confirms API is running |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Analyze multiple transactions (JSON) |
| `POST` | `/predict/csv` | Analyze transactions from CSV file |
| `POST` | `/predict/single` | Analyze one transaction |

---

### Example — Analyze Transactions

**Request:**
```bash
curl -X POST https://geetanjali09-fraud-detection-system-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"transaction_id": "TXN001", "amount": 150.00},
      {"transaction_id": "TXN002", "amount": 9500.00},
      {"transaction_id": "TXN003", "amount": 75.00}
    ]
  }'
```

**Response:**
```json
{
  "results": [
    {
      "transaction_id": "TXN001",
      "amount": 150.0,
      "z_score": 0.31,
      "anomaly_flag": false,
      "risk_score": 11.2,
      "risk_level": "LOW"
    },
    {
      "transaction_id": "TXN002",
      "amount": 9500.0,
      "z_score": 3.84,
      "anomaly_flag": true,
      "risk_score": 89.6,
      "risk_level": "HIGH"
    },
    {
      "transaction_id": "TXN003",
      "amount": 75.0,
      "z_score": 0.18,
      "anomaly_flag": false,
      "risk_score": 7.4,
      "risk_level": "LOW"
    }
  ],
  "summary": {
    "total_transactions": 3,
    "flagged_transactions": 1,
    "flag_rate_pct": 33.33,
    "avg_risk_score": 36.1,
    "by_risk_level": {"LOW": 2, "MEDIUM": 0, "HIGH": 1}
  }
}
```

### Response Fields Explained

| Field | Type | Meaning |
|---|---|---|
| `transaction_id` | string | Your original ID returned unchanged |
| `amount` | number | Transaction amount |
| `z_score` | number | Standard deviations from average (higher = more unusual) |
| `anomaly_flag` | boolean | `true` if Isolation Forest flagged this |
| `risk_score` | 0–100 | Final combined risk score |
| `risk_level` | string | `LOW`, `MEDIUM`, or `HIGH` |

---

## 🚀 Run Locally

### Step 1 — Clone the repo
```bash
git clone https://github.com/Geetanjali098/Fraud-Detection-System.git
cd Fraud-Detection-System
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Start the backend (Terminal 1)
```bash
uvicorn backend.main:app --reload
```
✅ API running at: `http://localhost:8000`
✅ API docs at: `http://localhost:8000/docs`

### Step 4 — Start the frontend (Terminal 2)
```bash
streamlit run frontend/app.py
```
✅ Dashboard at: `http://localhost:8501`

### Step 5 — Test it
Upload `data/sample_data.csv` → click **Analyze CSV** → see 4 HIGH risk transactions flagged ✅

---

## ☁️ Deploy — Hugging Face + Streamlit Cloud

This project uses a **split deployment** strategy:

```
Backend  → Hugging Face Spaces  (free, always on, Docker, ML-focused)
Frontend → Streamlit Cloud      (free forever, purpose-built for Streamlit)
```

---

### 🔷 Part 1 — Backend on Hugging Face

#### Why Hugging Face for the Backend?
- ✅ Free forever — no credit card needed
- ✅ Never sleeps — always available
- ✅ Docker support — full control over environment
- ✅ Built for ML apps — scikit-learn loads without timeout issues
- ✅ Auto-rebuilds when you push code

#### Dockerfile (place in repo root)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

> ⚠️ Three rules for Hugging Face Docker:
> 1. Always `WORKDIR /app` — not `/backend`
> 2. Always `ENV PYTHONPATH=/app` — fixes ModuleNotFoundError
> 3. Always port `7860` — Hugging Face requires this

#### Step-by-Step

**Step 1** — Go to [huggingface.co/spaces](https://huggingface.co/spaces)
→ **Create new Space** → SDK: **Docker** → Visibility: **Public**

**Step 2** — Clone your Space locally:
```bash
git lfs install
git clone https://huggingface.co/spaces/Geetanjali09/fraud-detection-system-api
cd fraud-detection-system-api
```

**Step 3** — Copy project files:
```powershell
Copy-Item -Path "..\Fraud-Detection-System\backend" -Destination "." -Recurse -Force
Copy-Item -Path "..\Fraud-Detection-System\data" -Destination "." -Recurse -Force
Copy-Item -Path "..\Fraud-Detection-System\requirements.txt" -Destination "." -Force
```

**Step 4** — Make sure `backend/__init__.py` exists:
```bash
# Create empty __init__.py
echo. > backend/__init__.py
```

**Step 5** — Push to Hugging Face:
```bash
git add .
git commit -m "deploy: fraud detection backend"
git remote set-url origin https://Geetanjali09:YOUR_HF_TOKEN@huggingface.co/spaces/Geetanjali09/fraud-detection-system-api
git push origin main
```

**Step 6** — Wait 3–5 minutes for build → verify:
```
https://geetanjali09-fraud-detection-system-api.hf.space/health
→ {"status": "healthy"} ✅
```

> 💡 The App tab on Hugging Face will show "Starting..." forever for a pure API.
> This is normal — always test using `/health` directly.

---

### 🔷 Part 2 — Frontend on Streamlit Cloud

#### Why Streamlit Cloud for the Frontend?
- ✅ 100% free forever
- ✅ Never sleeps
- ✅ Purpose-built for Streamlit apps
- ✅ Deploys directly from GitHub in one click
- ✅ Supports secrets for environment variables

#### Step-by-Step

**Step 1** — Make sure `frontend/app.py` reads API URL from environment:
```python
import os
import streamlit as st
API_BASE = st.secrets.get("API_BASE", os.getenv("API_BASE", "http://localhost:8000"))
```

**Step 2** — Push to GitHub:
```bash
git add frontend/app.py
git commit -m "fix: read API_BASE from streamlit secrets"
git push origin main
```

**Step 3** — Go to [share.streamlit.io](https://share.streamlit.io) → **Sign in with GitHub**

**Step 4** — Click **"New app"** → fill in:

| Field | Value |
|---|---|
| **Repository** | `Geetanjali098/Fraud-Detection-System` |
| **Branch** | `main` |
| **Main file path** | `frontend/app.py` |

**Step 5** — Click **"Advanced settings"** → paste in Secrets:
```toml
API_BASE = "https://geetanjali09-fraud-detection-system-api.hf.space"
```

**Step 6** — Click **"Deploy"** → wait 2 minutes → your app is live! 🚀

---

### 🔷 Final Architecture

```
GitHub Repo (Geetanjali098/Fraud-Detection-System)
         │                          │
         ▼                          ▼
Hugging Face Space           Streamlit Cloud
─────────────────            ───────────────
Backend (FastAPI)            Frontend (Streamlit)
Docker container             Managed Python env
Port 7860                    Auto port
Free + Always on ✅          Free + Always on ✅
         │                          │
         └──────────────────────────┘
              API_BASE connects them
              (set as Streamlit secret)

Share this URL with anyone:
https://geetanjali098-fraud-detection-system.streamlit.app
```

---

### ⚠️ Common Deployment Issues

| Issue | Platform | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'backend'` | Hugging Face | Add `ENV PYTHONPATH=/app` in Dockerfile + create `backend/__init__.py` |
| App tab shows "Starting..." forever | Hugging Face | Normal for APIs — test `/health` directly |
| Build takes 5+ minutes | Hugging Face | Normal — scikit-learn is large, just wait |
| Frontend can't reach backend | Streamlit Cloud | Check `API_BASE` secret is set correctly |
| `st.secrets` key error | Streamlit Cloud | Add `API_BASE` in Advanced Settings → Secrets |
| Port error on Hugging Face | Hugging Face | Always use port `7860` in Dockerfile |

---

## 📊 About the Sample Data

`data/sample_data.csv` contains **50 transactions** for testing:

```
Normal transactions (46):    $45 – $310    everyday spending
Suspicious transactions (4): $8,750 – $15,200  clear outliers
```

The 4 suspicious ones: **TXN016** ($9,800), **TXN023** ($11,500),
**TXN034** ($8,750), **TXN045** ($15,200)

After analysis, all 4 should appear as **HIGH risk** with scores above 70.

---

## 🧠 What I Learned

**Machine Learning:**
- How to combine a statistical method (Z-score) with an unsupervised ML
  algorithm (Isolation Forest) for better accuracy than either alone
- How Isolation Forest detects anomalies without labelled training data
- How to normalise and weight multiple signals into one interpretable score

**Software Engineering:**
- How to build a production-ready REST API with validation, error handling,
  logging, and auto-generated documentation
- Why separating concerns matters — ML logic, API routing, and utilities
  in separate files makes code easier to maintain and debug
- How to use lazy imports to keep startup time fast on serverless platforms

**Deployment:**
- How to containerize a Python ML app with Docker
- How to deploy a FastAPI backend on Hugging Face Spaces using Docker
- How to deploy a Streamlit frontend on Streamlit Cloud
- How environment variables and secrets connect two separate deployed services
- Why `PYTHONPATH` matters in Docker and how to set it correctly

---

## 🤝 Contributing

Found a bug or want to add a feature?

```bash
git checkout -b feature/your-feature-name
git commit -m "add: description of your change"
git push origin feature/your-feature-name
# Open a Pull Request on GitHub
```

---

## 👩‍💻 Author

**Geetanjali**
🔗 [github.com/Geetanjali098](https://github.com/Geetanjali098)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

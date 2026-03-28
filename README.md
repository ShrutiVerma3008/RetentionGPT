# RetentionAI 🧠

> **AI-powered Customer Churn Prediction & Retention Intelligence**
>
> Hackathon submission — built for the BFSI sector

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.ai)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-purple)](https://langchain.com)

---

## 🎯 What is RetentionAI?

RetentionAI is a **real-time churn prediction and intervention platform** for Indian banks. It combines:

- **XGBoost** churn prediction with **SHAP** explainability
- **BERT** emotion detection from customer complaints
- **Digital Twin** counterfactual simulation ("what-if" scenarios)
- **LangChain** 4-module orchestration for personalized outreach generation
- A premium interactive **dashboard** with 3 interaction modes

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RetentionAI Platform                     │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Frontend    │    │  FastAPI     │    │  ML Layer    │  │
│  │  Dashboard   │◄──►│  Backend     │◄──►│  XGBoost     │  │
│  │  (HTML/CSS/  │    │  (7 APIs)    │    │  + SHAP      │  │
│  │   Vanilla JS)│    │              │    │  + BERT      │  │
│  └──────────────┘    └──────┬───────┘    └──────────────┘  │
│                             │                               │
│                    ┌────────▼────────┐                      │
│                    │  LangChain      │                      │
│                    │  Orchestrator   │                      │
│                    │  (4 Modules)    │                      │
│                    └─────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
retentionAI/
├── Retentiongpt_dashboard.html     # Interactive frontend (3 modes)
├── backend/
│   ├── main.py                     # FastAPI — 7 API endpoints
│   ├── requirements.txt
│   ├── models/
│   │   ├── churn_predictor.py      # XGBoost + SHAP wrapper
│   │   ├── emotion_detector.py     # DistilBERT emotion classifier
│   │   └── xgb_churn.pkl           # Trained model (after training)
│   └── services/
│       ├── digital_twin.py         # Counterfactual simulation engine
│       └── langchain_orchestrator.py  # 4-module LLM pipeline
├── ml/
│   ├── generate_dataset.py         # Synthetic data generator
│   ├── train_xgboost.py            # Training + evaluation + SHAP plots
│   └── RetentionAI_ML_Pipeline.ipynb  # End-to-end demo notebook
├── data/
│   └── synthetic_customers.csv     # Generated dataset (500 rows)
├── plots/                          # SHAP + AUC plots (after training)
├── Dockerfile
├── docker-compose.yml
├── .env.template
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd retentionAI

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r backend/requirements.txt
```

### 2. Configure Environment

```bash
cp .env.template .env
# Edit .env — add OPENAI_API_KEY if you have one (optional)
```

### 3. Generate Data + Train Model

```bash
# Generate synthetic dataset (500 customers)
python ml/generate_dataset.py

# Train XGBoost model + produce SHAP plots
python ml/train_xgboost.py
```

Expected output:
```
✅  Dataset saved → data/synthetic_customers.csv
✅  Model saved  → backend/models/xgb_churn.pkl
📈  Test AUC: 0.9134
```

### 4. Start the API

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

### 5. Open the Dashboard

Open `Retentiongpt_dashboard.html` in your browser, or serve it locally:

```bash
python -m http.server 5500
# Open: http://localhost:5500/Retentiongpt_dashboard.html
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Service liveness check |
| `GET`  | `/api/portfolio/summary` | Portfolio KPIs (347 at risk, AUC, etc.) |
| `GET`  | `/api/customers` | All at-risk customers with scores |
| `GET`  | `/api/customers/{id}` | Customer detail + SHAP explanations |
| `POST` | `/api/simulate` | Digital twin counterfactual scenario |
| `POST` | `/api/outreach/generate` | LangChain outreach draft |
| `POST` | `/api/predict` | Real-time churn score for new record |

### Example: Simulate an intervention

```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "pk", "action": "call"}'
```

Response:
```json
{
  "customer_id": "pk",
  "simulation": {
    "new_score": 54.2,
    "delta_pts": "-27 pts",
    "revenue_retained": "₹3,12,480",
    "recommendation": "Highly Recommended",
    "confidence": 83
  }
}
```

---

## 🐳 Docker

```bash
# Copy env
cp .env.template .env

# Build + run
docker-compose up --build

# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

---

## 📊 ML Pipeline

### Model: XGBoost Classifier
- **Features**: 12 (demographics, engagement, NPS, complaints, emotion)
- **Training**: 400 rows | **Test**: 100 rows | **CV**: 5-fold Stratified
- **Class imbalance**: Handled via `scale_pos_weight`
- **Metric**: AUC-ROC ~0.91, Average Precision ~0.88

### Explainability: SHAP TreeExplainer
- Global feature importance (beeswarm + bar)
- Per-customer waterfall plots
- Human-readable risk reason generation

### Emotion: DistilBERT (zero-shot)
- 5 classes: Satisfied, Confused, Frustrated, Price-sensitive, Comparison-shopping
- Heuristic fallback for offline use

### Digital Twin: Counterfactual Simulation
- Logistic effectiveness curves
- Customer-context sensitivity modifiers
- CLV-weighted revenue retention estimates

---

## 🤖 LangChain 4-Module Pipeline

```
Customer Context
      │
      ▼
┌─────────────────────────────┐
│ Module 1: Risk Analyst      │ ← Why is this customer at risk?
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Module 2: Empathy Engine    │ ← What is their emotional state?
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Module 3: Action Selector   │ ← What is the Next Best Action?
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Module 4: Outreach Writer   │ ← Generate personalized message
└─────────────────────────────┘
```

---

## 🎬 Demo Script (for judges)

### Step 1 — Overview Mode
> "The Overview dashboard shows 347 customers currently at churn risk, with ₹2.4 Cr revenue at stake. The ML model runs continuously, updating risk tiers in real time."

### Step 2 — Action Mode
> "Clicking 'Action Mode' shows our intelligent default — Priya Krishnamurthy, highest risk × highest CLV. The SHAP model tells us exactly *why* she's at risk: 3 unresolved complaints, 62% drop in transactions. The Next Best Action card recommends a Priority RM Call with 78% confidence. The AI Outreach Draft is pre-generated by our LangChain pipeline."

### Step 3 — Simulation Mode
> "The Digital Twin lets us run what-if scenarios. Without intervention, churn probability stays at 81%. A discount offer reduces it to 72% — partial. A Priority RM Call drops it to 54% and retains ₹3.2L in expected revenue. A Premier Upgrade achieves the biggest reduction to 40%."

---

## 🏆 Hackathon Highlights

- ✅ **End-to-end AI pipeline** — from raw data to personalized outreach
- ✅ **Explainable AI** — SHAP + human-readable risk reasons
- ✅ **Emotion-aware** — BERT emotion detection
- ✅ **Digital Twin** — mathematically-grounded simulations
- ✅ **Production-ready** — FastAPI + Docker + CI-friendly
- ✅ **Demo-safe** — works fully offline without OpenAI key

---

## 👥 Team

Built at the intersection of ML, NLP, and fintech UX.

---

*Made with ❤️ for Indian BFSI sector transformation*
#   R e t e n t i o n G P T  
 
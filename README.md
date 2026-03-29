<div align="center">

<!-- ANIMATED HEADER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=RetentionAI&fontSize=72&fontColor=ffffff&fontAlignY=38&desc=Predict.%20Personalise.%20Retain.&descAlignY=58&descSize=20&animation=fadeIn" width="100%"/>

<!-- BADGES -->
<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-EA4335?style=for-the-badge&logo=xgboost&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![BERT](https://img.shields.io/badge/BERT-Emotion%20AI-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white)

<br/><br/>

<!-- TYPEWRITER EFFECT USING SVG -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=A78BFA&center=true&vCenter=true&multiline=true&width=700&height=80&lines=🏦+Built+for+Indian+BFSI+Sector;🧠+XGBoost+%2B+SHAP+%2B+BERT+%2B+LangChain;💡+Stop+Churn+Before+It+Happens" alt="Typing SVG" />

<br/>

> **RetentionAI** is an end-to-end, real-time churn prediction and personalised intervention platform for Indian banks.  
> It doesn't just tell you *who* will leave — it tells you *why*, *when*, and *how to keep them*.

<br/>

[![⭐ Star this repo](https://img.shields.io/github/stars/yourusername/retentionAI?style=social)](https://github.com/yourusername/retentionAI)
&nbsp;&nbsp;
[![🍴 Fork](https://img.shields.io/github/forks/yourusername/retentionAI?style=social)](https://github.com/yourusername/retentionAI/fork)

</div>

---

## 🌊 The Problem We're Solving

<table>
<tr>
<td width="60%">

Indian banks lose **₹47,000 Cr+** annually to customer attrition.

The traditional response? **React after the customer leaves.**

Relationship managers call customers *who've already decided to go*.  
Marketing blasts generic offers to everyone.  
Operations reports churn only after the quarter ends.

**RetentionAI flips this entirely.**

We identify customers drifting toward the exit **weeks in advance** — and arm the bank with the exact personalised action to bring them back.

</td>
<td width="40%" align="center">

```
Without RetentionAI       With RetentionAI
─────────────────         ─────────────────
Customer leaves   →       Risk detected early
Bank notices late →       AI explains why
Generic call sent →       Personalised action
₹3.2L lost        →       ₹3.2L retained ✅
```

</td>
</tr>
</table>

---

## ✨ What Makes Us Different

<div align="center">

| Feature | What It Does | Why It Matters |
|:-------:|:------------|:--------------|
| 🎯 **XGBoost + SHAP** | Predicts churn with AUC ~0.91 + explains each prediction in plain English | No black box — every score is justified |
| 🧠 **BERT Emotion Engine** | Detects emotional state from complaint text (Frustrated / Price-sensitive / Confused…) | Outreach tone adapts to customer mood |
| 👯 **Digital Twin Simulation** | "What-if" scenario modelling — test interventions before you make them | Prioritise actions with highest ROI |
| 🤖 **LangChain 4-Module Pipeline** | Auto-generates personalised outreach drafts (call scripts, emails, SMS) | RM gets a ready-to-send message, not a blank screen |
| 📊 **3-Mode Live Dashboard** | Overview → Action → Simulation — one screen for the full journey | Investigation to intervention in under 60 seconds |

</div>

---

## 🏗️ Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                     RetentionAI Platform                        ║
║                                                                  ║
║   ┌─────────────┐      ┌──────────────┐      ┌───────────────┐  ║
║   │  Dashboard  │◄────►│   FastAPI    │◄────►│   ML Layer    │  ║
║   │  (3 Modes)  │      │  7 Endpoints │      │  XGBoost      │  ║
║   │  HTML/CSS/  │      │              │      │  + SHAP       │  ║
║   │  Vanilla JS │      │              │      │  + DistilBERT │  ║
║   └─────────────┘      └──────┬───────┘      └───────────────┘  ║
║                               │                                  ║
║                    ┌──────────▼──────────┐                       ║
║                    │  LangChain Pipeline │                       ║
║                    │  Risk → Empathy →   │                       ║
║                    │  Action → Outreach  │                       ║
║                    └─────────────────────┘                       ║
╚══════════════════════════════════════════════════════════════════╝
```

**Data Flow:**
```
Raw Customer Data
      │
      ▼
  Feature Engineering (12 features)
      │
      ▼
  XGBoost Classifier ──► SHAP Explainer ──► Human-readable Risk Reasons
      │
      ▼
  DistilBERT Emotion Classifier
      │
      ▼
  Digital Twin Simulation Engine
      │
      ▼
  LangChain 4-Module Orchestrator
      │
      ▼
  Personalised Outreach (Email / Call Script / SMS)
```

---

## 🤖 The LangChain 4-Module Pipeline

```
  Customer Context
        │
        ▼
  ╔═══════════════════╗
  ║  Module 1         ║   "She has 3 unresolved complaints.
  ║  Risk Analyst 🔍  ║    Transaction frequency dropped 62%
  ║                   ║    in the last 60 days."
  ╚════════╤══════════╝
           │
           ▼
  ╔═══════════════════╗
  ║  Module 2         ║   "Emotional state: FRUSTRATED.
  ║  Empathy Engine 💙║    Last complaint: EMI deduction
  ║                   ║    without notice."
  ╚════════╤══════════╝
           │
           ▼
  ╔═══════════════════╗
  ║  Module 3         ║   "Next Best Action: Priority RM Call.
  ║  Action Selector ⚡║    Confidence: 78%.
  ║                   ║    Expected churn drop: -27 pts."
  ╚════════╤══════════╝
           │
           ▼
  ╔═══════════════════╗
  ║  Module 4         ║   Generates ready-to-send
  ║  Outreach Writer✍️║    personalised message draft
  ║                   ║    in Hindi / English.
  ╚═══════════════════╝
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/retentionAI.git
cd retentionAI

python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r backend/requirements.txt
```

### 2. Configure Environment
```bash
cp .env.template .env
# Add OPENAI_API_KEY (optional — system works fully offline without it)
```

### 3. Generate Data & Train Model
```bash
python ml/generate_dataset.py
python ml/train_xgboost.py
```

Expected output:
```
✅  Dataset saved  →  data/synthetic_customers.csv  (500 rows)
✅  Model saved    →  backend/models/xgb_churn.pkl
📈  Test AUC: 0.9134    |    Average Precision: 0.88
```

### 4. Start the API
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
📖 Interactive docs: **http://localhost:8000/docs**

### 5. Open the Dashboard
```bash
python -m http.server 5500
# Open: http://localhost:5500/Retentiongpt_dashboard.html
```

---

## 🐳 Docker (One Command)
```bash
cp .env.template .env
docker-compose up --build
# API → http://localhost:8000
# Docs → http://localhost:8000/docs
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|:------:|:---------|:------------|
| `GET` | `/health` | Service liveness |
| `GET` | `/api/portfolio/summary` | Portfolio KPIs — 347 at risk, ₹2.4Cr revenue stake |
| `GET` | `/api/customers` | All at-risk customers with churn scores |
| `GET` | `/api/customers/{id}` | Customer detail + SHAP explanation |
| `POST` | `/api/simulate` | Digital Twin — what-if intervention scenario |
| `POST` | `/api/outreach/generate` | LangChain personalised outreach draft |
| `POST` | `/api/predict` | Real-time churn score for new customer record |

### Example — Digital Twin Simulation
```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "pk", "action": "priority_call"}'
```
```json
{
  "customer_id": "pk",
  "simulation": {
    "current_churn_score": 81.2,
    "new_score": 54.2,
    "delta_pts": "-27 pts",
    "revenue_retained": "₹3,12,480",
    "recommendation": "Highly Recommended",
    "confidence": 83
  }
}
```

---

## 📊 ML Pipeline Details

```
┌──────────────────────────────────────────────────────┐
│                  XGBoost Classifier                  │
│                                                      │
│  Features (12):                                      │
│  ├── Demographics  (age, segment, city_tier)         │
│  ├── Engagement    (txn_frequency, recency, drops)   │
│  ├── Sentiment     (NPS, complaints, emotion_score)  │
│  └── Product       (products_held, digital_usage)    │
│                                                      │
│  Training:  400 rows  |  Test: 100 rows              │
│  CV:        5-fold Stratified                        │
│  AUC-ROC:   ~0.91     |  Avg Precision: ~0.88        │
│  Imbalance: scale_pos_weight                         │
└──────────────────────────────────────────────────────┘
```

**SHAP Explainability** — every prediction comes with a human-readable reason:
```
⚠️ Priya is at HIGH RISK because:
   → Transaction frequency dropped 62% (last 60 days)
   → 3 unresolved complaints (score: -0.34 SHAP)
   → NPS dropped from 8 → 3 (score: -0.29 SHAP)
   → No digital engagement for 45 days (score: -0.18 SHAP)
```

---

## 📂 Project Structure

```
retentionAI/
├── 📄 Retentiongpt_dashboard.html     # Frontend — 3 interaction modes
├── 🐳 Dockerfile
├── 🐳 docker-compose.yml
├── ⚙️  .env.template
│
├── backend/
│   ├── main.py                        # FastAPI — 7 endpoints
│   ├── requirements.txt
│   ├── models/
│   │   ├── churn_predictor.py         # XGBoost + SHAP wrapper
│   │   ├── emotion_detector.py        # DistilBERT classifier
│   │   └── xgb_churn.pkl              # Trained model
│   └── services/
│       ├── digital_twin.py            # Counterfactual simulation
│       └── langchain_orchestrator.py  # 4-module LLM pipeline
│
└── ml/
    ├── generate_dataset.py            # Synthetic data generator
    ├── train_xgboost.py               # Training + SHAP plots
    └── RetentionAI_ML_Pipeline.ipynb  # End-to-end demo notebook
```

---

## 🎬 Demo Walkthrough (For Judges)

**Step 1 — Overview Mode**
> *"347 customers are currently at churn risk with ₹2.4 Cr revenue at stake. The model updates risk tiers in real time as new transactions and complaint data flows in."*

**Step 2 — Action Mode**
> *"The AI surfaces Priya Krishnamurthy — highest risk × highest CLV. SHAP tells us exactly why: 3 unresolved complaints, 62% transaction drop. Next Best Action: Priority RM Call, confidence 78%. The LangChain pipeline has already drafted the outreach message — the RM just hits send."*

**Step 3 — Simulation Mode**
> *"Before the RM makes the call, the Digital Twin runs the simulation:*
> - *Discount offer → churn drops to 72% (partial)*
> - *Priority RM Call → drops to 54%, retains ₹3.2L*
> - *Premier Upgrade → drops to 40% (best outcome)*
>
> *We recommend the call. The expected revenue retained is ₹3,12,480."*

---

## 🏆 Hackathon Highlights

```
✅  End-to-end AI pipeline     From raw data → to personalised outreach
✅  Explainable AI (XAI)       SHAP + human-readable risk reasons
✅  Emotion-aware outreach     BERT detects frustration, price sensitivity
✅  Digital Twin simulation    Test interventions before committing
✅  Production-ready           FastAPI + Docker + CI-friendly structure
✅  Demo-safe offline mode     Works 100% without OpenAI API key
✅  Indian BFSI context        ₹ CLV, Hindi/English outreach, RBI-aware
```

---

## 👩‍💻 Team

<div align="center">

Built with ❤️ at **iDEA Hackathon 2.0** — PSBs Hackathon Series 2026  
*An Initiative of Government of India, Ministry of Finance, Department of Financial Services*

*Presented by Union Bank of India × K.J. Somaiya College of Engineering*

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer&animation=fadeIn" width="100%"/>

**⭐ If this project helped you, please star the repo!**

`RetentionAI` — *Predict. Personalise. Retain.*

</div>

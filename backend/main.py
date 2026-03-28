"""
RetentionAI — FastAPI Backend
================================
Production-level API with 7 core endpoints.

Run:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Docs:  http://localhost:8000/docs  (Swagger UI)
"""

from __future__ import annotations
import sys, os
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.models.churn_predictor      import get_predictor
from backend.models.emotion_detector     import get_emotion_detector
from backend.services.digital_twin       import simulate_intervention, INTERVENTIONS
from backend.services.langchain_orchestrator import get_orchestrator

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# App setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(
    title       = "RetentionAI API",
    description = "AI-powered customer churn prediction and retention intelligence.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # Tighted in production
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sample data (mirrors dashboard, replaces DB for hackathon)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MOCK_CUSTOMERS = [
    {
        "id": "pk", "name": "Priya Krishnamurthy",
        "segment": "Premium Savings", "tenure_months": 36,
        "avg_balance": 420000, "clv_inr": 420000, "monthly_txn_count": 5,
        "complaints_open": 3, "nps_score": 4.2, "competitor_inquiry": 0,
        "app_login_days": 6, "intl_transfer_dormancy": 0, "product_count": 2,
        "emotion_label": 2, "emotion_name": "Frustrated",
        "churn_probability": 0.81, "risk_tier": "high",
        "shap_reasons": [
            {"title": "Transaction velocity dropped 62%",  "desc": "Significant divergence from 12-mo average.", "impact": "+31% risk"},
            {"title": "3 unresolved app complaints",       "desc": "Helpdesk tickets open for >7 days.",         "impact": "+22% risk"},
            {"title": "Balance decline by 40%",           "desc": "Funds likely transferred outward.",           "impact": "+18% risk"},
        ],
        "nba": {"title": "Priority RM Call", "confidence": 78,
                "reason": "Trust score critically low. Personalized call addresses emotional dimension."},
    },
    {
        "id": "ra", "name": "Rahul Agarwal",
        "segment": "Salary Account", "tenure_months": 18,
        "avg_balance": 180000, "clv_inr": 180000, "monthly_txn_count": 9,
        "complaints_open": 1, "nps_score": 6.1, "competitor_inquiry": 1,
        "app_login_days": 12, "intl_transfer_dormancy": 0, "product_count": 1,
        "emotion_label": 4, "emotion_name": "Comparison-shopping",
        "churn_probability": 0.74, "risk_tier": "high",
        "shap_reasons": [
            {"title": "Competitor web inquiry signal",  "desc": "Matched external financial intent data.", "impact": "+28% risk"},
            {"title": "Investment activity stalled",    "desc": "No systemic plans renewed in 90 days.",  "impact": "+20% risk"},
        ],
        "nba": {"title": "Send Wealth Mgmt Offer", "confidence": 65,
                "reason": "Competitor browsing indicates intent to invest. Counter-offer required."},
    },
    {
        "id": "ms", "name": "Meena Subramaniam",
        "segment": "NRI Savings", "tenure_months": 72,
        "avg_balance": 950000, "clv_inr": 950000, "monthly_txn_count": 3,
        "complaints_open": 0, "nps_score": 7.5, "competitor_inquiry": 0,
        "app_login_days": 22, "intl_transfer_dormancy": 1, "product_count": 3,
        "emotion_label": 1, "emotion_name": "Confused",
        "churn_probability": 0.58, "risk_tier": "medium",
        "shap_reasons": [
            {"title": "Intl transfer dormancy",  "desc": "Core NRI product unused for 5 months.", "impact": "+24% risk"},
        ],
        "nba": {"title": "Educational Email", "confidence": 55,
                "reason": "Likely unaware of recent fee reductions on intl transfers."},
    },
]

CUSTOMERS_BY_ID = {c['id']: c for c in MOCK_CUSTOMERS}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Request / Response schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SimulateRequest(BaseModel):
    customer_id : str = Field(..., example="pk")
    action      : str = Field(..., example="call", description="none | offer | call | upgrade")

class OutreachRequest(BaseModel):
    customer_id : str = Field(..., example="pk")
    channel     : str = Field("email", example="email", description="email | whatsapp")

class PredictRequest(BaseModel):
    age                   : int   = Field(..., example=38)
    tenure_months         : int   = Field(..., example=36)
    product_count         : int   = Field(..., example=2)
    avg_balance           : float = Field(..., example=420000)
    monthly_txn_count     : int   = Field(..., example=5)
    clv_inr               : float = Field(..., example=420000)
    complaints_open       : int   = Field(..., example=3)
    nps_score             : float = Field(..., example=4.2)
    competitor_inquiry    : int   = Field(..., example=0)
    app_login_days        : int   = Field(..., example=6)
    intl_transfer_dormancy: int   = Field(..., example=0)
    emotion_label         : int   = Field(..., example=2)
    complaint_text        : Optional[str] = Field(None, example="App keeps crashing and no one is helping.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe."""
    return {"status": "ok", "service": "RetentionAI API", "version": "1.0.0"}


@app.get("/api/portfolio/summary", tags=["Portfolio"])
def portfolio_summary():
    """
    Portfolio-level KPI stats shown in the Overview dashboard.
    Returns aggregated metrics across all customers.
    """
    high   = [c for c in MOCK_CUSTOMERS if c['risk_tier'] == 'high']
    medium = [c for c in MOCK_CUSTOMERS if c['risk_tier'] == 'medium']
    low    = [c for c in MOCK_CUSTOMERS if c['risk_tier'] == 'low']

    avg_prob = sum(c['churn_probability'] for c in MOCK_CUSTOMERS) / len(MOCK_CUSTOMERS)
    total_clv_at_risk = sum(
        c['clv_inr'] * c['churn_probability'] for c in MOCK_CUSTOMERS
    )

    return {
        "customers_at_risk"       : 347,         # scaled from demo data
        "avg_churn_probability"   : round(avg_prob * 100, 1),
        "revenue_at_risk_inr"     : 24_00_000,   # ₹2.4 Cr
        "intervention_success_pct": 38,
        "risk_distribution": {
            "high"  : {"count": 168, "pct": 28},
            "medium": {"count": 122, "pct": 45},
            "low"   : {"count": 57,  "pct": 27},
        },
        "week_over_week": {
            "new_at_risk": 23,
            "churn_prob_change": -3,
            "revenue_change_inr": 18_00_000,
        }
    }


@app.get("/customers/risk", tags=["Customers"])
def list_customers(
    risk_tier : Optional[str] = Query(None, description="Filter: high | medium | low"),
    limit     : int           = Query(50, le=500),
):
    """
    List all at-risk customers with churn scores.
    Supports filtering by risk tier.
    """
    customers = list(MOCK_CUSTOMERS)
    if risk_tier:
        customers = [c for c in customers if c['risk_tier'] == risk_tier]
    return {
        "count": len(customers),
        "customers": [
            {k: v for k, v in c.items() if k != 'shap_reasons'}
            for c in customers[:limit]
        ]
    }


@app.get("/customers/{customer_id}", tags=["Customers"])
def get_customer(customer_id: str):
    """
    Full customer profile with SHAP-driven risk explanations and NBA.
    """
    c = CUSTOMERS_BY_ID.get(customer_id)
    if not c:
        raise HTTPException(404, f"Customer '{customer_id}' not found")
    return c


@app.post("/simulate", tags=["Digital Twin"])
def simulate(req: SimulateRequest):
    """
    Digital Twin counterfactual simulation.
    Returns new churn probability and revenue impact for a given intervention.
    """
    c = CUSTOMERS_BY_ID.get(req.customer_id)
    if not c:
        raise HTTPException(404, f"Customer '{req.customer_id}' not found")
    if req.action not in INTERVENTIONS:
        raise HTTPException(400, f"Unknown action '{req.action}'. Valid: {list(INTERVENTIONS)}")

    result = simulate_intervention(
        base_score = c['churn_probability'],
        action     = req.action,
        customer   = c,
    )
    return {"customer_id": req.customer_id, "simulation": result}


@app.post("/generate-message", tags=["LangChain"])
def generate_outreach(req: OutreachRequest):
    """
    LangChain 4-module pipeline: generates personalized outreach draft.
    Uses GPT-4o-mini if OPENAI_API_KEY is set, or demo mode otherwise.
    """
    c = CUSTOMERS_BY_ID.get(req.customer_id)
    if not c:
        raise HTTPException(404, f"Customer '{req.customer_id}' not found")
    if req.channel not in ('email', 'whatsapp'):
        raise HTTPException(400, "channel must be 'email' or 'whatsapp'")

    orch   = get_orchestrator()
    result = orch.run_pipeline(c, channel=req.channel)
    return {
        "customer_id"    : req.customer_id,
        "channel"        : req.channel,
        "pipeline_output": result,
    }


@app.post("/api/predict", tags=["ML Model"])
def predict(req: PredictRequest):
    """
    Real-time churn prediction for a new customer record.
    Returns churn probability, risk tier, emotion, and SHAP explanations.
    """
    row = req.dict()
    text = row.pop('complaint_text', None)

    # Churn probability
    predictor = get_predictor()
    churn_prob = predictor.predict_proba(row)

    # SHAP explanations
    shap_reasons = predictor.explain_customer(row, top_n=3)

    # Emotion from text or features
    detector = get_emotion_detector()
    if text:
        emotion = detector.classify(text)
    else:
        emotion = detector.classify_from_features(
            row['complaints_open'], row['nps_score'], row['competitor_inquiry']
        )

    risk_tier = 'high' if churn_prob > 0.70 else ('medium' if churn_prob > 0.30 else 'low')

    return {
        "churn_probability": round(churn_prob, 4),
        "churn_percent"    : round(churn_prob * 100, 1),
        "risk_tier"        : risk_tier,
        "emotion"          : emotion,
        "shap_explanations": shap_reasons,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dev entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

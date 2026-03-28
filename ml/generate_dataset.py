"""
RetentionAI — Synthetic Dataset Generator
==========================================
Generates 500 synthetic bank customer records with realistic correlations
to churn. Features are designed to produce meaningful SHAP values.

Run: python ml/generate_dataset.py
Output: data/synthetic_customers.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 500

# ── Core demographics ──────────────────────────────────────────────────────
age              = np.random.normal(38, 12, N).clip(21, 70).astype(int)
tenure_months    = np.random.exponential(30, N).clip(1, 120).astype(int)

# ── Product & balance signals ──────────────────────────────────────────────
product_count    = np.random.choice([1, 2, 3, 4], N, p=[0.35, 0.40, 0.18, 0.07])
avg_balance      = (np.random.lognormal(10.5, 0.8, N)).clip(5_000, 2_000_000).astype(int)
monthly_txn_count= np.random.poisson(14, N).clip(0, 60)
clv_inr          = (avg_balance * np.random.uniform(0.3, 0.9, N)).astype(int)

# ── Engagement & satisfaction ──────────────────────────────────────────────
complaints_open  = np.random.choice([0, 1, 2, 3, 4], N, p=[0.55, 0.25, 0.12, 0.05, 0.03])
nps_score        = np.random.normal(6.2, 2.1, N).clip(0, 10).round(1)
competitor_inquiry = np.random.choice([0, 1], N, p=[0.70, 0.30])
app_login_days   = np.random.normal(18, 8, N).clip(0, 30).astype(int)  # days/month
intl_transfer_dormancy = np.random.choice([0, 1], N, p=[0.65, 0.35])

# ── Emotion labels (0-4 ordinal) ───────────────────────────────────────────
# 0=Satisfied, 1=Confused, 2=Frustrated, 3=Price-sensitive, 4=Comparison-shopping
emotion_raw = (
    (complaints_open > 1).astype(int) * 2 +          # Frustrated
    (competitor_inquiry == 1).astype(int) * 2 +       # Comparison-shopping
    (nps_score < 5).astype(int)                        # Price-sensitive
)
emotion_label = np.clip(emotion_raw, 0, 4)
emotion_names = {0:'Satisfied', 1:'Confused', 2:'Frustrated',
                 3:'Price-sensitive', 4:'Comparison-shopping'}

# ── Churn label (logistic prob then threshold) ─────────────────────────────
churn_log_odds = (
    -1.0                                                             # intercept (reduced from -3.0)
    + 0.30  * complaints_open                                        # each complaint adds 0.3
    - 0.10  * (nps_score - 5)                                        # below 5 = risk
    + 1.20  * competitor_inquiry                                     # strong signal
    - 0.03  * (monthly_txn_count - 10).clip(-10, 10)                 # low txn = risk
    - 0.00001 * avg_balance                                          # low balance = risk
    - 0.02  * tenure_months                                          # longer = more loyal
    + 0.40  * (product_count == 1).astype(int)                       # single product = risky
    + 0.50  * intl_transfer_dormancy                                 # dormant NRI
    - 0.04  * app_login_days                                         # disengaged = risk
)
churn_prob = 1 / (1 + np.exp(-churn_log_odds))
churn_prob += np.random.normal(0, 0.05, N)           # noise
churn_prob = churn_prob.clip(0.01, 0.99)
churn = (churn_prob > 0.50).astype(int)              # ~30% churn rate

# ── Risk tier ─────────────────────────────────────────────────────────────
risk_tier = pd.cut(churn_prob, bins=[0, 0.30, 0.70, 1.0],
                   labels=['low', 'medium', 'high'])

# ── Assemble DataFrame ─────────────────────────────────────────────────────
df = pd.DataFrame({
    'customer_id'           : [f'CUST{str(i).zfill(4)}' for i in range(1, N+1)],
    'age'                   : age,
    'tenure_months'         : tenure_months,
    'product_count'         : product_count,
    'avg_balance'           : avg_balance,
    'monthly_txn_count'     : monthly_txn_count,
    'clv_inr'               : clv_inr,
    'complaints_open'       : complaints_open,
    'nps_score'             : nps_score,
    'competitor_inquiry'    : competitor_inquiry,
    'app_login_days'        : app_login_days,
    'intl_transfer_dormancy': intl_transfer_dormancy,
    'emotion_label'         : emotion_label,
    'emotion_name'          : [emotion_names[min(e, 4)] for e in emotion_label],
    'churn_probability'     : churn_prob.round(4),
    'risk_tier'             : risk_tier,
    'churn'                 : churn,
})

# Save
out_path = Path(__file__).parent.parent / 'data' / 'synthetic_customers.csv'
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print(f"✅  Dataset saved → {out_path}")
print(f"    Rows: {len(df)} | Churn rate: {churn.mean()*100:.1f}%")
print(f"    Risk distribution:\n{df['risk_tier'].value_counts().to_string()}")
print(f"\nSample:\n{df.head(3).to_string()}")

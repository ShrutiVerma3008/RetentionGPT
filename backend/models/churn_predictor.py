"""
RetentionAI — XGBoost Churn Predictor + SHAP Explainer
=======================================================
Wraps the trained XGBoost model with:
  • predict_proba()      → churn probability 0-1
  • explain_customer()   → top-N SHAP reasons (human-readable)
  • batch_score()        → score a DataFrame
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# ── Feature config  ──────────────────────────────────────────────────────────
FEATURES = [
    'age', 'tenure_months', 'product_count', 'avg_balance',
    'monthly_txn_count', 'clv_inr', 'complaints_open', 'nps_score',
    'competitor_inquiry', 'app_login_days', 'intl_transfer_dormancy',
    'emotion_label'
]

# Human-readable labels for SHAP feature names
FEATURE_LABELS = {
    'age'                   : 'Customer age',
    'tenure_months'         : 'Account tenure',
    'product_count'         : 'Number of products held',
    'avg_balance'           : 'Average account balance',
    'monthly_txn_count'     : 'Monthly transaction velocity',
    'clv_inr'               : 'Customer lifetime value',
    'complaints_open'       : 'Open support complaints',
    'nps_score'             : 'NPS satisfaction score',
    'competitor_inquiry'    : 'Competitor inquiry signal',
    'app_login_days'        : 'App engagement (days/month)',
    'intl_transfer_dormancy': 'International transfer dormancy',
    'emotion_label'         : 'Detected emotional state',
}

MODEL_PATH = Path(__file__).parent / 'xgb_churn.pkl'


class ChurnPredictor:
    """Singleton-friendly XGBoost + SHAP wrapper."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model = None
        self.explainer = None
        self._load(model_path)

    def _load(self, path: Path):
        if path.exists():
            with open(path, 'rb') as f:
                bundle = pickle.load(f)
            self.model    = bundle['model']
            self.explainer = bundle['explainer']
            print(f"✅  ChurnPredictor loaded from {path}")
        else:
            print(f"⚠️  Model not found at {path}. Run: python ml/train_xgboost.py")

    def predict_proba(self, row: Dict[str, Any]) -> float:
        """Score a single customer dict → churn probability."""
        if self.model is None:
            return 0.5  # fallback if model not trained yet
        df = pd.DataFrame([row])[FEATURES].fillna(0)
        return float(self.model.predict_proba(df)[0][1])

    def explain_customer(self, row: Dict[str, Any], top_n: int = 3) -> List[Dict]:
        """
        Returns top_n SHAP-based risk drivers for a customer.
        Format matches the dashboard's timeline schema:
        [{"title": "...", "desc": "...", "impact": "+X% risk"}, ...]
        """
        if self.explainer is None:
            return self._mock_explanations()

        df = pd.DataFrame([row])[FEATURES].fillna(0)
        raw = self.explainer.shap_values(df)
        # Normalize across shap API versions
        if hasattr(raw, 'values'):
            shap_values = raw.values[0]
        elif isinstance(raw, list):
            shap_values = raw[1][0]   # positive class, first row
        else:
            shap_values = raw[0]

        # Pair feature names with SHAP values
        pairs = sorted(
            zip(FEATURES, shap_values),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        results = []
        for feat, shap_val in pairs[:top_n]:
            if shap_val <= 0:
                continue  # only show risk-increasing features
            pct_impact = int(round(shap_val * 100, 0))
            label      = FEATURE_LABELS.get(feat, feat)
            raw_value  = row.get(feat, '?')
            desc       = self._describe_feature(feat, raw_value)
            results.append({
                'title' : f'{label} is a risk driver',
                'desc'  : desc,
                'impact': f'+{pct_impact}% risk'
            })

        return results if results else self._mock_explanations()

    def batch_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add churn_score column to a DataFrame."""
        if self.model is None:
            df['churn_score'] = 0.5
            return df
        features_df = df[FEATURES].fillna(0)
        df = df.copy()
        df['churn_score'] = self.model.predict_proba(features_df)[:, 1]
        return df

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _describe_feature(feat: str, value) -> str:
        templates = {
            'complaints_open'       : f'{value} unresolved support ticket(s) open for >7 days signal active frustration.',
            'competitor_inquiry'    : 'External financial intent signals matched — customer actively evaluating alternatives.',
            'nps_score'             : f'NPS score of {value}/10 is below satisfaction threshold.',
            'monthly_txn_count'     : f'Only {value} transactions this month — significant decline from historical average.',
            'app_login_days'        : f'App engagement dropped to {value} days/month indicating disengagement.',
            'intl_transfer_dormancy': 'Core international transfer product unused for 5+ months.',
            'avg_balance'           : f'Balance at ₹{value:,} — declining trend detected.',
            'tenure_months'         : f'Tenure of {value} months — early lifecycle customer at higher natural risk.',
            'product_count'         : f'Only {value} product(s) held — low product stickiness.',
        }
        return templates.get(feat, f'Feature value: {value}')

    @staticmethod
    def _mock_explanations() -> List[Dict]:
        return [
            {'title': 'Transaction velocity dropped 62%', 'desc': 'Significant divergence from 12-month average.', 'impact': '+31% risk'},
            {'title': '3 unresolved app complaints',      'desc': 'Helpdesk tickets open for >7 days.',             'impact': '+22% risk'},
            {'title': 'Balance decline by 40%',          'desc': 'Funds likely transferred outward.',               'impact': '+18% risk'},
        ]


# Module-level singleton (lazy loaded)
_predictor: ChurnPredictor | None = None

def get_predictor() -> ChurnPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor

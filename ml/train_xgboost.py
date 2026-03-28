"""
RetentionAI — XGBoost Training Script
=======================================
Trains the churn prediction model and saves it as a pickle bundle.

Run: python ml/train_xgboost.py
Requires: pip install xgboost shap scikit-learn pandas

Output:
  backend/models/xgb_churn.pkl  — model + explainer bundle
  plots/shap_beeswarm.png       — SHAP summary plot
  plots/auc_curve.png           — AUC-ROC curve
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')   # headless
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_PATH = ROOT / 'data' / 'synthetic_customers.csv'
MODEL_OUT = ROOT / 'backend' / 'models' / 'xgb_churn.pkl'
PLOTS_DIR = ROOT / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    'age', 'tenure_months', 'product_count', 'avg_balance',
    'monthly_txn_count', 'clv_inr', 'complaints_open', 'nps_score',
    'competitor_inquiry', 'app_login_days', 'intl_transfer_dormancy',
    'emotion_label'
]
TARGET = 'churn'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Load data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 60)
print("  RetentionAI — XGBoost Training Pipeline")
print("=" * 60)

if not DATA_PATH.exists():
    print(f"⚠️  Dataset not found at {DATA_PATH}")
    print("    Run: python ml/generate_dataset.py  first.")
    raise SystemExit(1)

df = pd.read_csv(DATA_PATH)
print(f"\n📂  Loaded {len(df)} rows from {DATA_PATH.name}")
print(f"    Churn rate: {df[TARGET].mean()*100:.1f}%")
print(f"    Features:   {len(FEATURES)}")

X = df[FEATURES].fillna(0)
y = df[TARGET]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Train/test split
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n📊  Train: {len(X_train)} | Test: {len(X_test)}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Model definition + cross-validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
model = xgb.XGBClassifier(
    n_estimators     = 200,
    max_depth        = 5,
    learning_rate    = 0.08,
    subsample        = 0.85,
    colsample_bytree = 0.85,
    scale_pos_weight = (y == 0).sum() / (y == 1).sum(),  # handle imbalance
    eval_metric      = 'logloss',
    random_state     = 42,
    verbosity        = 0,
)

print("\n🔄  Running 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"    CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Final training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n🚀  Training final model on full train set...")
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
y_pred     = model.predict(X_test)
y_proba    = model.predict_proba(X_test)[:, 1]
auc        = roc_auc_score(y_test, y_proba)
avg_prec   = average_precision_score(y_test, y_proba)

print(f"\n{'='*60}")
print(f"  📈  Model Evaluation")
print(f"{'='*60}")
print(f"  AUC-ROC           : {auc:.4f}")
print(f"  Avg Precision     : {avg_prec:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\n  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. SHAP Explainability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n🔍  Computing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values_raw = explainer.shap_values(X_test)
# In newer SHAP versions, shap_values returns an Explanation object or ndarray
# Normalize to 2D array for positive class
if hasattr(shap_values_raw, 'values'):
    shap_values = shap_values_raw.values
elif isinstance(shap_values_raw, list):
    shap_values = shap_values_raw[1]  # positive class
else:
    shap_values = shap_values_raw

# Beeswarm plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=FEATURES,
                  show=False, plot_size=(10, 6))
plt.title('RetentionAI — SHAP Feature Impact (Global)', fontsize=14, pad=16)
plt.tight_layout()
out_beeswarm = PLOTS_DIR / 'shap_beeswarm.png'
plt.savefig(out_beeswarm, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved → {out_beeswarm}")

# Bar summary
plt.figure(figsize=(10, 5))
shap.summary_plot(shap_values, X_test, feature_names=FEATURES,
                  plot_type='bar', show=False)

plt.title('RetentionAI — Mean |SHAP| Feature Importance', fontsize=14, pad=16)
plt.tight_layout()
out_bar = PLOTS_DIR / 'shap_importance.png'
plt.savefig(out_bar, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved → {out_bar}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. AUC-ROC + Precision-Recall curves
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax1.plot(fpr, tpr, color='#6366f1', lw=2.5, label=f'AUC = {auc:.3f}')
ax1.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4, label='Random')
ax1.fill_between(fpr, tpr, alpha=0.08, color='#6366f1')
ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
ax1.set_title('AUC-ROC Curve', fontsize=13); ax1.legend()
ax1.grid(True, alpha=0.3)

# PR curve
prec, rec, _ = precision_recall_curve(y_test, y_proba)
ax2.plot(rec, prec, color='#10b981', lw=2.5, label=f'AP = {avg_prec:.3f}')
ax2.fill_between(rec, prec, alpha=0.08, color='#10b981')
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve', fontsize=13); ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('RetentionAI — Model Evaluation', fontsize=15, y=1.02)
plt.tight_layout()
out_curves = PLOTS_DIR / 'auc_pr_curves.png'
plt.savefig(out_curves, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved → {out_curves}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Save model bundle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
bundle = {
    'model'     : model,
    'explainer' : explainer,
    'features'  : FEATURES,
    'auc'       : round(float(auc), 4),
    'avg_prec'  : round(float(avg_prec), 4),
}
with open(MODEL_OUT, 'wb') as f:
    pickle.dump(bundle, f)

print(f"\n✅  Model saved → {MODEL_OUT}")
print(f"{'='*60}")
print(f"  Run the API server: uvicorn backend.main:app --reload")
print(f"{'='*60}\n")

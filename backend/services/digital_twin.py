"""
RetentionAI — Digital Twin Counterfactual Simulation Engine
=============================================================
Implements mathematically-grounded counterfactual logic:
"If we take action X, what will the new churn probability be?"

Model design:
─────────────────────────────────────────────────────────────
Each intervention has:
  • Base reduction capacity   (max possible delta)
  • Effective reduction       (actual delta, function of customer state)
  • Floor probability         (can't reduce below a minimum)
  • Revenue estimation        (based on CLV and probability delta)

Formulae inspired by:
  Cox (2017) — Intervention counterfactuals in retention modelling
  Fudenberg & Tirole (1991) — Game-theoretic customer switching model
─────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import math
from typing import Dict, Any


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Intervention parameter table
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVENTIONS: Dict[str, Dict[str, Any]] = {
    'none': {
        'label'       : 'Baseline (No Action)',
        'max_delta'   : 0.0,
        'floor'       : 0.0,
        'cost_score'  : 0,        # 0-100, higher = costlier
        'description' : 'No intervention. Customer churn trajectory unchanged.',
        'recommendation': 'Baseline — loss imminent if no action taken.',
    },
    'offer': {
        'label'       : 'Automated Discount Offer',
        'max_delta'   : 0.18,     # max 18pt reduction
        'floor'       : 0.30,     # floor at 30%
        'cost_score'  : 20,
        'sensitivity' : {         # feature sensitivity modifiers
            'price'   : 1.4,      # price-sensitive customers respond more
            'loyal'   : 0.7,      # long-tenure customers respond less (not just $$)
        },
        'description' : 'Email/SMS automated coupon — 50% fee waiver for 6 months.',
        'recommendation': 'Effective for price-sensitive customers. Low trust impact.',
    },
    'call': {
        'label'       : 'Priority RM Call',
        'max_delta'   : 0.32,
        'floor'       : 0.20,
        'cost_score'  : 55,
        'sensitivity' : {
            'frustrated' : 1.5,   # frustrated customers respond most to human touch
            'loyal'      : 1.2,
        },
        'description' : 'Senior Relationship Manager personalized call.',
        'recommendation': 'Highest trust impact. Recommended for emotional customers.',
    },
    'upgrade': {
        'label'       : 'Premier Tier Upgrade',
        'max_delta'   : 0.48,
        'floor'       : 0.12,
        'cost_score'  : 80,
        'sensitivity' : {
            'highclv'   : 1.3,    # high CLV customers value status
            'comparison': 1.6,    # comparison shoppers value premium features
        },
        'description' : 'Waive requirements and fast-track Premier Banking upgrade.',
        'recommendation': 'Best ROI for high CLV customers. Requires manager sign-off.',
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core simulation function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def simulate_intervention(
    base_score: float,             # current churn probability [0-1]
    action: str,                   # intervention key
    customer: Dict[str, Any] | None = None  # optional customer context
) -> Dict[str, Any]:
    """
    Compute the counterfactual outcome of an intervention.

    Args:
        base_score: Current model churn probability (0.0 → 1.0)
        action:     One of 'none', 'offer', 'call', 'upgrade'
        customer:   Dict with keys: emotion_name, clv_inr, tenure_months

    Returns:
        {
          new_score, delta, delta_pct, revenue_retained,
          roi_score, label, description, recommendation
        }
    """
    ctx    = customer or {}
    action = action.lower().strip()

    if action not in INTERVENTIONS:
        action = 'none'

    params = INTERVENTIONS[action]

    # ── Compute effective delta ────────────────────────────────────────────
    raw_delta    = _compute_effective_delta(base_score, action, params, ctx)
    new_score    = max(params['floor'], base_score - raw_delta)
    actual_delta = base_score - new_score

    # ── Revenue model ──────────────────────────────────────────────────────
    clv = float(ctx.get('clv_inr', 200_000))
    # Expected revenue retained = delta_probability × CLV × discounted_factor
    discount_factor = 0.85   # 15% acquisition cost offset
    revenue_retained = int(actual_delta * clv * discount_factor)

    # ── ROI Score (0-100) ──────────────────────────────────────────────────
    roi_score = _compute_roi(actual_delta, revenue_retained, params['cost_score'])

    return {
        'action'           : action,
        'new_score'        : round(new_score * 100, 1),          # as percentage
        'delta'            : round(-actual_delta * 100, 1),       # negative = reduction
        'delta_pts'        : f"-{round(actual_delta * 100, 0):.0f} pts",
        'revenue_retained' : f"₹{revenue_retained:,}",
        'roi_score'        : roi_score,
        'label'            : _score_label(new_score),
        'color_class'      : _color_class(new_score),
        'description'      : params['description'],
        'recommendation'   : params['recommendation'],
        'confidence'       : _confidence(action, base_score, ctx),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Internal helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_effective_delta(base: float, action: str,
                               params: dict, ctx: dict) -> float:
    """Apply sensitivity modifiers to the raw max_delta."""
    max_d = params['max_delta']
    sens  = params.get('sensitivity', {})

    # Base effectiveness curves: high base risk? interventions are more effective
    # Logistic-shaped effectiveness vs base score
    effectiveness = _logistic(base, steepness=8, midpoint=0.55)

    mod = 1.0
    emotion = ctx.get('emotion_name', '').lower()
    tenure  = float(ctx.get('tenure_months', 24))
    clv     = float(ctx.get('clv_inr', 200_000))

    if action == 'offer':
        if 'price' in emotion:               mod *= sens.get('price', 1.0)
        if tenure > 36:                      mod *= sens.get('loyal', 1.0)

    elif action == 'call':
        if 'frustrat' in emotion:            mod *= sens.get('frustrated', 1.0)
        if tenure > 24:                      mod *= sens.get('loyal', 1.0)

    elif action == 'upgrade':
        if clv > 300_000:                    mod *= sens.get('highclv', 1.0)
        if 'comparison' in emotion:          mod *= sens.get('comparison', 1.0)

    return min(max_d, max_d * effectiveness * mod)


def _logistic(x: float, steepness: float = 8, midpoint: float = 0.5) -> float:
    """Sigmoid function: maps churn probability to effectiveness multiplier."""
    return 1 / (1 + math.exp(-steepness * (x - midpoint)))


def _compute_roi(delta: float, revenue: int, cost_score: int) -> int:
    """Composite ROI score 0-100."""
    if cost_score == 0:
        return 0
    roi = (delta * 100 * 0.6) + (min(revenue / 50_000, 10) * 2) - (cost_score * 0.2)
    return max(0, min(100, int(roi)))


def _score_label(score: float) -> str:
    if score >= 0.70: return 'Critical Churn Risk'
    if score >= 0.45: return 'Elevated Risk'
    if score >= 0.25: return 'Moderate Risk'
    return 'Stabilized'


def _color_class(score: float) -> str:
    if score >= 0.70: return 'high'
    if score >= 0.45: return 'med'
    return 'low'


def _confidence(action: str, base: float, ctx: dict) -> int:
    """Model confidence 50-95% based on data availability."""
    conf = 60
    if ctx: conf += 10
    if action != 'none': conf += 8
    if base > 0.6:    conf += 7
    return min(95, conf)

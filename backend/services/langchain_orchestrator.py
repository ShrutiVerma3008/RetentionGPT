"""
RetentionAI — LangChain Orchestrator (4 Prompt Modules)
=========================================================
Sequential LangChain pipeline connecting the 4 GPT-powered modules:

  Module 1 — Risk Analyst    : Why is customer at risk?
  Module 2 — Empathy Engine  : What is the emotional state?
  Module 3 — Action Selector : What is the Next Best Action?
  Module 4 — Outreach Writer : Personalized email / WhatsApp

Fallback: If OPENAI_API_KEY is not set, returns realistic synthetic
responses for demo purposes (hackathon-safe).

Usage:
    from backend.services.langchain_orchestrator import get_orchestrator
    orch = get_orchestrator()
    result = orch.run_pipeline(customer, channel="email")
"""

from __future__ import annotations
import os
import re
from typing import Dict, Any

# ── Try importing LangChain ────────────────────────────────────────────────
try:
    from langchain_community.chat_models import ChatOpenAI          # type: ignore
    from langchain.prompts import PromptTemplate           # type: ignore
    from langchain.chains import LLMChain                  # type: ignore
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt Templates (Module 1-4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

M1_RISK_ANALYST = PromptTemplate(
    input_variables=['name', 'segment', 'tenure', 'risk_factors'],
    template="""You are a senior financial risk analyst at a top Indian private bank.

Customer Profile:
- Name: {name}
- Segment: {segment}
- Tenure: {tenure} months
- Top Risk Drivers: {risk_factors}

In 2 concise sentences, summarize the primary reason this customer is at high churn risk.
Be specific, data-driven, and professional. Do not use bullet points.
"""
) if LANGCHAIN_AVAILABLE else None


M2_EMPATHY_ENGINE = PromptTemplate(
    input_variables=['name', 'emotion', 'risk_summary'],
    template="""You are an expert in customer psychology and emotional intelligence for banking.

Customer emotional state: {emotion}
Risk context: {risk_summary}

In 1-2 sentences, describe the emotional journey of {name} and what they are truly feeling.
Focus on empathy, not technical analysis.
"""
) if LANGCHAIN_AVAILABLE else None


M3_ACTION_SELECTOR = PromptTemplate(
    input_variables=['name', 'emotion', 'risk_summary', 'emotional_context', 'clv'],
    template="""You are a Customer Success Strategy Director at a leading bank.

Customer: {name}
Emotional State: {emotion}
Risk Summary: {risk_summary}
Emotional Context: {emotional_context}
Customer Lifetime Value: ₹{clv}

Select the single best retention action from:
  A) Priority RM Call
  B) Automated Discount Offer
  C) Product Upgrade / Tier Change
  D) Educational Content / Feature Demo

Respond with only the action name and a 1-sentence justification.
"""
) if LANGCHAIN_AVAILABLE else None


M4_OUTREACH_WRITER = PromptTemplate(
    input_variables=['name', 'segment', 'action', 'emotion', 'channel', 'bank_name'],
    template="""You are a world-class retention copywriter for {bank_name} bank.

Write a personalized {channel} message for:
- Customer Name: {name}
- Segment: {segment}
- Recommended Action: {action}
- Emotional State: {emotion}

Rules:
- {channel} should be warm, personal, and professional
- If email: include Subject line + 3 short paragraphs
- If WhatsApp: concise, friendly, max 80 words, add 1 emoji
- Address the emotional state subtly (don't directly call it out)
- End with a clear, low-friction call to action
"""
) if LANGCHAIN_AVAILABLE else None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Orchestrator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RetentionOrchestrator:
    """
    4-module sequential LangChain pipeline.
    Falls back to templated demo responses if OPENAI_API_KEY is absent.
    """

    def __init__(self):
        self.llm = None
        self.chains: Dict[str, Any] = {}
        self._init_chains()

    def _init_chains(self):
        api_key = os.getenv('OPENAI_API_KEY', '')
        if LANGCHAIN_AVAILABLE and api_key:
            try:
                self.llm = ChatOpenAI(
                    model_name='gpt-4o-mini',
                    temperature=0.7,
                    openai_api_key=api_key,
                )
                self.chains = {
                    'm1': LLMChain(llm=self.llm, prompt=M1_RISK_ANALYST),
                    'm2': LLMChain(llm=self.llm, prompt=M2_EMPATHY_ENGINE),
                    'm3': LLMChain(llm=self.llm, prompt=M3_ACTION_SELECTOR),
                    'm4': LLMChain(llm=self.llm, prompt=M4_OUTREACH_WRITER),
                }
                print("✅  LangChain orchestrator: GPT-4o-mini connected")
            except Exception as e:
                print(f"⚠️  LangChain init failed: {e}. Using demo mode.")
        else:
            print("ℹ️  LangChain orchestrator: Demo mode (no OPENAI_API_KEY)")

    # ── Main entry point ───────────────────────────────────────────────────
    def run_pipeline(
        self,
        customer: Dict[str, Any],
        channel: str = 'email',
        bank_name: str = 'HD Bank',
    ) -> Dict[str, Any]:
        """
        Runs all 4 modules sequentially.
        Returns dict with outputs from each module + final outreach draft.
        """
        name     = customer.get('name', 'Valued Customer')
        segment  = customer.get('segment', 'Premium Savings')
        tenure   = str(customer.get('tenure_months', 24))
        emotion  = customer.get('emotion_name', 'Frustrated')
        clv      = str(customer.get('clv_inr', 200_000))
        risk_factors = _format_risk_factors(customer.get('shap_reasons', []))

        if self.chains:
            return self._run_live(name, segment, tenure, emotion, clv,
                                   risk_factors, channel, bank_name)
        return self._run_demo(name, segment, emotion, channel, bank_name)

    def generate_outreach(
        self,
        customer: Dict[str, Any],
        channel: str = 'email',
        bank_name: str = 'HD Bank',
    ) -> str:
        """Convenience: returns only the outreach message string."""
        result = self.run_pipeline(customer, channel, bank_name)
        return result.get('outreach_draft', '')

    # ── Live LangChain execution ───────────────────────────────────────────
    def _run_live(self, name, segment, tenure, emotion, clv,
                   risk_factors, channel, bank_name) -> Dict[str, Any]:

        # Module 1 — Risk Analyst
        m1_out = self.chains['m1'].run(
            name=name, segment=segment, tenure=tenure, risk_factors=risk_factors
        )

        # Module 2 — Empathy Engine
        m2_out = self.chains['m2'].run(
            name=name, emotion=emotion, risk_summary=m1_out
        )

        # Module 3 — Action Selector
        m3_out = self.chains['m3'].run(
            name=name, emotion=emotion, risk_summary=m1_out,
            emotional_context=m2_out, clv=clv
        )

        # Module 4 — Outreach Writer
        m4_out = self.chains['m4'].run(
            name=name, segment=segment, action=m3_out,
            emotion=emotion, channel=channel, bank_name=bank_name
        )

        return {
            'module_1_risk_analysis'  : m1_out.strip(),
            'module_2_empathy'        : m2_out.strip(),
            'module_3_action'         : m3_out.strip(),
            'module_4_outreach'       : m4_out.strip(),
            'outreach_draft'          : m4_out.strip(),
            'mode'                    : 'gpt',
        }

    # ── Demo / fallback responses ──────────────────────────────────────────
    def _run_demo(self, name, segment, emotion, channel, bank_name) -> Dict[str, Any]:
        first = name.split()[0]
        emotion_lower = emotion.lower()

        action_map = {
            'frustrated'         : 'Priority RM Call',
            'price-sensitive'    : 'Offer 50% fee waiver for 6 months',
            'comparison-shopping': 'Premier Tier Early Upgrade',
            'confused'           : 'Educational email + dedicated helpline',
            'satisfied'          : 'Loyalty rewards programme enrollment',
        }
        action = next((v for k, v in action_map.items() if k in emotion_lower),
                      'Priority RM Call')

        if channel == 'email':
            draft = (
                f"Subject: We'd love to make things right for you, {first}\n\n"
                f"Hi {first},\n\n"
                f"I'm Rajesh from the Relationship Management team at {bank_name}. "
                f"I noticed you've been with us as a {segment} member and wanted to "
                f"personally reach out to ensure your experience with us has been exceptional.\n\n"
                f"I'd love to walk you through some exclusive benefits available to you "
                f"and resolve anything that may not be working as expected. "
                f"Would you be available for a 10-minute call this week?\n\n"
                f"Looking forward to connecting.\nWarm regards,\nRajesh Kumar\n"
                f"Senior Relationship Manager, {bank_name}"
            )
        else:
            draft = (
                f"Hi {first} 👋 This is Rajesh from {bank_name}. "
                f"Just checking in as a valued {segment} member — we have something "
                f"special for you! Could we connect for 5 mins this week? "
                f"Reply YES and I'll call you at your preferred time."
            )

        return {
            'module_1_risk_analysis': f'{name} shows multiple disengagement signals consistent with pre-churn behaviour in their {segment} segment.',
            'module_2_empathy'      : f'{first} appears {emotion_lower}, likely feeling undervalued despite their long tenure.',
            'module_3_action'       : action,
            'module_4_outreach'     : draft,
            'outreach_draft'        : draft,
            'mode'                  : 'demo',
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _format_risk_factors(reasons: list) -> str:
    if not reasons:
        return 'Transaction velocity drop, open complaints, balance decline'
    return '; '.join(r.get('title', '') for r in reasons[:3])


# Module-level singleton
_orchestrator: RetentionOrchestrator | None = None

def get_orchestrator() -> RetentionOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RetentionOrchestrator()
    return _orchestrator

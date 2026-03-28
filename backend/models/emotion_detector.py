"""
RetentionAI — BERT Emotion Detector
=====================================
Uses DistilBERT zero-shot classification to detect customer emotion
from free-text complaint/feedback text.

Emotion classes (aligned with dashboard tags):
  0 → Satisfied
  1 → Confused
  2 → Frustrated
  3 → Price-sensitive
  4 → Comparison-shopping

Usage:
    from backend.models.emotion_detector import get_emotion_detector
    detector = get_emotion_detector()
    result = detector.classify("This app keeps crashing and my issue is unresolved.")
    # → {"emotion": "Frustrated", "confidence": 0.87, "label_id": 2}
"""

from __future__ import annotations
import re
from typing import Dict, Any

EMOTION_CLASSES = [
    'Satisfied',
    'Confused',
    'Frustrated',
    'Price-sensitive',
    'Comparison-shopping',
]

# ── Keyword heuristic fallback (works without transformers installed) ──────────
KEYWORD_MAP = {
    'Frustrated'         : ['frustrated', 'angry', 'unresolved', 'crash', 'broken',
                            'terrible', 'worst', 'useless', 'disgusting', 'sick of'],
    'Confused'           : ['confused', 'unclear', "don't understand", 'how to',
                            'what is', 'help me', 'guide', 'explain'],
    'Price-sensitive'    : ['expensive', 'fee', 'charge', 'hidden', 'costly',
                            'cheaper', 'discount', 'waive', 'high cost'],
    'Comparison-shopping': ['other bank', 'competitor', 'comparing', 'hdfc', 'sbi',
                            'icici', 'axis', 'better rates', 'switching'],
    'Satisfied'          : ['great', 'excellent', 'happy', 'love', 'best', 'perfect',
                            'thanks', 'wonderful'],
}


class EmotionDetector:
    """
    Emotion classifier with two modes:
      1. BERT mode  — uses transformers pipeline (if installed)
      2. Heuristic  — keyword scoring fallback (always available)
    """

    def __init__(self):
        self._pipeline = None
        self._try_load_bert()

    def _try_load_bert(self):
        try:
            from transformers import pipeline          # type: ignore
            self._pipeline = pipeline(
                "zero-shot-classification",
                model="typeform/distilbert-base-uncased-mnli",
                device=-1,                             # CPU
            )
            print("✅  EmotionDetector: DistilBERT loaded (zero-shot)")
        except Exception as e:
            print(f"ℹ️  EmotionDetector: BERT unavailable ({e}), using heuristic fallback")

    # ── Public API ──────────────────────────────────────────────────────────
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify emotion from free-text.
        Returns: {"emotion": str, "confidence": float, "label_id": int, "mode": str}
        """
        if not text or not text.strip():
            return self._build_result('Satisfied', 1.0, 0, 'default')

        if self._pipeline is not None:
            return self._bert_classify(text)
        return self._heuristic_classify(text)

    def classify_from_features(self, complaints: int, nps: float,
                                competitor_inquiry: int) -> Dict[str, Any]:
        """
        Classify emotion from structured features (no text needed).
        Used when only tabular data is available.
        """
        if competitor_inquiry == 1:
            return self._build_result('Comparison-shopping', 0.82, 4, 'feature-rule')
        if complaints >= 2:
            return self._build_result('Frustrated', 0.75 + complaints * 0.05, 2, 'feature-rule')
        if nps < 5:
            return self._build_result('Price-sensitive', 0.68, 3, 'feature-rule')
        if nps < 7:
            return self._build_result('Confused', 0.60, 1, 'feature-rule')
        return self._build_result('Satisfied', 0.78, 0, 'feature-rule')

    # ── Internal classifiers ────────────────────────────────────────────────
    def _bert_classify(self, text: str) -> Dict[str, Any]:
        result = self._pipeline(text, candidate_labels=EMOTION_CLASSES, multi_label=False)
        emotion    = result['labels'][0]
        confidence = result['scores'][0]
        label_id   = EMOTION_CLASSES.index(emotion)
        return self._build_result(emotion, confidence, label_id, 'bert')

    def _heuristic_classify(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        scores: Dict[str, int] = {e: 0 for e in EMOTION_CLASSES}

        for emotion, keywords in KEYWORD_MAP.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    scores[emotion] += 1

        best_emotion = max(scores, key=lambda e: scores[e])
        if scores[best_emotion] == 0:
            best_emotion = 'Confused'   # graceful default

        total = sum(scores.values()) or 1
        confidence = round(scores[best_emotion] / total, 2)
        label_id   = EMOTION_CLASSES.index(best_emotion)
        return self._build_result(best_emotion, confidence, label_id, 'heuristic')

    @staticmethod
    def _build_result(emotion: str, confidence: float,
                      label_id: int, mode: str) -> Dict[str, Any]:
        return {
            'emotion'   : emotion,
            'confidence': round(min(confidence, 1.0), 3),
            'label_id'  : label_id,
            'mode'      : mode,
        }


# Module-level singleton
_detector: EmotionDetector | None = None

def get_emotion_detector() -> EmotionDetector:
    global _detector
    if _detector is None:
        _detector = EmotionDetector()
    return _detector

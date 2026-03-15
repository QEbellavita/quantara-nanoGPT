"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Emotional Domain Processor
===============================================================================
Processes emotional events to derive mood distributions, volatility, and
recovery rates used by the User Profile Engine DNA fingerprint.

Integrates with:
- Neural Workflow AI Engine
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.
===============================================================================
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List

from . import BaseDomainProcessor


# ---------------------------------------------------------------------------
# Emotion family mappings
# ---------------------------------------------------------------------------

EMOTION_FAMILIES: Dict[str, str] = {
    # Positive
    "joy": "positive", "happiness": "positive", "excitement": "positive",
    "contentment": "positive", "satisfaction": "positive", "love": "positive",
    "gratitude": "positive", "pride": "positive", "optimism": "positive",
    "amusement": "positive", "admiration": "positive", "approval": "positive",
    "caring": "positive", "desire": "positive", "relief": "positive",
    # Negative
    "sadness": "negative", "anger": "negative", "fear": "negative",
    "disgust": "negative", "frustration": "negative", "anxiety": "negative",
    "grief": "negative", "disappointment": "negative", "disapproval": "negative",
    "remorse": "negative", "embarrassment": "negative",
    # Neutral
    "neutral": "neutral", "surprise": "neutral", "curiosity": "neutral",
    "confusion": "neutral", "realization": "neutral",
}


def _family(emotion: str) -> str:
    return EMOTION_FAMILIES.get(emotion.lower(), "neutral")


class EmotionalProcessor(BaseDomainProcessor):
    """Derives emotional DNA metrics from raw emotion detection events."""

    domain = "emotional"

    def compute(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        mood_dist: Counter = Counter()
        family_dist: Counter = Counter()
        confidences: List[float] = []

        for event in events:
            payload: Dict[str, Any] = {}
            raw = event.get("payload")
            if isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    payload = {}
            elif isinstance(raw, dict):
                payload = raw

            emotion = str(payload.get("emotion", payload.get("label", "neutral"))).lower()
            mood_dist[emotion] += 1
            family_dist[_family(emotion)] += 1

            conf = payload.get("confidence", payload.get("score", 1.0))
            try:
                confidences.append(float(conf))
            except (TypeError, ValueError):
                confidences.append(1.0)

        # Dominant emotion / family
        dominant_emotion = mood_dist.most_common(1)[0][0] if mood_dist else "neutral"
        dominant_family = family_dist.most_common(1)[0][0] if family_dist else "neutral"

        # Emotional range
        emotional_range = len(mood_dist)

        # Average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Volatility: ratio of consecutive emotion changes
        emotion_sequence = []
        for event in sorted(events, key=lambda e: e.get("timestamp", 0)):
            raw = event.get("payload")
            payload = {}
            if isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(raw, dict):
                payload = raw
            emotion_sequence.append(
                str(payload.get("emotion", payload.get("label", "neutral"))).lower()
            )

        if len(emotion_sequence) > 1:
            changes = sum(
                1 for a, b in zip(emotion_sequence, emotion_sequence[1:]) if a != b
            )
            volatility = changes / (len(emotion_sequence) - 1)
        else:
            volatility = 0.0

        # Recovery rate: negative -> positive family transitions
        family_sequence = [_family(e) for e in emotion_sequence]
        if len(family_sequence) > 1:
            neg_to_pos = sum(
                1 for a, b in zip(family_sequence, family_sequence[1:])
                if a == "negative" and b == "positive"
            )
            negative_count = family_sequence.count("negative")
            recovery_rate = neg_to_pos / negative_count if negative_count > 0 else 0.0
        else:
            recovery_rate = 0.0

        score = min(1.0, len(events) / 100) * avg_confidence

        return {
            "score": round(score, 4),
            "event_count": len(events),
            "metrics": {
                "mood_distribution": dict(mood_dist),
                "family_distribution": dict(family_dist),
                "dominant_emotion": dominant_emotion,
                "dominant_family": dominant_family,
                "emotional_range": emotional_range,
                "avg_confidence": round(avg_confidence, 4),
                "volatility": round(volatility, 4),
                "recovery_rate": round(recovery_rate, 4),
            },
        }

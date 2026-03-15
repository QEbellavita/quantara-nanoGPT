"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Linguistic Domain Processor
===============================================================================
Analyses text payloads in events to compute vocabulary complexity, sentence
length, and tone distributions for the User Profile Engine DNA fingerprint.

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
import re
from collections import Counter
from typing import Any, Dict, List

from . import BaseDomainProcessor


class LinguisticProcessor(BaseDomainProcessor):
    """Derives linguistic DNA metrics from text-bearing event payloads."""

    domain = "linguistic"

    def compute(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        sentence_lengths: List[int] = []
        all_words: List[str] = []
        unique_words: set = set()
        tone_dist: Counter = Counter()

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

            text = str(
                payload.get("text", payload.get("message", payload.get("content", "")))
            ).strip()

            if text:
                words = _tokenise(text)
                sentence_lengths.append(len(words))
                all_words.extend(words)
                unique_words.update(words)

            tone = str(payload.get("tone", payload.get("sentiment", ""))).lower()
            if tone:
                tone_dist[tone] += 1

        avg_sentence_length = (
            sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
        )
        vocabulary_complexity = (
            len(unique_words) / len(all_words) if all_words else 0.0
        )
        dominant_tone = tone_dist.most_common(1)[0][0] if tone_dist else "neutral"

        score = min(1.0, len(events) / 100)

        return {
            "score": round(score, 4),
            "event_count": len(events),
            "metrics": {
                "avg_sentence_length": round(avg_sentence_length, 4),
                "vocabulary_complexity": round(vocabulary_complexity, 4),
                "tone_distribution": dict(tone_dist),
                "dominant_tone": dominant_tone,
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """Simple whitespace + punctuation tokeniser (lowercase)."""
    return [w.lower() for w in re.findall(r"\b\w+\b", text)]

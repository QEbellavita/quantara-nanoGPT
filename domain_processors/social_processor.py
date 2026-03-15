"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Social Domain Processor
===============================================================================
Processes social interaction events to quantify engagement frequency,
sharing behaviour, and feedback loops for the User Profile Engine DNA
fingerprint.

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
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from . import BaseDomainProcessor


class SocialProcessor(BaseDomainProcessor):
    """Derives social DNA metrics from interaction and sharing events."""

    domain = "social"

    def compute(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        interaction_count = len(events)
        shares = 0
        feedback_received = 0
        interaction_types: Counter = Counter()
        active_days: Set[str] = set()

        for event in events:
            event_type = str(event.get("event_type", "interaction")).lower()
            interaction_types[event_type] += 1

            payload: Dict[str, Any] = {}
            raw = event.get("payload")
            if isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    payload = {}
            elif isinstance(raw, dict):
                payload = raw

            if payload.get("shared") or event_type in ("share", "shared"):
                shares += 1
            if payload.get("feedback") is not None or event_type in (
                "feedback", "feedback_received"
            ):
                feedback_received += 1

            ts = event.get("timestamp")
            if ts is not None:
                try:
                    day_str = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
                        "%Y-%m-%d"
                    )
                    active_days.add(day_str)
                except (TypeError, ValueError, OSError):
                    pass

        daily_interaction_rate = (
            interaction_count / len(active_days) if active_days else float(interaction_count)
        )

        score = min(1.0, len(events) / 100)

        return {
            "score": round(score, 4),
            "event_count": interaction_count,
            "metrics": {
                "interaction_count": interaction_count,
                "shares": shares,
                "feedback_received": feedback_received,
                "interaction_types": dict(interaction_types),
                "daily_interaction_rate": round(daily_interaction_rate, 4),
            },
        }

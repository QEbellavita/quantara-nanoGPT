"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Behavioral Domain Processor
===============================================================================
Processes session lifecycle events to compute completion rates, intervention
responses, and engagement streaks for the User Profile Engine DNA fingerprint.

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
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from . import BaseDomainProcessor


class BehavioralProcessor(BaseDomainProcessor):
    """Derives behavioral DNA metrics from session-lifecycle events."""

    domain = "behavioral"

    def compute(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        started = 0
        completed = 0
        abandoned = 0
        intervention_triggered = 0
        intervention_responded = 0

        active_days: Set[str] = set()

        for event in events:
            event_type = str(event.get("event_type", "")).lower()

            if event_type == "session_started":
                started += 1
            elif event_type == "session_completed":
                completed += 1
            elif event_type == "session_abandoned":
                abandoned += 1

            payload: Dict[str, Any] = {}
            raw = event.get("payload")
            if isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    payload = {}
            elif isinstance(raw, dict):
                payload = raw

            if payload.get("intervention_triggered") or event_type == "intervention_triggered":
                intervention_triggered += 1
            if payload.get("intervention_responded") or event_type == "intervention_responded":
                intervention_responded += 1

            ts = event.get("timestamp")
            if ts is not None:
                try:
                    day_str = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
                        "%Y-%m-%d"
                    )
                    active_days.add(day_str)
                except (TypeError, ValueError, OSError):
                    pass

        completion_rate = completed / started if started > 0 else 0.0
        intervention_response_rate = (
            intervention_responded / intervention_triggered
            if intervention_triggered > 0
            else 0.0
        )
        engagement_streak = _max_consecutive_days(active_days)

        score = min(1.0, len(events) / 100)

        return {
            "score": round(score, 4),
            "event_count": len(events),
            "metrics": {
                "completion_rate": round(completion_rate, 4),
                "intervention_response_rate": round(intervention_response_rate, 4),
                "engagement_streak": engagement_streak,
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_consecutive_days(day_strings: Set[str]) -> int:
    """Return the length of the longest run of consecutive calendar days."""
    if not day_strings:
        return 0

    from datetime import date, timedelta

    days = sorted(date.fromisoformat(d) for d in day_strings)
    max_streak = 1
    current = 1
    for prev, curr in zip(days, days[1:]):
        if curr - prev == timedelta(days=1):
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 1
    return max_streak

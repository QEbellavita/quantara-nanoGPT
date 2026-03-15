"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Aspirational Domain Processor
===============================================================================
Processes goal-setting, growth, and values events to quantify ambition
fulfilment and core values alignment for the User Profile Engine DNA
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
from typing import Any, Dict, List

from . import BaseDomainProcessor


class AspirationalProcessor(BaseDomainProcessor):
    """Derives aspirational DNA metrics from goal and values events."""

    domain = "aspirational"

    def compute(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        goals_set = 0
        goals_completed = 0
        growth_areas: Counter = Counter()
        value_mentions: Counter = Counter()

        for event in events:
            event_type = str(event.get("event_type", "")).lower()

            payload: Dict[str, Any] = {}
            raw = event.get("payload")
            if isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    payload = {}
            elif isinstance(raw, dict):
                payload = raw

            # Goal tracking
            if event_type in ("goal_set", "goal_created") or payload.get("goal_set"):
                goals_set += 1
            if event_type in ("goal_completed", "goal_achieved") or payload.get(
                "goal_completed"
            ):
                goals_completed += 1

            # Growth areas
            area = payload.get("growth_area", payload.get("area"))
            if area:
                growth_areas[str(area).lower()] += 1

            # Values
            values = payload.get("values", payload.get("core_values", []))
            if isinstance(values, str):
                values = [values]
            for v in values:
                value_mentions[str(v).lower()] += 1

        goal_completion_rate = goals_completed / goals_set if goals_set > 0 else 0.0
        active_growth_areas = list(growth_areas.keys())
        core_values = [v for v, _ in value_mentions.most_common(5)]

        score = min(1.0, len(events) / 100)

        return {
            "score": round(score, 4),
            "event_count": len(events),
            "metrics": {
                "goals_set": goals_set,
                "goals_completed": goals_completed,
                "goal_completion_rate": round(goal_completion_rate, 4),
                "active_growth_areas": active_growth_areas,
                "core_values": core_values,
            },
        }

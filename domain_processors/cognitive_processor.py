"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Cognitive Domain Processor
===============================================================================
Processes cognitive technique events to quantify diversity of mental exercises,
average completion times, and retention / success rates for the User Profile
Engine DNA fingerprint.

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


class CognitiveProcessor(BaseDomainProcessor):
    """Derives cognitive DNA metrics from mental-technique usage events."""

    domain = "cognitive"

    def compute(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        technique_usage: Counter = Counter()
        completion_times: List[float] = []
        outcome_success = 0

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

            technique = str(
                payload.get("technique", payload.get("technique_name", "unknown"))
            ).lower()
            technique_usage[technique] += 1

            ct = payload.get("completion_time", payload.get("duration"))
            if ct is not None:
                try:
                    completion_times.append(float(ct))
                except (TypeError, ValueError):
                    pass

            outcome = str(payload.get("outcome", payload.get("result", ""))).lower()
            if outcome in ("success", "completed", "passed"):
                outcome_success += 1

        technique_diversity = len(technique_usage)
        avg_completion_time = (
            sum(completion_times) / len(completion_times) if completion_times else None
        )
        technique_retention_rate = outcome_success / len(events)

        score = min(1.0, len(events) / 100)

        return {
            "score": round(score, 4),
            "event_count": len(events),
            "metrics": {
                "technique_usage": dict(technique_usage),
                "technique_diversity": technique_diversity,
                "avg_completion_time": (
                    round(avg_completion_time, 4) if avg_completion_time is not None else None
                ),
                "technique_retention_rate": round(technique_retention_rate, 4),
            },
        }

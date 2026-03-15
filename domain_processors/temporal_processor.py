"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Temporal Domain Processor
===============================================================================
Analyses event timestamps to build hour/day distributions, identify peak
activity hours, and infer the user's chronotype for the User Profile Engine
DNA fingerprint.

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
from typing import Any, Dict, List

from . import BaseDomainProcessor

# Hour ranges used for chronotype classification
MORNING_HOURS = set(range(5, 12))   # 05:00 – 11:59
EVENING_HOURS = set(range(18, 24))  # 18:00 – 23:59


class TemporalProcessor(BaseDomainProcessor):
    """Derives temporal DNA metrics from event timestamp data."""

    domain = "temporal"

    def compute(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        hour_dist: Counter = Counter()
        day_dist: Counter = Counter()

        for event in events:
            ts = event.get("timestamp")
            if ts is None:
                continue
            try:
                dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                hour_dist[dt.hour] += 1
                # 0=Monday … 6=Sunday
                day_dist[dt.strftime("%A")] += 1
            except (TypeError, ValueError, OSError):
                continue

        # Peak hours: top 3 most active hours
        peak_hours = [h for h, _ in hour_dist.most_common(3)]

        # Chronotype
        morning_activity = sum(hour_dist[h] for h in MORNING_HOURS)
        evening_activity = sum(hour_dist[h] for h in EVENING_HOURS)
        total_activity = sum(hour_dist.values()) or 1
        morning_ratio = morning_activity / total_activity
        evening_ratio = evening_activity / total_activity

        if morning_ratio > 0.4 and morning_ratio > evening_ratio:
            chronotype = "morning"
        elif evening_ratio > 0.4 and evening_ratio > morning_ratio:
            chronotype = "evening"
        else:
            chronotype = "balanced"

        score = min(1.0, len(events) / 100)

        return {
            "score": round(score, 4),
            "event_count": len(events),
            "metrics": {
                "hour_distribution": dict(hour_dist),
                "day_distribution": dict(day_dist),
                "peak_hours": peak_hours,
                "chronotype": chronotype,
            },
        }

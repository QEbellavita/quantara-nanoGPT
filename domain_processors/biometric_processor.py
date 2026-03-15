"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Biometric Domain Processor
===============================================================================
Processes physiological sensor events (HR, HRV, EDA) to derive baseline
metrics and stress ratios for the User Profile Engine DNA fingerprint.

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
from typing import Any, Dict, List, Optional

from . import BaseDomainProcessor

# Heart-rate threshold above which an event is considered high-stress.
HR_STRESS_THRESHOLD = 100.0


class BiometricProcessor(BaseDomainProcessor):
    """Derives biometric DNA metrics from HR / HRV / EDA sensor events."""

    domain = "biometric"

    def compute(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        hr_values: List[float] = []
        hrv_values: List[float] = []
        eda_values: List[float] = []
        high_hr_count = 0

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

            hr = _to_float(payload.get("hr", payload.get("heart_rate")))
            hrv = _to_float(payload.get("hrv", payload.get("heart_rate_variability")))
            eda = _to_float(payload.get("eda", payload.get("electrodermal_activity")))

            if hr is not None:
                hr_values.append(hr)
                if hr > HR_STRESS_THRESHOLD:
                    high_hr_count += 1
            if hrv is not None:
                hrv_values.append(hrv)
            if eda is not None:
                eda_values.append(eda)

        resting_hr = _mean(hr_values)
        hr_min = min(hr_values) if hr_values else None
        hr_max = max(hr_values) if hr_values else None
        hrv_baseline = _mean(hrv_values)
        eda_baseline = _mean(eda_values)
        stress_ratio = high_hr_count / len(events) if events else 0.0

        score = min(1.0, len(events) / 100)

        return {
            "score": round(score, 4),
            "event_count": len(events),
            "metrics": {
                "resting_hr": _round_opt(resting_hr),
                "hr_min": _round_opt(hr_min),
                "hr_max": _round_opt(hr_max),
                "hrv_baseline": _round_opt(hrv_baseline),
                "eda_baseline": _round_opt(eda_baseline),
                "stress_ratio": round(stress_ratio, 4),
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _round_opt(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    return round(value, ndigits) if value is not None else None

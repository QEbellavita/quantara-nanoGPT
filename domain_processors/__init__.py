"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Domain Processors Package
===============================================================================
Abstract base class and registry for all 8 DNA domain processors.
Each processor ingests raw events from ProfileDB and emits a normalised score
plus domain-specific metric breakdowns consumed by the User Profile Engine and
Evolution Engine.

Integrates with:
- Neural Workflow AI Engine
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.
===============================================================================
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseDomainProcessor(abc.ABC):
    """Abstract processor that converts a list of raw events into a domain score.

    Sub-classes must declare a ``domain`` class variable (str) and implement
    ``compute(events)``.
    """

    domain: str = ""  # e.g. 'emotional', 'biometric', …

    @abc.abstractmethod
    def compute(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process *events* and return a result dict.

        Returns
        -------
        dict with keys:
            score       : float  in [0.0, 1.0]
            event_count : int
            metrics     : dict   domain-specific breakdown
        """

    def get_empty_score(self) -> Dict[str, Any]:
        """Return a zero-score result for when there are no events."""
        return {"score": 0.0, "metrics": {}, "event_count": 0}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def get_all_processors() -> List[BaseDomainProcessor]:
    """Return one instance of every registered domain processor."""
    from .emotional_processor import EmotionalProcessor
    from .biometric_processor import BiometricProcessor
    from .cognitive_processor import CognitiveProcessor
    from .behavioral_processor import BehavioralProcessor
    from .temporal_processor import TemporalProcessor
    from .linguistic_processor import LinguisticProcessor
    from .social_processor import SocialProcessor
    from .aspirational_processor import AspirationalProcessor

    return [
        EmotionalProcessor(),
        BiometricProcessor(),
        CognitiveProcessor(),
        BehavioralProcessor(),
        TemporalProcessor(),
        LinguisticProcessor(),
        SocialProcessor(),
        AspirationalProcessor(),
    ]

"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Intelligence Publisher
===============================================================================
Personalizes therapy, coaching, and workflow intelligence from user fingerprints
and publishes actionable payloads to the ProfileEventBus for downstream
consumption by the Neural Workflow AI Engine and connected services.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.
===============================================================================
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from profile_event_bus import ProfileEventBus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mode(stage: int) -> str:
    """Return 'active' for stage >= 3, else 'advisory'."""
    return "active" if stage >= 3 else "advisory"


# ---------------------------------------------------------------------------
# TherapyPersonalizer
# ---------------------------------------------------------------------------

class TherapyPersonalizer:
    """
    Derives therapy personalization signals from the behavioral and cognitive
    DNA strands of a user fingerprint.

    Connected to:
    - Neural Workflow AI Engine
    - ML Training & Prediction Systems
    - Backend APIs (cases, workflows, analytics)
    """

    def compute(self, fingerprint: Dict[str, Any], stage: int) -> Dict[str, Any]:
        """
        Compute a therapy personalization payload.

        Parameters
        ----------
        fingerprint : dict
            User genetic fingerprint keyed by domain (behavioral, cognitive, …).
        stage : int
            Current profile evolution stage (1-5).

        Returns
        -------
        dict
            Therapy personalization signals.
        """
        behavioral = fingerprint.get("behavioral", {})
        cognitive = fingerprint.get("cognitive", {})

        behavioral_metrics = behavioral.get("metrics", {})
        cognitive_metrics = cognitive.get("metrics", {})

        # --- Technique scores from cognitive DNA ---
        technique_usage: Dict[str, int] = cognitive_metrics.get("technique_usage", {})
        retention_rate: float = float(cognitive_metrics.get("technique_retention_rate", 0.0))

        technique_scores: Dict[str, float] = {
            technique: min(1.0, count / 20) * retention_rate
            for technique, count in technique_usage.items()
        }

        # --- Behavioral rates ---
        completion_rate: float = float(behavioral_metrics.get("completion_rate", 0.0))
        intervention_response_rate: float = float(
            behavioral_metrics.get("intervention_response_rate", 0.0)
        )

        # --- Preferred step type ---
        preferred_step_type = "cognitive" if retention_rate > 0.7 else "calming"

        return {
            "technique_scores": technique_scores,
            "completion_rate": round(completion_rate, 4),
            "intervention_response_rate": round(intervention_response_rate, 4),
            "preferred_step_type": preferred_step_type,
            "weight_change_bound": 0.3,
            "mode": _get_mode(stage),
        }


# ---------------------------------------------------------------------------
# CoachingPersonalizer
# ---------------------------------------------------------------------------

class CoachingPersonalizer:
    """
    Derives coaching personalization signals from the linguistic, social, and
    aspirational DNA strands of a user fingerprint.

    Connected to:
    - Neural Workflow AI Engine
    - ML Training & Prediction Systems
    - All Dashboard Data Integration
    """

    def compute(self, fingerprint: Dict[str, Any], stage: int) -> Dict[str, Any]:
        """
        Compute a coaching personalization payload.

        Parameters
        ----------
        fingerprint : dict
            User genetic fingerprint keyed by domain.
        stage : int
            Current profile evolution stage (1-5).

        Returns
        -------
        dict
            Coaching personalization signals.
        """
        linguistic = fingerprint.get("linguistic", {})
        aspirational = fingerprint.get("aspirational", {})

        linguistic_metrics = linguistic.get("metrics", {})
        aspirational_metrics = aspirational.get("metrics", {})

        avg_sentence_length: float = float(linguistic_metrics.get("avg_sentence_length", 0.0))
        vocabulary_complexity: float = float(linguistic_metrics.get("vocabulary_complexity", 0.0))

        # --- Preferred tone ---
        preferred_tone = "direct" if avg_sentence_length < 10 else "supportive"

        # --- Vocabulary level ---
        vocabulary_level = "complex" if vocabulary_complexity > 0.6 else "simple"

        # --- Response depth ---
        response_depth = "detailed" if avg_sentence_length > 15 else "brief"

        # --- Active goals from aspirational core values ---
        active_goals = list(aspirational_metrics.get("core_values", []))

        return {
            "preferred_tone": preferred_tone,
            "vocabulary_level": vocabulary_level,
            "response_depth": response_depth,
            "active_goals": active_goals,
            "mode": _get_mode(stage),
        }


# ---------------------------------------------------------------------------
# WorkflowPrioritizer
# ---------------------------------------------------------------------------

class WorkflowPrioritizer:
    """
    Derives workflow prioritization signals from the emotional and behavioral
    DNA strands of a user fingerprint.

    Connected to:
    - Neural Workflow AI Engine (Phases 1-5)
    - Backend APIs (cases, workflows, analytics)
    - Real-time data from customer service, distribution, etc.
    """

    def compute(self, fingerprint: Dict[str, Any], stage: int) -> Dict[str, Any]:
        """
        Compute a workflow prioritization payload.

        Parameters
        ----------
        fingerprint : dict
            User genetic fingerprint keyed by domain.
        stage : int
            Current profile evolution stage (1-5).

        Returns
        -------
        dict
            Workflow prioritization signals with emotional_readiness_score in [0, 1].
        """
        emotional = fingerprint.get("emotional", {})
        behavioral = fingerprint.get("behavioral", {})

        emotional_metrics = emotional.get("metrics", {})
        behavioral_metrics = behavioral.get("metrics", {})

        # --- Stability (inverse of volatility) ---
        volatility: float = float(emotional_metrics.get("volatility", 0.0))
        stability: float = 1.0 - volatility

        # --- Valence from dominant emotion family ---
        dominant_family: str = str(emotional_metrics.get("dominant_family", "neutral")).lower()
        if dominant_family == "positive":
            valence = 1.0
        elif dominant_family == "negative":
            valence = 0.3
        else:
            valence = 0.5

        # --- Momentum from behavioral metrics ---
        completion_rate: float = float(behavioral_metrics.get("completion_rate", 0.0))
        streak: int = int(behavioral_metrics.get("engagement_streak", 0))
        momentum: float = completion_rate * 0.6 + min(streak, 10) / 10 * 0.4

        # --- Composite readiness score ---
        emotional_readiness_score: float = (
            stability * 0.4 + valence * 0.3 + momentum * 0.3
        )
        # Clamp to [0, 1]
        emotional_readiness_score = max(0.0, min(1.0, emotional_readiness_score))

        return {
            "emotional_readiness_score": round(emotional_readiness_score, 4),
            "stability": round(stability, 4),
            "valence": round(valence, 4),
            "momentum": round(momentum, 4),
            "mode": _get_mode(stage),
        }


# ---------------------------------------------------------------------------
# IntelligencePublisher
# ---------------------------------------------------------------------------

class IntelligencePublisher:
    """
    Orchestrates therapy, coaching, and workflow personalization for a user
    and publishes each intelligence payload to the ProfileEventBus.

    Optionally delivers payloads via an outbound connector (e.g.
    NeuralEcosystemConnector) for webhook-based integration with external
    Quantara services.

    Connected to:
    - Neural Workflow AI Engine (Phases 1-5)
    - ML Training & Prediction Systems
    - Backend APIs (cases, workflows, analytics)
    - All Dashboard Data Integration
    - Real-time data from customer service, distribution, etc.
    """

    def __init__(self, bus: ProfileEventBus, connector: Optional[Any] = None) -> None:
        self._bus = bus
        self._connector = connector
        self._therapy = TherapyPersonalizer()
        self._coaching = CoachingPersonalizer()
        self._workflow = WorkflowPrioritizer()

    def publish_for_user(
        self,
        user_id: str,
        fingerprint: Dict[str, Any],
        stage: int,
        confidence: float,
    ) -> None:
        """
        Compute and publish intelligence payloads for *user_id*.

        Four payloads are published to the bus under the topics:
        - ``intelligence.therapy``
        - ``intelligence.coaching``
        - ``intelligence.workflow``
        - ``intelligence.calibration``

        If a connector is configured the same payloads are forwarded via
        outbound webhooks.

        Parameters
        ----------
        user_id : str
            Unique identifier for the target user.
        fingerprint : dict
            Full user genetic fingerprint keyed by domain.
        stage : int
            Current profile evolution stage (1-5).
        confidence : float
            Overall fingerprint confidence score in [0, 1].
        """
        timestamp = time.time()

        therapy_payload = self._therapy.compute(fingerprint, stage)
        coaching_payload = self._coaching.compute(fingerprint, stage)
        workflow_payload = self._workflow.compute(fingerprint, stage)
        calibration_payload = {
            "user_id": user_id,
            "stage": stage,
            "confidence": round(confidence, 4),
            "timestamp": timestamp,
            "mode": _get_mode(stage),
        }

        events = [
            ("intelligence.therapy", {**therapy_payload, "user_id": user_id, "timestamp": timestamp}),
            ("intelligence.coaching", {**coaching_payload, "user_id": user_id, "timestamp": timestamp}),
            ("intelligence.workflow", {**workflow_payload, "user_id": user_id, "timestamp": timestamp}),
            ("intelligence.calibration", calibration_payload),
        ]

        for topic, payload in events:
            self._bus.publish(topic, payload)
            logger.debug(
                "IntelligencePublisher: published topic=%r user_id=%r stage=%d",
                topic,
                user_id,
                stage,
            )

        # Outbound connector delivery
        if self._connector is not None:
            for topic, payload in events:
                try:
                    self._connector.deliver(topic, payload)
                except Exception:
                    logger.exception(
                        "IntelligencePublisher: connector delivery failed for topic=%r user_id=%r",
                        topic,
                        user_id,
                    )

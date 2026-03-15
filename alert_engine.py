"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Alert Engine
===============================================================================
Reactive and predictive alert detection for the Quantara emotional AI platform.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.
===============================================================================
"""

import logging
import math
import time
from collections import defaultdict
from typing import Dict, List, Optional

from profile_event_bus import ProfileEventBus

logger = logging.getLogger(__name__)

# Emotion family classifications
NEGATIVE_FAMILIES = {"Sadness", "Anger", "Fear", "Self-Conscious"}
POSITIVE_FAMILIES = {"Joy", "Love", "Calm"}

# ---------------------------------------------------------------------------
# ReactiveDetector
# ---------------------------------------------------------------------------


class ReactiveDetector:
    """
    Detects emotional distress and recovery patterns from a stream of events.

    Checks 6 pattern types:
    1. emotional_spiral  — 5+ negative emotions within 2 hours
    2. rapid_cycling     — 8+ consecutive emotion family changes within 1 hour
    3. sustained_stress  — biometric stress_ratio > 0.5 for 30+ minutes
    4. emotional_flatline — same single emotion for 24+ hours
    5. engagement_drop   — no events for 3+ days
    6. recovery_detected — transition from 2+ negative to 2+ positive emotions
    """

    def check(self, events: List[Dict], user_id: str) -> List[Dict]:
        """
        Check a list of events for alert patterns.

        Parameters
        ----------
        events : List[Dict]
            Recent events for the user. Each event is expected to have at
            minimum a 'timestamp' (Unix epoch float) and relevant domain
            fields depending on the pattern being checked.
        user_id : str
            Identifier for the user.

        Returns
        -------
        List[Dict]
            Zero or more alert dicts, each containing:
            user_id, alert_type, detection_method, severity, message,
            recommended_action, confidence, timestamp.
        """
        alerts: List[Dict] = []
        now = time.time()

        alerts.extend(self._check_emotional_spiral(events, user_id, now))
        alerts.extend(self._check_rapid_cycling(events, user_id, now))
        alerts.extend(self._check_sustained_stress(events, user_id, now))
        alerts.extend(self._check_emotional_flatline(events, user_id, now))
        alerts.extend(self._check_engagement_drop(events, user_id, now))
        alerts.extend(self._check_recovery_detected(events, user_id, now))

        return alerts

    # ------------------------------------------------------------------
    # Pattern checkers
    # ------------------------------------------------------------------

    def _check_emotional_spiral(
        self, events: List[Dict], user_id: str, now: float
    ) -> List[Dict]:
        """5+ negative emotions (by family) within a 2-hour window."""
        two_hours_ago = now - 7200
        negative_events = [
            e for e in events
            if e.get("timestamp", 0) >= two_hours_ago
            and e.get("emotion_family") in NEGATIVE_FAMILIES
        ]
        if len(negative_events) >= 5:
            return [
                self._make_alert(
                    user_id=user_id,
                    alert_type="emotional_spiral",
                    severity="high",
                    message=(
                        f"Detected {len(negative_events)} negative emotional events "
                        "within the last 2 hours — potential emotional spiral."
                    ),
                    recommended_action=(
                        "Consider reaching out to a mental health professional or "
                        "using a grounding technique."
                    ),
                    confidence=min(1.0, len(negative_events) / 10),
                )
            ]
        return []

    def _check_rapid_cycling(
        self, events: List[Dict], user_id: str, now: float
    ) -> List[Dict]:
        """8+ consecutive emotion family changes within 1 hour."""
        one_hour_ago = now - 3600
        recent = sorted(
            [e for e in events if e.get("timestamp", 0) >= one_hour_ago],
            key=lambda e: e.get("timestamp", 0),
        )
        if len(recent) < 2:
            return []

        changes = 0
        prev_family = recent[0].get("emotion_family")
        for e in recent[1:]:
            curr_family = e.get("emotion_family")
            if curr_family and curr_family != prev_family:
                changes += 1
                prev_family = curr_family

        if changes >= 8:
            return [
                self._make_alert(
                    user_id=user_id,
                    alert_type="rapid_cycling",
                    severity="high",
                    message=(
                        f"Detected {changes} emotion family changes within the last "
                        "hour — rapid emotional cycling."
                    ),
                    recommended_action=(
                        "Suggest mindfulness or emotional stabilisation exercises."
                    ),
                    confidence=min(1.0, changes / 15),
                )
            ]
        return []

    def _check_sustained_stress(
        self, events: List[Dict], user_id: str, now: float
    ) -> List[Dict]:
        """Biometric events with stress_ratio > 0.5 spanning 30+ minutes."""
        stress_events = sorted(
            [
                e for e in events
                if e.get("stress_ratio") is not None
                and e.get("stress_ratio", 0) > 0.5
            ],
            key=lambda e: e.get("timestamp", 0),
        )
        if len(stress_events) < 2:
            return []

        earliest = stress_events[0].get("timestamp", 0)
        latest = stress_events[-1].get("timestamp", 0)
        duration_minutes = (latest - earliest) / 60

        if duration_minutes >= 30:
            avg_stress = sum(e.get("stress_ratio", 0) for e in stress_events) / len(
                stress_events
            )
            return [
                self._make_alert(
                    user_id=user_id,
                    alert_type="sustained_stress",
                    severity="high",
                    message=(
                        f"Sustained high stress detected over {duration_minutes:.0f} "
                        f"minutes (avg stress ratio: {avg_stress:.2f})."
                    ),
                    recommended_action=(
                        "Recommend a break, breathing exercise, or relaxation activity."
                    ),
                    confidence=min(1.0, duration_minutes / 120),
                )
            ]
        return []

    def _check_emotional_flatline(
        self, events: List[Dict], user_id: str, now: float
    ) -> List[Dict]:
        """Same single emotion for 24+ hours."""
        twenty_four_hours_ago = now - 86400
        recent = [
            e for e in events
            if e.get("timestamp", 0) >= twenty_four_hours_ago
            and e.get("emotion") is not None
        ]
        if len(recent) < 2:
            return []

        emotions = {e.get("emotion") for e in recent}
        if len(emotions) == 1:
            the_emotion = next(iter(emotions))
            return [
                self._make_alert(
                    user_id=user_id,
                    alert_type="emotional_flatline",
                    severity="medium",
                    message=(
                        f"User has reported only '{the_emotion}' for the last 24 hours "
                        "— possible emotional flatline."
                    ),
                    recommended_action=(
                        "Encourage varied activities and social engagement to promote "
                        "emotional range."
                    ),
                    confidence=0.75,
                )
            ]
        return []

    def _check_engagement_drop(
        self, events: List[Dict], user_id: str, now: float
    ) -> List[Dict]:
        """No events for 3+ days."""
        if not events:
            return [
                self._make_alert(
                    user_id=user_id,
                    alert_type="engagement_drop",
                    severity="low",
                    message="No events recorded — user has been inactive.",
                    recommended_action=(
                        "Send a gentle re-engagement notification."
                    ),
                    confidence=0.6,
                )
            ]
        three_days_ago = now - 3 * 86400
        latest_ts = max(e.get("timestamp", 0) for e in events)
        if latest_ts < three_days_ago:
            gap_days = (now - latest_ts) / 86400
            return [
                self._make_alert(
                    user_id=user_id,
                    alert_type="engagement_drop",
                    severity="low",
                    message=(
                        f"No events in the last {gap_days:.1f} days — potential "
                        "engagement drop."
                    ),
                    recommended_action=(
                        "Send a gentle re-engagement notification."
                    ),
                    confidence=min(1.0, gap_days / 14),
                )
            ]
        return []

    def _check_recovery_detected(
        self, events: List[Dict], user_id: str, now: float
    ) -> List[Dict]:
        """Transition from 2+ negative to 2+ positive emotions."""
        emotion_events = sorted(
            [e for e in events if e.get("emotion_family") is not None],
            key=lambda e: e.get("timestamp", 0),
        )
        if len(emotion_events) < 4:
            return []

        midpoint = len(emotion_events) // 2
        first_half = emotion_events[:midpoint]
        second_half = emotion_events[midpoint:]

        negative_first = sum(
            1 for e in first_half if e.get("emotion_family") in NEGATIVE_FAMILIES
        )
        positive_second = sum(
            1 for e in second_half if e.get("emotion_family") in POSITIVE_FAMILIES
        )

        if negative_first >= 2 and positive_second >= 2:
            return [
                self._make_alert(
                    user_id=user_id,
                    alert_type="recovery_detected",
                    severity="positive",
                    message=(
                        f"Recovery pattern detected: transitioned from {negative_first} "
                        f"negative to {positive_second} positive emotional states."
                    ),
                    recommended_action=(
                        "Reinforce positive trajectory with encouragement and reflection."
                    ),
                    confidence=min(
                        1.0,
                        (negative_first + positive_second) / (len(emotion_events) * 0.8),
                    ),
                )
            ]
        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_alert(
        self,
        user_id: str,
        alert_type: str,
        severity: str,
        message: str,
        recommended_action: str,
        confidence: float,
    ) -> Dict:
        return {
            "user_id": user_id,
            "alert_type": alert_type,
            "detection_method": "reactive",
            "severity": severity,
            "message": message,
            "recommended_action": recommended_action,
            "confidence": confidence,
            "timestamp": time.time(),
        }


# ---------------------------------------------------------------------------
# PredictiveDetector
# ---------------------------------------------------------------------------


class PredictiveDetector:
    """
    Detects recurring distress fingerprints via cosine similarity.

    Stores up to 3 historical signature vectors per user and fires when
    a new fingerprint closely resembles a stored one (cosine similarity > 0.8).
    Requires at least 3 stored signatures before making predictions.
    """

    def __init__(self) -> None:
        # user_id -> list of 6-dim vectors (sliding window of max 3)
        self._signatures: Dict[str, List[List[float]]] = defaultdict(list)

    def store_signature(self, user_id: str, fingerprint: Dict) -> None:
        """
        Extract a 6-dim vector from a fingerprint dict and append it.

        Maintains a sliding window of at most 3 vectors per user.

        Parameters
        ----------
        user_id : str
        fingerprint : Dict
            Expected keys (nested): emotional.volatility, emotional.dominant_family,
            biometric.stress_ratio, biometric.resting_hr,
            behavioral.completion_rate, behavioral.peak_hours_variance.
        """
        vector = self._extract_vector(fingerprint)
        sigs = self._signatures[user_id]
        sigs.append(vector)
        if len(sigs) > 3:
            self._signatures[user_id] = sigs[-3:]

    def check(self, user_id: str, fingerprint: Dict) -> List[Dict]:
        """
        Compare fingerprint against stored signatures.

        Returns an alert list if cosine similarity with any stored signature
        exceeds 0.8. Returns empty list if fewer than 3 signatures are stored.

        Parameters
        ----------
        user_id : str
        fingerprint : Dict

        Returns
        -------
        List[Dict]
            Alert dicts with detection_method='predictive'.
        """
        sigs = self._signatures.get(user_id, [])
        if len(sigs) < 3:
            return []

        current_vector = self._extract_vector(fingerprint)
        max_similarity = max(
            self._cosine_similarity(current_vector, stored) for stored in sigs
        )

        if max_similarity > 0.8:
            return [
                {
                    "user_id": user_id,
                    "alert_type": "predictive_distress",
                    "detection_method": "predictive",
                    "severity": "medium",
                    "message": (
                        f"Current emotional fingerprint closely resembles a historical "
                        f"distress pattern (similarity: {max_similarity:.2f})."
                    ),
                    "recommended_action": (
                        "Review recent context and consider proactive intervention."
                    ),
                    "confidence": max_similarity,
                    "timestamp": time.time(),
                }
            ]
        return []

    def _extract_vector(self, fingerprint: Dict) -> List[float]:
        """
        Extract a 6-dimensional feature vector from a fingerprint dict.

        Dimensions:
          0 — emotional.volatility
          1 — 1.0 if dominant_family is negative, else 0.0
          2 — biometric.stress_ratio
          3 — normalised resting_hr (40–120 bpm → 0–1)
          4 — behavioral.completion_rate
          5 — behavioral.peak_hours_variance
        """
        emotional = fingerprint.get("emotional", {})
        biometric = fingerprint.get("biometric", {})
        behavioral = fingerprint.get("behavioral", {})

        volatility = float(emotional.get("volatility", 0.0))
        dominant_family = emotional.get("dominant_family", "")
        neg_flag = 1.0 if dominant_family in NEGATIVE_FAMILIES else 0.0
        stress_ratio = float(biometric.get("stress_ratio", 0.0))

        resting_hr = float(biometric.get("resting_hr", 60.0))
        # Normalise 40–120 bpm to 0–1, clamped
        hr_normalised = max(0.0, min(1.0, (resting_hr - 40.0) / 80.0))

        completion_rate = float(behavioral.get("completion_rate", 0.0))
        peak_hours_variance = float(behavioral.get("peak_hours_variance", 0.0))

        return [
            volatility,
            neg_flag,
            stress_ratio,
            hr_normalised,
            completion_rate,
            peak_hours_variance,
        ]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Returns 0.0 if either vector has zero magnitude.
        """
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# AlertThrottler
# ---------------------------------------------------------------------------


class AlertThrottler:
    """
    Prevents alert spam by enforcing cooldowns and stage-based suppression.

    Cooldown key: ``{user_id}:{alert_type}:{method}`` — reactive and
    predictive subscriptions carry independent cooldowns.

    Stage rules:
    - Stage 1: only 'positive' severity alerts pass.
    - Low severity is suppressed while a high severity alert is active for the
      same (user_id, alert_type).

    Parameters
    ----------
    cooldown_seconds : int
        How long (in seconds) to suppress a repeated alert. Default 4 hours.
    """

    def __init__(self, cooldown_seconds: int = 14400) -> None:
        self._cooldown_seconds = cooldown_seconds
        # cooldown_key -> last fired timestamp
        self._last_fired: Dict[str, float] = {}
        # (user_id, alert_type) -> (severity, timestamp) of most recent high alert
        self._active_high: Dict[tuple, float] = {}

    def should_fire(
        self,
        user_id: str,
        alert_type: str,
        method: str,
        severity: str,
        user_stage: int = 2,
    ) -> bool:
        """
        Decide whether an alert should be delivered.

        Parameters
        ----------
        user_id : str
        alert_type : str
        method : str
            'reactive' or 'predictive'
        severity : str
            'high', 'medium', 'low', or 'positive'
        user_stage : int
            Current evolution stage (1–5). Stage 1 only receives positive alerts.

        Returns
        -------
        bool
        """
        now = time.time()

        # Stage 1: only positive alerts
        if user_stage == 1 and severity != "positive":
            return False

        # Low suppressed while high is active for same (user_id, alert_type)
        if severity == "low":
            high_key = (user_id, alert_type)
            high_ts = self._active_high.get(high_key)
            if high_ts is not None and (now - high_ts) < self._cooldown_seconds:
                return False

        # Cooldown check (method-scoped key)
        cooldown_key = f"{user_id}:{alert_type}:{method}"
        last_ts = self._last_fired.get(cooldown_key)
        if last_ts is not None and (now - last_ts) < self._cooldown_seconds:
            return False

        # All checks passed — record the firing
        self._last_fired[cooldown_key] = now
        if severity == "high":
            self._active_high[(user_id, alert_type)] = now

        return True


# ---------------------------------------------------------------------------
# AlertEngine
# ---------------------------------------------------------------------------


class AlertEngine:
    """
    Orchestrates reactive and predictive alert detection via the event bus.

    Subscribes to ``event.*`` (async) to buffer recent events per user and run
    ReactiveDetector. Also subscribes to ``alert.reactive`` (sync) to feed
    PredictiveDetector signatures.

    Parameters
    ----------
    bus : ProfileEventBus
    db : optional
        ProfileDB instance (reserved for future persistence).
    cooldown_seconds : int
        Alert throttler cooldown in seconds (default 4 hours).
    """

    MAX_BUFFER = 500

    def __init__(
        self,
        bus: ProfileEventBus,
        db=None,
        cooldown_seconds: int = 14400,
    ) -> None:
        self._bus = bus
        self._db = db
        self._reactive = ReactiveDetector()
        self._predictive = PredictiveDetector()
        self._throttler = AlertThrottler(cooldown_seconds=cooldown_seconds)

        # user_id -> deque-like list of recent events (max MAX_BUFFER)
        self._buffers: Dict[str, List[Dict]] = defaultdict(list)

        # Subscribe to all events asynchronously
        self._bus.subscribe("event.*", self._on_event, mode="async")

        # Subscribe to reactive alerts synchronously to update signatures
        self._bus.subscribe("alert.reactive", self._on_reactive_alert, mode="sync")

        logger.info("AlertEngine initialised (cooldown=%ds)", cooldown_seconds)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_event(self, topic: str, payload: Dict) -> None:
        """Buffer the incoming event and run reactive detection."""
        user_id = payload.get("user_id")
        if not user_id:
            return

        buf = self._buffers[user_id]
        buf.append(payload)
        if len(buf) > self.MAX_BUFFER:
            self._buffers[user_id] = buf[-self.MAX_BUFFER:]

        alerts = self._reactive.check(self._buffers[user_id], user_id)
        for alert in alerts:
            user_stage = payload.get("user_stage", 2)
            if self._throttler.should_fire(
                user_id=user_id,
                alert_type=alert["alert_type"],
                method="reactive",
                severity=alert["severity"],
                user_stage=user_stage,
            ):
                self._bus.publish("alert.reactive", alert)

    def _on_reactive_alert(self, topic: str, payload: Dict) -> None:
        """Store fingerprint from reactive alert payloads for predictive model."""
        user_id = payload.get("user_id")
        fingerprint = payload.get("fingerprint")
        if user_id and fingerprint:
            self._predictive.store_signature(user_id, fingerprint)

    # ------------------------------------------------------------------
    # Predictive check (called externally with fresh fingerprint)
    # ------------------------------------------------------------------

    def check_predictive(self, user_id: str, fingerprint: Dict) -> None:
        """
        Run predictive detection and publish matching alerts to 'alert.predictive'.

        Only alerts with confidence > 0.7 are considered. Throttling is applied
        independently from reactive alerts.

        Parameters
        ----------
        user_id : str
        fingerprint : Dict
            Profile fingerprint dict consumed by PredictiveDetector._extract_vector.
        """
        self._predictive.store_signature(user_id, fingerprint)
        alerts = self._predictive.check(user_id, fingerprint)
        for alert in alerts:
            if alert["confidence"] <= 0.7:
                continue
            if self._throttler.should_fire(
                user_id=user_id,
                alert_type=alert["alert_type"],
                method="predictive",
                severity=alert["severity"],
            ):
                self._bus.publish("alert.predictive", alert)

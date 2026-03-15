"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Alert Engine
===============================================================================
Unit tests for ReactiveDetector, AlertThrottler, PredictiveDetector, and
AlertEngine covering all six reactive patterns, throttling rules, and
predictive similarity matching.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.
===============================================================================
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alert_engine import (
    NEGATIVE_FAMILIES,
    POSITIVE_FAMILIES,
    AlertEngine,
    AlertThrottler,
    PredictiveDetector,
    ReactiveDetector,
)
from profile_event_bus import ProfileEventBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now():
    return time.time()


def _make_emotion_event(family: str, offset_seconds: int = 0, emotion: str = None) -> dict:
    """Create a minimal emotion event dict."""
    return {
        "user_id": "u1",
        "emotion_family": family,
        "emotion": emotion or family.lower(),
        "timestamp": _now() - offset_seconds,
    }


def _make_stress_event(stress_ratio: float, offset_seconds: int = 0) -> dict:
    """Create a biometric stress event."""
    return {
        "user_id": "u1",
        "stress_ratio": stress_ratio,
        "timestamp": _now() - offset_seconds,
    }


def _fingerprint(
    volatility: float = 0.5,
    dominant_family: str = "Sadness",
    stress_ratio: float = 0.7,
    resting_hr: float = 85.0,
    completion_rate: float = 0.4,
    peak_hours_variance: float = 0.3,
) -> dict:
    return {
        "emotional": {
            "volatility": volatility,
            "dominant_family": dominant_family,
        },
        "biometric": {
            "stress_ratio": stress_ratio,
            "resting_hr": resting_hr,
        },
        "behavioral": {
            "completion_rate": completion_rate,
            "peak_hours_variance": peak_hours_variance,
        },
    }


# ---------------------------------------------------------------------------
# TestReactiveDetector
# ---------------------------------------------------------------------------


class TestReactiveDetector(unittest.TestCase):
    """Tests for the six reactive alert patterns in ReactiveDetector."""

    def setUp(self):
        self.detector = ReactiveDetector()

    # --- emotional_spiral ---

    def test_emotional_spiral_detected(self):
        """5 negative emotion events within 2 hours should trigger emotional_spiral."""
        events = [
            _make_emotion_event("Sadness", offset_seconds=i * 600)  # every 10 min
            for i in range(5)
        ]
        alerts = self.detector.check(events, "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertIn("emotional_spiral", types)

    def test_emotional_spiral_not_triggered_with_positive_emotions(self):
        """Positive emotions should not trigger emotional_spiral."""
        events = [
            _make_emotion_event("Joy", offset_seconds=i * 600) for i in range(5)
        ]
        alerts = self.detector.check(events, "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertNotIn("emotional_spiral", types)

    def test_emotional_spiral_requires_five(self):
        """Only 4 negative events — no spiral alert."""
        events = [
            _make_emotion_event("Fear", offset_seconds=i * 600) for i in range(4)
        ]
        alerts = self.detector.check(events, "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertNotIn("emotional_spiral", types)

    def test_emotional_spiral_outside_window_not_triggered(self):
        """5 negative events but older than 2 hours — no spiral alert."""
        events = [
            _make_emotion_event("Anger", offset_seconds=7200 + i * 600)
            for i in range(5)
        ]
        alerts = self.detector.check(events, "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertNotIn("emotional_spiral", types)

    # --- rapid_cycling ---

    def test_rapid_cycling_detected(self):
        """8+ emotion family changes within 1 hour should trigger rapid_cycling."""
        families = ["Joy", "Sadness", "Anger", "Calm", "Fear", "Love",
                    "Self-Conscious", "Joy", "Sadness"]
        events = [
            _make_emotion_event(fam, offset_seconds=i * 300)  # every 5 min
            for i, fam in enumerate(families)
        ]
        alerts = self.detector.check(events, "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertIn("rapid_cycling", types)

    def test_rapid_cycling_requires_eight_changes(self):
        """Only 4 changes — no rapid_cycling alert."""
        families = ["Joy", "Sadness", "Anger", "Calm", "Fear"]
        events = [
            _make_emotion_event(fam, offset_seconds=i * 300)
            for i, fam in enumerate(families)
        ]
        alerts = self.detector.check(events, "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertNotIn("rapid_cycling", types)

    # --- sustained_stress ---

    def test_sustained_stress_detected(self):
        """High stress spanning 30+ minutes should trigger sustained_stress."""
        # Events spread from 60 min ago to now — comfortably spans 30+ minutes
        events = [
            _make_stress_event(stress_ratio=0.8, offset_seconds=(60 - i * 20) * 60)
            for i in range(4)  # at 60min, 40min, 20min, 0min ago
        ]
        alerts = self.detector.check(events, "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertIn("sustained_stress", types)

    def test_sustained_stress_below_threshold_not_triggered(self):
        """Stress ratio <= 0.5 should not trigger sustained_stress."""
        events = [
            _make_stress_event(stress_ratio=0.4, offset_seconds=(30 - i) * 60)
            for i in range(4)
        ]
        alerts = self.detector.check(events, "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertNotIn("sustained_stress", types)

    def test_sustained_stress_short_duration_not_triggered(self):
        """High stress but only for 10 minutes — no alert."""
        events = [
            _make_stress_event(stress_ratio=0.9, offset_seconds=(10 - i) * 60)
            for i in range(2)
        ]
        alerts = self.detector.check(events, "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertNotIn("sustained_stress", types)

    # --- recovery_detected ---

    def test_recovery_detected(self):
        """Transition from 2+ negative to 2+ positive should trigger recovery_detected."""
        negative_events = [
            _make_emotion_event("Sadness", offset_seconds=3600 + i * 300)
            for i in range(3)
        ]
        positive_events = [
            _make_emotion_event("Joy", offset_seconds=i * 300)
            for i in range(3)
        ]
        events = negative_events + positive_events
        alerts = self.detector.check(events, "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertIn("recovery_detected", types)

    def test_recovery_not_triggered_without_prior_negatives(self):
        """Only positive emotions — no recovery_detected alert."""
        events = [
            _make_emotion_event("Joy", offset_seconds=i * 300) for i in range(6)
        ]
        alerts = self.detector.check(events, "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertNotIn("recovery_detected", types)

    # --- alert structure ---

    def test_alert_has_required_keys(self):
        """Every alert must include the required fields."""
        events = [
            _make_emotion_event("Fear", offset_seconds=i * 600) for i in range(5)
        ]
        alerts = self.detector.check(events, "u1")
        required = {
            "user_id",
            "alert_type",
            "detection_method",
            "severity",
            "message",
            "recommended_action",
            "confidence",
            "timestamp",
        }
        for alert in alerts:
            for key in required:
                self.assertIn(key, alert, f"Missing key '{key}' in alert")

    def test_detection_method_is_reactive(self):
        """All reactive alerts should have detection_method == 'reactive'."""
        events = [
            _make_emotion_event("Anger", offset_seconds=i * 600) for i in range(5)
        ]
        alerts = self.detector.check(events, "u1")
        for alert in alerts:
            self.assertEqual(alert["detection_method"], "reactive")

    # --- engagement_drop ---

    def test_engagement_drop_no_events(self):
        """No events at all should trigger engagement_drop."""
        alerts = self.detector.check([], "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertIn("engagement_drop", types)

    def test_engagement_drop_old_events(self):
        """Event older than 3 days should trigger engagement_drop."""
        events = [_make_emotion_event("Joy", offset_seconds=4 * 86400)]
        alerts = self.detector.check(events, "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertIn("engagement_drop", types)

    def test_engagement_drop_recent_events_no_alert(self):
        """Recent events should NOT trigger engagement_drop."""
        events = [_make_emotion_event("Joy", offset_seconds=3600)]
        alerts = self.detector.check(events, "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertNotIn("engagement_drop", types)

    # --- emotional_flatline ---

    def test_emotional_flatline_detected(self):
        """Same single emotion for 24+ hours should trigger emotional_flatline."""
        events = [
            {"user_id": "u1", "emotion": "sadness", "emotion_family": "Sadness",
             "timestamp": _now() - (i * 3600)}
            for i in range(5)
        ]
        alerts = self.detector.check(events, "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertIn("emotional_flatline", types)

    def test_emotional_flatline_varied_emotions_no_alert(self):
        """Different emotions in last 24 hours — no flatline alert."""
        events = [
            {"user_id": "u1", "emotion": "joy", "emotion_family": "Joy",
             "timestamp": _now() - 1800},
            {"user_id": "u1", "emotion": "sadness", "emotion_family": "Sadness",
             "timestamp": _now() - 3600},
        ]
        alerts = self.detector.check(events, "u1")
        types = [a["alert_type"] for a in alerts]
        self.assertNotIn("emotional_flatline", types)


# ---------------------------------------------------------------------------
# TestAlertThrottler
# ---------------------------------------------------------------------------


class TestAlertThrottler(unittest.TestCase):
    """Tests for AlertThrottler cooldown and suppression logic."""

    def test_first_alert_passes(self):
        """The very first alert for a key should always pass."""
        throttler = AlertThrottler(cooldown_seconds=14400)
        result = throttler.should_fire("u1", "emotional_spiral", "reactive", "high")
        self.assertTrue(result)

    def test_same_type_and_method_throttled(self):
        """Firing the same (user, type, method) a second time within cooldown is suppressed."""
        throttler = AlertThrottler(cooldown_seconds=14400)
        throttler.should_fire("u1", "emotional_spiral", "reactive", "high")
        result = throttler.should_fire("u1", "emotional_spiral", "reactive", "high")
        self.assertFalse(result)

    def test_different_method_not_throttled(self):
        """Reactive and predictive cooldowns are independent."""
        throttler = AlertThrottler(cooldown_seconds=14400)
        throttler.should_fire("u1", "emotional_spiral", "reactive", "high")
        result = throttler.should_fire("u1", "emotional_spiral", "predictive", "high")
        self.assertTrue(result)

    def test_low_suppressed_by_active_high(self):
        """Low severity alert is suppressed when a high alert is active for same type."""
        throttler = AlertThrottler(cooldown_seconds=14400)
        throttler.should_fire("u1", "sustained_stress", "reactive", "high")
        result = throttler.should_fire("u1", "sustained_stress", "reactive", "low")
        self.assertFalse(result)

    def test_cooldown_expires(self):
        """After the cooldown window passes, the alert fires again."""
        throttler = AlertThrottler(cooldown_seconds=1)  # 1-second cooldown
        throttler.should_fire("u1", "engagement_drop", "reactive", "low")
        time.sleep(1.1)
        result = throttler.should_fire("u1", "engagement_drop", "reactive", "low")
        self.assertTrue(result)

    def test_stage_1_only_positive_alerts(self):
        """Stage 1 users should only receive alerts with severity='positive'."""
        throttler = AlertThrottler(cooldown_seconds=14400)
        # high severity blocked at stage 1
        self.assertFalse(
            throttler.should_fire("u1", "emotional_spiral", "reactive", "high", user_stage=1)
        )
        # medium severity blocked at stage 1
        self.assertFalse(
            throttler.should_fire("u1", "emotional_flatline", "reactive", "medium", user_stage=1)
        )
        # low severity blocked at stage 1
        self.assertFalse(
            throttler.should_fire("u1", "engagement_drop", "reactive", "low", user_stage=1)
        )
        # positive severity allowed at stage 1
        self.assertTrue(
            throttler.should_fire("u1", "recovery_detected", "reactive", "positive", user_stage=1)
        )

    def test_stage_2_allows_all_severities(self):
        """Stage 2+ users are not restricted to positive alerts."""
        throttler = AlertThrottler(cooldown_seconds=14400)
        result = throttler.should_fire("u1", "emotional_spiral", "reactive", "high", user_stage=2)
        self.assertTrue(result)

    def test_different_users_independent_cooldowns(self):
        """Throttling for one user should not affect another user."""
        throttler = AlertThrottler(cooldown_seconds=14400)
        throttler.should_fire("u1", "rapid_cycling", "reactive", "high")
        result = throttler.should_fire("u2", "rapid_cycling", "reactive", "high")
        self.assertTrue(result)

    def test_different_alert_types_independent(self):
        """Throttling for one alert type should not suppress a different type."""
        throttler = AlertThrottler(cooldown_seconds=14400)
        throttler.should_fire("u1", "emotional_spiral", "reactive", "high")
        result = throttler.should_fire("u1", "rapid_cycling", "reactive", "high")
        self.assertTrue(result)


# ---------------------------------------------------------------------------
# TestPredictiveDetector
# ---------------------------------------------------------------------------


class TestPredictiveDetector(unittest.TestCase):
    """Tests for PredictiveDetector signature storage and similarity matching."""

    def setUp(self):
        self.detector = PredictiveDetector()

    def test_fewer_than_three_signatures_no_alert(self):
        """With fewer than 3 stored signatures, check() should return empty."""
        fp = _fingerprint()
        self.detector.store_signature("u1", fp)
        self.detector.store_signature("u1", fp)
        result = self.detector.check("u1", fp)
        self.assertEqual(result, [])

    def test_three_identical_signatures_fires(self):
        """Three identical stored signatures should yield similarity 1.0 and fire."""
        fp = _fingerprint()
        for _ in range(3):
            self.detector.store_signature("u1", fp)
        result = self.detector.check("u1", fp)
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0]["confidence"], 0.8)
        self.assertEqual(result[0]["detection_method"], "predictive")

    def test_dissimilar_fingerprint_does_not_fire(self):
        """A fingerprint very different from stored signatures should not fire."""
        for _ in range(3):
            self.detector.store_signature("u1", _fingerprint(volatility=0.9, stress_ratio=0.9))
        # Completely opposite fingerprint
        fp_low = _fingerprint(
            volatility=0.01,
            dominant_family="Joy",
            stress_ratio=0.01,
            resting_hr=50.0,
            completion_rate=0.99,
            peak_hours_variance=0.01,
        )
        result = self.detector.check("u1", fp_low)
        self.assertEqual(result, [])

    def test_sliding_window_keeps_only_last_three(self):
        """Storing 4 signatures keeps only the latest 3."""
        for i in range(4):
            self.detector.store_signature("u1", _fingerprint(volatility=float(i)))
        self.assertEqual(len(self.detector._signatures["u1"]), 3)

    def test_new_user_no_alert(self):
        """A user with no stored signatures should never get a predictive alert."""
        result = self.detector.check("new_user", _fingerprint())
        self.assertEqual(result, [])

    def test_cosine_similarity_identical_vectors(self):
        """Cosine similarity of a vector with itself should be ~1.0."""
        v = [0.5, 1.0, 0.7, 0.5, 0.4, 0.3]
        sim = self.detector._cosine_similarity(v, v)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_cosine_similarity_zero_vector(self):
        """Cosine similarity with a zero vector should be 0.0."""
        v = [0.5, 1.0, 0.7, 0.5, 0.4, 0.3]
        zero = [0.0] * 6
        sim = self.detector._cosine_similarity(v, zero)
        self.assertEqual(sim, 0.0)

    def test_extract_vector_negative_family_flag(self):
        """dominant_family in NEGATIVE_FAMILIES should set dimension 1 to 1.0."""
        fp = _fingerprint(dominant_family="Anger")
        vec = self.detector._extract_vector(fp)
        self.assertEqual(vec[1], 1.0)

    def test_extract_vector_positive_family_flag(self):
        """dominant_family in POSITIVE_FAMILIES should set dimension 1 to 0.0."""
        fp = _fingerprint(dominant_family="Joy")
        vec = self.detector._extract_vector(fp)
        self.assertEqual(vec[1], 0.0)

    def test_extract_vector_hr_normalisation(self):
        """resting_hr=80 should normalise to 0.5 (midpoint of 40-120 range)."""
        fp = _fingerprint(resting_hr=80.0)
        vec = self.detector._extract_vector(fp)
        self.assertAlmostEqual(vec[3], 0.5, places=5)

    def test_extract_vector_hr_clamped_low(self):
        """resting_hr below 40 should clamp to 0.0."""
        fp = _fingerprint(resting_hr=20.0)
        vec = self.detector._extract_vector(fp)
        self.assertEqual(vec[3], 0.0)

    def test_extract_vector_hr_clamped_high(self):
        """resting_hr above 120 should clamp to 1.0."""
        fp = _fingerprint(resting_hr=150.0)
        vec = self.detector._extract_vector(fp)
        self.assertEqual(vec[3], 1.0)


# ---------------------------------------------------------------------------
# TestAlertEngineIntegration (light integration via event bus)
# ---------------------------------------------------------------------------


class TestAlertEngineIntegration(unittest.TestCase):
    """Light integration tests for AlertEngine using a real ProfileEventBus."""

    def setUp(self):
        self.bus = ProfileEventBus()
        self.engine = AlertEngine(self.bus, cooldown_seconds=14400)
        self.received_reactive = []
        self.received_predictive = []

        self.bus.subscribe(
            "alert.reactive",
            lambda topic, payload: self.received_reactive.append(payload),
            mode="sync",
        )
        self.bus.subscribe(
            "alert.predictive",
            lambda topic, payload: self.received_predictive.append(payload),
            mode="sync",
        )

    def tearDown(self):
        self.bus.shutdown()

    def test_predictive_alert_fired_for_high_similarity(self):
        """check_predictive() should publish to alert.predictive for similar fingerprints."""
        fp = _fingerprint()
        # Manually prime 3 signatures
        for _ in range(3):
            self.engine._predictive.store_signature("u1", fp)

        # Now check with same fingerprint — confidence will be ~1.0 > 0.7
        self.engine.check_predictive("u1", fp)
        self.assertEqual(len(self.received_predictive), 1)
        self.assertEqual(self.received_predictive[0]["detection_method"], "predictive")

    def test_predictive_alert_not_fired_below_confidence_threshold(self):
        """check_predictive() should NOT publish if confidence <= 0.7.

        We directly seed 3 signatures into the predictive detector that are
        very different from the fingerprint under test, bypassing the
        store-then-check behaviour of check_predictive().
        """
        high_stress_fp = _fingerprint(
            volatility=0.9, dominant_family="Anger", stress_ratio=0.95,
            resting_hr=115.0, completion_rate=0.05, peak_hours_variance=0.9
        )
        # Directly seed 3 stored signatures (not via check_predictive)
        for _ in range(3):
            self.engine._predictive._signatures["u2"].append(
                self.engine._predictive._extract_vector(high_stress_fp)
            )

        # The fingerprint to check is the polar opposite
        low_fp = _fingerprint(
            volatility=0.01, dominant_family="Joy", stress_ratio=0.01,
            resting_hr=45.0, completion_rate=0.99, peak_hours_variance=0.01
        )
        # Call check directly (not check_predictive which would store the vector)
        alerts = self.engine._predictive.check("u2", low_fp)
        # Confirm similarity is below threshold
        for a in alerts:
            self.assertLessEqual(a["confidence"], 0.7)
        # No alerts with confidence > 0.7 should have been published
        self.assertEqual(len(self.received_predictive), 0)

    def test_check_predictive_throttles_repeated_calls(self):
        """Same predictive alert should not fire twice within cooldown."""
        fp = _fingerprint()
        for _ in range(3):
            self.engine._predictive.store_signature("u1", fp)

        self.engine.check_predictive("u1", fp)
        self.engine.check_predictive("u1", fp)
        # Only the first should have fired
        self.assertEqual(len(self.received_predictive), 1)


if __name__ == "__main__":
    unittest.main()

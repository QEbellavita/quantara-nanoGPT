"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Intelligence Publisher
===============================================================================
Unit tests for IntelligencePublisher, TherapyPersonalizer, CoachingPersonalizer,
and WorkflowPrioritizer. Covers computation of personalization signals and
pub/sub delivery to the ProfileEventBus.

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
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from profile_event_bus import ProfileEventBus
from intelligence_publisher import (
    TherapyPersonalizer,
    CoachingPersonalizer,
    WorkflowPrioritizer,
    IntelligencePublisher,
    _get_mode,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_fingerprint(
    *,
    technique_usage=None,
    retention_rate=0.5,
    completion_rate=0.8,
    intervention_response_rate=0.6,
    engagement_streak=5,
    avg_sentence_length=12.0,
    vocabulary_complexity=0.55,
    core_values=None,
    volatility=0.2,
    dominant_family="positive",
) -> dict:
    """Build a minimal fingerprint dict suitable for personalizer tests."""
    if technique_usage is None:
        technique_usage = {"mindfulness": 10, "cbt": 5, "breathing": 20}
    if core_values is None:
        core_values = ["growth", "resilience", "compassion"]

    return {
        "behavioral": {
            "metrics": {
                "completion_rate": completion_rate,
                "intervention_response_rate": intervention_response_rate,
                "engagement_streak": engagement_streak,
            }
        },
        "cognitive": {
            "metrics": {
                "technique_usage": technique_usage,
                "technique_retention_rate": retention_rate,
            }
        },
        "linguistic": {
            "metrics": {
                "avg_sentence_length": avg_sentence_length,
                "vocabulary_complexity": vocabulary_complexity,
            }
        },
        "aspirational": {
            "metrics": {
                "core_values": core_values,
            }
        },
        "emotional": {
            "metrics": {
                "volatility": volatility,
                "dominant_family": dominant_family,
            }
        },
        "social": {
            "metrics": {}
        },
    }


# ---------------------------------------------------------------------------
# _get_mode helper
# ---------------------------------------------------------------------------

class TestGetMode(unittest.TestCase):
    def test_stage_3_is_active(self):
        self.assertEqual(_get_mode(3), "active")

    def test_stage_4_is_active(self):
        self.assertEqual(_get_mode(4), "active")

    def test_stage_5_is_active(self):
        self.assertEqual(_get_mode(5), "active")

    def test_stage_2_is_advisory(self):
        self.assertEqual(_get_mode(2), "advisory")

    def test_stage_1_is_advisory(self):
        self.assertEqual(_get_mode(1), "advisory")


# ---------------------------------------------------------------------------
# TherapyPersonalizer
# ---------------------------------------------------------------------------

class TestTherapyPersonalizer(unittest.TestCase):
    def setUp(self):
        self.personalizer = TherapyPersonalizer()

    def test_technique_scores_present(self):
        """technique_scores should contain entries for every technique in cognitive DNA."""
        fp = _make_fingerprint(
            technique_usage={"mindfulness": 10, "cbt": 5},
            retention_rate=0.8,
        )
        result = self.personalizer.compute(fp, stage=3)
        self.assertIn("technique_scores", result)
        self.assertIn("mindfulness", result["technique_scores"])
        self.assertIn("cbt", result["technique_scores"])

    def test_technique_score_formula(self):
        """technique_score = min(1, count/20) * retention_rate."""
        fp = _make_fingerprint(
            technique_usage={"mindfulness": 10},
            retention_rate=0.6,
        )
        result = self.personalizer.compute(fp, stage=1)
        expected = min(1.0, 10 / 20) * 0.6  # = 0.5 * 0.6 = 0.3
        self.assertAlmostEqual(result["technique_scores"]["mindfulness"], expected, places=5)

    def test_technique_score_capped_at_one(self):
        """Count >= 20 should not push score above retention_rate."""
        fp = _make_fingerprint(
            technique_usage={"breathing": 40},
            retention_rate=1.0,
        )
        result = self.personalizer.compute(fp, stage=3)
        self.assertLessEqual(result["technique_scores"]["breathing"], 1.0)

    def test_mode_active_for_stage_3(self):
        """Stage 3 should produce mode='active'."""
        fp = _make_fingerprint()
        result = self.personalizer.compute(fp, stage=3)
        self.assertEqual(result["mode"], "active")

    def test_mode_advisory_for_stage_1(self):
        """Stage 1 should produce mode='advisory'."""
        fp = _make_fingerprint()
        result = self.personalizer.compute(fp, stage=1)
        self.assertEqual(result["mode"], "advisory")

    def test_preferred_step_type_cognitive_high_retention(self):
        """retention_rate > 0.7 → preferred_step_type = 'cognitive'."""
        fp = _make_fingerprint(retention_rate=0.9)
        result = self.personalizer.compute(fp, stage=1)
        self.assertEqual(result["preferred_step_type"], "cognitive")

    def test_preferred_step_type_calming_low_retention(self):
        """retention_rate <= 0.7 → preferred_step_type = 'calming'."""
        fp = _make_fingerprint(retention_rate=0.5)
        result = self.personalizer.compute(fp, stage=1)
        self.assertEqual(result["preferred_step_type"], "calming")

    def test_weight_change_bound(self):
        """weight_change_bound should always be 0.3."""
        fp = _make_fingerprint()
        result = self.personalizer.compute(fp, stage=2)
        self.assertEqual(result["weight_change_bound"], 0.3)

    def test_completion_rate_and_intervention_response_rate_present(self):
        fp = _make_fingerprint(completion_rate=0.75, intervention_response_rate=0.5)
        result = self.personalizer.compute(fp, stage=1)
        self.assertAlmostEqual(result["completion_rate"], 0.75, places=4)
        self.assertAlmostEqual(result["intervention_response_rate"], 0.5, places=4)

    def test_empty_fingerprint_returns_valid_dict(self):
        """Empty fingerprint should not raise — defaults are used."""
        result = self.personalizer.compute({}, stage=1)
        self.assertIn("technique_scores", result)
        self.assertEqual(result["technique_scores"], {})


# ---------------------------------------------------------------------------
# CoachingPersonalizer
# ---------------------------------------------------------------------------

class TestCoachingPersonalizer(unittest.TestCase):
    def setUp(self):
        self.personalizer = CoachingPersonalizer()

    def test_preferred_tone_direct_short_sentences(self):
        """avg_sentence_length < 10 → preferred_tone = 'direct'."""
        fp = _make_fingerprint(avg_sentence_length=7.0)
        result = self.personalizer.compute(fp, stage=1)
        self.assertEqual(result["preferred_tone"], "direct")

    def test_preferred_tone_supportive_long_sentences(self):
        """avg_sentence_length >= 10 → preferred_tone = 'supportive'."""
        fp = _make_fingerprint(avg_sentence_length=14.0)
        result = self.personalizer.compute(fp, stage=2)
        self.assertEqual(result["preferred_tone"], "supportive")

    def test_vocabulary_level_complex(self):
        """vocabulary_complexity > 0.6 → vocabulary_level = 'complex'."""
        fp = _make_fingerprint(vocabulary_complexity=0.75)
        result = self.personalizer.compute(fp, stage=1)
        self.assertEqual(result["vocabulary_level"], "complex")

    def test_vocabulary_level_simple(self):
        """vocabulary_complexity <= 0.6 → vocabulary_level = 'simple'."""
        fp = _make_fingerprint(vocabulary_complexity=0.4)
        result = self.personalizer.compute(fp, stage=1)
        self.assertEqual(result["vocabulary_level"], "simple")

    def test_response_depth_detailed(self):
        """avg_sentence_length > 15 → response_depth = 'detailed'."""
        fp = _make_fingerprint(avg_sentence_length=20.0)
        result = self.personalizer.compute(fp, stage=1)
        self.assertEqual(result["response_depth"], "detailed")

    def test_response_depth_brief(self):
        """avg_sentence_length <= 15 → response_depth = 'brief'."""
        fp = _make_fingerprint(avg_sentence_length=10.0)
        result = self.personalizer.compute(fp, stage=1)
        self.assertEqual(result["response_depth"], "brief")

    def test_active_goals_from_core_values(self):
        """active_goals should reflect aspirational core_values."""
        values = ["growth", "balance", "courage"]
        fp = _make_fingerprint(core_values=values)
        result = self.personalizer.compute(fp, stage=2)
        self.assertEqual(result["active_goals"], values)

    def test_mode_active_stage_3(self):
        fp = _make_fingerprint()
        result = self.personalizer.compute(fp, stage=3)
        self.assertEqual(result["mode"], "active")

    def test_mode_advisory_stage_2(self):
        fp = _make_fingerprint()
        result = self.personalizer.compute(fp, stage=2)
        self.assertEqual(result["mode"], "advisory")

    def test_empty_fingerprint_returns_valid_dict(self):
        result = self.personalizer.compute({}, stage=1)
        self.assertIn("preferred_tone", result)
        self.assertIn("active_goals", result)


# ---------------------------------------------------------------------------
# WorkflowPrioritizer
# ---------------------------------------------------------------------------

class TestWorkflowPrioritizer(unittest.TestCase):
    def setUp(self):
        self.prioritizer = WorkflowPrioritizer()

    def test_readiness_score_in_range(self):
        """emotional_readiness_score must be in [0, 1]."""
        fp = _make_fingerprint()
        result = self.prioritizer.compute(fp, stage=2)
        score = result["emotional_readiness_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_positive_family_valence(self):
        """positive dominant_family → valence = 1.0 (reflected in a higher score)."""
        fp_pos = _make_fingerprint(dominant_family="positive", volatility=0.0)
        fp_neg = _make_fingerprint(dominant_family="negative", volatility=0.0)
        pos_result = self.prioritizer.compute(fp_pos, stage=1)
        neg_result = self.prioritizer.compute(fp_neg, stage=1)
        self.assertGreater(
            pos_result["emotional_readiness_score"],
            neg_result["emotional_readiness_score"],
        )

    def test_negative_family_lower_score(self):
        """negative dominant_family → valence = 0.3."""
        fp = _make_fingerprint(dominant_family="negative", volatility=0.0, completion_rate=0.0, engagement_streak=0)
        result = self.prioritizer.compute(fp, stage=1)
        # stability=1.0*0.4 + valence=0.3*0.3 + momentum=0.0*0.3 = 0.4 + 0.09 = 0.49
        self.assertAlmostEqual(result["emotional_readiness_score"], 0.49, places=4)

    def test_neutral_family_valence(self):
        """neutral dominant_family → valence = 0.5."""
        fp = _make_fingerprint(dominant_family="neutral", volatility=0.0, completion_rate=0.0, engagement_streak=0)
        result = self.prioritizer.compute(fp, stage=1)
        # stability=1.0*0.4 + valence=0.5*0.3 + momentum=0.0*0.3 = 0.4 + 0.15 = 0.55
        self.assertAlmostEqual(result["emotional_readiness_score"], 0.55, places=4)

    def test_high_volatility_lowers_score(self):
        """Higher volatility reduces stability and therefore readiness score."""
        fp_stable = _make_fingerprint(volatility=0.0)
        fp_volatile = _make_fingerprint(volatility=0.9)
        stable_result = self.prioritizer.compute(fp_stable, stage=1)
        volatile_result = self.prioritizer.compute(fp_volatile, stage=1)
        self.assertGreater(
            stable_result["emotional_readiness_score"],
            volatile_result["emotional_readiness_score"],
        )

    def test_streak_capped_at_10(self):
        """engagement_streak > 10 should not further increase momentum beyond cap."""
        fp_10 = _make_fingerprint(engagement_streak=10, completion_rate=1.0)
        fp_50 = _make_fingerprint(engagement_streak=50, completion_rate=1.0)
        r10 = self.prioritizer.compute(fp_10, stage=1)
        r50 = self.prioritizer.compute(fp_50, stage=1)
        self.assertEqual(r10["emotional_readiness_score"], r50["emotional_readiness_score"])

    def test_result_keys_present(self):
        fp = _make_fingerprint()
        result = self.prioritizer.compute(fp, stage=2)
        for key in ("emotional_readiness_score", "stability", "valence", "momentum", "mode"):
            self.assertIn(key, result)

    def test_empty_fingerprint_returns_valid_dict(self):
        result = self.prioritizer.compute({}, stage=1)
        score = result["emotional_readiness_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ---------------------------------------------------------------------------
# IntelligencePublisher
# ---------------------------------------------------------------------------

class TestIntelligencePublisher(unittest.TestCase):
    def setUp(self):
        self.bus = ProfileEventBus()
        self.publisher = IntelligencePublisher(self.bus)

    def tearDown(self):
        self.bus.shutdown()

    def _collect_events(self, pattern="intelligence.*"):
        """Subscribe and return a list that accumulates (topic, payload) tuples."""
        received = []

        def handler(topic, payload):
            received.append((topic, payload))

        self.bus.subscribe(pattern, handler, mode="sync")
        return received

    def test_publishes_therapy_event(self):
        """publish_for_user should emit an intelligence.therapy event."""
        received = self._collect_events()
        fp = _make_fingerprint()
        self.publisher.publish_for_user("u1", fp, stage=3, confidence=0.9)
        topics = [t for t, _ in received]
        self.assertIn("intelligence.therapy", topics)

    def test_publishes_coaching_event(self):
        """publish_for_user should emit an intelligence.coaching event."""
        received = self._collect_events()
        fp = _make_fingerprint()
        self.publisher.publish_for_user("u1", fp, stage=2, confidence=0.75)
        topics = [t for t, _ in received]
        self.assertIn("intelligence.coaching", topics)

    def test_publishes_workflow_event(self):
        """publish_for_user should emit an intelligence.workflow event."""
        received = self._collect_events()
        fp = _make_fingerprint()
        self.publisher.publish_for_user("u1", fp, stage=1, confidence=0.6)
        topics = [t for t, _ in received]
        self.assertIn("intelligence.workflow", topics)

    def test_publishes_calibration_event(self):
        """publish_for_user should emit an intelligence.calibration event."""
        received = self._collect_events()
        fp = _make_fingerprint()
        self.publisher.publish_for_user("u1", fp, stage=4, confidence=0.88)
        topics = [t for t, _ in received]
        self.assertIn("intelligence.calibration", topics)

    def test_all_four_events_published(self):
        """Exactly 4 intelligence events should be published."""
        received = self._collect_events()
        fp = _make_fingerprint()
        self.publisher.publish_for_user("u2", fp, stage=3, confidence=0.85)
        self.assertEqual(len(received), 4)

    def test_payload_contains_user_id(self):
        """Every published payload should carry the user_id."""
        received = self._collect_events()
        fp = _make_fingerprint()
        self.publisher.publish_for_user("user-42", fp, stage=2, confidence=0.7)
        for topic, payload in received:
            self.assertEqual(
                payload.get("user_id"), "user-42",
                msg=f"user_id missing or wrong in topic={topic!r}",
            )

    def test_therapy_payload_fields(self):
        """intelligence.therapy payload should contain expected fields."""
        received = self._collect_events()
        fp = _make_fingerprint()
        self.publisher.publish_for_user("u3", fp, stage=3, confidence=0.9)
        therapy_events = [p for t, p in received if t == "intelligence.therapy"]
        self.assertEqual(len(therapy_events), 1)
        payload = therapy_events[0]
        for field in ("technique_scores", "completion_rate", "preferred_step_type", "mode"):
            self.assertIn(field, payload)

    def test_coaching_payload_fields(self):
        """intelligence.coaching payload should contain expected fields."""
        received = self._collect_events()
        fp = _make_fingerprint()
        self.publisher.publish_for_user("u4", fp, stage=2, confidence=0.8)
        coaching_events = [p for t, p in received if t == "intelligence.coaching"]
        self.assertEqual(len(coaching_events), 1)
        payload = coaching_events[0]
        for field in ("preferred_tone", "vocabulary_level", "response_depth", "active_goals", "mode"):
            self.assertIn(field, payload)

    def test_workflow_payload_fields(self):
        """intelligence.workflow payload should contain expected fields."""
        received = self._collect_events()
        fp = _make_fingerprint()
        self.publisher.publish_for_user("u5", fp, stage=1, confidence=0.65)
        workflow_events = [p for t, p in received if t == "intelligence.workflow"]
        self.assertEqual(len(workflow_events), 1)
        payload = workflow_events[0]
        self.assertIn("emotional_readiness_score", payload)
        score = payload["emotional_readiness_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_connector_called_for_each_event(self):
        """If a connector is provided its deliver() method is called for each event."""

        class MockConnector:
            def __init__(self):
                self.calls = []

            def deliver(self, topic, payload):
                self.calls.append(topic)

        connector = MockConnector()
        publisher = IntelligencePublisher(self.bus, connector=connector)
        fp = _make_fingerprint()
        publisher.publish_for_user("u6", fp, stage=3, confidence=0.9)

        self.assertEqual(len(connector.calls), 4)
        self.assertIn("intelligence.therapy", connector.calls)
        self.assertIn("intelligence.coaching", connector.calls)
        self.assertIn("intelligence.workflow", connector.calls)
        self.assertIn("intelligence.calibration", connector.calls)

    def test_connector_failure_does_not_prevent_bus_delivery(self):
        """A connector that raises should not prevent bus subscribers from receiving events."""

        class BrokenConnector:
            def deliver(self, topic, payload):
                raise RuntimeError("connector down")

        received = self._collect_events()
        publisher = IntelligencePublisher(self.bus, connector=BrokenConnector())
        fp = _make_fingerprint()
        # Should not raise even if connector fails
        publisher.publish_for_user("u7", fp, stage=2, confidence=0.7)
        self.assertEqual(len(received), 4)

    def test_no_connector(self):
        """Publisher without a connector should still publish all bus events."""
        received = self._collect_events()
        fp = _make_fingerprint()
        self.publisher.publish_for_user("u8", fp, stage=5, confidence=1.0)
        self.assertEqual(len(received), 4)


if __name__ == "__main__":
    unittest.main()

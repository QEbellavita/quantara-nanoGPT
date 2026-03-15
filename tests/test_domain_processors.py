"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Domain Processors Test Suite
===============================================================================
Comprehensive tests for BaseDomainProcessor and all 8 DNA domain processors.
Verifies abstract enforcement, empty-event handling, domain names, and key
metric computations.

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
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain_processors import BaseDomainProcessor, get_all_processors
from domain_processors.emotional_processor import EmotionalProcessor
from domain_processors.biometric_processor import BiometricProcessor
from domain_processors.cognitive_processor import CognitiveProcessor
from domain_processors.behavioral_processor import BehavioralProcessor
from domain_processors.temporal_processor import TemporalProcessor
from domain_processors.linguistic_processor import LinguisticProcessor
from domain_processors.social_processor import SocialProcessor
from domain_processors.aspirational_processor import AspirationalProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event(payload: dict, event_type: str = "generic", timestamp: float = None) -> dict:
    """Build a minimal event dict with JSON-serialised payload."""
    return {
        "event_type": event_type,
        "payload": json.dumps(payload),
        "timestamp": timestamp if timestamp is not None else time.time(),
    }


# ---------------------------------------------------------------------------
# BaseDomainProcessor
# ---------------------------------------------------------------------------

class TestBaseDomainProcessor:
    def test_is_abstract(self):
        """Cannot instantiate BaseDomainProcessor directly."""
        with pytest.raises(TypeError):
            BaseDomainProcessor()  # type: ignore[abstract]

    def test_has_domain_class_var(self):
        assert hasattr(BaseDomainProcessor, "domain")

    def test_get_all_processors_returns_eight(self):
        processors = get_all_processors()
        assert len(processors) == 8

    def test_all_processors_are_base_instances(self):
        for p in get_all_processors():
            assert isinstance(p, BaseDomainProcessor)

    def test_all_processors_have_unique_domains(self):
        domains = [p.domain for p in get_all_processors()]
        assert len(set(domains)) == 8

    def test_get_empty_score_structure(self):
        """get_empty_score must return the canonical zero-score dict."""
        # Use a concrete processor to access the method
        p = EmotionalProcessor()
        empty = p.get_empty_score()
        assert empty == {"score": 0.0, "metrics": {}, "event_count": 0}


# ---------------------------------------------------------------------------
# EmotionalProcessor
# ---------------------------------------------------------------------------

class TestEmotionalProcessor:
    def setup_method(self):
        self.proc = EmotionalProcessor()

    def test_domain_name(self):
        assert self.proc.domain == "emotional"

    def test_empty_events_returns_zero(self):
        result = self.proc.compute([])
        assert result["score"] == 0.0
        assert result["event_count"] == 0

    def test_basic_metrics_present(self):
        events = [
            _event({"emotion": "joy", "confidence": 0.9}),
            _event({"emotion": "sadness", "confidence": 0.7}),
            _event({"emotion": "joy", "confidence": 0.8}),
        ]
        result = self.proc.compute(events)
        m = result["metrics"]
        assert "mood_distribution" in m
        assert "family_distribution" in m
        assert "dominant_emotion" in m
        assert "dominant_family" in m
        assert "emotional_range" in m
        assert "avg_confidence" in m
        assert "volatility" in m
        assert "recovery_rate" in m

    def test_dominant_emotion(self):
        events = [
            _event({"emotion": "joy", "confidence": 0.9}),
            _event({"emotion": "joy", "confidence": 0.8}),
            _event({"emotion": "sadness", "confidence": 0.7}),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["dominant_emotion"] == "joy"

    def test_dominant_family(self):
        events = [
            _event({"emotion": "joy", "confidence": 1.0}),
            _event({"emotion": "happiness", "confidence": 1.0}),
            _event({"emotion": "sadness", "confidence": 1.0}),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["dominant_family"] == "positive"

    def test_emotional_range(self):
        events = [
            _event({"emotion": "joy"}),
            _event({"emotion": "sadness"}),
            _event({"emotion": "anger"}),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["emotional_range"] == 3

    def test_avg_confidence(self):
        events = [
            _event({"emotion": "joy", "confidence": 0.8}),
            _event({"emotion": "sadness", "confidence": 0.6}),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["avg_confidence"] == pytest.approx(0.7, abs=0.001)

    def test_score_scales_with_event_count(self):
        events = [_event({"emotion": "joy", "confidence": 1.0}) for _ in range(50)]
        result = self.proc.compute(events)
        assert result["score"] == pytest.approx(0.5, abs=0.01)

    def test_score_capped_at_one(self):
        events = [_event({"emotion": "joy", "confidence": 1.0}) for _ in range(200)]
        result = self.proc.compute(events)
        assert result["score"] == pytest.approx(1.0)

    def test_event_count_matches(self):
        events = [_event({"emotion": "joy"}) for _ in range(7)]
        result = self.proc.compute(events)
        assert result["event_count"] == 7

    def test_volatility_all_same_emotion(self):
        events = [_event({"emotion": "joy"}) for _ in range(5)]
        result = self.proc.compute(events)
        assert result["metrics"]["volatility"] == 0.0

    def test_recovery_rate_neg_to_pos(self):
        now = time.time()
        events = [
            _event({"emotion": "sadness"}, timestamp=now),
            _event({"emotion": "joy"}, timestamp=now + 1),
            _event({"emotion": "anger"}, timestamp=now + 2),
            _event({"emotion": "happiness"}, timestamp=now + 3),
        ]
        result = self.proc.compute(events)
        # Two negative events, two transitions neg->pos (sadness->joy, anger->happiness)
        assert result["metrics"]["recovery_rate"] == pytest.approx(1.0, abs=0.01)

    def test_payload_as_dict(self):
        """Processor handles pre-parsed dict payloads as well as JSON strings."""
        events = [{"event_type": "emotion", "payload": {"emotion": "joy", "confidence": 0.9},
                   "timestamp": time.time()}]
        result = self.proc.compute(events)
        assert result["metrics"]["dominant_emotion"] == "joy"


# ---------------------------------------------------------------------------
# BiometricProcessor
# ---------------------------------------------------------------------------

class TestBiometricProcessor:
    def setup_method(self):
        self.proc = BiometricProcessor()

    def test_domain_name(self):
        assert self.proc.domain == "biometric"

    def test_empty_events_returns_zero(self):
        result = self.proc.compute([])
        assert result["score"] == 0.0

    def test_basic_metrics_present(self):
        events = [_event({"hr": 72, "hrv": 45, "eda": 0.3})]
        result = self.proc.compute(events)
        m = result["metrics"]
        for key in ("resting_hr", "hr_min", "hr_max", "hrv_baseline", "eda_baseline", "stress_ratio"):
            assert key in m

    def test_resting_hr(self):
        events = [
            _event({"hr": 60}),
            _event({"hr": 80}),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["resting_hr"] == pytest.approx(70.0, abs=0.1)

    def test_hr_min_max(self):
        events = [
            _event({"hr": 55}),
            _event({"hr": 90}),
            _event({"hr": 72}),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["hr_min"] == pytest.approx(55.0)
        assert result["metrics"]["hr_max"] == pytest.approx(90.0)

    def test_stress_ratio(self):
        events = [
            _event({"hr": 110}),  # high
            _event({"hr": 60}),
            _event({"hr": 120}),  # high
            _event({"hr": 70}),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["stress_ratio"] == pytest.approx(0.5, abs=0.01)

    def test_score_scaling(self):
        events = [_event({"hr": 70}) for _ in range(50)]
        result = self.proc.compute(events)
        assert result["score"] == pytest.approx(0.5, abs=0.01)

    def test_missing_hr_fields(self):
        """Events without HR data should not crash."""
        events = [_event({"eda": 0.4}), _event({"hrv": 30})]
        result = self.proc.compute(events)
        assert result["metrics"]["resting_hr"] is None
        assert result["metrics"]["hrv_baseline"] is not None


# ---------------------------------------------------------------------------
# CognitiveProcessor
# ---------------------------------------------------------------------------

class TestCognitiveProcessor:
    def setup_method(self):
        self.proc = CognitiveProcessor()

    def test_domain_name(self):
        assert self.proc.domain == "cognitive"

    def test_empty_events_returns_zero(self):
        result = self.proc.compute([])
        assert result["score"] == 0.0

    def test_basic_metrics_present(self):
        events = [_event({"technique": "mindfulness", "outcome": "success"})]
        result = self.proc.compute(events)
        m = result["metrics"]
        for key in ("technique_usage", "technique_diversity", "avg_completion_time",
                    "technique_retention_rate"):
            assert key in m

    def test_technique_usage_counter(self):
        events = [
            _event({"technique": "breathing"}),
            _event({"technique": "breathing"}),
            _event({"technique": "mindfulness"}),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["technique_usage"]["breathing"] == 2
        assert result["metrics"]["technique_usage"]["mindfulness"] == 1

    def test_technique_diversity(self):
        events = [
            _event({"technique": "a"}),
            _event({"technique": "b"}),
            _event({"technique": "c"}),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["technique_diversity"] == 3

    def test_avg_completion_time(self):
        events = [
            _event({"technique": "a", "completion_time": 10}),
            _event({"technique": "a", "completion_time": 20}),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["avg_completion_time"] == pytest.approx(15.0)

    def test_retention_rate(self):
        events = [
            _event({"technique": "a", "outcome": "success"}),
            _event({"technique": "b", "outcome": "failed"}),
            _event({"technique": "c", "outcome": "completed"}),
            _event({"technique": "d", "outcome": "aborted"}),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["technique_retention_rate"] == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# BehavioralProcessor
# ---------------------------------------------------------------------------

class TestBehavioralProcessor:
    def setup_method(self):
        self.proc = BehavioralProcessor()

    def test_domain_name(self):
        assert self.proc.domain == "behavioral"

    def test_empty_events_returns_zero(self):
        result = self.proc.compute([])
        assert result["score"] == 0.0

    def test_basic_metrics_present(self):
        events = [_event({}, event_type="session_started")]
        result = self.proc.compute(events)
        m = result["metrics"]
        for key in ("completion_rate", "intervention_response_rate", "engagement_streak"):
            assert key in m

    def test_completion_rate(self):
        events = [
            _event({}, event_type="session_started"),
            _event({}, event_type="session_started"),
            _event({}, event_type="session_completed"),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["completion_rate"] == pytest.approx(0.5, abs=0.01)

    def test_completion_rate_no_starts(self):
        events = [_event({}, event_type="session_completed")]
        result = self.proc.compute(events)
        assert result["metrics"]["completion_rate"] == 0.0

    def test_intervention_response_rate(self):
        events = [
            _event({}, event_type="intervention_triggered"),
            _event({}, event_type="intervention_triggered"),
            _event({}, event_type="intervention_responded"),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["intervention_response_rate"] == pytest.approx(0.5, abs=0.01)

    def test_engagement_streak_consecutive(self):
        base = 1_700_000_000.0  # ~2023-11-14 UTC
        day_seconds = 86400
        events = [
            _event({}, timestamp=base),
            _event({}, timestamp=base + day_seconds),
            _event({}, timestamp=base + 2 * day_seconds),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["engagement_streak"] >= 3

    def test_engagement_streak_gap(self):
        base = 1_700_000_000.0
        day_seconds = 86400
        events = [
            _event({}, timestamp=base),
            _event({}, timestamp=base + day_seconds),
            # gap of 2 days
            _event({}, timestamp=base + 4 * day_seconds),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["engagement_streak"] == 2


# ---------------------------------------------------------------------------
# TemporalProcessor
# ---------------------------------------------------------------------------

class TestTemporalProcessor:
    def setup_method(self):
        self.proc = TemporalProcessor()

    def test_domain_name(self):
        assert self.proc.domain == "temporal"

    def test_empty_events_returns_zero(self):
        result = self.proc.compute([])
        assert result["score"] == 0.0

    def test_basic_metrics_present(self):
        events = [_event({}, timestamp=time.time())]
        result = self.proc.compute(events)
        m = result["metrics"]
        for key in ("hour_distribution", "day_distribution", "peak_hours", "chronotype"):
            assert key in m

    def test_hour_distribution_populated(self):
        # Use a fixed timestamp: 2024-01-15 08:00 UTC = Monday, hour 8
        ts = 1705305600.0  # 2024-01-15 08:00 UTC
        events = [_event({}, timestamp=ts)]
        result = self.proc.compute(events)
        assert 8 in result["metrics"]["hour_distribution"]

    def test_chronotype_morning(self):
        # Force morning activity (hour 8)
        morning_ts = 1705305600.0  # 08:00 UTC
        events = [_event({}, timestamp=morning_ts) for _ in range(10)]
        result = self.proc.compute(events)
        assert result["metrics"]["chronotype"] == "morning"

    def test_chronotype_evening(self):
        # Force evening activity (2024-01-15 20:00 UTC = hour 20)
        evening_ts = 1705348800.0
        events = [_event({}, timestamp=evening_ts) for _ in range(10)]
        result = self.proc.compute(events)
        assert result["metrics"]["chronotype"] == "evening"

    def test_peak_hours_at_most_three(self):
        events = [_event({}, timestamp=time.time()) for _ in range(5)]
        result = self.proc.compute(events)
        assert len(result["metrics"]["peak_hours"]) <= 3


# ---------------------------------------------------------------------------
# LinguisticProcessor
# ---------------------------------------------------------------------------

class TestLinguisticProcessor:
    def setup_method(self):
        self.proc = LinguisticProcessor()

    def test_domain_name(self):
        assert self.proc.domain == "linguistic"

    def test_empty_events_returns_zero(self):
        result = self.proc.compute([])
        assert result["score"] == 0.0

    def test_basic_metrics_present(self):
        events = [_event({"text": "Hello world", "tone": "positive"})]
        result = self.proc.compute(events)
        m = result["metrics"]
        for key in ("avg_sentence_length", "vocabulary_complexity", "tone_distribution",
                    "dominant_tone"):
            assert key in m

    def test_avg_sentence_length(self):
        events = [
            _event({"text": "one two three"}),   # 3 words
            _event({"text": "one two three four five"}),  # 5 words
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["avg_sentence_length"] == pytest.approx(4.0, abs=0.1)

    def test_vocabulary_complexity_all_unique(self):
        events = [_event({"text": "alpha beta gamma delta"})]
        result = self.proc.compute(events)
        assert result["metrics"]["vocabulary_complexity"] == pytest.approx(1.0)

    def test_vocabulary_complexity_all_same(self):
        events = [_event({"text": "the the the the"})]
        result = self.proc.compute(events)
        assert result["metrics"]["vocabulary_complexity"] == pytest.approx(0.25)

    def test_tone_distribution(self):
        events = [
            _event({"text": "a", "tone": "positive"}),
            _event({"text": "b", "tone": "positive"}),
            _event({"text": "c", "tone": "negative"}),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["tone_distribution"]["positive"] == 2
        assert result["metrics"]["dominant_tone"] == "positive"

    def test_no_text_events(self):
        """Events without text should not crash and return zero avg_sentence_length."""
        events = [_event({"tone": "neutral"})]
        result = self.proc.compute(events)
        assert result["metrics"]["avg_sentence_length"] == 0.0


# ---------------------------------------------------------------------------
# SocialProcessor
# ---------------------------------------------------------------------------

class TestSocialProcessor:
    def setup_method(self):
        self.proc = SocialProcessor()

    def test_domain_name(self):
        assert self.proc.domain == "social"

    def test_empty_events_returns_zero(self):
        result = self.proc.compute([])
        assert result["score"] == 0.0

    def test_basic_metrics_present(self):
        events = [_event({}, event_type="interaction")]
        result = self.proc.compute(events)
        m = result["metrics"]
        for key in ("interaction_count", "shares", "feedback_received",
                    "interaction_types", "daily_interaction_rate"):
            assert key in m

    def test_interaction_count(self):
        events = [_event({}) for _ in range(5)]
        result = self.proc.compute(events)
        assert result["metrics"]["interaction_count"] == 5

    def test_shares_counted(self):
        events = [
            _event({"shared": True}),
            _event({"shared": False}),
            _event({}, event_type="share"),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["shares"] == 2

    def test_feedback_received(self):
        events = [
            _event({"feedback": "great"}),
            _event({}, event_type="feedback_received"),
            _event({}),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["feedback_received"] == 2

    def test_interaction_types_counter(self):
        events = [
            _event({}, event_type="like"),
            _event({}, event_type="like"),
            _event({}, event_type="comment"),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["interaction_types"]["like"] == 2
        assert result["metrics"]["interaction_types"]["comment"] == 1

    def test_daily_interaction_rate(self):
        base = 1_700_000_000.0
        day = 86400
        events = [
            _event({}, timestamp=base),
            _event({}, timestamp=base),
            _event({}, timestamp=base + day),
        ]
        result = self.proc.compute(events)
        # 3 interactions over 2 days = 1.5
        assert result["metrics"]["daily_interaction_rate"] == pytest.approx(1.5, abs=0.01)


# ---------------------------------------------------------------------------
# AspirationalProcessor
# ---------------------------------------------------------------------------

class TestAspirationalProcessor:
    def setup_method(self):
        self.proc = AspirationalProcessor()

    def test_domain_name(self):
        assert self.proc.domain == "aspirational"

    def test_empty_events_returns_zero(self):
        result = self.proc.compute([])
        assert result["score"] == 0.0

    def test_basic_metrics_present(self):
        events = [_event({}, event_type="goal_set")]
        result = self.proc.compute(events)
        m = result["metrics"]
        for key in ("goals_set", "goals_completed", "goal_completion_rate",
                    "active_growth_areas", "core_values"):
            assert key in m

    def test_goals_set_count(self):
        events = [
            _event({}, event_type="goal_set"),
            _event({}, event_type="goal_set"),
            _event({}, event_type="goal_created"),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["goals_set"] == 3

    def test_goals_completed_count(self):
        events = [
            _event({}, event_type="goal_completed"),
            _event({}, event_type="goal_achieved"),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["goals_completed"] == 2

    def test_goal_completion_rate(self):
        events = [
            _event({}, event_type="goal_set"),
            _event({}, event_type="goal_set"),
            _event({}, event_type="goal_completed"),
        ]
        result = self.proc.compute(events)
        assert result["metrics"]["goal_completion_rate"] == pytest.approx(0.5, abs=0.01)

    def test_goal_completion_rate_no_goals_set(self):
        events = [_event({}, event_type="goal_completed")]
        result = self.proc.compute(events)
        assert result["metrics"]["goal_completion_rate"] == 0.0

    def test_active_growth_areas(self):
        events = [
            _event({"growth_area": "mindfulness"}),
            _event({"growth_area": "fitness"}),
        ]
        result = self.proc.compute(events)
        assert "mindfulness" in result["metrics"]["active_growth_areas"]
        assert "fitness" in result["metrics"]["active_growth_areas"]

    def test_core_values_top_five(self):
        values = ["integrity", "growth", "kindness", "balance", "creativity", "courage"]
        events = [
            _event({"values": values[:3]}),
            _event({"values": values[3:]}),
            _event({"core_values": ["integrity"]}),  # extra weight
        ]
        result = self.proc.compute(events)
        assert len(result["metrics"]["core_values"]) <= 5
        # 'integrity' appears twice so should be in core_values
        assert "integrity" in result["metrics"]["core_values"]

    def test_score_scaling(self):
        events = [_event({}) for _ in range(100)]
        result = self.proc.compute(events)
        assert result["score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistryIntegration:
    """Smoke-test all processors end-to-end through the registry."""

    def test_all_processors_handle_empty_events(self):
        for proc in get_all_processors():
            result = proc.compute([])
            assert result["score"] == 0.0, f"{proc.domain} failed on empty events"
            assert result["event_count"] == 0

    def test_all_processors_return_required_keys(self):
        events = [_event({"emotion": "joy", "hr": 70, "text": "hello"}, timestamp=time.time())]
        for proc in get_all_processors():
            result = proc.compute(events)
            assert "score" in result, f"{proc.domain} missing 'score'"
            assert "event_count" in result, f"{proc.domain} missing 'event_count'"
            assert "metrics" in result, f"{proc.domain} missing 'metrics'"
            assert isinstance(result["score"], float)
            assert 0.0 <= result["score"] <= 1.0, f"{proc.domain} score out of range"

# tests/test_personalization.py
"""Tests for profile-based emotion personalization."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_classifier import FAMILY_NAMES, EMOTION_FAMILIES


class TestFamilyScoresInAnalyzer:
    """Test that MultimodalEmotionAnalyzer.analyze() returns family_scores."""

    @pytest.fixture
    def analyzer(self):
        from emotion_classifier import MultimodalEmotionAnalyzer
        return MultimodalEmotionAnalyzer(use_sentence_transformer=True)

    def test_family_scores_present(self, analyzer):
        """analyze() result should contain family_scores dict."""
        result = analyzer.analyze("I feel happy today")
        assert 'family_scores' in result
        assert isinstance(result['family_scores'], dict)

    def test_family_scores_keys(self, analyzer):
        """family_scores keys should be the 9 family names."""
        result = analyzer.analyze("I am so excited")
        assert set(result['family_scores'].keys()) == set(FAMILY_NAMES)

    def test_family_scores_sum_to_one(self, analyzer):
        """family_scores values should sum to ~1.0 (softmax output)."""
        result = analyzer.analyze("This is a test")
        total = sum(result['family_scores'].values())
        assert abs(total - 1.0) < 0.01

    def test_family_scores_all_positive(self, analyzer):
        """All family_scores should be >= 0."""
        result = analyzer.analyze("I feel nothing")
        for score in result['family_scores'].values():
            assert score >= 0.0


class TestApplyProfilePersonalization:
    """Test the tiebreaker personalization function."""

    def _make_snapshot(self, emotion_prior, event_count=50):
        from user_profile_engine import ProfileSnapshot
        dominant = max(emotion_prior, key=emotion_prior.get)
        return ProfileSnapshot(
            user_id='test', evolution_stage=3, overall_confidence=0.7,
            emotion_prior=emotion_prior, dominant_family=dominant,
            event_count=event_count,
            family_counts={f: int(v * event_count) for f, v in emotion_prior.items()},
            last_updated=1710000000.0,
        )

    def test_skip_when_no_snapshot(self):
        from emotion_classifier import apply_profile_personalization
        result = {'family': 'Joy', 'confidence': 0.8, 'family_scores': {'Joy': 0.8, 'Sadness': 0.1}}
        out = apply_profile_personalization(result, None)
        assert out['personalized'] is False

    def test_skip_when_cold_start(self):
        from emotion_classifier import apply_profile_personalization
        snap = self._make_snapshot({'Joy': 1.0}, event_count=5)
        result = {'family': 'Joy', 'confidence': 0.8, 'family_scores': {'Joy': 0.8, 'Sadness': 0.1}}
        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is False

    def test_skip_when_no_family_scores(self):
        from emotion_classifier import apply_profile_personalization
        snap = self._make_snapshot({'Joy': 0.5, 'Sadness': 0.5})
        result = {'family': 'Joy', 'confidence': 0.8}
        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is False

    def test_skip_when_confident(self):
        from emotion_classifier import apply_profile_personalization
        snap = self._make_snapshot({'Sadness': 0.8, 'Joy': 0.2})
        result = {
            'dominant_emotion': 'joy', 'family': 'Joy', 'confidence': 0.7,
            'family_scores': {'Joy': 0.7, 'Sadness': 0.2, 'Anger': 0.05, 'Fear': 0.05},
            'scores': {'joy': 0.7},
        }
        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is False

    def test_tiebreak_changes_winner(self):
        from emotion_classifier import apply_profile_personalization
        snap = self._make_snapshot({'Fear': 0.7, 'Sadness': 0.1, 'Joy': 0.1, 'Anger': 0.1})
        result = {
            'dominant_emotion': 'sadness', 'family': 'Sadness', 'confidence': 0.30,
            'family_scores': {'Sadness': 0.30, 'Fear': 0.28, 'Joy': 0.20, 'Anger': 0.12, 'Calm': 0.05, 'Love': 0.02, 'Self-Conscious': 0.01, 'Surprise': 0.01, 'Neutral': 0.01},
            'scores': {'sadness': 0.15, 'grief': 0.10, 'boredom': 0.05, 'fear': 0.14, 'anxiety': 0.10, 'worry': 0.04},
        }
        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is True
        assert out['family'] == 'Fear'
        assert 'profile_tiebreak' in out['personalization_reason']

    def test_tiebreak_confirms_winner(self):
        from emotion_classifier import apply_profile_personalization
        snap = self._make_snapshot({'Joy': 0.6, 'Sadness': 0.2, 'Anger': 0.2})
        result = {
            'dominant_emotion': 'joy', 'family': 'Joy', 'confidence': 0.35,
            'family_scores': {'Joy': 0.35, 'Sadness': 0.33, 'Anger': 0.15, 'Fear': 0.10, 'Calm': 0.03, 'Love': 0.02, 'Self-Conscious': 0.01, 'Surprise': 0.005, 'Neutral': 0.005},
            'scores': {'joy': 0.20, 'excitement': 0.10, 'enthusiasm': 0.05},
        }
        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is True
        assert out['family'] == 'Joy'
        assert 'profile_consulted' in out['personalization_reason']

    def test_family_scores_unchanged(self):
        from emotion_classifier import apply_profile_personalization
        snap = self._make_snapshot({'Fear': 0.7, 'Sadness': 0.3})
        original_scores = {'Sadness': 0.30, 'Fear': 0.28, 'Joy': 0.20, 'Anger': 0.12, 'Calm': 0.05, 'Love': 0.02, 'Self-Conscious': 0.01, 'Surprise': 0.01, 'Neutral': 0.01}
        result = {
            'dominant_emotion': 'sadness', 'family': 'Sadness', 'confidence': 0.30,
            'family_scores': dict(original_scores),
            'scores': {'sadness': 0.15, 'fear': 0.14},
        }
        apply_profile_personalization(result, snap)
        assert result['family_scores'] == original_scores

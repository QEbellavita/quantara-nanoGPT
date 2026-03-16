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

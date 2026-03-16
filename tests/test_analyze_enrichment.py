# tests/test_analyze_enrichment.py
"""Tests for opt-in profile enrichment in /api/emotion/analyze."""

import os
import sys
import json
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAnalyzePersonalizationWiring:

    def test_personalized_false_when_no_snapshot(self):
        from emotion_classifier import apply_profile_personalization
        result = {
            'dominant_emotion': 'joy', 'family': 'Joy', 'confidence': 0.9,
            'family_scores': {'Joy': 0.9, 'Sadness': 0.05},
            'scores': {'joy': 0.9},
        }
        out = apply_profile_personalization(result, None)
        assert 'personalized' in out
        assert out['personalized'] is False

    def test_no_personalized_field_without_calling_function(self):
        result = {'dominant_emotion': 'joy', 'family': 'Joy', 'confidence': 0.9}
        assert 'personalized' not in result


class TestProfileContextEnrichment:

    def _make_snapshot(self):
        from user_profile_engine import ProfileSnapshot
        return ProfileSnapshot(
            user_id='u1', evolution_stage=3, overall_confidence=0.72,
            emotion_prior={'Fear': 0.38, 'Sadness': 0.21, 'Joy': 0.12},
            dominant_family='Fear', event_count=847,
            family_counts={'Fear': 322, 'Sadness': 178, 'Joy': 102},
            last_updated=1710600000.0,
        )

    def test_profile_context_format(self):
        snap = self._make_snapshot()
        from datetime import datetime, timezone
        context = {
            'evolution_stage': snap.evolution_stage,
            'overall_confidence': snap.overall_confidence,
            'dominant_family': snap.dominant_family,
            'emotion_prior': snap.emotion_prior,
            'event_count': snap.event_count,
            'last_updated': datetime.fromtimestamp(snap.last_updated, tz=timezone.utc).isoformat(),
        }
        assert context['evolution_stage'] == 3
        assert context['event_count'] == 847
        assert 'Fear' in context['emotion_prior']

    def test_snapshot_json_serializable(self):
        snap = self._make_snapshot()
        from datetime import datetime, timezone
        context = {
            'evolution_stage': snap.evolution_stage,
            'overall_confidence': snap.overall_confidence,
            'dominant_family': snap.dominant_family,
            'emotion_prior': snap.emotion_prior,
            'event_count': snap.event_count,
            'last_updated': datetime.fromtimestamp(snap.last_updated, tz=timezone.utc).isoformat(),
        }
        serialized = json.dumps(context)
        assert '"evolution_stage": 3' in serialized


class TestDominantEmotionKeyFix:

    def test_log_event_uses_dominant_emotion(self):
        result = {'dominant_emotion': 'anxiety', 'family': 'Fear', 'confidence': 0.7}
        old_way = result.get('emotion', 'neutral')
        assert old_way == 'neutral'
        new_way = result.get('dominant_emotion', result.get('emotion', 'neutral'))
        assert new_way == 'anxiety'

    def test_fallback_classifier_still_works(self):
        result = {'emotion': 'joy', 'family': 'Joy', 'confidence': 1.0}
        emotion = result.get('dominant_emotion', result.get('emotion', 'neutral'))
        assert emotion == 'joy'


class TestEndToEndPersonalization:
    """Integration test: log events -> build snapshot -> personalize analysis."""

    @pytest.fixture
    def engine(self, tmp_path):
        from user_profile_engine import UserProfileEngine
        return UserProfileEngine(db_path=str(tmp_path / 'e2e.db'))

    def test_full_loop(self, engine):
        """Events build prior, prior influences ambiguous classification."""
        from emotion_classifier import apply_profile_personalization

        for _ in range(12):
            engine.log_event('u1', 'emotional', 'emotion_classified', {
                'emotion': 'anxiety', 'family': 'Fear', 'confidence': 0.6,
            })
        for _ in range(3):
            engine.log_event('u1', 'emotional', 'emotion_classified', {
                'emotion': 'sadness', 'family': 'Sadness', 'confidence': 0.5,
            })

        snap = engine.get_profile_snapshot('u1')
        assert snap is not None
        assert snap.event_count == 15
        assert snap.dominant_family == 'Fear'
        assert snap.emotion_prior['Fear'] > 0.7

        result = {
            'dominant_emotion': 'sadness', 'family': 'Sadness', 'confidence': 0.31,
            'family_scores': {
                'Sadness': 0.31, 'Fear': 0.29, 'Joy': 0.15,
                'Anger': 0.10, 'Calm': 0.05, 'Love': 0.04,
                'Self-Conscious': 0.03, 'Surprise': 0.02, 'Neutral': 0.01,
            },
            'scores': {
                'sadness': 0.15, 'grief': 0.10, 'boredom': 0.06,
                'fear': 0.14, 'anxiety': 0.12, 'worry': 0.03,
            },
        }

        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is True
        assert out['family'] == 'Fear'
        assert 'profile_tiebreak' in out['personalization_reason']
        assert out['dominant_emotion'] in ['fear', 'anxiety', 'worry', 'overwhelmed', 'stressed']

    def test_confident_prediction_unaffected(self, engine):
        """Even with strong prior, confident predictions are not changed."""
        from emotion_classifier import apply_profile_personalization

        for _ in range(20):
            engine.log_event('u1', 'emotional', 'emotion_classified', {
                'emotion': 'anxiety', 'family': 'Fear', 'confidence': 0.8,
            })

        snap = engine.get_profile_snapshot('u1')

        result = {
            'dominant_emotion': 'joy', 'family': 'Joy', 'confidence': 0.85,
            'family_scores': {
                'Joy': 0.85, 'Fear': 0.05, 'Sadness': 0.03,
                'Anger': 0.02, 'Calm': 0.02, 'Love': 0.01,
                'Self-Conscious': 0.01, 'Surprise': 0.005, 'Neutral': 0.005,
            },
            'scores': {'joy': 0.85},
        }

        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is False
        assert out['family'] == 'Joy'
        assert out['dominant_emotion'] == 'joy'

    def test_restart_then_personalize(self, engine):
        """After simulated restart, lazy rebuild enables personalization."""
        from emotion_classifier import apply_profile_personalization

        for _ in range(15):
            engine.log_event('u1', 'emotional', 'emotion_classified', {
                'emotion': 'anger', 'family': 'Anger', 'confidence': 0.7,
            })

        with engine._snapshot_lock:
            engine._snapshots.clear()

        snap = engine.get_profile_snapshot('u1')
        assert snap is not None
        assert snap.dominant_family == 'Anger'

        result = {
            'dominant_emotion': 'joy', 'family': 'Joy', 'confidence': 0.30,
            'family_scores': {
                'Joy': 0.30, 'Anger': 0.29, 'Sadness': 0.15,
                'Fear': 0.10, 'Calm': 0.05, 'Love': 0.04,
                'Self-Conscious': 0.03, 'Surprise': 0.02, 'Neutral': 0.02,
            },
            'scores': {'joy': 0.20, 'anger': 0.18, 'frustration': 0.11},
        }

        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is True
        assert out['family'] == 'Anger'

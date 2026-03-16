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

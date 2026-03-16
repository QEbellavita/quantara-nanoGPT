# tests/test_transition_tracker.py
"""Tests for EmotionTransitionTracker pattern detection and persistence."""

import pytest
import json
import os
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_transition_tracker import EmotionTransitionTracker, EmotionRecord


class TestRecord:
    """Test record() adds entries correctly."""

    @pytest.fixture
    def tracker(self, tmp_path):
        return EmotionTransitionTracker(
            persist_dir=str(tmp_path / 'transitions'),
            auto_persist=False,
        )

    def test_record_adds_entry(self, tracker):
        """record() should add an entry to the user's history."""
        result = tracker.record('user1', 'joy', 'Joy', confidence=0.9)

        assert result['user_id'] == 'user1'
        assert result['total_records'] == 1
        assert result['recorded']['emotion'] == 'joy'

    def test_record_multiple_entries(self, tracker):
        """Multiple records should accumulate."""
        tracker.record('user1', 'joy')
        tracker.record('user1', 'sadness')
        result = tracker.record('user1', 'anger')

        assert result['total_records'] == 3

    def test_record_auto_resolves_family(self, tracker):
        """Family should auto-resolve from emotion when not provided."""
        result = tracker.record('user1', 'excitement')

        assert result['recorded']['family'] == 'Joy'

    def test_record_returns_alerts(self, tracker):
        """record() return value should include alerts list."""
        result = tracker.record('user1', 'joy')

        assert 'alerts' in result
        assert isinstance(result['alerts'], list)


class TestGetTrajectory:
    """Test get_trajectory() returns correct structure."""

    @pytest.fixture
    def tracker(self, tmp_path):
        t = EmotionTransitionTracker(
            persist_dir=str(tmp_path / 'transitions'),
            auto_persist=False,
        )
        now = datetime.now(timezone.utc)
        # Add records within the last 24 hours
        for i, emotion in enumerate(['joy', 'sadness', 'anger', 'calm']):
            ts = (now - timedelta(hours=4 - i)).isoformat()
            t.record('user1', emotion, timestamp=ts)
        return t

    def test_trajectory_has_expected_keys(self, tracker):
        """get_trajectory() should return all expected keys."""
        result = tracker.get_trajectory('user1', window_hours=24)

        expected_keys = [
            'user_id', 'window_hours', 'timeline', 'record_count',
            'family_distribution', 'emotion_distribution',
            'avg_confidence', 'valence',
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_trajectory_timeline_length(self, tracker):
        """Timeline should contain all records in the window."""
        result = tracker.get_trajectory('user1', window_hours=24)

        assert result['record_count'] == 4
        assert len(result['timeline']) == 4

    def test_trajectory_valence_has_fractions(self, tracker):
        """Valence should have positive, negative, and neutral fractions."""
        result = tracker.get_trajectory('user1', window_hours=24)

        valence = result['valence']
        assert 'positive_fraction' in valence
        assert 'negative_fraction' in valence
        assert 'neutral_fraction' in valence

    def test_trajectory_empty_user(self, tracker):
        """Unknown user should return empty trajectory."""
        result = tracker.get_trajectory('nonexistent', window_hours=24)

        assert result['record_count'] == 0
        assert result['timeline'] == []


class TestDetectPatternsRapidCycling:
    """Test rapid_cycling pattern: 3+ family changes within 30 minutes."""

    @pytest.fixture
    def tracker(self, tmp_path):
        return EmotionTransitionTracker(
            persist_dir=str(tmp_path / 'transitions'),
            auto_persist=False,
        )

    def test_rapid_cycling_detected(self, tracker):
        """3+ family changes in 30 min should trigger rapid_cycling."""
        now = datetime.now(timezone.utc)
        # 4 entries alternating families within the last 10 minutes
        emotions = [
            ('joy', 'Joy'),
            ('anger', 'Anger'),
            ('sadness', 'Sadness'),
            ('excitement', 'Joy'),
        ]
        for i, (emotion, family) in enumerate(emotions):
            ts = (now - timedelta(minutes=10 - i * 2)).isoformat()
            tracker.record('user1', emotion, family, timestamp=ts)

        alerts = tracker.detect_patterns('user1')
        pattern_types = [a['pattern'] for a in alerts]

        assert 'rapid_cycling' in pattern_types

    def test_no_rapid_cycling_with_stable_emotions(self, tracker):
        """Same family repeatedly should NOT trigger rapid_cycling."""
        now = datetime.now(timezone.utc)
        for i in range(5):
            ts = (now - timedelta(minutes=10 - i * 2)).isoformat()
            tracker.record('user1', 'joy', 'Joy', timestamp=ts)

        alerts = tracker.detect_patterns('user1')
        pattern_types = [a['pattern'] for a in alerts]

        assert 'rapid_cycling' not in pattern_types


class TestDetectPatternsNegativeSpiral:
    """Test negative_spiral pattern: 3+ consecutive negative emotions."""

    @pytest.fixture
    def tracker(self, tmp_path):
        return EmotionTransitionTracker(
            persist_dir=str(tmp_path / 'transitions'),
            auto_persist=False,
        )

    def test_negative_spiral_detected(self, tracker):
        """3 consecutive negative-family emotions should trigger negative_spiral."""
        now = datetime.now(timezone.utc)
        negative_emotions = [
            ('anger', 'Anger'),
            ('fear', 'Fear'),
            ('sadness', 'Sadness'),
        ]
        for i, (emotion, family) in enumerate(negative_emotions):
            ts = (now - timedelta(minutes=5 - i)).isoformat()
            tracker.record('user1', emotion, family, timestamp=ts)

        alerts = tracker.detect_patterns('user1')
        pattern_types = [a['pattern'] for a in alerts]

        assert 'negative_spiral' in pattern_types

    def test_no_negative_spiral_with_positive_break(self, tracker):
        """A positive emotion breaking the streak should prevent detection."""
        now = datetime.now(timezone.utc)
        emotions = [
            ('anger', 'Anger'),
            ('joy', 'Joy'),       # breaks the streak
            ('sadness', 'Sadness'),
        ]
        for i, (emotion, family) in enumerate(emotions):
            ts = (now - timedelta(minutes=5 - i)).isoformat()
            tracker.record('user1', emotion, family, timestamp=ts)

        alerts = tracker.detect_patterns('user1')
        pattern_types = [a['pattern'] for a in alerts]

        assert 'negative_spiral' not in pattern_types


class TestDetectPatternsEmotionalFlatline:
    """Test emotional_flatline pattern: same emotion sustained 2+ hours."""

    @pytest.fixture
    def tracker(self, tmp_path):
        return EmotionTransitionTracker(
            persist_dir=str(tmp_path / 'transitions'),
            auto_persist=False,
        )

    def test_emotional_flatline_detected(self, tracker):
        """Same emotion for 2+ hours should trigger emotional_flatline."""
        now = datetime.now(timezone.utc)
        # Spread 'calm' over 3 hours
        for i in range(4):
            ts = (now - timedelta(hours=3 - i)).isoformat()
            tracker.record('user1', 'calm', 'Calm', timestamp=ts)

        alerts = tracker.detect_patterns('user1')
        pattern_types = [a['pattern'] for a in alerts]

        assert 'emotional_flatline' in pattern_types

    def test_no_flatline_with_variety(self, tracker):
        """Different emotions should NOT trigger flatline."""
        now = datetime.now(timezone.utc)
        emotions = ['joy', 'calm', 'excitement', 'hope']
        for i, emotion in enumerate(emotions):
            ts = (now - timedelta(hours=3 - i)).isoformat()
            tracker.record('user1', emotion, timestamp=ts)

        alerts = tracker.detect_patterns('user1')
        pattern_types = [a['pattern'] for a in alerts]

        assert 'emotional_flatline' not in pattern_types


class TestDetectPatternsPositiveRecovery:
    """Test positive_recovery pattern: negative -> positive transition."""

    @pytest.fixture
    def tracker(self, tmp_path):
        return EmotionTransitionTracker(
            persist_dir=str(tmp_path / 'transitions'),
            auto_persist=False,
        )

    def test_positive_recovery_detected(self, tracker):
        """Transition from negative family to positive should trigger positive_recovery."""
        now = datetime.now(timezone.utc)
        tracker.record('user1', 'anger', 'Anger',
                       timestamp=(now - timedelta(minutes=5)).isoformat())
        tracker.record('user1', 'joy', 'Joy',
                       timestamp=now.isoformat())

        alerts = tracker.detect_patterns('user1')
        pattern_types = [a['pattern'] for a in alerts]

        assert 'positive_recovery' in pattern_types

    def test_no_recovery_for_neutral_transition(self, tracker):
        """Negative to neutral should NOT trigger positive_recovery."""
        now = datetime.now(timezone.utc)
        tracker.record('user1', 'anger', 'Anger',
                       timestamp=(now - timedelta(minutes=5)).isoformat())
        tracker.record('user1', 'neutral', 'Neutral',
                       timestamp=now.isoformat())

        alerts = tracker.detect_patterns('user1')
        pattern_types = [a['pattern'] for a in alerts]

        assert 'positive_recovery' not in pattern_types


class TestGetDashboardSummary:
    """Test get_dashboard_summary() returns expected keys."""

    @pytest.fixture
    def tracker(self, tmp_path):
        t = EmotionTransitionTracker(
            persist_dir=str(tmp_path / 'transitions'),
            auto_persist=False,
        )
        now = datetime.now(timezone.utc)
        for i, emotion in enumerate(['joy', 'sadness', 'calm']):
            ts = (now - timedelta(minutes=30 - i * 10)).isoformat()
            t.record('user1', emotion, timestamp=ts)
        return t

    def test_dashboard_summary_has_expected_keys(self, tracker):
        """Summary should contain all top-level keys."""
        summary = tracker.get_dashboard_summary('user1')

        expected_keys = [
            'user_id', 'generated_at', 'session', 'current_state',
            'trajectory_24h', 'transition_matrix_24h',
            'total_transitions_24h', 'alerts', 'alert_summary',
            'recommendations',
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_dashboard_session_has_record_info(self, tracker):
        """Session block should have first/last record and total count."""
        summary = tracker.get_dashboard_summary('user1')

        session = summary['session']
        assert 'first_record' in session
        assert 'last_record' in session
        assert session['total_records'] == 3

    def test_alert_summary_has_severity_counts(self, tracker):
        """Alert summary should have total, critical, warning, info counts."""
        summary = tracker.get_dashboard_summary('user1')

        alert_summary = summary['alert_summary']
        assert 'total' in alert_summary
        assert 'critical' in alert_summary
        assert 'warning' in alert_summary
        assert 'info' in alert_summary

    def test_recommendations_is_list(self, tracker):
        """Recommendations should be a non-empty list of strings."""
        summary = tracker.get_dashboard_summary('user1')

        assert isinstance(summary['recommendations'], list)
        assert len(summary['recommendations']) >= 1
        assert all(isinstance(r, str) for r in summary['recommendations'])


class TestThreadSafety:
    """Test concurrent record() calls don't corrupt state."""

    @pytest.fixture
    def tracker(self, tmp_path):
        return EmotionTransitionTracker(
            persist_dir=str(tmp_path / 'transitions'),
            auto_persist=False,
        )

    def test_concurrent_records_same_user(self, tracker):
        """Multiple threads recording to same user should not lose entries."""
        num_threads = 10
        records_per_thread = 50
        errors = []

        def record_emotions(thread_id):
            try:
                for i in range(records_per_thread):
                    tracker.record(
                        'shared_user',
                        'joy' if i % 2 == 0 else 'sadness',
                        confidence=0.8,
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_emotions, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

        expected_total = num_threads * records_per_thread
        actual_total = len(tracker._history.get('shared_user', []))
        assert actual_total == expected_total

    def test_concurrent_records_different_users(self, tracker):
        """Concurrent writes to different users should be independent."""
        num_users = 5
        records_per_user = 20
        errors = []

        def record_for_user(user_id):
            try:
                for _ in range(records_per_user):
                    tracker.record(user_id, 'calm', 'Calm')
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_for_user, args=(f'user_{i}',))
            for i in range(num_users)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        for i in range(num_users):
            assert len(tracker._history[f'user_{i}']) == records_per_user


class TestJSONPersistence:
    """Test save/load cycle with JSON persistence."""

    def test_persist_and_load_cycle(self, tmp_path):
        """Data persisted to disk should survive a new tracker instance."""
        persist_dir = str(tmp_path / 'transitions')

        # Create tracker, add records, persist
        tracker1 = EmotionTransitionTracker(
            persist_dir=persist_dir,
            auto_persist=True,
        )
        tracker1.record('user_a', 'joy', 'Joy', confidence=0.95)
        tracker1.record('user_a', 'sadness', 'Sadness', confidence=0.80)
        tracker1.record('user_b', 'anger', 'Anger', confidence=0.70)
        tracker1.persist_all()

        # Create a fresh tracker loading from same directory
        tracker2 = EmotionTransitionTracker(
            persist_dir=persist_dir,
            auto_persist=False,
        )

        # Verify user_a records loaded
        assert 'user_a' in tracker2._history
        assert len(tracker2._history['user_a']) == 2
        assert tracker2._history['user_a'][0].emotion == 'joy'
        assert tracker2._history['user_a'][1].emotion == 'sadness'
        assert tracker2._history['user_a'][0].confidence == 0.95

        # Verify user_b loaded
        assert 'user_b' in tracker2._history
        assert len(tracker2._history['user_b']) == 1
        assert tracker2._history['user_b'][0].emotion == 'anger'

    def test_persist_creates_json_files(self, tmp_path):
        """Persistence should create .json files in the persist directory."""
        persist_dir = str(tmp_path / 'transitions')

        tracker = EmotionTransitionTracker(
            persist_dir=persist_dir,
            auto_persist=True,
        )
        tracker.record('user_x', 'calm', 'Calm')
        tracker.persist_all()

        json_path = tmp_path / 'transitions' / 'user_x.json'
        assert json_path.exists()

        with open(json_path, 'r') as f:
            data = json.load(f)

        assert data['user_id'] == 'user_x'
        assert len(data['records']) == 1
        assert data['records'][0]['emotion'] == 'calm'

    def test_clear_user_removes_persisted_data(self, tmp_path):
        """clear_user() should remove both in-memory and persisted data."""
        persist_dir = str(tmp_path / 'transitions')

        tracker = EmotionTransitionTracker(
            persist_dir=persist_dir,
            auto_persist=True,
        )
        tracker.record('user_z', 'joy', 'Joy')
        tracker.persist_all()

        json_path = tmp_path / 'transitions' / 'user_z.json'
        assert json_path.exists()

        tracker.clear_user('user_z')

        assert 'user_z' not in tracker._history
        assert not json_path.exists()

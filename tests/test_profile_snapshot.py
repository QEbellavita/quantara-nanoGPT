"""
Tests for ProfileSnapshot dataclass and snapshot update logic
in UserProfileEngine.
"""

import os
import sys
import threading
import time

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from user_profile_engine import UserProfileEngine, ProfileSnapshot


# ─── TestProfileSnapshotCreation ──────────────────────────────────────────────

class TestProfileSnapshotCreation:
    """Verify snapshot creation and non-creation scenarios."""

    @pytest.fixture
    def engine(self, tmp_path):
        db_path = str(tmp_path / 'test.db')
        eng = UserProfileEngine(db_path=db_path)
        yield eng
        eng.close()

    def test_no_snapshot_before_events(self, engine):
        """get_profile_snapshot returns None when no events logged."""
        snap = engine.get_profile_snapshot('u1')
        assert snap is None

    def test_snapshot_created_on_emotion_event(self, engine):
        """Logging an emotion_classified event creates a snapshot."""
        engine.log_event(
            user_id='u1',
            domain='emotional',
            event_type='emotion_classified',
            payload={'family': 'joy'},
        )
        snap = engine.get_profile_snapshot('u1')
        assert snap is not None
        assert isinstance(snap, ProfileSnapshot)
        assert snap.user_id == 'u1'
        assert snap.dominant_family == 'joy'
        assert snap.event_count == 1

    def test_non_emotion_event_no_snapshot(self, engine):
        """Non-emotion events do not create a snapshot."""
        engine.log_event(
            user_id='u1',
            domain='behavioral',
            event_type='session_start',
            payload={'duration': 10},
        )
        snap = engine.get_profile_snapshot('u1')
        assert snap is None


# ─── TestProfileSnapshotPrior ─────────────────────────────────────────────────

class TestProfileSnapshotPrior:
    """Verify emotion_prior distribution accuracy."""

    @pytest.fixture
    def engine(self, tmp_path):
        db_path = str(tmp_path / 'test.db')
        eng = UserProfileEngine(db_path=db_path)
        yield eng
        eng.close()

    def test_single_family_prior(self, engine):
        """Single family should have prior of 1.0."""
        engine.log_event('u1', 'emotional', 'emotion_classified', {'family': 'joy'})
        snap = engine.get_profile_snapshot('u1')
        assert snap.emotion_prior == {'joy': 1.0}

    def test_multi_family_prior(self, engine):
        """Multiple families should have proportional priors."""
        engine.log_event('u1', 'emotional', 'emotion_classified', {'family': 'joy'})
        engine.log_event('u1', 'emotional', 'emotion_classified', {'family': 'joy'})
        engine.log_event('u1', 'emotional', 'emotion_classified', {'family': 'sadness'})
        snap = engine.get_profile_snapshot('u1')
        assert snap.emotion_prior['joy'] == pytest.approx(2 / 3)
        assert snap.emotion_prior['sadness'] == pytest.approx(1 / 3)
        assert snap.dominant_family == 'joy'
        assert snap.event_count == 3

    def test_prior_sums_to_one(self, engine):
        """Priors should always sum to 1.0."""
        for fam in ['joy', 'sadness', 'anger', 'fear', 'surprise']:
            engine.log_event('u1', 'emotional', 'emotion_classified', {'family': fam})
        snap = engine.get_profile_snapshot('u1')
        assert sum(snap.emotion_prior.values()) == pytest.approx(1.0)


# ─── TestProfileSnapshotRestart ───────────────────────────────────────────────

class TestProfileSnapshotRestart:
    """Verify snapshot rebuild from DB after engine restart."""

    @pytest.fixture
    def engine(self, tmp_path):
        db_path = str(tmp_path / 'test.db')
        eng = UserProfileEngine(db_path=db_path)
        yield eng
        eng.close()

    def test_rebuild_from_db(self, tmp_path):
        """Snapshot should rebuild from DB events on a fresh engine."""
        db_path = str(tmp_path / 'rebuild.db')
        eng1 = UserProfileEngine(db_path=db_path)
        eng1.log_event('u1', 'emotional', 'emotion_classified', {'family': 'joy'})
        eng1.log_event('u1', 'emotional', 'emotion_classified', {'family': 'sadness'})
        eng1.close()

        eng2 = UserProfileEngine(db_path=db_path)
        try:
            snap = eng2.get_profile_snapshot('u1')
            assert snap is not None
            assert snap.event_count == 2
            assert snap.emotion_prior['joy'] == pytest.approx(0.5)
            assert snap.emotion_prior['sadness'] == pytest.approx(0.5)
        finally:
            eng2.close()

    def test_rebuild_returns_none_for_no_events(self, tmp_path):
        """Rebuild returns None when there are no emotion events in DB."""
        db_path = str(tmp_path / 'empty.db')
        eng1 = UserProfileEngine(db_path=db_path)
        eng1.log_event('u1', 'behavioral', 'session_start', {'x': 1})
        eng1.close()

        eng2 = UserProfileEngine(db_path=db_path)
        try:
            snap = eng2.get_profile_snapshot('u1')
            assert snap is None
        finally:
            eng2.close()


# ─── TestProfileSnapshotThreadSafety ──────────────────────────────────────────

class TestProfileSnapshotThreadSafety:
    """Verify thread-safe concurrent snapshot updates."""

    @pytest.fixture
    def engine(self, tmp_path):
        db_path = str(tmp_path / 'test.db')
        eng = UserProfileEngine(db_path=db_path)
        yield eng
        eng.close()

    def test_concurrent_updates(self, engine):
        """4 threads x 25 events should result in 100 total events."""
        families = ['joy', 'sadness', 'anger', 'fear']
        errors = []

        def worker(family):
            try:
                for _ in range(25):
                    engine.log_event(
                        'u1', 'emotional', 'emotion_classified',
                        {'family': family},
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f,)) for f in families]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        snap = engine.get_profile_snapshot('u1')
        assert snap is not None
        assert snap.event_count == 100
        assert sum(snap.family_counts.values()) == 100
        assert sum(snap.emotion_prior.values()) == pytest.approx(1.0)


# ─── TestProfileSnapshotDeleteUser ────────────────────────────────────────────

class TestProfileSnapshotDeleteUser:
    """Verify delete_user clears snapshot."""

    @pytest.fixture
    def engine(self, tmp_path):
        db_path = str(tmp_path / 'test.db')
        eng = UserProfileEngine(db_path=db_path)
        yield eng
        eng.close()

    def test_delete_clears_snapshot(self, engine):
        """Deleting a user should clear their snapshot."""
        engine.log_event('u1', 'emotional', 'emotion_classified', {'family': 'joy'})
        snap = engine.get_profile_snapshot('u1')
        assert snap is not None

        engine.delete_user('u1')
        snap = engine.get_profile_snapshot('u1')
        assert snap is None

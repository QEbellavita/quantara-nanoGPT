"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Profile Retention Manager Tests
===============================================================================
Tests for RetentionManager: tiered aggregation and storage-ceiling enforcement.

Integrates with:
- Neural Workflow AI Engine
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
===============================================================================
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from profile_db import ProfileDB
from profile_retention import RetentionManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    pdb = ProfileDB(db_path=str(tmp_path / "retention_test.db"))
    yield pdb
    pdb.close()


@pytest.fixture
def manager(db):
    return RetentionManager(db, ceiling_per_user=500_000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_events(db, user_id: str, count: int, age_seconds: float, domain: str = "emotion"):
    """Log *count* events with timestamps ``age_seconds`` ago."""
    ts = time.time() - age_seconds
    for i in range(count):
        # Spread across the window so they land in different buckets if desired
        db.log_event(
            user_id=user_id,
            domain=domain,
            event_type=f"test_event_{i}",
            payload={"index": i},
            source="batch",
            confidence=0.8,
            timestamp=ts + i,  # 1-second offsets
            wait=True,
        )


# ---------------------------------------------------------------------------
# Tests — aggregation of old events
# ---------------------------------------------------------------------------

class TestAggregateOldEvents:
    """Running run_aggregation on a user with old events produces summaries."""

    def test_aggregate_40_day_old_events(self, db):
        """100 events older than 30 days should be collapsed to aggregates."""
        user_id = "user_aggregate_40d"
        age = 40 * 24 * 3600  # 40 days in seconds

        _log_events(db, user_id, 100, age)
        before_count = db.get_event_count(user_id)
        assert before_count == 100

        manager = RetentionManager(db)
        manager.run_aggregation(user_id)

        after_count = db.get_event_count(user_id)
        # Some aggregated summary events must have been created
        assert after_count > 0, "Should still have summary events after aggregation"
        # Raw events (100) should have been replaced with far fewer summaries
        assert after_count < before_count, (
            "Aggregation should reduce the total number of stored events"
        )

    def test_aggregate_creates_summary_events(self, db):
        """After aggregation there should be events with '_aggregate' in type."""
        user_id = "user_summary_check"
        age = 45 * 24 * 3600  # 45 days

        _log_events(db, user_id, 50, age, domain="biometric")

        manager = RetentionManager(db)
        manager.run_aggregation(user_id)

        events = db.get_events(user_id, limit=1000)
        aggregate_events = [e for e in events if "_aggregate" in e.get("event_type", "")]
        assert len(aggregate_events) > 0, "Expected at least one aggregate summary event"

    def test_recent_events_are_not_touched(self, db):
        """Events within the last 30 days must not be aggregated."""
        user_id = "user_recent_safe"
        _log_events(db, user_id, 20, age_seconds=1 * 24 * 3600)  # 1 day old

        manager = RetentionManager(db)
        manager.run_aggregation(user_id)

        count = db.get_event_count(user_id)
        assert count == 20, "Recent events (< 30 days) should remain untouched"

    def test_aggregate_multi_domain(self, db):
        """Events from different domains are aggregated independently."""
        user_id = "user_multi_domain"
        age = 60 * 24 * 3600
        for domain in ("emotion", "biometric", "social"):
            _log_events(db, user_id, 30, age, domain=domain)

        before = db.get_event_count(user_id)
        assert before == 90

        manager = RetentionManager(db)
        manager.run_aggregation(user_id)

        after = db.get_event_count(user_id)
        assert after < before

    def test_no_events_is_safe(self, db):
        """run_aggregation on a user with no events should not raise."""
        manager = RetentionManager(db)
        manager.run_aggregation("ghost_user")  # should not raise


# ---------------------------------------------------------------------------
# Tests — storage ceiling enforcement
# ---------------------------------------------------------------------------

class TestStorageCeilingEnforcement:
    """enforce_ceiling triggers aggregation when events exceed the cap."""

    def test_ceiling_triggers_aggregation(self, db):
        """60 events with ceiling=50 should cause aggregation of old events."""
        user_id = "user_ceiling_50"
        manager = RetentionManager(db, ceiling_per_user=50)

        # Log 60 events — 50 old (> 7 days), 10 recent
        _log_events(db, user_id, 50, age_seconds=10 * 24 * 3600)  # 10 days old
        _log_events(db, user_id, 10, age_seconds=1 * 24 * 3600)   # 1 day old

        before = db.get_event_count(user_id)
        assert before == 60

        manager.enforce_ceiling(user_id)

        after = db.get_event_count(user_id)
        assert after <= 50, (
            f"After enforce_ceiling with cap=50, expected ≤50 events, got {after}"
        )

    def test_under_ceiling_is_noop(self, db):
        """If count is within ceiling no aggregation occurs."""
        user_id = "user_under_ceiling"
        manager = RetentionManager(db, ceiling_per_user=500)

        _log_events(db, user_id, 10, age_seconds=30 * 24 * 3600)
        manager.enforce_ceiling(user_id)

        # Count must remain 10 — nothing was aggregated
        assert db.get_event_count(user_id) == 10

    def test_ceiling_attribute_respected(self, db):
        """RetentionManager stores and uses the supplied ceiling value."""
        manager = RetentionManager(db, ceiling_per_user=1234)
        assert manager.ceiling_per_user == 1234

    def test_default_ceiling(self, db):
        """Default ceiling is 500,000."""
        manager = RetentionManager(db)
        assert manager.ceiling_per_user == 500_000

    def test_large_overage_is_reduced(self, db):
        """With low ceiling and many old events the count drops substantially."""
        user_id = "user_large_overage"
        manager = RetentionManager(db, ceiling_per_user=5)

        _log_events(db, user_id, 100, age_seconds=20 * 24 * 3600)
        before = db.get_event_count(user_id)
        assert before == 100

        manager.enforce_ceiling(user_id)

        after = db.get_event_count(user_id)
        assert after < before, "Event count should decrease after ceiling enforcement"


# ---------------------------------------------------------------------------
# unittest-style fallback (allows running with python -m unittest)
# ---------------------------------------------------------------------------

class TestAggregateOldEventsUnittest(unittest.TestCase):
    """Mirror of TestAggregateOldEvents using unittest so the file is runnable
    without pytest if needed."""

    def setUp(self):
        import tempfile, pathlib
        self._tmpdir = tempfile.mkdtemp()
        self.db = ProfileDB(db_path=os.path.join(self._tmpdir, "test.db"))
        self.manager = RetentionManager(self.db)

    def tearDown(self):
        self.db.close()

    def test_aggregate_creates_fewer_events(self):
        user_id = "ut_aggregate_fewer"
        _log_events(self.db, user_id, 100, age_seconds=40 * 24 * 3600)
        before = self.db.get_event_count(user_id)
        self.manager.run_aggregation(user_id)
        after = self.db.get_event_count(user_id)
        self.assertGreater(before, after)


class TestStorageCeilingEnforcementUnittest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self.db = ProfileDB(db_path=os.path.join(self._tmpdir, "test_ceil.db"))

    def tearDown(self):
        self.db.close()

    def test_enforce_ceiling_reduces_count(self):
        user_id = "ut_ceiling"
        manager = RetentionManager(self.db, ceiling_per_user=50)
        _log_events(self.db, user_id, 60, age_seconds=10 * 24 * 3600)
        manager.enforce_ceiling(user_id)
        after = self.db.get_event_count(user_id)
        self.assertLessEqual(after, 50)


if __name__ == "__main__":
    unittest.main()

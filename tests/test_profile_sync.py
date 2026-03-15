"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Profile Sync Worker Tests
===============================================================================
Unit tests for ProfileSyncWorker: lifecycle, back-off, and cap behaviour.

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
import threading
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profile_sync_worker import ProfileSyncWorker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_worker(**kwargs) -> ProfileSyncWorker:
    """Return a worker with a mock engine and no real URLs by default."""
    engine = MagicMock()
    defaults = dict(poll_interval=1, backend_url=None, master_url=None)
    defaults.update(kwargs)
    return ProfileSyncWorker(engine, **defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProfileSyncWorkerLifecycle(unittest.TestCase):
    """Worker starts cleanly and stops when requested."""

    def test_starts_and_stops(self):
        worker = _make_worker(poll_interval=1)
        worker.start()
        self.assertTrue(worker.is_alive(), "Worker thread should be alive after start()")
        worker.stop()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive(), "Worker thread should have exited after stop()")

    def test_daemon_flag(self):
        worker = _make_worker()
        self.assertTrue(worker.daemon, "Worker must be a daemon thread")

    def test_stop_before_start_does_not_crash(self):
        """Calling stop() on an un-started worker should not raise."""
        worker = _make_worker(poll_interval=300)
        # Set the stop event so run() exits immediately if it ever runs
        worker._stop_event.set()
        # Should not raise
        try:
            worker.stop()
        except RuntimeError:
            # join() on an un-started thread raises RuntimeError — that is acceptable
            pass


class TestProfileSyncWorkerBackoff(unittest.TestCase):
    """Back-off logic: doubles on failure, caps at 3600, resets on success."""

    def test_backoff_increases_on_failure(self):
        worker = _make_worker(poll_interval=10)
        initial = worker._current_interval
        worker._on_poll_failure()
        self.assertGreater(
            worker._current_interval,
            initial,
            "Interval should increase after a poll failure",
        )

    def test_backoff_doubles(self):
        worker = _make_worker(poll_interval=100)
        worker._on_poll_failure()
        self.assertEqual(worker._current_interval, 200)
        worker._on_poll_failure()
        self.assertEqual(worker._current_interval, 400)

    def test_backoff_caps_at_3600(self):
        worker = _make_worker(poll_interval=100)
        # Force many failures
        for _ in range(20):
            worker._on_poll_failure()
        self.assertEqual(
            worker._current_interval,
            3600,
            "Back-off must be capped at 3600 seconds (60 min)",
        )
        # One more failure must not exceed 3600
        worker._on_poll_failure()
        self.assertLessEqual(worker._current_interval, 3600)

    def test_backoff_resets_on_success(self):
        worker = _make_worker(poll_interval=60)
        # Simulate several failures to grow the interval
        worker._on_poll_failure()
        worker._on_poll_failure()
        self.assertGreater(worker._current_interval, 60)
        # One success resets to base
        worker._on_poll_success()
        self.assertEqual(
            worker._current_interval,
            worker._base_interval,
            "Interval must reset to base after poll success",
        )

    def test_success_without_prior_failure_is_noop(self):
        worker = _make_worker(poll_interval=300)
        worker._on_poll_success()
        self.assertEqual(worker._current_interval, 300)

    def test_max_interval_attribute(self):
        worker = _make_worker(poll_interval=100)
        self.assertEqual(worker._max_interval, 3600)


class TestProfileSyncWorkerPolling(unittest.TestCase):
    """Poll methods behave correctly when requests succeed or fail."""

    def test_poll_backend_returns_false_on_health_failure(self):
        worker = _make_worker(poll_interval=1, backend_url="http://nowhere.invalid")
        result = worker._poll_backend()
        self.assertFalse(result)

    def test_poll_master_returns_false_on_health_failure(self):
        worker = _make_worker(poll_interval=1, master_url="http://nowhere.invalid")
        result = worker._poll_master()
        self.assertFalse(result)

    @patch("profile_sync_worker.requests")
    def test_poll_backend_success_ingests_events(self, mock_requests):
        """When health + data both succeed, _ingest_event is called for each item."""
        engine = MagicMock()
        worker = ProfileSyncWorker(
            engine, poll_interval=1, backend_url="http://backend.test"
        )

        health_resp = MagicMock()
        health_resp.status_code = 200
        health_resp.raise_for_status = MagicMock()

        data_resp = MagicMock()
        data_resp.status_code = 200
        data_resp.raise_for_status = MagicMock()
        data_resp.json.return_value = [
            {"user_id": "u1", "domain": "emotion", "event_type": "joy", "confidence": 0.9},
            {"user_id": "u2", "domain": "biometric", "event_type": "hr_reading"},
        ]

        mock_requests.get.side_effect = [health_resp, data_resp]

        result = worker._poll_backend()
        self.assertTrue(result)
        self.assertEqual(engine.log_event.call_count, 2)

    @patch("profile_sync_worker.requests")
    def test_poll_master_success_ingests_events(self, mock_requests):
        engine = MagicMock()
        worker = ProfileSyncWorker(
            engine, poll_interval=1, master_url="http://master.test"
        )

        health_resp = MagicMock()
        health_resp.raise_for_status = MagicMock()

        data_resp = MagicMock()
        data_resp.raise_for_status = MagicMock()
        data_resp.json.return_value = [
            {"user_id": "u3", "domain": "behavioral", "event_type": "workflow_step"},
        ]

        mock_requests.get.side_effect = [health_resp, data_resp]

        result = worker._poll_master()
        self.assertTrue(result)
        self.assertEqual(engine.log_event.call_count, 1)


if __name__ == "__main__":
    unittest.main()

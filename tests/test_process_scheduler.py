"""
QUANTARA NEURAL ECOSYSTEM - Process Scheduler

Unit tests for ProcessScheduler: debounce, count threshold, and
independent-user behaviour.  All timers are kept intentionally short so
the suite completes in a few seconds.
"""

import threading
import time
import unittest

from process_scheduler import ProcessScheduler


class TestProcessSchedulerDebounce(unittest.TestCase):
    """Debounce fires after inactivity window elapses."""

    def test_debounce_triggers_after_delay(self):
        """After 0.2 s debounce, a single event should fire within 0.4 s."""
        calls = []
        scheduler = ProcessScheduler(
            process_fn=lambda uid: calls.append(uid),
            debounce_seconds=0.2,
            count_threshold=1000,   # effectively disabled
            periodic_seconds=9999,  # effectively disabled
        )
        scheduler.start()
        try:
            scheduler.notify_event("user_A")
            time.sleep(0.4)          # well past the 0.2 s debounce window
            self.assertIn("user_A", calls, "Debounce should have fired for user_A")
        finally:
            scheduler.stop()

    def test_debounce_resets_on_new_event(self):
        """New events keep resetting the debounce window."""
        calls = []
        scheduler = ProcessScheduler(
            process_fn=lambda uid: calls.append(uid),
            debounce_seconds=0.3,
            count_threshold=1000,
            periodic_seconds=9999,
        )
        scheduler.start()
        try:
            # Send an event, then immediately send another before the window
            # expires — the debounce clock should reset.
            scheduler.notify_event("user_B")
            time.sleep(0.15)                    # < debounce_seconds
            scheduler.notify_event("user_B")    # resets the clock
            time.sleep(0.15)                    # total ~0.30 s from last event

            # Should NOT have fired yet (last event was only 0.15 s ago)
            self.assertEqual(
                calls.count("user_B"),
                0,
                "Debounce should not have fired: window was just reset",
            )

            # Now wait for the full window to expire after the second event
            time.sleep(0.25)
            self.assertIn("user_B", calls, "Debounce should have fired after reset")
        finally:
            scheduler.stop()


class TestProcessSchedulerCountThreshold(unittest.TestCase):
    """Count threshold fires immediately without waiting for the debounce window."""

    def test_count_threshold_triggers_immediately(self):
        """Reaching threshold=5 should fire synchronously inside notify_event."""
        calls = []
        fired_event = threading.Event()

        def process_fn(uid):
            calls.append(uid)
            fired_event.set()

        scheduler = ProcessScheduler(
            process_fn=process_fn,
            debounce_seconds=9999,   # effectively disabled
            count_threshold=5,
            periodic_seconds=9999,
        )
        scheduler.start()
        try:
            for _ in range(4):
                scheduler.notify_event("user_C")
            self.assertEqual(len(calls), 0, "Should not fire before threshold")

            scheduler.notify_event("user_C")     # 5th event — hits threshold
            fired = fired_event.wait(timeout=0.5)
            self.assertTrue(fired, "process_fn should have been called at threshold")
            self.assertIn("user_C", calls)
        finally:
            scheduler.stop()

    def test_count_resets_after_threshold(self):
        """After a threshold fire the count resets; re-accumulating reaches it again."""
        calls = []

        scheduler = ProcessScheduler(
            process_fn=lambda uid: calls.append(uid),
            debounce_seconds=9999,
            count_threshold=3,
            periodic_seconds=9999,
        )
        scheduler.start()
        try:
            for _ in range(3):
                scheduler.notify_event("user_D")
            time.sleep(0.1)
            first_count = calls.count("user_D")
            self.assertEqual(first_count, 1, "Should fire exactly once at first threshold")

            for _ in range(3):
                scheduler.notify_event("user_D")
            time.sleep(0.1)
            self.assertEqual(calls.count("user_D"), 2, "Should fire again at second threshold")
        finally:
            scheduler.stop()


class TestProcessSchedulerIndependentUsers(unittest.TestCase):
    """Events for different users are tracked and processed independently."""

    def test_different_users_are_independent(self):
        """user_X reaching threshold must not affect user_Y's state."""
        calls = []
        scheduler = ProcessScheduler(
            process_fn=lambda uid: calls.append(uid),
            debounce_seconds=9999,
            count_threshold=3,
            periodic_seconds=9999,
        )
        scheduler.start()
        try:
            # user_X hits threshold
            for _ in range(3):
                scheduler.notify_event("user_X")
            time.sleep(0.1)
            self.assertIn("user_X", calls)
            self.assertNotIn("user_Y", calls, "user_Y should not have been processed")

            # user_Y accumulates events independently
            for _ in range(2):
                scheduler.notify_event("user_Y")
            time.sleep(0.1)
            self.assertNotIn("user_Y", calls, "user_Y count=2 < threshold=3, should not fire")

            scheduler.notify_event("user_Y")  # 3rd event hits threshold
            time.sleep(0.1)
            self.assertIn("user_Y", calls, "user_Y should have fired at threshold")
        finally:
            scheduler.stop()

    def test_debounce_independent_per_user(self):
        """Debounce windows for separate users do not interfere with each other."""
        calls = []
        scheduler = ProcessScheduler(
            process_fn=lambda uid: calls.append(uid),
            debounce_seconds=0.2,
            count_threshold=1000,
            periodic_seconds=9999,
        )
        scheduler.start()
        try:
            scheduler.notify_event("user_E")
            time.sleep(0.05)
            scheduler.notify_event("user_F")

            # After 0.4 s, both should have fired
            time.sleep(0.4)
            self.assertIn("user_E", calls)
            self.assertIn("user_F", calls)
        finally:
            scheduler.stop()


class TestProcessSchedulerErrorHandling(unittest.TestCase):
    """Errors in process_fn are caught; other users continue to be processed."""

    def test_exception_in_process_fn_does_not_crash(self):
        """A raising process_fn must not propagate and crash the scheduler."""
        good_calls = []

        def flaky_fn(uid):
            if uid == "bad_user":
                raise RuntimeError("simulated failure")
            good_calls.append(uid)

        scheduler = ProcessScheduler(
            process_fn=flaky_fn,
            debounce_seconds=0.2,
            count_threshold=1000,
            periodic_seconds=9999,
        )
        scheduler.start()
        try:
            scheduler.notify_event("bad_user")
            scheduler.notify_event("good_user")
            time.sleep(0.5)
            self.assertIn(
                "good_user",
                good_calls,
                "good_user should still be processed even if bad_user raised",
            )
        finally:
            scheduler.stop()


if __name__ == "__main__":
    unittest.main()

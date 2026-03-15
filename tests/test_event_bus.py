"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Profile Event Bus
===============================================================================
Unit tests for ProfileEventBus: TopicMatcher pattern semantics and full
pub/sub lifecycle covering sync delivery, async delivery, wildcard routing,
unsubscribe, error isolation, and multi-subscriber fan-out.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.
===============================================================================
"""

import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from profile_event_bus import ProfileEventBus, TopicMatcher


# ---------------------------------------------------------------------------
# TopicMatcher tests
# ---------------------------------------------------------------------------


class TestTopicMatcher(unittest.TestCase):
    """Tests for TopicMatcher.matches() glob pattern semantics."""

    def test_exact_match(self):
        """An exact topic string matches its identical pattern."""
        self.assertTrue(TopicMatcher.matches("profile.updated", "profile.updated"))

    def test_no_match(self):
        """A pattern that does not match returns False."""
        self.assertFalse(TopicMatcher.matches("profile.updated", "profile.deleted"))

    def test_wildcard_event_star(self):
        """'event.*' matches 'event.created' but not a bare 'event'."""
        self.assertTrue(TopicMatcher.matches("event.*", "event.created"))
        self.assertTrue(TopicMatcher.matches("event.*", "event.deleted"))
        self.assertFalse(TopicMatcher.matches("event.*", "event"))

    def test_different_prefix_no_match(self):
        """A pattern with one prefix does not match a different prefix."""
        self.assertFalse(TopicMatcher.matches("event.*", "profile.created"))

    def test_profile_domain_wildcard(self):
        """'profile.domain.*' matches third-level topics under that prefix."""
        self.assertTrue(TopicMatcher.matches("profile.domain.*", "profile.domain.update"))
        self.assertTrue(TopicMatcher.matches("profile.domain.*", "profile.domain.sync"))
        self.assertFalse(TopicMatcher.matches("profile.domain.*", "emotion.domain.update"))

    def test_full_wildcard(self):
        """'*' matches any single-segment or multi-segment topic via fnmatch."""
        self.assertTrue(TopicMatcher.matches("*", "anything"))
        self.assertTrue(TopicMatcher.matches("*", "profile.updated"))
        self.assertTrue(TopicMatcher.matches("*", "a.b.c.d"))


# ---------------------------------------------------------------------------
# ProfileEventBus tests
# ---------------------------------------------------------------------------


class TestProfileEventBus(unittest.TestCase):
    """Tests for ProfileEventBus pub/sub lifecycle."""

    def setUp(self):
        self.bus = ProfileEventBus()

    def tearDown(self):
        self.bus.shutdown()

    # --- sync delivery ---

    def test_sync_subscriber_receives_event(self):
        """A sync subscriber's callback is invoked with the correct topic and payload."""
        received = []

        def handler(topic, payload):
            received.append((topic, payload))

        self.bus.subscribe("profile.updated", handler, mode="sync")
        self.bus.publish("profile.updated", {"user_id": "u1", "score": 0.9})

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][0], "profile.updated")
        self.assertEqual(received[0][1]["user_id"], "u1")

    def test_wildcard_subscriber_receives_matching_events(self):
        """A wildcard pattern routes matching topics to the subscriber."""
        received_topics = []

        def handler(topic, payload):
            received_topics.append(topic)

        self.bus.subscribe("profile.*", handler, mode="sync")

        self.bus.publish("profile.updated", {})
        self.bus.publish("profile.deleted", {})
        self.bus.publish("emotion.spike", {})  # should NOT be received

        self.assertIn("profile.updated", received_topics)
        self.assertIn("profile.deleted", received_topics)
        self.assertNotIn("emotion.spike", received_topics)

    # --- async delivery ---

    def test_async_subscriber_receives_event(self):
        """An async subscriber's callback is invoked on its worker thread."""
        received_event = threading.Event()
        received = []

        def handler(topic, payload):
            received.append((topic, payload))
            received_event.set()

        self.bus.subscribe("emotion.spike", handler, mode="async")
        self.bus.publish("emotion.spike", {"intensity": 0.95})

        triggered = received_event.wait(timeout=3.0)
        self.assertTrue(triggered, "Async callback was not invoked within timeout")
        self.assertEqual(received[0][0], "emotion.spike")
        self.assertEqual(received[0][1]["intensity"], 0.95)

    # --- unsubscribe ---

    def test_unsubscribe_stops_delivery(self):
        """After unsubscribing, the callback is no longer invoked."""
        received = []

        def handler(topic, payload):
            received.append((topic, payload))

        sub_id = self.bus.subscribe("profile.updated", handler, mode="sync")
        self.bus.publish("profile.updated", {"seq": 1})  # should be received

        self.bus.unsubscribe(sub_id)
        self.bus.publish("profile.updated", {"seq": 2})  # should NOT be received

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][1]["seq"], 1)

    def test_unsubscribe_unknown_id_is_silent(self):
        """Calling unsubscribe with a non-existent ID does not raise."""
        try:
            self.bus.unsubscribe("non-existent-id-00000")
        except Exception as exc:
            self.fail(f"unsubscribe raised unexpectedly: {exc}")

    # --- error isolation ---

    def test_subscriber_exception_does_not_break_others(self):
        """If one sync subscriber raises, other subscribers still receive the event."""
        second_received = []

        def bad_handler(topic, payload):
            raise RuntimeError("intentional failure")

        def good_handler(topic, payload):
            second_received.append((topic, payload))

        self.bus.subscribe("profile.updated", bad_handler, mode="sync")
        self.bus.subscribe("profile.updated", good_handler, mode="sync")

        self.bus.publish("profile.updated", {"data": "test"})

        self.assertEqual(len(second_received), 1)
        self.assertEqual(second_received[0][1]["data"], "test")

    # --- multiple subscribers ---

    def test_multiple_subscribers_same_topic(self):
        """All subscribers for the same topic each receive the published event."""
        results = {1: [], 2: [], 3: []}

        for idx in (1, 2, 3):
            def make_handler(i):
                def handler(topic, payload):
                    results[i].append(payload)
                return handler
            self.bus.subscribe("sensor.reading", make_handler(idx), mode="sync")

        self.bus.publish("sensor.reading", {"value": 42})

        for idx in (1, 2, 3):
            self.assertEqual(len(results[idx]), 1, f"Subscriber {idx} did not receive event")
            self.assertEqual(results[idx][0]["value"], 42)

    # --- async unsubscribe ---

    def test_async_unsubscribe_stops_delivery(self):
        """After unsubscribing an async subscriber it stops receiving further events."""
        received = []
        first_event = threading.Event()

        def handler(topic, payload):
            received.append(payload)
            first_event.set()

        sub_id = self.bus.subscribe("profile.sync", handler, mode="async")
        self.bus.publish("profile.sync", {"seq": 1})

        # Wait for the first delivery to confirm the worker is running.
        self.assertTrue(first_event.wait(timeout=3.0), "First async event not received")

        self.bus.unsubscribe(sub_id)
        # Small pause to let the poison pill be processed before publishing again.
        time.sleep(0.05)
        self.bus.publish("profile.sync", {"seq": 2})
        time.sleep(0.1)  # Allow time for any erroneous delivery.

        # Only the first event should have been received.
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["seq"], 1)


if __name__ == "__main__":
    unittest.main()

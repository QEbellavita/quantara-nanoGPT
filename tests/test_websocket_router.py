"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - WebSocket Router
===============================================================================
Unit tests for WebSocketRouter: screen subscription mapping, batch buffering,
connection lifecycle, topic-based routing, screen changes, and disconnection.

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

from profile_event_bus import ProfileEventBus
from websocket_router import (
    BatchBuffer,
    DEFAULT_TOPICS,
    ScreenSubscriptionMap,
    WebSocketRouter,
)


# ---------------------------------------------------------------------------
# ScreenSubscriptionMap tests
# ---------------------------------------------------------------------------


class TestScreenSubscriptionMap(unittest.TestCase):
    """Tests for ScreenSubscriptionMap.get_topics() screen → pattern mapping."""

    def test_dna_strands_returns_domain_wildcard(self):
        topics = ScreenSubscriptionMap.get_topics("dna_strands")
        self.assertIn("profile.domain.*", topics)

    def test_evolution_timeline_returns_stage_and_snapshot(self):
        topics = ScreenSubscriptionMap.get_topics("evolution_timeline")
        self.assertIn("profile.stage.*", topics)
        self.assertIn("profile.snapshot.*", topics)

    def test_backgrounded_returns_empty(self):
        topics = ScreenSubscriptionMap.get_topics("backgrounded")
        self.assertEqual(topics, [])

    def test_unknown_screen_returns_default_topics(self):
        topics = ScreenSubscriptionMap.get_topics("some_unknown_screen_xyz")
        self.assertEqual(topics, DEFAULT_TOPICS)

    def test_coaching_session_topics(self):
        topics = ScreenSubscriptionMap.get_topics("coaching_session")
        self.assertIn("intelligence.coaching", topics)
        self.assertIn("alert.*", topics)

    def test_therapist_dashboard_topics(self):
        topics = ScreenSubscriptionMap.get_topics("therapist_dashboard")
        self.assertIn("profile.updated", topics)
        self.assertIn("alert.*", topics)
        self.assertIn("intelligence.therapy", topics)

    def test_3d_fingerprint_topics(self):
        topics = ScreenSubscriptionMap.get_topics("3d_fingerprint")
        self.assertIn("profile.updated", topics)


# ---------------------------------------------------------------------------
# BatchBuffer tests
# ---------------------------------------------------------------------------


class TestBatchBuffer(unittest.TestCase):
    """Tests for BatchBuffer event buffering, FIFO trimming, flush and expiry."""

    def test_buffers_single_event(self):
        buf = BatchBuffer()
        buf.add("u1", {"msg": "hello"})
        events = buf.flush("u1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["msg"], "hello")

    def test_flush_clears_buffer(self):
        buf = BatchBuffer()
        buf.add("u1", {"seq": 1})
        buf.flush("u1")
        # Second flush should return empty list.
        events = buf.flush("u1")
        self.assertEqual(events, [])

    def test_flush_unknown_user_returns_empty(self):
        buf = BatchBuffer()
        self.assertEqual(buf.flush("nobody"), [])

    def test_max_size_keeps_newest(self):
        buf = BatchBuffer(max_size=3)
        for i in range(6):
            buf.add("u1", {"seq": i})
        events = buf.flush("u1")
        self.assertEqual(len(events), 3)
        # Newest 3 are seq 3, 4, 5.
        seqs = [e["seq"] for e in events]
        self.assertEqual(seqs, [3, 4, 5])

    def test_multiple_users_independent_buffers(self):
        buf = BatchBuffer()
        buf.add("u1", {"x": 1})
        buf.add("u2", {"x": 2})
        u1_events = buf.flush("u1")
        u2_events = buf.flush("u2")
        self.assertEqual(u1_events[0]["x"], 1)
        self.assertEqual(u2_events[0]["x"], 2)

    def test_cleanup_removes_expired_buffers(self):
        buf = BatchBuffer(expire_seconds=1)
        buf.add("u1", {"seq": 1})
        # Manually backdate the created_at timestamp.
        buf._buffers["u1"]["created_at"] = time.time() - 2
        buf.cleanup()
        self.assertEqual(buf.flush("u1"), [])

    def test_cleanup_keeps_fresh_buffers(self):
        buf = BatchBuffer(expire_seconds=300)
        buf.add("u1", {"seq": 1})
        buf.cleanup()
        events = buf.flush("u1")
        self.assertEqual(len(events), 1)


# ---------------------------------------------------------------------------
# WebSocketRouter tests
# ---------------------------------------------------------------------------


class _CapturingRouter(WebSocketRouter):
    """WebSocketRouter subclass that captures emitted events instead of calling socketio."""

    def __init__(self, bus):
        super().__init__(bus, socketio=None)
        self.emitted: list = []

    def _emit(self, sid, event, data, namespace):
        self.emitted.append({"sid": sid, "event": event, "data": data, "namespace": namespace})


class TestWebSocketRouter(unittest.TestCase):
    """Tests for WebSocketRouter routing, screen change, and disconnect."""

    def setUp(self):
        self.bus = ProfileEventBus()
        self.router = _CapturingRouter(self.bus)

    def tearDown(self):
        self.bus.shutdown()

    # --- helper ----------------------------------------------------------

    def _wait_for_emit(self, count: int = 1, timeout: float = 3.0) -> bool:
        """Block until at least `count` events have been emitted or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if len(self.router.emitted) >= count:
                return True
            time.sleep(0.02)
        return False

    # --- routing ---------------------------------------------------------

    def test_routes_event_to_matching_screen(self):
        """An event whose topic matches the screen's patterns is emitted to the sid."""
        self.router.connect("sid-1", "user-A", "3d_fingerprint")
        self.bus.publish("profile.updated", {"user_id": "user-A", "score": 0.8})

        delivered = self._wait_for_emit(1)
        self.assertTrue(delivered, "Event not emitted within timeout")
        self.assertEqual(self.router.emitted[0]["sid"], "sid-1")
        self.assertEqual(self.router.emitted[0]["event"], "profile_event")

    def test_does_not_route_non_matching_topic(self):
        """An event whose topic does NOT match the screen's patterns is not emitted."""
        # evolution_timeline subscribes to profile.stage.* and profile.snapshot.*
        self.router.connect("sid-1", "user-A", "evolution_timeline")
        self.bus.publish("profile.updated", {"user_id": "user-A"})

        # Give the async worker time to process.
        time.sleep(0.3)
        self.assertEqual(len(self.router.emitted), 0)

    def test_backgrounded_screen_receives_no_events(self):
        """A client on the 'backgrounded' screen receives no events."""
        self.router.connect("sid-1", "user-A", "backgrounded")
        self.bus.publish("profile.updated", {"user_id": "user-A"})
        self.bus.publish("alert.urgent", {"user_id": "user-A"})
        time.sleep(0.3)
        self.assertEqual(len(self.router.emitted), 0)

    def test_event_without_user_id_is_dropped(self):
        """Events missing user_id in the payload are silently ignored."""
        self.router.connect("sid-1", "user-A", "3d_fingerprint")
        self.bus.publish("profile.updated", {"score": 0.5})  # no user_id
        time.sleep(0.3)
        self.assertEqual(len(self.router.emitted), 0)

    # --- screen change ---------------------------------------------------

    def test_screen_change_updates_topic_filter(self):
        """After change_screen the connection receives events for the new screen only."""
        self.router.connect("sid-1", "user-A", "evolution_timeline")
        # Switch to 3d_fingerprint which subscribes to profile.updated.
        self.router.change_screen("sid-1", "3d_fingerprint")
        self.bus.publish("profile.updated", {"user_id": "user-A", "v": 2})

        delivered = self._wait_for_emit(1)
        self.assertTrue(delivered, "Event after screen change not emitted")
        self.assertEqual(self.router.emitted[0]["sid"], "sid-1")

    def test_change_screen_to_backgrounded_stops_events(self):
        """Changing to 'backgrounded' screen stops event delivery."""
        self.router.connect("sid-1", "user-A", "3d_fingerprint")
        self.router.change_screen("sid-1", "backgrounded")
        self.bus.publish("profile.updated", {"user_id": "user-A"})
        time.sleep(0.3)
        self.assertEqual(len(self.router.emitted), 0)

    def test_change_screen_unknown_sid_is_safe(self):
        """change_screen with an unknown sid does not raise."""
        try:
            self.router.change_screen("ghost-sid", "3d_fingerprint")
        except Exception as exc:
            self.fail(f"change_screen raised unexpectedly: {exc}")

    # --- disconnect ------------------------------------------------------

    def test_disconnect_cleans_up_connection(self):
        """After disconnect the sid is removed and events are no longer routed."""
        self.router.connect("sid-1", "user-A", "3d_fingerprint")
        self.router.disconnect("sid-1")
        self.bus.publish("profile.updated", {"user_id": "user-A"})
        time.sleep(0.3)
        self.assertEqual(len(self.router.emitted), 0)

    def test_disconnect_removes_user_sid_mapping(self):
        """After the last sid disconnects the user entry is removed from _user_sids."""
        self.router.connect("sid-1", "user-A", "3d_fingerprint")
        self.router.disconnect("sid-1")
        self.assertNotIn("user-A", self.router._user_sids)

    def test_disconnect_unknown_sid_is_safe(self):
        """Calling disconnect with an unknown sid does not raise."""
        try:
            self.router.disconnect("ghost-sid")
        except Exception as exc:
            self.fail(f"disconnect raised unexpectedly: {exc}")

    def test_multiple_sids_same_user_partial_disconnect(self):
        """Disconnecting one sid of a multi-sid user leaves the other active."""
        self.router.connect("sid-1", "user-A", "3d_fingerprint")
        self.router.connect("sid-2", "user-A", "3d_fingerprint")
        self.router.disconnect("sid-1")
        self.assertIn("user-A", self.router._user_sids)
        self.assertIn("sid-2", self.router._user_sids["user-A"])

    # --- buffering -------------------------------------------------------

    def test_events_buffered_for_offline_user_replayed_on_connect(self):
        """Events published before a user connects are buffered and flushed on connect."""
        # Publish before any connection exists.
        self.bus.publish("profile.updated", {"user_id": "user-B", "buffered": True})
        # Give the async worker time to buffer the event.
        time.sleep(0.3)

        # Now connect — buffered events should be replayed.
        self.router.connect("sid-2", "user-B", "3d_fingerprint")
        delivered = self._wait_for_emit(1)
        self.assertTrue(delivered, "Buffered event was not replayed on connect")
        self.assertEqual(self.router.emitted[0]["sid"], "sid-2")


if __name__ == "__main__":
    unittest.main()

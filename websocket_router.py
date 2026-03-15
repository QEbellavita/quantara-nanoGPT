"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - WebSocket Router
===============================================================================
Smart screen-based WebSocket routing with topic filtering and batch buffering.
Routes real-time profile, emotion, and biometric events to connected clients
based on the currently active screen, buffering events for offline sessions.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.
===============================================================================
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set

from profile_event_bus import ProfileEventBus, TopicMatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Screen → topic subscriptions
# ---------------------------------------------------------------------------

SCREEN_TOPICS: Dict[str, List[str]] = {
    "dna_strands": ["profile.domain.*"],
    "evolution_timeline": ["profile.stage.*", "profile.snapshot.*"],
    "3d_fingerprint": ["profile.updated"],
    "coaching_session": ["intelligence.coaching", "alert.*"],
    "therapist_dashboard": ["profile.updated", "alert.*", "intelligence.therapy"],
    "backgrounded": [],
}

DEFAULT_TOPICS: List[str] = ["profile.updated", "alert.*"]


class ScreenSubscriptionMap:
    """Maps screen names to their relevant event topic patterns."""

    @staticmethod
    def get_topics(screen: str) -> List[str]:
        """
        Return the list of topic patterns for the given screen name.

        Falls back to DEFAULT_TOPICS for unrecognised screen names.
        """
        return SCREEN_TOPICS.get(screen, DEFAULT_TOPICS)


# ---------------------------------------------------------------------------
# Batch buffer
# ---------------------------------------------------------------------------


class BatchBuffer:
    """
    Per-user event buffer for clients that are not currently connected.

    Events are buffered up to max_size (oldest are dropped when full) and
    expire after expire_seconds of inactivity so stale data is never replayed.
    """

    def __init__(self, max_size: int = 100, expire_seconds: int = 300) -> None:
        self.max_size = max_size
        self.expire_seconds = expire_seconds
        # {user_id: {'events': [...], 'created_at': float}}
        self._buffers: Dict[str, Dict[str, Any]] = {}

    def add(self, user_id: str, event: Any) -> None:
        """Append an event for user_id, trimming to max_size (keep newest)."""
        if user_id not in self._buffers:
            self._buffers[user_id] = {"events": [], "created_at": time.time()}
        self._buffers[user_id]["events"].append(event)
        # Keep only the newest max_size entries
        if len(self._buffers[user_id]["events"]) > self.max_size:
            self._buffers[user_id]["events"] = self._buffers[user_id]["events"][-self.max_size:]

    def flush(self, user_id: str) -> List[Any]:
        """Return and remove all buffered events for user_id."""
        entry = self._buffers.pop(user_id, None)
        if entry is None:
            return []
        return entry["events"]

    def cleanup(self) -> None:
        """Remove buffers older than expire_seconds."""
        now = time.time()
        expired = [
            uid
            for uid, entry in self._buffers.items()
            if now - entry["created_at"] > self.expire_seconds
        ]
        for uid in expired:
            del self._buffers[uid]
            logger.debug("BatchBuffer: expired buffer for user_id=%s", uid)


# ---------------------------------------------------------------------------
# WebSocket Router
# ---------------------------------------------------------------------------


class WebSocketRouter:
    """
    Routes bus events to connected WebSocket clients based on their active
    screen's topic subscriptions.

    Each socket connection (sid) is associated with a user_id and a screen.
    When the bus emits an event the router checks which connected sids match
    the topic pattern for their current screen and emits only to those clients.
    Events for users with no active connection are buffered and replayed on
    reconnect.
    """

    # Bus topic patterns this router subscribes to.
    _BUS_PATTERNS = ["profile.*", "alert.*", "intelligence.*"]

    def __init__(self, bus: ProfileEventBus, socketio: Any = None) -> None:
        self._bus = bus
        self._socketio = socketio

        # {sid: {'user_id': str, 'screen': str, 'topics': List[str]}}
        self._connections: Dict[str, Dict[str, Any]] = {}
        # {user_id: set(sids)}
        self._user_sids: Dict[str, Set[str]] = {}
        # Pending events for disconnected users
        self._batch_buffer = BatchBuffer()

        # Subscribe to all relevant bus namespaces in async mode so delivery
        # does not block the publishing thread.
        for pattern in self._BUS_PATTERNS:
            self._bus.subscribe(pattern, self._on_bus_event, mode="async")

        logger.info("WebSocketRouter: initialised, listening on patterns=%s", self._BUS_PATTERNS)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, sid: str, user_id: str, screen: str) -> None:
        """
        Register a new WebSocket connection.

        Any events buffered for the user while they were offline are flushed
        and emitted immediately after registration.
        """
        topics = ScreenSubscriptionMap.get_topics(screen)
        self._connections[sid] = {
            "user_id": user_id,
            "screen": screen,
            "topics": topics,
        }
        if user_id not in self._user_sids:
            self._user_sids[user_id] = set()
        self._user_sids[user_id].add(sid)

        logger.info(
            "WebSocketRouter: connect sid=%s user_id=%s screen=%s topics=%s",
            sid, user_id, screen, topics,
        )

        # Replay any buffered events for this user.
        buffered = self._batch_buffer.flush(user_id)
        for event in buffered:
            self._emit(sid, event.get("event", "profile_event"), event.get("data", event), "/")

    def change_screen(self, sid: str, screen: str) -> None:
        """Update the active screen (and therefore the subscribed topics) for a connection."""
        conn = self._connections.get(sid)
        if conn is None:
            logger.warning("WebSocketRouter: change_screen called for unknown sid=%s", sid)
            return
        new_topics = ScreenSubscriptionMap.get_topics(screen)
        conn["screen"] = screen
        conn["topics"] = new_topics
        logger.info(
            "WebSocketRouter: change_screen sid=%s screen=%s topics=%s",
            sid, screen, new_topics,
        )

    def disconnect(self, sid: str) -> None:
        """Remove a WebSocket connection and clean up tracking state."""
        conn = self._connections.pop(sid, None)
        if conn is None:
            logger.debug("WebSocketRouter: disconnect called for unknown sid=%s", sid)
            return
        user_id = conn["user_id"]
        sids = self._user_sids.get(user_id, set())
        sids.discard(sid)
        if not sids:
            self._user_sids.pop(user_id, None)
        logger.info("WebSocketRouter: disconnect sid=%s user_id=%s", sid, user_id)

    # ------------------------------------------------------------------
    # Bus event handler
    # ------------------------------------------------------------------

    def _on_bus_event(self, topic: str, payload: dict) -> None:
        """
        Invoked for every event the bus delivers to this router.

        Looks up the user_id from the payload, finds all connected sids for
        that user, checks each sid's topic patterns and emits when matched.
        Events for users with no active sids are buffered.
        """
        user_id = payload.get("user_id")
        if not user_id:
            logger.debug("WebSocketRouter: dropping event with no user_id (topic=%s)", topic)
            return

        sids = self._user_sids.get(user_id, set())
        if not sids:
            # No connected clients — buffer the event for later replay.
            self._batch_buffer.add(user_id, {"event": "profile_event", "data": payload, "topic": topic})
            logger.debug(
                "WebSocketRouter: buffering event for offline user_id=%s topic=%s", user_id, topic
            )
            return

        for sid in list(sids):
            conn = self._connections.get(sid)
            if conn is None:
                continue
            # Check whether any of this connection's topic patterns match the event.
            if any(TopicMatcher.matches(pattern, topic) for pattern in conn["topics"]):
                self._emit(sid, "profile_event", payload, "/")
            else:
                logger.debug(
                    "WebSocketRouter: sid=%s skipped (screen=%s, topic=%s not in %s)",
                    sid, conn["screen"], topic, conn["topics"],
                )

    # ------------------------------------------------------------------
    # Emit helper (override in tests / subclasses)
    # ------------------------------------------------------------------

    def _emit(self, sid: str, event: str, data: Any, namespace: str) -> None:
        """Emit a SocketIO event to a single client. Override in tests."""
        if self._socketio is not None:
            try:
                self._socketio.emit(event, data, to=sid, namespace=namespace)
            except Exception:
                logger.exception(
                    "WebSocketRouter: emit failed sid=%s event=%s", sid, event
                )
        else:
            logger.debug(
                "WebSocketRouter: no socketio, would emit event=%s sid=%s data=%s",
                event, sid, data,
            )

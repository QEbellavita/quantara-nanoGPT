"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Profile Event Bus
===============================================================================
Core pub/sub event bus with synchronous and asynchronous delivery modes.
Foundation layer for all real-time profile, emotion, and biometric event
routing across the Quantara Neural Ecosystem.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.
===============================================================================
"""

import fnmatch
import logging
import queue
import threading
import uuid
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class TopicMatcher:
    """Static utility for matching topic strings against glob-style patterns."""

    @staticmethod
    def matches(pattern: str, topic: str) -> bool:
        """
        Return True if `topic` matches `pattern` using fnmatch glob semantics.

        Examples:
            matches('event.*', 'event.created')  -> True
            matches('profile.domain.*', 'profile.domain.update') -> True
            matches('*', 'anything.at.all') -> True
        """
        return fnmatch.fnmatch(topic, pattern)


class ProfileEventBus:
    """
    Pub/sub event bus supporting synchronous and asynchronous callback delivery.

    Subscribers register a topic pattern (glob) and a callback. When an event
    is published the bus fans out to every matching subscriber. Sync subscribers
    are called inline on the publishing thread; async subscribers receive events
    on a dedicated daemon worker thread via a per-subscriber queue.
    """

    # Sentinel value used to terminate async worker threads.
    _POISON_PILL = object()

    def __init__(self) -> None:
        # sub_id -> {'pattern': str, 'callback': Callable, 'mode': str,
        #            'queue': Optional[Queue], 'thread': Optional[Thread]}
        self._subscribers: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._running = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(
        self,
        topic_pattern: str,
        callback: Callable,
        mode: str = "sync",
    ) -> str:
        """
        Register a subscriber for events whose topic matches `topic_pattern`.

        Parameters
        ----------
        topic_pattern : str
            Glob-style pattern (e.g. 'profile.*', 'emotion.high', '*').
        callback : Callable
            Called with (topic: str, payload: dict). Must be thread-safe for
            async mode (it runs on a dedicated worker thread).
        mode : str
            'sync'  — callback invoked on the publisher's thread.
            'async' — callback invoked on a dedicated daemon thread.

        Returns
        -------
        str
            Unique subscription ID that can be passed to `unsubscribe()`.
        """
        sub_id = str(uuid.uuid4())
        entry: dict = {
            "pattern": topic_pattern,
            "callback": callback,
            "mode": mode,
            "queue": None,
            "thread": None,
        }

        if mode == "async":
            worker_queue: queue.Queue = queue.Queue()
            thread = threading.Thread(
                target=self._async_worker_loop,
                args=(sub_id, callback, worker_queue),
                daemon=True,
                name=f"event-bus-worker-{sub_id[:8]}",
            )
            entry["queue"] = worker_queue
            entry["thread"] = thread
            thread.start()

        with self._lock:
            self._subscribers[sub_id] = entry

        logger.debug(
            "ProfileEventBus: subscribed id=%s pattern=%r mode=%s",
            sub_id,
            topic_pattern,
            mode,
        )
        return sub_id

    def unsubscribe(self, sub_id: str) -> None:
        """
        Remove a subscriber by its subscription ID.

        For async subscribers the worker thread is stopped gracefully via a
        poison-pill message. This method returns immediately and does not wait
        for the thread to terminate.
        """
        with self._lock:
            entry = self._subscribers.pop(sub_id, None)

        if entry is None:
            logger.debug("ProfileEventBus: unsubscribe called for unknown id=%s", sub_id)
            return

        if entry["mode"] == "async" and entry["queue"] is not None:
            entry["queue"].put(self._POISON_PILL)

        logger.debug("ProfileEventBus: unsubscribed id=%s", sub_id)

    def publish(self, topic: str, payload: dict) -> None:
        """
        Publish an event to all subscribers whose pattern matches `topic`.

        Sync subscribers are called immediately on the calling thread (errors
        are caught and logged so they cannot break delivery to other subscribers).
        Async subscribers receive the event via their worker queue.

        Parameters
        ----------
        topic : str
            The event topic string (e.g. 'profile.updated', 'emotion.spike').
        payload : dict
            Arbitrary event data.
        """
        if not self._running:
            logger.warning("ProfileEventBus: publish called after shutdown, ignoring.")
            return

        with self._lock:
            snapshot = list(self._subscribers.values())

        for entry in snapshot:
            if not TopicMatcher.matches(entry["pattern"], topic):
                continue

            if entry["mode"] == "sync":
                try:
                    entry["callback"](topic, payload)
                except Exception:
                    logger.exception(
                        "ProfileEventBus: sync subscriber raised an exception "
                        "(pattern=%r, topic=%r)", entry["pattern"], topic
                    )
            else:
                if entry["queue"] is not None:
                    entry["queue"].put((topic, payload))

    def shutdown(self) -> None:
        """
        Stop all async worker threads by sending them poison pills.

        After calling `shutdown()` any subsequent `publish()` calls are silently
        dropped. This method does not wait for worker threads to drain their
        queues.
        """
        self._running = False
        with self._lock:
            entries = list(self._subscribers.values())

        for entry in entries:
            if entry["mode"] == "async" and entry["queue"] is not None:
                entry["queue"].put(self._POISON_PILL)

        logger.debug("ProfileEventBus: shutdown complete, %d subscribers notified.", len(entries))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _async_worker_loop(
        self,
        sub_id: str,
        callback: Callable,
        worker_queue: queue.Queue,
    ) -> None:
        """
        Drain `worker_queue` and invoke `callback` for each event.

        Terminates when a poison-pill sentinel is dequeued. Per-event
        exceptions are caught and logged so the worker loop stays alive.
        """
        logger.debug("ProfileEventBus: async worker started for sub_id=%s", sub_id)
        while True:
            item = worker_queue.get()
            if item is self._POISON_PILL:
                logger.debug("ProfileEventBus: async worker stopping for sub_id=%s", sub_id)
                break
            topic, payload = item
            try:
                callback(topic, payload)
            except Exception:
                logger.exception(
                    "ProfileEventBus: async subscriber raised an exception "
                    "(sub_id=%s, topic=%r)", sub_id, topic
                )

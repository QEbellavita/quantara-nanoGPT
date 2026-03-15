"""
QUANTARA NEURAL ECOSYSTEM - Process Scheduler

Schedules when process() runs via debounced, count-threshold,
and periodic triggers. Independent from the event bus.
"""

import threading
import time
import logging

logger = logging.getLogger(__name__)


class ProcessScheduler:
    """
    Schedules process_fn(user_id) calls using three trigger mechanisms:

    1. Debounce: fires after debounce_seconds of inactivity (count > 0).
    2. Count threshold: fires immediately when event count >= count_threshold.
    3. Periodic: fires for all pending users every periodic_seconds.
    """

    def __init__(
        self,
        process_fn,
        debounce_seconds: float = 30.0,
        count_threshold: int = 20,
        periodic_seconds: float = 300.0,
    ):
        self.process_fn = process_fn
        self.debounce_seconds = debounce_seconds
        self.count_threshold = count_threshold
        self.periodic_seconds = periodic_seconds

        # {user_id: {"count": int, "last_event_time": float}}
        self._pending: dict = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._debounce_thread: threading.Thread | None = None
        self._periodic_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start background debounce and periodic threads (both daemon)."""
        self._stop_event.clear()

        self._debounce_thread = threading.Thread(
            target=self._debounce_loop, name="ProcessScheduler-debounce", daemon=True
        )
        self._periodic_thread = threading.Thread(
            target=self._periodic_loop, name="ProcessScheduler-periodic", daemon=True
        )

        self._debounce_thread.start()
        self._periodic_thread.start()
        logger.debug(
            "ProcessScheduler started (debounce=%.1fs, threshold=%d, periodic=%.1fs)",
            self.debounce_seconds,
            self.count_threshold,
            self.periodic_seconds,
        )

    def stop(self):
        """Signal threads to stop and wait for them to finish."""
        self._stop_event.set()
        if self._debounce_thread is not None:
            self._debounce_thread.join()
        if self._periodic_thread is not None:
            self._periodic_thread.join()
        logger.debug("ProcessScheduler stopped.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify_event(self, user_id):
        """
        Record an event for user_id.

        Increments the event count and updates last_event_time.
        If count reaches count_threshold, _process_user is called immediately.
        """
        with self._lock:
            if user_id not in self._pending:
                self._pending[user_id] = {"count": 0, "last_event_time": 0.0}
            self._pending[user_id]["count"] += 1
            self._pending[user_id]["last_event_time"] = time.monotonic()
            current_count = self._pending[user_id]["count"]

        if current_count >= self.count_threshold:
            logger.debug(
                "Count threshold reached for user %s (%d events), processing immediately.",
                user_id,
                current_count,
            )
            self._process_user(user_id)

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    def _debounce_loop(self):
        """Poll every 0.1 s; fire process for users whose inactivity window has elapsed."""
        while not self._stop_event.is_set():
            now = time.monotonic()
            users_to_process = []

            with self._lock:
                for user_id, state in list(self._pending.items()):
                    if state["count"] > 0:
                        elapsed = now - state["last_event_time"]
                        if elapsed >= self.debounce_seconds:
                            users_to_process.append(user_id)

            for user_id in users_to_process:
                logger.debug(
                    "Debounce elapsed for user %s, processing.", user_id
                )
                self._process_user(user_id)

            self._stop_event.wait(timeout=0.1)

    def _periodic_loop(self):
        """Sleep periodic_seconds, then process all users with pending events."""
        while not self._stop_event.is_set():
            # Use wait so stop() wakes us up promptly.
            self._stop_event.wait(timeout=self.periodic_seconds)
            if self._stop_event.is_set():
                break

            with self._lock:
                users_to_process = [
                    uid for uid, state in self._pending.items() if state["count"] > 0
                ]

            if users_to_process:
                logger.debug(
                    "Periodic tick: processing %d pending user(s).", len(users_to_process)
                )
                for user_id in users_to_process:
                    self._process_user(user_id)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _process_user(self, user_id):
        """
        Reset the event count for user_id and invoke process_fn(user_id).

        Errors raised by process_fn are caught and logged so that one
        failing user does not block others.
        """
        with self._lock:
            if user_id not in self._pending or self._pending[user_id]["count"] == 0:
                # Already processed by another trigger; nothing to do.
                return
            self._pending[user_id]["count"] = 0

        try:
            self.process_fn(user_id)
        except Exception:
            logger.exception("process_fn raised an exception for user %s", user_id)

"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Profile Sync Worker
===============================================================================
Background daemon thread that periodically polls the Quantara Backend and
Master APIs for recent emotion events and workflow data, injecting them into
the User Profile Engine.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.
===============================================================================
"""

import logging
import threading
import time
from typing import Optional

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

logger = logging.getLogger(__name__)


class ProfileSyncWorker(threading.Thread):
    """Daemon thread that polls the Backend and Master APIs on a configurable
    interval, logging discovered events into the User Profile Engine.

    Implements exponential back-off on poll failures (doubles each failure,
    capped at ``_max_interval``) and resets to the base interval on success.
    """

    def __init__(
        self,
        engine,
        poll_interval: int = 300,
        backend_url: Optional[str] = None,
        master_url: Optional[str] = None,
    ):
        """Initialise the sync worker.

        Parameters
        ----------
        engine:
            A :class:`UserProfileEngine` instance used to log incoming events.
        poll_interval:
            Seconds between poll cycles in the normal (no-failure) case.
        backend_url:
            Base URL for the Quantara Backend service, e.g.
            ``"http://localhost:3001"``.  When *None* the backend poll is
            skipped.
        master_url:
            Base URL for the Quantara Master service, e.g.
            ``"http://localhost:4000"``.  When *None* the master poll is
            skipped.
        """
        super().__init__(name="ProfileSyncWorker", daemon=True)
        self.engine = engine
        self._base_interval: int = poll_interval
        self._current_interval: int = poll_interval
        self._max_interval: int = 3600
        self._backend_url: Optional[str] = backend_url.rstrip("/") if backend_url else None
        self._master_url: Optional[str] = master_url.rstrip("/") if master_url else None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ #
    # Public lifecycle API                                                 #
    # ------------------------------------------------------------------ #

    def start(self) -> None:  # type: ignore[override]
        """Start the daemon thread."""
        logger.info(
            "ProfileSyncWorker starting (interval=%ds, backend=%s, master=%s)",
            self._base_interval,
            self._backend_url,
            self._master_url,
        )
        super().start()

    def stop(self) -> None:
        """Signal the worker to stop and wait for it to exit cleanly."""
        logger.info("ProfileSyncWorker stop requested")
        self._stop_event.set()
        self.join(timeout=self._current_interval + 5)

    # ------------------------------------------------------------------ #
    # Thread run loop                                                      #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Poll backend and master in a loop until :meth:`stop` is called."""
        logger.info("ProfileSyncWorker run loop started")
        while not self._stop_event.is_set():
            success = True
            if self._backend_url:
                if not self._poll_backend():
                    success = False
            if self._master_url:
                if not self._poll_master():
                    success = False

            if success:
                self._on_poll_success()
            else:
                self._on_poll_failure()

            # Sleep in small increments so stop() is responsive
            deadline = time.monotonic() + self._current_interval
            while not self._stop_event.is_set() and time.monotonic() < deadline:
                self._stop_event.wait(timeout=min(1.0, deadline - time.monotonic()))

        logger.info("ProfileSyncWorker run loop exited")

    # ------------------------------------------------------------------ #
    # Poll helpers                                                         #
    # ------------------------------------------------------------------ #

    def _poll_backend(self) -> bool:
        """Poll the Backend service health endpoint then fetch recent emotion events.

        Returns
        -------
        bool
            ``True`` if both requests succeeded, ``False`` on any error.
        """
        base = self._backend_url
        try:
            health_resp = requests.get(f"{base}/api/health", timeout=5)
            health_resp.raise_for_status()
            logger.debug("Backend health OK (%d)", health_resp.status_code)
        except Exception as exc:
            logger.warning("Backend health check failed: %s", exc)
            return False

        try:
            data_resp = requests.get(f"{base}/api/emotion/recent", timeout=10)
            data_resp.raise_for_status()
            events = data_resp.json()
            if isinstance(events, list):
                logger.info("Backend: received %d recent emotion events", len(events))
                for evt in events:
                    self._ingest_event(evt, source="sync")
            else:
                logger.debug("Backend emotion/recent returned non-list payload")
        except Exception as exc:
            logger.warning("Backend emotion/recent fetch failed: %s", exc)
            return False

        return True

    def _poll_master(self) -> bool:
        """Poll the Master service health endpoint then fetch recent workflow events.

        Returns
        -------
        bool
            ``True`` if both requests succeeded, ``False`` on any error.
        """
        base = self._master_url
        try:
            health_resp = requests.get(f"{base}/api/v1/health", timeout=5)
            health_resp.raise_for_status()
            logger.debug("Master health OK (%d)", health_resp.status_code)
        except Exception as exc:
            logger.warning("Master health check failed: %s", exc)
            return False

        try:
            data_resp = requests.get(f"{base}/api/v1/workflows/recent", timeout=10)
            data_resp.raise_for_status()
            events = data_resp.json()
            if isinstance(events, list):
                logger.info("Master: received %d recent workflow events", len(events))
                for evt in events:
                    self._ingest_event(evt, source="sync")
            else:
                logger.debug("Master workflows/recent returned non-list payload")
        except Exception as exc:
            logger.warning("Master workflows/recent fetch failed: %s", exc)
            return False

        return True

    # ------------------------------------------------------------------ #
    # Back-off helpers                                                     #
    # ------------------------------------------------------------------ #

    def _on_poll_failure(self) -> None:
        """Double the current poll interval, capped at :attr:`_max_interval`."""
        new_interval = min(self._current_interval * 2, self._max_interval)
        if new_interval != self._current_interval:
            logger.warning(
                "ProfileSyncWorker: poll failure — backing off from %ds to %ds",
                self._current_interval,
                new_interval,
            )
        self._current_interval = new_interval

    def _on_poll_success(self) -> None:
        """Reset the poll interval back to the configured base."""
        if self._current_interval != self._base_interval:
            logger.info(
                "ProfileSyncWorker: poll success — resetting interval to %ds",
                self._base_interval,
            )
        self._current_interval = self._base_interval

    # ------------------------------------------------------------------ #
    # Event ingestion                                                      #
    # ------------------------------------------------------------------ #

    def _ingest_event(self, evt: dict, source: str = "sync") -> None:
        """Attempt to log a raw API event payload into the profile engine."""
        try:
            user_id = evt.get("user_id") or evt.get("userId")
            domain = evt.get("domain", "emotion")
            event_type = evt.get("event_type") or evt.get("type", "sync_event")
            payload = evt.get("payload") or evt.get("data")
            confidence = evt.get("confidence", 1.0)
            if user_id:
                self.engine.log_event(
                    user_id=str(user_id),
                    domain=domain,
                    event_type=event_type,
                    payload=payload,
                    source=source,
                    confidence=confidence,
                )
        except Exception as exc:
            logger.debug("ProfileSyncWorker: could not ingest event: %s", exc)

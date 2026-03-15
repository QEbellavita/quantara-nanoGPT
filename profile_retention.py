"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Profile Retention Manager
===============================================================================
Tiered event aggregation and storage-ceiling enforcement for the User Profile
database.  Keeps the events table manageable by rolling up old raw events into
summarised "aggregate" records while preserving statistical fidelity.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.
===============================================================================
"""

import json
import logging
import sqlite3
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Tier boundaries (seconds)
_30_DAYS  = 30  * 24 * 3600
_90_DAYS  = 90  * 24 * 3600
_180_DAYS = 180 * 24 * 3600
_7_DAYS   =  7  * 24 * 3600

# Aggregation window sizes (seconds)
_HOURLY  = 3600
_DAILY   = 86400
_WEEKLY  = 7 * 86400


class RetentionManager:
    """Tiered event aggregation and ceiling enforcement for the profile DB.

    Parameters
    ----------
    db:
        A :class:`~profile_db.ProfileDB` instance to operate on.
    ceiling_per_user:
        Maximum number of raw events kept per user before
        :meth:`enforce_ceiling` triggers early hourly aggregation.
    """

    def __init__(self, db, ceiling_per_user: int = 500_000):
        self.db = db
        self.ceiling_per_user = ceiling_per_user

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run_aggregation(self, user_id: str) -> None:
        """Apply the full three-tier aggregation policy for *user_id*.

        - 30–90 days old  → hourly summaries
        - 90–180 days old → daily summaries
        - >180 days old   → weekly summaries
        """
        now = time.time()

        # Tier 1: 30-90 days → hourly
        start_90  = now - _90_DAYS
        start_30  = now - _30_DAYS
        self._aggregate_range(user_id, start_90, start_30, _HOURLY, "hourly")

        # Tier 2: 90-180 days → daily
        start_180 = now - _180_DAYS
        self._aggregate_range(user_id, start_180, start_90, _DAILY, "daily")

        # Tier 3: >180 days → weekly
        # Use a very old epoch as lower bound so we capture everything older
        epoch = 0.0
        self._aggregate_range(user_id, epoch, start_180, _WEEKLY, "weekly")

        logger.info("RetentionManager.run_aggregation complete for user %s", user_id)

    def enforce_ceiling(self, user_id: str) -> None:
        """Enforce the per-user event ceiling.

        If the total event count exceeds :attr:`ceiling_per_user`, aggregate
        everything older than 7 days down to hourly summaries so that the
        count is reduced.
        """
        count = self.db.get_event_count(user_id)
        if count <= self.ceiling_per_user:
            logger.debug(
                "enforce_ceiling: user %s has %d events (ceiling %d) — no action",
                user_id, count, self.ceiling_per_user,
            )
            return

        logger.warning(
            "enforce_ceiling: user %s has %d events > ceiling %d — aggregating",
            user_id, count, self.ceiling_per_user,
        )
        now = time.time()
        cutoff = now - _7_DAYS
        # Aggregate everything older than 7 days to hourly
        self._aggregate_range(user_id, 0.0, cutoff, _HOURLY, "hourly")
        new_count = self.db.get_event_count(user_id)
        logger.info(
            "enforce_ceiling: user %s reduced from %d to %d events",
            user_id, count, new_count,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _aggregate_range(
        self,
        user_id: str,
        start: float,
        end: float,
        window_seconds: int,
        summary_type: str,
    ) -> int:
        """Group raw events in [*start*, *end*) by domain + time window and
        replace them with a single summary event per group.

        Parameters
        ----------
        user_id:
            Target user.
        start, end:
            Unix timestamp boundaries (inclusive start, exclusive end).
        window_seconds:
            Size of each aggregation bucket in seconds.
        summary_type:
            Human-readable label stored in the summary event's event_type
            (e.g. ``"hourly"``, ``"daily"``, ``"weekly"``).

        Returns
        -------
        int
            Number of summary events created.
        """
        # Fetch all raw events in the range (excluding already-aggregated ones)
        conn = self._read_conn()
        try:
            rows = conn.execute(
                "SELECT event_id, domain, timestamp, payload, confidence "
                "FROM events "
                "WHERE user_id = ? AND timestamp >= ? AND timestamp < ? "
                "  AND event_type NOT LIKE '%_aggregate' "
                "ORDER BY domain, timestamp",
                (user_id, start, end),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return 0

        # Group by (domain, bucket_index)
        groups: dict = {}
        for row in rows:
            event_id, domain, ts, payload_json, confidence = (
                row[0], row[1], row[2], row[3], row[4]
            )
            bucket = int(ts // window_seconds)
            key = (domain, bucket)
            if key not in groups:
                groups[key] = {
                    "event_ids": [],
                    "domain": domain,
                    "bucket_start": bucket * window_seconds,
                    "confidences": [],
                    "payloads": [],
                }
            groups[key]["event_ids"].append(event_id)
            if confidence is not None:
                groups[key]["confidences"].append(confidence)
            if payload_json:
                try:
                    groups[key]["payloads"].append(json.loads(payload_json))
                except (json.JSONDecodeError, TypeError):
                    pass

        summaries_created = 0
        for key, group in groups.items():
            event_ids = group["event_ids"]
            # Nothing to aggregate if it's already a single event
            if len(event_ids) < 1:
                continue

            avg_confidence = (
                sum(group["confidences"]) / len(group["confidences"])
                if group["confidences"] else 1.0
            )
            summary_payload = {
                "summary_type": summary_type,
                "window_seconds": window_seconds,
                "event_count": len(event_ids),
                "avg_confidence": round(avg_confidence, 4),
            }

            # Insert the summary event
            self.db.log_event(
                user_id=user_id,
                domain=group["domain"],
                event_type=f"{summary_type}_aggregate",
                payload=summary_payload,
                source="retention",
                confidence=avg_confidence,
                timestamp=group["bucket_start"],
                wait=True,
            )
            summaries_created += 1

            # Delete the original events
            self._delete_events(event_ids)

        logger.debug(
            "_aggregate_range: user=%s range=[%.0f,%.0f) window=%ds → %d summaries",
            user_id, start, end, window_seconds, summaries_created,
        )
        return summaries_created

    # ------------------------------------------------------------------ #
    # Low-level DB helpers                                                 #
    # ------------------------------------------------------------------ #

    def _read_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _delete_events(self, event_ids: list) -> None:
        """Delete a batch of events by their integer IDs."""
        if not event_ids:
            return
        placeholders = ",".join("?" * len(event_ids))
        self.db._enqueue_write(
            f"DELETE FROM events WHERE event_id IN ({placeholders})",
            tuple(event_ids),
            wait=True,
        )

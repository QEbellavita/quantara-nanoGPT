"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Profile Database Layer
===============================================================================
SQLite-backed storage for user events, profiles, evolution snapshots, and
cross-domain synergies.  Uses WAL mode for concurrent reads and a single
dedicated writer thread so that Flask request threads never block on disk I/O.

Integrates with:
- Neural Workflow AI Engine
- User Profile Engine
- Evolution Engine
- Domain Processors
===============================================================================
"""

import json
import logging
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

VALID_DOMAINS: Tuple[str, ...] = (
    "emotion",
    "biometric",
    "social",
    "cognitive",
    "behavioral",
    "environmental",
    "physiological",
    "linguistic",
)

VALID_SOURCES: Tuple[str, ...] = (
    "api",
    "websocket",
    "batch",
    "sensor",
    "manual",
    "sync",
    "ml_pipeline",
    "external",
)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL,
    timestamp   REAL    NOT NULL,
    domain      TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    payload     TEXT,
    source      TEXT,
    confidence  REAL
);

CREATE INDEX IF NOT EXISTS idx_events_user_domain_ts
    ON events (user_id, domain, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_user_ts
    ON events (user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_user_type
    ON events (user_id, event_type);

CREATE TABLE IF NOT EXISTS profiles (
    user_id          TEXT    PRIMARY KEY,
    created_at       REAL    NOT NULL,
    fingerprint_json TEXT,
    schema_version   INTEGER DEFAULT 1,
    confidence       REAL    DEFAULT 0.0,
    evolution_stage  INTEGER DEFAULT 1,
    evolution_count  INTEGER DEFAULT 0,
    last_evolved     REAL,
    last_synced      REAL
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT    NOT NULL,
    timestamp        REAL    NOT NULL,
    snapshot_type    TEXT,
    fingerprint_json TEXT,
    stage            INTEGER,
    confidence       REAL,
    trends_json      TEXT
);

CREATE INDEX IF NOT EXISTS idx_snapshots_user_ts
    ON snapshots (user_id, timestamp);

CREATE TABLE IF NOT EXISTS synergies (
    synergy_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        TEXT    NOT NULL,
    domain_a       TEXT    NOT NULL,
    domain_b       TEXT    NOT NULL,
    correlation    REAL,
    insight        TEXT,
    detected_at    REAL,
    last_confirmed REAL
);

CREATE INDEX IF NOT EXISTS idx_synergies_user
    ON synergies (user_id);
"""


# ---------------------------------------------------------------------------
# ProfileDBWriter  (single writer thread)
# ---------------------------------------------------------------------------

class ProfileDBWriter:
    """Daemon thread that serialises all database writes."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._queue: "list[tuple]" = []
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="ProfileDBWriter")
        self._thread.start()

    # -- public api ----------------------------------------------------------

    def enqueue(self, sql: str, params: tuple = (), *, wait: bool = False) -> Optional[Any]:
        """Put a write on the queue.  If *wait* is True block until it has
        been executed and return the lastrowid (or raise the exception)."""
        done_event = threading.Event() if wait else None
        result_box: list = []
        error_box: list = []

        with self._lock:
            self._queue.append((sql, params, done_event, result_box, error_box))
        self._event.set()

        if done_event is not None:
            done_event.wait()
            if error_box:
                raise error_box[0]
            return result_box[0] if result_box else None
        return None

    def stop(self):
        self._running = False
        self._event.set()
        self._thread.join(timeout=5)

    # -- internals -----------------------------------------------------------

    def _run(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            while self._running or self._queue:
                self._event.wait(timeout=0.05)
                self._event.clear()
                self._drain(conn)
        finally:
            self._drain(conn)
            conn.close()

    def _drain(self, conn: sqlite3.Connection):
        with self._lock:
            batch = list(self._queue)
            self._queue.clear()

        for sql, params, done_event, result_box, error_box in batch:
            try:
                cur = conn.execute(sql, params)
                conn.commit()
                result_box.append(cur.lastrowid)
            except Exception as exc:
                logger.exception("ProfileDBWriter error")
                error_box.append(exc)
            finally:
                if done_event is not None:
                    done_event.set()


# ---------------------------------------------------------------------------
# ProfileDB
# ---------------------------------------------------------------------------

class ProfileDB:
    """High-level interface for the profile database.

    Read operations use a per-call connection (safe from any thread).
    Write operations are serialised through the ``ProfileDBWriter`` thread.
    """

    def __init__(self, db_path: str = "profile.db"):
        self.db_path = db_path
        self._init_schema()
        self._writer = ProfileDBWriter(db_path)

    # -- lifecycle -----------------------------------------------------------

    def _init_schema(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript(_SCHEMA_SQL)
        conn.close()

    def close(self):
        """Shut down the writer thread."""
        self._writer.stop()

    # -- helpers -------------------------------------------------------------

    def _read_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _enqueue_write(self, sql: str, params: tuple = (), *, wait: bool = False):
        return self._writer.enqueue(sql, params, wait=wait)

    # ======================================================================
    # EVENT CRUD
    # ======================================================================

    def log_event(
        self,
        user_id: str,
        domain: str,
        event_type: str,
        payload: Optional[dict] = None,
        source: str = "api",
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
        wait: bool = True,
    ) -> int:
        """Insert an event and return its ``event_id``."""
        ts = timestamp if timestamp is not None else time.time()
        payload_json = json.dumps(payload) if payload is not None else None
        row_id = self._enqueue_write(
            "INSERT INTO events (user_id, timestamp, domain, event_type, payload, source, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, ts, domain, event_type, payload_json, source, confidence),
            wait=wait,
        )
        return row_id

    def get_events(
        self,
        user_id: str,
        domain: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        clauses = ["user_id = ?"]
        params: list = [user_id]
        if domain is not None:
            clauses.append("domain = ?")
            params.append(domain)
        if start_time is not None:
            clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time is not None:
            clauses.append("timestamp <= ?")
            params.append(end_time)

        where = " AND ".join(clauses)
        sql = (
            f"SELECT * FROM events WHERE {where} "
            f"ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        )
        params += [limit, offset]

        conn = self._read_conn()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_event_count(self, user_id: str, domain: Optional[str] = None) -> int:
        clauses = ["user_id = ?"]
        params: list = [user_id]
        if domain is not None:
            clauses.append("domain = ?")
            params.append(domain)
        where = " AND ".join(clauses)
        conn = self._read_conn()
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM events WHERE {where}", params).fetchone()
            return row[0]
        finally:
            conn.close()

    def get_domain_event_counts(self, user_id: str) -> Dict[str, int]:
        conn = self._read_conn()
        try:
            rows = conn.execute(
                "SELECT domain, COUNT(*) as cnt FROM events WHERE user_id = ? GROUP BY domain",
                (user_id,),
            ).fetchall()
            return {r["domain"]: r["cnt"] for r in rows}
        finally:
            conn.close()

    def get_domain_sources(self, user_id: str, domain: str) -> List[str]:
        conn = self._read_conn()
        try:
            rows = conn.execute(
                "SELECT DISTINCT source FROM events WHERE user_id = ? AND domain = ?",
                (user_id, domain),
            ).fetchall()
            return [r["source"] for r in rows]
        finally:
            conn.close()

    def get_latest_event_time(self, user_id: str, domain: Optional[str] = None) -> Optional[float]:
        clauses = ["user_id = ?"]
        params: list = [user_id]
        if domain is not None:
            clauses.append("domain = ?")
            params.append(domain)
        where = " AND ".join(clauses)
        conn = self._read_conn()
        try:
            row = conn.execute(f"SELECT MAX(timestamp) FROM events WHERE {where}", params).fetchone()
            return row[0]
        finally:
            conn.close()

    # ======================================================================
    # PROFILE CRUD
    # ======================================================================

    def get_or_create_profile(self, user_id: str) -> Dict[str, Any]:
        conn = self._read_conn()
        try:
            row = conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
            if row:
                return dict(row)
        finally:
            conn.close()

        now = time.time()
        self._enqueue_write(
            "INSERT OR IGNORE INTO profiles (user_id, created_at) VALUES (?, ?)",
            (user_id, now),
            wait=True,
        )
        conn = self._read_conn()
        try:
            row = conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
            return dict(row)
        finally:
            conn.close()

    def update_profile(self, user_id: str, **fields) -> None:
        allowed = {
            "fingerprint_json", "schema_version", "confidence",
            "evolution_stage", "evolution_count", "last_evolved", "last_synced",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        params = tuple(updates.values()) + (user_id,)
        self._enqueue_write(
            f"UPDATE profiles SET {set_clause} WHERE user_id = ?",
            params,
            wait=True,
        )

    def delete_user(self, user_id: str) -> None:
        """Purge a user from all four tables."""
        for table in ("events", "profiles", "snapshots", "synergies"):
            self._enqueue_write(
                f"DELETE FROM {table} WHERE user_id = ?",
                (user_id,),
                wait=True,
            )

    # ======================================================================
    # SNAPSHOT CRUD
    # ======================================================================

    def save_snapshot(
        self,
        user_id: str,
        snapshot_type: str = "evolution",
        fingerprint_json: Optional[str] = None,
        stage: int = 1,
        confidence: float = 0.0,
        trends_json: Optional[str] = None,
        timestamp: Optional[float] = None,
        wait: bool = True,
    ) -> int:
        ts = timestamp if timestamp is not None else time.time()
        return self._enqueue_write(
            "INSERT INTO snapshots (user_id, timestamp, snapshot_type, fingerprint_json, "
            "stage, confidence, trends_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, ts, snapshot_type, fingerprint_json, stage, confidence, trends_json),
            wait=wait,
        )

    def get_snapshots(
        self,
        user_id: str,
        snapshot_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        clauses = ["user_id = ?"]
        params: list = [user_id]
        if snapshot_type is not None:
            clauses.append("snapshot_type = ?")
            params.append(snapshot_type)
        where = " AND ".join(clauses)
        params.append(limit)
        conn = self._read_conn()
        try:
            rows = conn.execute(
                f"SELECT * FROM snapshots WHERE {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_last_snapshot_time(self, user_id: str, snapshot_type: Optional[str] = None) -> Optional[float]:
        conn = self._read_conn()
        try:
            if snapshot_type is not None:
                row = conn.execute(
                    "SELECT MAX(timestamp) FROM snapshots WHERE user_id = ? AND snapshot_type = ?",
                    (user_id, snapshot_type),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT MAX(timestamp) FROM snapshots WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
            return row[0]
        finally:
            conn.close()

    # ======================================================================
    # SYNERGY CRUD
    # ======================================================================

    def save_synergy(
        self,
        user_id: str,
        domain_a: str,
        domain_b: str,
        correlation: float,
        insight: str = "",
        detected_at: Optional[float] = None,
        wait: bool = True,
    ) -> int:
        """Upsert a synergy row (matched on user + domain pair)."""
        now = time.time()
        dt = detected_at if detected_at is not None else now

        conn = self._read_conn()
        try:
            existing = conn.execute(
                "SELECT synergy_id FROM synergies "
                "WHERE user_id = ? AND domain_a = ? AND domain_b = ?",
                (user_id, domain_a, domain_b),
            ).fetchone()
        finally:
            conn.close()

        if existing:
            self._enqueue_write(
                "UPDATE synergies SET correlation = ?, insight = ?, last_confirmed = ? "
                "WHERE synergy_id = ?",
                (correlation, insight, now, existing["synergy_id"]),
                wait=wait,
            )
            return existing["synergy_id"]
        else:
            return self._enqueue_write(
                "INSERT INTO synergies (user_id, domain_a, domain_b, correlation, "
                "insight, detected_at, last_confirmed) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, domain_a, domain_b, correlation, insight, dt, now),
                wait=wait,
            )

    def get_synergies(self, user_id: str) -> List[Dict[str, Any]]:
        conn = self._read_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM synergies WHERE user_id = ? ORDER BY correlation DESC",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete_synergy(self, synergy_id: int) -> None:
        self._enqueue_write(
            "DELETE FROM synergies WHERE synergy_id = ?",
            (synergy_id,),
            wait=True,
        )

# User Profile & Genetic Fingerprint Engine — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a centralized User Profile Engine that ingests data from across the Quantara ecosystem, synthesizes it into an 8-domain genetic fingerprint, tracks user evolution through 5 stages, and serves it via REST API + WebSocket.

**Architecture:** Centralized engine with event-sourced ingestion. SQLite in WAL mode with a single writer thread. Domain processors compute DNA scores from events. Evolution engine manages stage transitions and snapshots. Flask Blueprint serves API endpoints.

**Tech Stack:** Python 3, Flask, SQLite3, scipy (Pearson correlation), threading, queue

**Spec:** `docs/superpowers/specs/2026-03-15-user-profile-genetic-fingerprint-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `profile_db.py` | SQLite schema, WAL mode, ProfileDBWriter thread, all queries |
| `domain_processors/__init__.py` | BaseDomainProcessor ABC, processor registry |
| `domain_processors/emotional_processor.py` | Emotional DNA: mood distribution, triggers, recovery, EQ |
| `domain_processors/biometric_processor.py` | Biometric DNA: HR/HRV/EDA baselines, stress curves |
| `domain_processors/cognitive_processor.py` | Cognitive DNA: learning style, technique retention |
| `domain_processors/behavioral_processor.py` | Behavioral DNA: habits, intervention response |
| `domain_processors/temporal_processor.py` | Temporal DNA: circadian patterns, session timing |
| `domain_processors/linguistic_processor.py` | Linguistic DNA: text analysis, tone, complexity |
| `domain_processors/social_processor.py` | Social DNA: interaction patterns, engagement |
| `domain_processors/aspirational_processor.py` | Aspirational DNA: goals, values, growth areas |
| `evolution_engine.py` | Stage transitions, confidence computation, synergy detection, snapshots |
| `user_profile_engine.py` | Orchestrator: EventLogger, ProfileSyncWorker, wires everything |
| `profile_api.py` | Flask Blueprint: all profile + frontend-compatible endpoints, auth, rate limiting |
| `profile_sync_worker.py` | Background thread polling Backend/Master APIs with exponential backoff |
| `profile_retention.py` | Event retention tiers, aggregation, storage ceiling enforcement |
| `tests/test_profile_db.py` | DB schema, writer thread, queries |
| `tests/test_profile_sync.py` | Sync worker polling, backoff, ecosystem integration |
| `tests/test_profile_retention.py` | Retention tiers, aggregation, ceiling |
| `tests/test_domain_processors.py` | All 8 domain processors |
| `tests/test_evolution_engine.py` | Stage transitions, confidence, synergies, snapshots |
| `tests/test_profile_engine.py` | Orchestrator integration tests |
| `tests/test_profile_api.py` | API endpoint tests |

---

## Chunk 1: Database Layer

### Task 1: Profile Database — Schema & Writer Thread

**Files:**
- Create: `profile_db.py`
- Create: `tests/test_profile_db.py`

- [ ] **Step 1: Write failing test for schema creation**

```python
# tests/test_profile_db.py
"""Tests for profile database layer."""

import pytest
import os
import sys
import sqlite3
import tempfile
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestProfileDBSchema:
    """Test database schema creation."""

    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / 'test_profile.db')

    def test_creates_events_table(self, db_path):
        from profile_db import ProfileDB
        db = ProfileDB(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'")
        assert cursor.fetchone() is not None
        conn.close()
        db.close()

    def test_creates_profiles_table(self, db_path):
        from profile_db import ProfileDB
        db = ProfileDB(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='profiles'")
        assert cursor.fetchone() is not None
        conn.close()
        db.close()

    def test_creates_snapshots_table(self, db_path):
        from profile_db import ProfileDB
        db = ProfileDB(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='snapshots'")
        assert cursor.fetchone() is not None
        conn.close()
        db.close()

    def test_creates_synergies_table(self, db_path):
        from profile_db import ProfileDB
        db = ProfileDB(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='synergies'")
        assert cursor.fetchone() is not None
        conn.close()
        db.close()

    def test_wal_mode_enabled(self, db_path):
        from profile_db import ProfileDB
        db = ProfileDB(db_path)
        conn = sqlite3.connect(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == 'wal'
        conn.close()
        db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_db.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'profile_db'"

- [ ] **Step 3: Implement ProfileDB schema**

```python
# profile_db.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Profile Database Layer
===============================================================================
SQLite database for user profiles, event logs, evolution snapshots, and synergies.
WAL mode enabled for concurrent read/write. All writes serialized through
ProfileDBWriter thread.

Integrates with:
- Neural Workflow AI Engine
- User Profile Engine
- Evolution Engine
- Domain Processors
===============================================================================
"""

import sqlite3
import json
import time
import threading
import queue
from typing import Optional, List, Dict, Any, Tuple


# Valid DNA domains
VALID_DOMAINS = (
    'biometric', 'cognitive', 'emotional', 'temporal',
    'behavioral', 'linguistic', 'social', 'aspirational',
)

# Valid sources
VALID_SOURCES = ('nanogpt', 'backend', 'frontend', 'master', 'external')

# Current fingerprint schema version
SCHEMA_VERSION = 1


class ProfileDB:
    """SQLite database for user profile storage.

    Uses WAL mode for concurrent reads. All writes go through
    the ProfileDBWriter queue for serialization.
    """

    def __init__(self, db_path: str = 'data/profile.db'):
        self.db_path = db_path
        self._write_queue = queue.Queue()
        self._writer_thread = None
        self._running = False
        self._init_db()
        self._start_writer()

    def _init_db(self):
        """Create tables and enable WAL mode."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                domain TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL
            );

            CREATE INDEX IF NOT EXISTS idx_events_user_domain_time
                ON events(user_id, domain, timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_user_time
                ON events(user_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_user_type
                ON events(user_id, event_type);

            CREATE TABLE IF NOT EXISTS profiles (
                user_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                fingerprint_json TEXT,
                schema_version INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.0,
                evolution_stage INTEGER DEFAULT 1,
                evolution_count INTEGER DEFAULT 0,
                last_evolved REAL,
                last_synced REAL
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                snapshot_type TEXT,
                fingerprint_json TEXT,
                stage INTEGER,
                confidence REAL,
                trends_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_user_time
                ON snapshots(user_id, timestamp);

            CREATE TABLE IF NOT EXISTS synergies (
                synergy_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                domain_a TEXT NOT NULL,
                domain_b TEXT NOT NULL,
                correlation REAL,
                insight TEXT,
                detected_at REAL,
                last_confirmed REAL
            );

            CREATE INDEX IF NOT EXISTS idx_synergies_user
                ON synergies(user_id);
        """)
        conn.close()

    def _start_writer(self):
        """Start the single writer thread."""
        self._running = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name='ProfileDBWriter'
        )
        self._writer_thread.start()

    def _writer_loop(self):
        """Process write queue sequentially."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        while self._running:
            try:
                item = self._write_queue.get(timeout=1.0)
                if item is None:
                    break
                sql, params, result_event, result_holder = item
                try:
                    cursor = conn.execute(sql, params)
                    conn.commit()
                    if result_holder is not None:
                        result_holder['lastrowid'] = cursor.lastrowid
                        result_holder['rowcount'] = cursor.rowcount
                except Exception as e:
                    if result_holder is not None:
                        result_holder['error'] = e
                finally:
                    if result_event:
                        result_event.set()
            except queue.Empty:
                continue
        conn.close()

    def _enqueue_write(self, sql: str, params: tuple = (), wait: bool = False) -> Optional[Dict]:
        """Enqueue a write operation. Optionally wait for result."""
        result_holder = {} if wait else None
        result_event = threading.Event() if wait else None
        self._write_queue.put((sql, params, result_event, result_holder))
        if wait and result_event:
            result_event.wait(timeout=10.0)
            if 'error' in result_holder:
                raise result_holder['error']
            return result_holder
        return None

    def _read_conn(self) -> sqlite3.Connection:
        """Get a read-only connection (caller must close)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- Event operations ---

    def log_event(self, user_id: str, domain: str, event_type: str,
                  payload: dict, source: str, confidence: float = None,
                  timestamp: float = None) -> Optional[int]:
        """Log an event to the event store. Returns event_id if wait=True."""
        ts = timestamp or time.time()
        payload_json = json.dumps(payload) if isinstance(payload, dict) else payload
        result = self._enqueue_write(
            "INSERT INTO events (user_id, timestamp, domain, event_type, payload, source, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, ts, domain, event_type, payload_json, source, confidence),
            wait=True
        )
        return result.get('lastrowid') if result else None

    def get_events(self, user_id: str, domain: str = None,
                   start_time: float = None, end_time: float = None,
                   limit: int = 100, offset: int = 0) -> List[Dict]:
        """Query events with optional filters."""
        conn = self._read_conn()
        sql = "SELECT * FROM events WHERE user_id = ?"
        params = [user_id]
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        if start_time:
            sql += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            sql += " AND timestamp <= ?"
            params.append(end_time)
        sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_event_count(self, user_id: str, domain: str = None) -> int:
        """Count events for a user, optionally filtered by domain."""
        conn = self._read_conn()
        sql = "SELECT COUNT(*) FROM events WHERE user_id = ?"
        params = [user_id]
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        count = conn.execute(sql, params).fetchone()[0]
        conn.close()
        return count

    def get_domain_event_counts(self, user_id: str) -> Dict[str, int]:
        """Get event counts per domain for a user."""
        conn = self._read_conn()
        rows = conn.execute(
            "SELECT domain, COUNT(*) as cnt FROM events WHERE user_id = ? GROUP BY domain",
            (user_id,)
        ).fetchall()
        conn.close()
        return {row['domain']: row['cnt'] for row in rows}

    def get_domain_sources(self, user_id: str, domain: str) -> List[str]:
        """Get distinct sources for a domain."""
        conn = self._read_conn()
        rows = conn.execute(
            "SELECT DISTINCT source FROM events WHERE user_id = ? AND domain = ?",
            (user_id, domain)
        ).fetchall()
        conn.close()
        return [row['source'] for row in rows]

    def get_latest_event_time(self, user_id: str, domain: str) -> Optional[float]:
        """Get timestamp of most recent event in a domain."""
        conn = self._read_conn()
        row = conn.execute(
            "SELECT MAX(timestamp) FROM events WHERE user_id = ? AND domain = ?",
            (user_id, domain)
        ).fetchone()
        conn.close()
        return row[0] if row and row[0] else None

    # --- Profile operations ---

    def get_or_create_profile(self, user_id: str) -> Dict:
        """Get existing profile or create a new one."""
        conn = self._read_conn()
        row = conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
        conn.close()
        if row:
            return dict(row)
        now = time.time()
        self._enqueue_write(
            "INSERT OR IGNORE INTO profiles (user_id, created_at, schema_version, confidence, evolution_stage, evolution_count) "
            "VALUES (?, ?, ?, 0.0, 1, 0)",
            (user_id, now, SCHEMA_VERSION),
            wait=True
        )
        conn = self._read_conn()
        row = conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
        conn.close()
        return dict(row) if row else {'user_id': user_id, 'created_at': now, 'confidence': 0.0, 'evolution_stage': 1, 'evolution_count': 0}

    def update_profile(self, user_id: str, fingerprint: dict = None,
                       confidence: float = None, evolution_stage: int = None,
                       evolution_count: int = None):
        """Update profile fields."""
        updates = []
        params = []
        if fingerprint is not None:
            updates.append("fingerprint_json = ?")
            params.append(json.dumps(fingerprint))
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        if evolution_stage is not None:
            updates.append("evolution_stage = ?")
            params.append(evolution_stage)
        if evolution_count is not None:
            updates.append("evolution_count = ?")
            params.append(evolution_count)
        if not updates:
            return
        updates.append("last_evolved = ?")
        params.append(time.time())
        params.append(user_id)
        self._enqueue_write(
            f"UPDATE profiles SET {', '.join(updates)} WHERE user_id = ?",
            tuple(params),
            wait=True
        )

    def delete_user(self, user_id: str):
        """Purge all data for a user across all 4 tables."""
        for table in ('events', 'profiles', 'snapshots', 'synergies'):
            self._enqueue_write(
                f"DELETE FROM {table} WHERE user_id = ?",
                (user_id,),
                wait=True
            )

    # --- Snapshot operations ---

    def save_snapshot(self, user_id: str, snapshot_type: str,
                      fingerprint: dict, stage: int, confidence: float,
                      trends: dict = None):
        """Save an evolution snapshot."""
        self._enqueue_write(
            "INSERT INTO snapshots (user_id, timestamp, snapshot_type, fingerprint_json, stage, confidence, trends_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, time.time(), snapshot_type, json.dumps(fingerprint),
             stage, confidence, json.dumps(trends) if trends else None),
            wait=True
        )

    def get_snapshots(self, user_id: str, snapshot_type: str = None,
                      limit: int = 100) -> List[Dict]:
        """Get evolution snapshots for a user."""
        conn = self._read_conn()
        sql = "SELECT * FROM snapshots WHERE user_id = ?"
        params = [user_id]
        if snapshot_type:
            sql += " AND snapshot_type = ?"
            params.append(snapshot_type)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_last_snapshot_time(self, user_id: str, snapshot_type: str) -> Optional[float]:
        """Get timestamp of most recent snapshot of given type."""
        conn = self._read_conn()
        row = conn.execute(
            "SELECT MAX(timestamp) FROM snapshots WHERE user_id = ? AND snapshot_type = ?",
            (user_id, snapshot_type)
        ).fetchone()
        conn.close()
        return row[0] if row and row[0] else None

    # --- Synergy operations ---

    def save_synergy(self, user_id: str, domain_a: str, domain_b: str,
                     correlation: float, insight: str):
        """Save or update a synergy between two domains."""
        now = time.time()
        conn = self._read_conn()
        existing = conn.execute(
            "SELECT synergy_id FROM synergies WHERE user_id = ? AND domain_a = ? AND domain_b = ?",
            (user_id, domain_a, domain_b)
        ).fetchone()
        conn.close()
        if existing:
            self._enqueue_write(
                "UPDATE synergies SET correlation = ?, insight = ?, last_confirmed = ? "
                "WHERE user_id = ? AND domain_a = ? AND domain_b = ?",
                (correlation, insight, now, user_id, domain_a, domain_b),
                wait=True
            )
        else:
            self._enqueue_write(
                "INSERT INTO synergies (user_id, domain_a, domain_b, correlation, insight, detected_at, last_confirmed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, domain_a, domain_b, correlation, insight, now, now),
                wait=True
            )

    def get_synergies(self, user_id: str) -> List[Dict]:
        """Get all synergies for a user."""
        conn = self._read_conn()
        rows = conn.execute(
            "SELECT * FROM synergies WHERE user_id = ?", (user_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def delete_synergy(self, user_id: str, domain_a: str, domain_b: str):
        """Remove a synergy."""
        self._enqueue_write(
            "DELETE FROM synergies WHERE user_id = ? AND domain_a = ? AND domain_b = ?",
            (user_id, domain_a, domain_b),
            wait=True
        )

    def close(self):
        """Shutdown the writer thread."""
        self._running = False
        self._write_queue.put(None)
        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_db.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Write tests for event logging and queries**

Append to `tests/test_profile_db.py`:

```python
class TestEventOperations:
    """Test event logging and querying."""

    @pytest.fixture
    def db(self, tmp_path):
        from profile_db import ProfileDB
        db = ProfileDB(str(tmp_path / 'test.db'))
        yield db
        db.close()

    def test_log_event_returns_id(self, db):
        event_id = db.log_event('user1', 'emotional', 'emotion_classified',
                                {'emotion': 'joy', 'confidence': 0.9}, 'nanogpt', 0.9)
        assert event_id is not None
        assert event_id > 0

    def test_get_events_returns_logged(self, db):
        db.log_event('user1', 'emotional', 'emotion_classified',
                     {'emotion': 'joy'}, 'nanogpt', 0.9)
        events = db.get_events('user1')
        assert len(events) == 1
        assert events[0]['domain'] == 'emotional'

    def test_get_events_filters_by_domain(self, db):
        db.log_event('user1', 'emotional', 'emotion_classified', {}, 'nanogpt')
        db.log_event('user1', 'biometric', 'hr_reading', {}, 'nanogpt')
        events = db.get_events('user1', domain='emotional')
        assert len(events) == 1

    def test_get_event_count(self, db):
        db.log_event('user1', 'emotional', 'e1', {}, 'nanogpt')
        db.log_event('user1', 'emotional', 'e2', {}, 'nanogpt')
        db.log_event('user1', 'biometric', 'b1', {}, 'nanogpt')
        assert db.get_event_count('user1') == 3
        assert db.get_event_count('user1', domain='emotional') == 2

    def test_get_domain_event_counts(self, db):
        db.log_event('user1', 'emotional', 'e1', {}, 'nanogpt')
        db.log_event('user1', 'biometric', 'b1', {}, 'nanogpt')
        db.log_event('user1', 'biometric', 'b2', {}, 'nanogpt')
        counts = db.get_domain_event_counts('user1')
        assert counts['emotional'] == 1
        assert counts['biometric'] == 2

    def test_user_isolation(self, db):
        db.log_event('user1', 'emotional', 'e1', {}, 'nanogpt')
        db.log_event('user2', 'emotional', 'e1', {}, 'nanogpt')
        assert db.get_event_count('user1') == 1
        assert db.get_event_count('user2') == 1


class TestProfileOperations:
    """Test profile CRUD."""

    @pytest.fixture
    def db(self, tmp_path):
        from profile_db import ProfileDB
        db = ProfileDB(str(tmp_path / 'test.db'))
        yield db
        db.close()

    def test_get_or_create_profile(self, db):
        profile = db.get_or_create_profile('user1')
        assert profile['user_id'] == 'user1'
        assert profile['evolution_stage'] == 1
        assert profile['confidence'] == 0.0

    def test_update_profile(self, db):
        db.get_or_create_profile('user1')
        db.update_profile('user1', confidence=0.5, evolution_stage=2)
        profile = db.get_or_create_profile('user1')
        assert profile['confidence'] == 0.5
        assert profile['evolution_stage'] == 2

    def test_delete_user_purges_all_tables(self, db):
        db.get_or_create_profile('user1')
        db.log_event('user1', 'emotional', 'e1', {}, 'nanogpt')
        db.save_snapshot('user1', 'weekly', {}, 1, 0.5)
        db.save_synergy('user1', 'emotional', 'biometric', 0.7, 'test')
        db.delete_user('user1')
        assert db.get_event_count('user1') == 0
        assert len(db.get_snapshots('user1')) == 0
        assert len(db.get_synergies('user1')) == 0


class TestSnapshotOperations:
    """Test snapshot CRUD."""

    @pytest.fixture
    def db(self, tmp_path):
        from profile_db import ProfileDB
        db = ProfileDB(str(tmp_path / 'test.db'))
        yield db
        db.close()

    def test_save_and_get_snapshot(self, db):
        db.save_snapshot('user1', 'weekly', {'emotional': {'score': 0.5}}, 2, 0.6)
        snapshots = db.get_snapshots('user1')
        assert len(snapshots) == 1
        assert snapshots[0]['snapshot_type'] == 'weekly'
        assert snapshots[0]['stage'] == 2

    def test_filter_by_type(self, db):
        db.save_snapshot('user1', 'weekly', {}, 1, 0.3)
        db.save_snapshot('user1', 'monthly', {}, 1, 0.3)
        weekly = db.get_snapshots('user1', snapshot_type='weekly')
        assert len(weekly) == 1
```

- [ ] **Step 6: Run all DB tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_db.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add profile_db.py tests/test_profile_db.py
git commit -m "feat(profile): add ProfileDB with schema, writer thread, and CRUD operations"
```

---

## Chunk 2: Domain Processors

### Task 2: Base Processor & Emotional DNA Processor

**Files:**
- Create: `domain_processors/__init__.py`
- Create: `domain_processors/emotional_processor.py`
- Create: `tests/test_domain_processors.py`

- [ ] **Step 1: Write failing test for base processor interface**

```python
# tests/test_domain_processors.py
"""Tests for DNA domain processors."""

import pytest
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBaseDomainProcessor:
    """Test base processor contract."""

    def test_base_processor_is_abstract(self):
        from domain_processors import BaseDomainProcessor
        with pytest.raises(TypeError):
            BaseDomainProcessor()

    def test_processor_has_domain_name(self):
        from domain_processors import BaseDomainProcessor
        assert hasattr(BaseDomainProcessor, 'domain')

    def test_processor_registry_returns_all_8(self):
        from domain_processors import get_all_processors
        processors = get_all_processors()
        assert len(processors) == 8
        domains = {p.domain for p in processors}
        assert domains == {
            'biometric', 'cognitive', 'emotional', 'temporal',
            'behavioral', 'linguistic', 'social', 'aspirational'
        }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_domain_processors.py::TestBaseDomainProcessor -v`
Expected: FAIL

- [ ] **Step 3: Implement base processor**

```python
# domain_processors/__init__.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - DNA Domain Processors
===============================================================================
Base class and registry for the 8 DNA domain processors.
Each processor computes a domain-specific score from the user's event log.

Integrates with:
- Neural Workflow AI Engine
- User Profile Engine
- Evolution Engine
===============================================================================
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseDomainProcessor(ABC):
    """Abstract base for DNA domain processors.

    Each processor reads events from its domain and computes
    a structured score dict with domain-specific metrics.
    """

    domain: str = ''  # Override in subclass

    @abstractmethod
    def compute(self, events: List[Dict]) -> Dict[str, Any]:
        """Compute domain score from events.

        Args:
            events: List of event dicts from this domain,
                    ordered by timestamp ascending.

        Returns:
            Dict with domain-specific metrics + 'score' (0.0-1.0).
        """
        pass

    def get_empty_score(self) -> Dict[str, Any]:
        """Return empty/default score when no events exist."""
        return {'score': 0.0, 'metrics': {}, 'event_count': 0}


def get_all_processors() -> List[BaseDomainProcessor]:
    """Return instances of all 8 domain processors."""
    from domain_processors.emotional_processor import EmotionalProcessor
    from domain_processors.biometric_processor import BiometricProcessor
    from domain_processors.cognitive_processor import CognitiveProcessor
    from domain_processors.behavioral_processor import BehavioralProcessor
    from domain_processors.temporal_processor import TemporalProcessor
    from domain_processors.linguistic_processor import LinguisticProcessor
    from domain_processors.social_processor import SocialProcessor
    from domain_processors.aspirational_processor import AspirationalProcessor
    return [
        EmotionalProcessor(),
        BiometricProcessor(),
        CognitiveProcessor(),
        BehavioralProcessor(),
        TemporalProcessor(),
        LinguisticProcessor(),
        SocialProcessor(),
        AspirationalProcessor(),
    ]
```

- [ ] **Step 4: Write failing test for EmotionalProcessor**

Append to `tests/test_domain_processors.py`:

```python
class TestEmotionalProcessor:
    """Test Emotional DNA computation."""

    @pytest.fixture
    def processor(self):
        from domain_processors.emotional_processor import EmotionalProcessor
        return EmotionalProcessor()

    def test_domain_name(self, processor):
        assert processor.domain == 'emotional'

    def test_empty_events_returns_zero(self, processor):
        result = processor.compute([])
        assert result['score'] == 0.0

    def test_computes_mood_distribution(self, processor):
        events = [
            {'payload': '{"emotion": "joy", "family": "Joy", "confidence": 0.9}', 'timestamp': time.time()},
            {'payload': '{"emotion": "joy", "family": "Joy", "confidence": 0.8}', 'timestamp': time.time()},
            {'payload': '{"emotion": "sadness", "family": "Sadness", "confidence": 0.7}', 'timestamp': time.time()},
        ]
        result = processor.compute(events)
        assert 'mood_distribution' in result['metrics']
        assert result['metrics']['mood_distribution']['joy'] == 2
        assert result['metrics']['dominant_emotion'] == 'joy'

    def test_computes_emotional_range(self, processor):
        events = [
            {'payload': '{"emotion": "joy", "family": "Joy"}', 'timestamp': time.time()},
            {'payload': '{"emotion": "anger", "family": "Anger"}', 'timestamp': time.time()},
            {'payload': '{"emotion": "calm", "family": "Calm"}', 'timestamp': time.time()},
        ]
        result = processor.compute(events)
        assert result['metrics']['emotional_range'] == 3

    def test_score_increases_with_more_events(self, processor):
        few = [{'payload': '{"emotion": "joy", "family": "Joy"}', 'timestamp': time.time()}]
        many = [{'payload': '{"emotion": "joy", "family": "Joy"}', 'timestamp': time.time()} for _ in range(50)]
        score_few = processor.compute(few)['score']
        score_many = processor.compute(many)['score']
        assert score_many > score_few
```

- [ ] **Step 5: Implement EmotionalProcessor**

```python
# domain_processors/emotional_processor.py
"""Emotional DNA processor — mood distribution, triggers, recovery, EQ."""

import json
from collections import Counter
from typing import Dict, List, Any

from domain_processors import BaseDomainProcessor


class EmotionalProcessor(BaseDomainProcessor):
    """Computes Emotional DNA from emotion classification events.

    Metrics: mood_distribution, dominant_emotion, dominant_family,
    emotional_range, avg_confidence, emotional_volatility.
    """

    domain = 'emotional'

    def compute(self, events: List[Dict]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        emotions = []
        families = []
        confidences = []

        for event in events:
            payload = json.loads(event['payload']) if isinstance(event['payload'], str) else event['payload']
            emotion = payload.get('emotion', 'neutral')
            family = payload.get('family', 'Neutral')
            conf = payload.get('confidence', 0.5)
            emotions.append(emotion)
            families.append(family)
            confidences.append(conf)

        mood_dist = dict(Counter(emotions))
        family_dist = dict(Counter(families))
        dominant_emotion = max(mood_dist, key=mood_dist.get)
        dominant_family = max(family_dist, key=family_dist.get)
        emotional_range = len(set(emotions))
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Volatility: how often emotions change between consecutive events
        changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
        volatility = changes / max(1, len(emotions) - 1) if len(emotions) > 1 else 0.0

        # Recovery: count transitions from negative to positive families
        negative_families = {'Sadness', 'Anger', 'Fear', 'Self-Conscious'}
        positive_families = {'Joy', 'Love', 'Calm'}
        recoveries = 0
        for i in range(1, len(families)):
            if families[i-1] in negative_families and families[i] in positive_families:
                recoveries += 1
        recovery_rate = recoveries / max(1, len(families) - 1) if len(families) > 1 else 0.0

        # Score: normalized by event count (more data = higher confidence)
        score = min(1.0, len(events) / 100) * avg_confidence

        return {
            'score': round(score, 4),
            'event_count': len(events),
            'metrics': {
                'mood_distribution': mood_dist,
                'family_distribution': family_dist,
                'dominant_emotion': dominant_emotion,
                'dominant_family': dominant_family,
                'emotional_range': emotional_range,
                'avg_confidence': round(avg_confidence, 4),
                'volatility': round(volatility, 4),
                'recovery_rate': round(recovery_rate, 4),
            }
        }
```

- [ ] **Step 6: Run tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_domain_processors.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add domain_processors/__init__.py domain_processors/emotional_processor.py tests/test_domain_processors.py
git commit -m "feat(profile): add BaseDomainProcessor and EmotionalProcessor"
```

### Task 3: Remaining 7 Domain Processors

**Files:**
- Create: `domain_processors/biometric_processor.py`
- Create: `domain_processors/cognitive_processor.py`
- Create: `domain_processors/behavioral_processor.py`
- Create: `domain_processors/temporal_processor.py`
- Create: `domain_processors/linguistic_processor.py`
- Create: `domain_processors/social_processor.py`
- Create: `domain_processors/aspirational_processor.py`
- Modify: `tests/test_domain_processors.py`

- [ ] **Step 1: Write tests for BiometricProcessor**

Append to `tests/test_domain_processors.py`:

```python
class TestBiometricProcessor:

    @pytest.fixture
    def processor(self):
        from domain_processors.biometric_processor import BiometricProcessor
        return BiometricProcessor()

    def test_domain_name(self, processor):
        assert processor.domain == 'biometric'

    def test_empty_events(self, processor):
        assert processor.compute([])['score'] == 0.0

    def test_computes_hr_baseline(self, processor):
        events = [
            {'payload': '{"hr": 72, "hrv": 55, "eda": 3.2}', 'timestamp': time.time()},
            {'payload': '{"hr": 68, "hrv": 60, "eda": 2.8}', 'timestamp': time.time()},
        ]
        result = processor.compute(events)
        assert 'resting_hr' in result['metrics']
        assert 60 < result['metrics']['resting_hr'] < 80
```

- [ ] **Step 2: Implement BiometricProcessor**

```python
# domain_processors/biometric_processor.py
"""Biometric DNA processor — HR, HRV, EDA baselines and stress curves."""

import json
from typing import Dict, List, Any

from domain_processors import BaseDomainProcessor


class BiometricProcessor(BaseDomainProcessor):
    """Computes Biometric DNA from biometric sensor events."""

    domain = 'biometric'

    def compute(self, events: List[Dict]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        hrs, hrvs, edas = [], [], []
        for event in events:
            payload = json.loads(event['payload']) if isinstance(event['payload'], str) else event['payload']
            if 'hr' in payload:
                hrs.append(payload['hr'])
            if 'hrv' in payload:
                hrvs.append(payload['hrv'])
            if 'eda' in payload:
                edas.append(payload['eda'])

        metrics = {}
        if hrs:
            metrics['resting_hr'] = round(sum(hrs) / len(hrs), 2)
            metrics['hr_min'] = min(hrs)
            metrics['hr_max'] = max(hrs)
        if hrvs:
            metrics['hrv_baseline'] = round(sum(hrvs) / len(hrvs), 2)
        if edas:
            metrics['eda_baseline'] = round(sum(edas) / len(edas), 2)

        # Stress response: ratio of high-HR events to total
        if hrs:
            high_hr_threshold = metrics.get('resting_hr', 72) * 1.2
            stress_events = sum(1 for h in hrs if h > high_hr_threshold)
            metrics['stress_ratio'] = round(stress_events / len(hrs), 4)

        score = min(1.0, len(events) / 100)
        return {'score': round(score, 4), 'event_count': len(events), 'metrics': metrics}
```

- [ ] **Step 3: Write tests for remaining 5 processors + implement them**

Each follows the same pattern. Tests verify: domain name, empty events return 0.0, computes expected metrics from sample events. Implementations:

```python
# domain_processors/cognitive_processor.py
"""Cognitive DNA processor — learning patterns, technique retention."""

import json
from collections import Counter
from typing import Dict, List, Any

from domain_processors import BaseDomainProcessor


class CognitiveProcessor(BaseDomainProcessor):
    domain = 'cognitive'

    def compute(self, events: List[Dict]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        techniques = []
        completion_times = []
        outcomes = []

        for event in events:
            payload = json.loads(event['payload']) if isinstance(event['payload'], str) else event['payload']
            if 'technique' in payload:
                techniques.append(payload['technique'])
            if 'completion_time' in payload:
                completion_times.append(payload['completion_time'])
            if 'outcome' in payload:
                outcomes.append(payload['outcome'])

        metrics = {}
        if techniques:
            tech_counts = dict(Counter(techniques))
            metrics['technique_usage'] = tech_counts
            metrics['technique_diversity'] = len(set(techniques))
        if completion_times:
            metrics['avg_completion_time'] = round(sum(completion_times) / len(completion_times), 2)
        if outcomes:
            success = sum(1 for o in outcomes if o == 'success')
            metrics['technique_retention_rate'] = round(success / len(outcomes), 4)

        score = min(1.0, len(events) / 100)
        return {'score': round(score, 4), 'event_count': len(events), 'metrics': metrics}
```

```python
# domain_processors/behavioral_processor.py
"""Behavioral DNA processor — habits, intervention response rates."""

import json
from collections import Counter
from typing import Dict, List, Any

from domain_processors import BaseDomainProcessor


class BehavioralProcessor(BaseDomainProcessor):
    domain = 'behavioral'

    def compute(self, events: List[Dict]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        sessions_started = 0
        sessions_completed = 0
        sessions_abandoned = 0
        technique_responses = []

        for event in events:
            payload = json.loads(event['payload']) if isinstance(event['payload'], str) else event['payload']
            etype = event.get('event_type', '')
            if etype == 'session_started':
                sessions_started += 1
            elif etype == 'session_completed':
                sessions_completed += 1
            elif etype == 'session_abandoned':
                sessions_abandoned += 1
            if 'response_positive' in payload:
                technique_responses.append(payload['response_positive'])

        metrics = {
            'sessions_started': sessions_started,
            'sessions_completed': sessions_completed,
            'sessions_abandoned': sessions_abandoned,
        }
        if sessions_started > 0:
            metrics['completion_rate'] = round(sessions_completed / sessions_started, 4)
        if technique_responses:
            metrics['intervention_response_rate'] = round(
                sum(1 for r in technique_responses if r) / len(technique_responses), 4
            )

        # Engagement streak: consecutive days with events
        days = set()
        for event in events:
            ts = event.get('timestamp', 0)
            days.add(int(ts // 86400))
        sorted_days = sorted(days)
        streak = 1
        max_streak = 1
        for i in range(1, len(sorted_days)):
            if sorted_days[i] == sorted_days[i-1] + 1:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        metrics['engagement_streak'] = max_streak if sorted_days else 0

        score = min(1.0, len(events) / 100)
        return {'score': round(score, 4), 'event_count': len(events), 'metrics': metrics}
```

```python
# domain_processors/temporal_processor.py
"""Temporal DNA processor — circadian patterns, session timing."""

import json
from collections import Counter
from datetime import datetime
from typing import Dict, List, Any

from domain_processors import BaseDomainProcessor


class TemporalProcessor(BaseDomainProcessor):
    domain = 'temporal'

    def compute(self, events: List[Dict]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        hours = []
        days_of_week = []

        for event in events:
            ts = event.get('timestamp', 0)
            dt = datetime.fromtimestamp(ts)
            hours.append(dt.hour)
            days_of_week.append(dt.strftime('%A'))

        hour_dist = dict(Counter(hours))
        day_dist = dict(Counter(days_of_week))

        # Peak hours: top 3 most active hours
        sorted_hours = sorted(hour_dist.items(), key=lambda x: x[1], reverse=True)
        peak_hours = [h for h, _ in sorted_hours[:3]]

        # Chronotype estimate
        morning = sum(hour_dist.get(h, 0) for h in range(5, 12))
        afternoon = sum(hour_dist.get(h, 0) for h in range(12, 18))
        evening = sum(hour_dist.get(h, 0) for h in range(18, 24))
        night = sum(hour_dist.get(h, 0) for h in range(0, 5))
        total = morning + afternoon + evening + night
        if total > 0:
            if morning / total > 0.4:
                chronotype = 'morning'
            elif evening / total > 0.4 or night / total > 0.2:
                chronotype = 'evening'
            else:
                chronotype = 'balanced'
        else:
            chronotype = 'unknown'

        metrics = {
            'hour_distribution': hour_dist,
            'day_distribution': day_dist,
            'peak_hours': peak_hours,
            'chronotype': chronotype,
        }

        score = min(1.0, len(events) / 100)
        return {'score': round(score, 4), 'event_count': len(events), 'metrics': metrics}
```

```python
# domain_processors/linguistic_processor.py
"""Linguistic DNA processor — text analysis, tone, vocabulary complexity."""

import json
import re
from collections import Counter
from typing import Dict, List, Any

from domain_processors import BaseDomainProcessor


class LinguisticProcessor(BaseDomainProcessor):
    domain = 'linguistic'

    def compute(self, events: List[Dict]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        sentence_lengths = []
        all_words = []
        tones = []

        for event in events:
            payload = json.loads(event['payload']) if isinstance(event['payload'], str) else event['payload']
            text = payload.get('text', '')
            if text:
                words = text.split()
                sentence_lengths.append(len(words))
                all_words.extend(w.lower() for w in words)
            if 'tone' in payload:
                tones.append(payload['tone'])

        metrics = {}
        if sentence_lengths:
            metrics['avg_sentence_length'] = round(sum(sentence_lengths) / len(sentence_lengths), 2)
        if all_words:
            unique_ratio = len(set(all_words)) / len(all_words)
            metrics['vocabulary_complexity'] = round(unique_ratio, 4)
            metrics['total_words_analyzed'] = len(all_words)
        if tones:
            tone_dist = dict(Counter(tones))
            metrics['tone_distribution'] = tone_dist
            metrics['dominant_tone'] = max(tone_dist, key=tone_dist.get)

        score = min(1.0, len(events) / 100)
        return {'score': round(score, 4), 'event_count': len(events), 'metrics': metrics}
```

```python
# domain_processors/social_processor.py
"""Social DNA processor — interaction patterns, engagement."""

import json
from collections import Counter
from typing import Dict, List, Any

from domain_processors import BaseDomainProcessor


class SocialProcessor(BaseDomainProcessor):
    domain = 'social'

    def compute(self, events: List[Dict]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        interaction_types = []
        shares = 0
        feedback_received = 0

        for event in events:
            payload = json.loads(event['payload']) if isinstance(event['payload'], str) else event['payload']
            etype = event.get('event_type', '')
            interaction_types.append(etype)
            if etype == 'content_shared':
                shares += 1
            elif etype == 'feedback_received':
                feedback_received += 1

        metrics = {
            'interaction_count': len(events),
            'shares': shares,
            'feedback_received': feedback_received,
            'interaction_types': dict(Counter(interaction_types)),
        }

        # Engagement frequency: events per day over the event window
        if len(events) >= 2:
            time_span = events[-1].get('timestamp', 0) - events[0].get('timestamp', 0)
            days = max(1, time_span / 86400)
            metrics['daily_interaction_rate'] = round(len(events) / days, 2)

        score = min(1.0, len(events) / 100)
        return {'score': round(score, 4), 'event_count': len(events), 'metrics': metrics}
```

```python
# domain_processors/aspirational_processor.py
"""Aspirational DNA processor — goals, values, growth areas."""

import json
from collections import Counter
from typing import Dict, List, Any

from domain_processors import BaseDomainProcessor


class AspirationalProcessor(BaseDomainProcessor):
    domain = 'aspirational'

    def compute(self, events: List[Dict]) -> Dict[str, Any]:
        if not events:
            return self.get_empty_score()

        goals_set = 0
        goals_completed = 0
        growth_areas = []
        values = []

        for event in events:
            payload = json.loads(event['payload']) if isinstance(event['payload'], str) else event['payload']
            etype = event.get('event_type', '')
            if etype == 'goal_set':
                goals_set += 1
            elif etype == 'goal_completed':
                goals_completed += 1
            if 'growth_area' in payload:
                growth_areas.append(payload['growth_area'])
            if 'value' in payload:
                values.append(payload['value'])

        metrics = {
            'goals_set': goals_set,
            'goals_completed': goals_completed,
        }
        if goals_set > 0:
            metrics['goal_completion_rate'] = round(goals_completed / goals_set, 4)
        if growth_areas:
            metrics['active_growth_areas'] = list(set(growth_areas))
        if values:
            value_counts = Counter(values)
            metrics['core_values'] = [v for v, _ in value_counts.most_common(5)]

        score = min(1.0, len(events) / 100)
        return {'score': round(score, 4), 'event_count': len(events), 'metrics': metrics}
```

- [ ] **Step 4: Write tests for each remaining processor**

Add test classes for each processor to `tests/test_domain_processors.py` following the same pattern: fixture creates instance, test domain name, test empty events, test key metrics from sample data.

- [ ] **Step 5: Run all processor tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_domain_processors.py -v`
Expected: All tests PASS

- [ ] **Step 6: Verify registry returns all 8**

Run: `cd /Users/bel/quantara-nanoGPT && python -c "from domain_processors import get_all_processors; ps = get_all_processors(); print(f'{len(ps)} processors: {[p.domain for p in ps]}')`
Expected: `8 processors: ['emotional', 'biometric', 'cognitive', 'behavioral', 'temporal', 'linguistic', 'social', 'aspirational']`

- [ ] **Step 7: Commit**

```bash
git add domain_processors/ tests/test_domain_processors.py
git commit -m "feat(profile): add all 8 DNA domain processors"
```

---

## Chunk 3: Evolution Engine

### Task 4: Evolution Engine — Stages, Confidence, Synergies, Snapshots

**Files:**
- Create: `evolution_engine.py`
- Create: `tests/test_evolution_engine.py`

- [ ] **Step 1: Write failing tests for confidence computation**

```python
# tests/test_evolution_engine.py
"""Tests for evolution engine — stages, confidence, synergies, snapshots."""

import pytest
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfidenceComputation:
    """Test per-domain and overall confidence."""

    @pytest.fixture
    def engine(self, tmp_path):
        from evolution_engine import EvolutionEngine
        from profile_db import ProfileDB
        db = ProfileDB(str(tmp_path / 'test.db'))
        eng = EvolutionEngine(db)
        yield eng
        db.close()

    def test_zero_events_zero_confidence(self, engine):
        conf = engine.compute_domain_confidence(event_count=0, latest_event_time=None, source_count=0)
        assert conf == 0.0

    def test_100_events_recent_multi_source(self, engine):
        conf = engine.compute_domain_confidence(
            event_count=100,
            latest_event_time=time.time(),
            source_count=2
        )
        assert conf == 1.0

    def test_single_source_reduces_confidence(self, engine):
        conf_multi = engine.compute_domain_confidence(100, time.time(), 2)
        conf_single = engine.compute_domain_confidence(100, time.time(), 1)
        assert conf_single < conf_multi

    def test_old_events_reduce_confidence(self, engine):
        recent = engine.compute_domain_confidence(100, time.time(), 2)
        old = engine.compute_domain_confidence(100, time.time() - 21 * 86400, 2)  # 3 weeks old
        assert old < recent

    def test_overall_confidence_weighted_mean(self, engine):
        domain_scores = {
            'emotional': {'score': 0.8, 'event_count': 50},
            'biometric': {'score': 0.6, 'event_count': 30},
        }
        domain_meta = {
            'emotional': {'latest_event_time': time.time(), 'source_count': 2},
            'biometric': {'latest_event_time': time.time(), 'source_count': 1},
        }
        conf = engine.compute_overall_confidence(domain_scores, domain_meta)
        assert 0.0 < conf < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_evolution_engine.py::TestConfidenceComputation -v`
Expected: FAIL

- [ ] **Step 3: Write failing tests for stage transitions**

Append to `tests/test_evolution_engine.py`:

```python
class TestStageTransitions:
    """Test evolution stage advancement and regression."""

    @pytest.fixture
    def engine(self, tmp_path):
        from evolution_engine import EvolutionEngine
        from profile_db import ProfileDB
        db = ProfileDB(str(tmp_path / 'test.db'))
        eng = EvolutionEngine(db)
        yield eng
        db.close()

    def test_nascent_is_default(self, engine):
        stage = engine.evaluate_stage(
            current_stage=1, confidence=0.0,
            total_events=0, domains_with_events=0,
            weeks_of_data=0, positive_patterns=False,
            synergy_count=0, stability_score=0.0,
            consecutive_met=0
        )
        assert stage['new_stage'] == 1

    def test_advance_to_awareness(self, engine):
        stage = engine.evaluate_stage(
            current_stage=1, confidence=0.35,
            total_events=60, domains_with_events=4,
            weeks_of_data=1, positive_patterns=False,
            synergy_count=0, stability_score=0.0,
            consecutive_met=3
        )
        assert stage['new_stage'] == 2

    def test_requires_3_consecutive_to_advance(self, engine):
        stage = engine.evaluate_stage(
            current_stage=1, confidence=0.35,
            total_events=60, domains_with_events=4,
            weeks_of_data=1, positive_patterns=False,
            synergy_count=0, stability_score=0.0,
            consecutive_met=2  # Not enough
        )
        assert stage['new_stage'] == 1

    def test_advance_to_regulation(self, engine):
        stage = engine.evaluate_stage(
            current_stage=2, confidence=0.6,
            total_events=200, domains_with_events=5,
            weeks_of_data=5, positive_patterns=True,
            synergy_count=0, stability_score=0.5,
            consecutive_met=3
        )
        assert stage['new_stage'] == 3

    def test_advance_to_integration(self, engine):
        stage = engine.evaluate_stage(
            current_stage=3, confidence=0.8,
            total_events=500, domains_with_events=6,
            weeks_of_data=9, positive_patterns=True,
            synergy_count=3, stability_score=0.7,
            consecutive_met=3
        )
        assert stage['new_stage'] == 4

    def test_advance_to_mastery(self, engine):
        stage = engine.evaluate_stage(
            current_stage=4, confidence=0.92,
            total_events=1000, domains_with_events=8,
            weeks_of_data=13, positive_patterns=True,
            synergy_count=5, stability_score=0.9,
            consecutive_met=3
        )
        assert stage['new_stage'] == 5

    def test_regress_on_sustained_negative(self, engine):
        stage = engine.evaluate_stage(
            current_stage=3, confidence=0.4,
            total_events=200, domains_with_events=5,
            weeks_of_data=6, positive_patterns=False,
            synergy_count=0, stability_score=0.2,
            consecutive_met=0,
            sustained_negative_weeks=3
        )
        assert stage['new_stage'] == 2
        assert 'recalibration' in stage['reason'].lower()

    def test_no_regress_below_nascent(self, engine):
        stage = engine.evaluate_stage(
            current_stage=1, confidence=0.1,
            total_events=10, domains_with_events=1,
            weeks_of_data=3, positive_patterns=False,
            synergy_count=0, stability_score=0.1,
            consecutive_met=0,
            sustained_negative_weeks=5
        )
        assert stage['new_stage'] == 1
```

- [ ] **Step 4: Implement EvolutionEngine**

```python
# evolution_engine.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Evolution Engine
===============================================================================
Manages user evolution stages, confidence computation, synergy detection,
and snapshot scheduling.

Integrates with:
- Neural Workflow AI Engine
- User Profile Engine
- Domain Processors
- Profile Database
===============================================================================
"""

import time
import json
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

try:
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


STAGE_NAMES = {
    1: 'Nascent',
    2: 'Awareness',
    3: 'Regulation',
    4: 'Integration',
    5: 'Mastery',
}


class EvolutionEngine:
    """Manages evolution stages, confidence, synergies, and snapshots."""

    def __init__(self, db):
        self.db = db

    # --- Confidence ---

    def compute_domain_confidence(self, event_count: int,
                                   latest_event_time: Optional[float],
                                   source_count: int) -> float:
        """Compute confidence for a single DNA domain.

        Formula: min(1.0, event_count/100) * recency_factor * source_factor
        """
        if event_count == 0:
            return 0.0

        volume_factor = min(1.0, event_count / 100)

        # Recency: 1.0 if within 7 days, decays 0.1/week, floor 0.3
        if latest_event_time:
            weeks_since = (time.time() - latest_event_time) / (7 * 86400)
            recency_factor = max(0.3, 1.0 - (max(0, weeks_since - 1) * 0.1))
        else:
            recency_factor = 0.3

        # Source diversity: 1.0 if 2+, 0.8 if single
        source_factor = 1.0 if source_count >= 2 else 0.8

        return round(volume_factor * recency_factor * source_factor, 4)

    def compute_overall_confidence(self, domain_scores: Dict[str, Dict],
                                    domain_meta: Dict[str, Dict]) -> float:
        """Compute overall fingerprint confidence as weighted mean of domain confidences."""
        total_weight = 0
        weighted_sum = 0

        for domain, scores in domain_scores.items():
            meta = domain_meta.get(domain, {})
            event_count = scores.get('event_count', 0)
            if event_count == 0:
                continue
            domain_conf = self.compute_domain_confidence(
                event_count,
                meta.get('latest_event_time'),
                meta.get('source_count', 0)
            )
            weighted_sum += domain_conf * event_count
            total_weight += event_count

        if total_weight == 0:
            return 0.0
        return round(weighted_sum / total_weight, 4)

    # --- Stage transitions ---

    def evaluate_stage(self, current_stage: int, confidence: float,
                       total_events: int, domains_with_events: int,
                       weeks_of_data: float, positive_patterns: bool,
                       synergy_count: int, stability_score: float,
                       consecutive_met: int,
                       sustained_negative_weeks: float = 0) -> Dict:
        """Evaluate whether user should advance/regress stages.

        Returns: {'new_stage': int, 'changed': bool, 'reason': str}
        """
        target_stage = current_stage

        # Check regression first: sustained negative patterns for 2+ weeks
        if current_stage > 1 and sustained_negative_weeks >= 2:
            target_stage = current_stage - 1
            return {
                'new_stage': target_stage,
                'changed': True,
                'reason': f'Recalibration: moving from {STAGE_NAMES[current_stage]} back to {STAGE_NAMES[target_stage]} for renewed growth. This is a normal part of the journey.',
                'stage_name': STAGE_NAMES[target_stage],
            }

        # Check advancement criteria
        if current_stage == 1:
            if total_events >= 50 and domains_with_events >= 3 and confidence >= 0.3:
                if consecutive_met >= 3:
                    target_stage = 2
        elif current_stage == 2:
            if weeks_of_data >= 4 and positive_patterns and confidence > 0.55:
                if consecutive_met >= 3:
                    target_stage = 3
        elif current_stage == 3:
            if synergy_count >= 3 and weeks_of_data >= 8 and confidence > 0.75:
                if consecutive_met >= 3:
                    target_stage = 4
        elif current_stage == 4:
            if weeks_of_data >= 12 and stability_score > 0.85 and confidence > 0.9:
                if consecutive_met >= 3:
                    target_stage = 5

        changed = target_stage != current_stage
        reason = ''
        if changed:
            reason = f'Advanced from {STAGE_NAMES[current_stage]} to {STAGE_NAMES[target_stage]}'

        return {
            'new_stage': target_stage,
            'changed': changed,
            'reason': reason,
            'stage_name': STAGE_NAMES[target_stage],
        }

    # --- Synergy Detection ---

    def detect_synergies(self, user_id: str,
                          daily_domain_scores: Dict[str, List[float]]) -> List[Dict]:
        """Detect cross-domain synergies using Pearson correlation.

        Args:
            daily_domain_scores: {domain: [daily_scores_over_28_days]}

        Returns:
            List of synergy dicts with domain_a, domain_b, correlation, insight
        """
        if not HAS_SCIPY:
            return []

        synergies = []
        domains = [d for d, scores in daily_domain_scores.items() if len(scores) >= 14]

        for i, domain_a in enumerate(domains):
            for domain_b in domains[i+1:]:
                scores_a = daily_domain_scores[domain_a]
                scores_b = daily_domain_scores[domain_b]
                # Align to same length
                min_len = min(len(scores_a), len(scores_b))
                if min_len < 14:
                    continue
                corr, p_value = pearsonr(scores_a[:min_len], scores_b[:min_len])
                if abs(corr) > 0.6 and p_value < 0.05:
                    if corr > 0:
                        insight = f"Your {domain_a} and {domain_b} patterns show positive correlation — improvements in one tend to accompany improvements in the other."
                    else:
                        insight = f"Your {domain_a} and {domain_b} patterns show inverse correlation — changes in one tend to coincide with opposite changes in the other."
                    synergies.append({
                        'domain_a': domain_a,
                        'domain_b': domain_b,
                        'correlation': round(corr, 4),
                        'p_value': round(p_value, 4),
                        'insight': insight,
                    })

        return synergies

    # --- Snapshots ---

    def check_missed_snapshots(self, user_id: str) -> List[str]:
        """Check for missed weekly/monthly snapshots and return types to catch up."""
        missed = []
        now = time.time()

        last_weekly = self.db.get_last_snapshot_time(user_id, 'weekly')
        if last_weekly:
            if now - last_weekly > 7 * 86400:
                missed.append('weekly')
        else:
            # No weekly snapshot ever taken — check if user has enough data
            profile = self.db.get_or_create_profile(user_id)
            if profile.get('created_at') and now - profile['created_at'] > 7 * 86400:
                missed.append('weekly')

        last_monthly = self.db.get_last_snapshot_time(user_id, 'monthly')
        if last_monthly:
            if now - last_monthly > 30 * 86400:
                missed.append('monthly')
        else:
            profile = self.db.get_or_create_profile(user_id)
            if profile.get('created_at') and now - profile['created_at'] > 30 * 86400:
                missed.append('monthly')

        return missed

    def get_stage_progress(self, current_stage: int, confidence: float,
                           total_events: int, domains_with_events: int,
                           weeks_of_data: float, positive_patterns: bool,
                           synergy_count: int, stability_score: float) -> Dict:
        """Get progress toward next stage as a criteria checklist."""
        if current_stage >= 5:
            return {'stage': 5, 'name': 'Mastery', 'progress': 1.0, 'criteria': [], 'at_max': True}

        next_stage = current_stage + 1
        criteria = []

        if current_stage == 1:
            criteria = [
                {'name': '50+ events', 'met': total_events >= 50, 'current': total_events, 'target': 50},
                {'name': '3+ domains', 'met': domains_with_events >= 3, 'current': domains_with_events, 'target': 3},
                {'name': 'Confidence >= 0.3', 'met': confidence >= 0.3, 'current': round(confidence, 2), 'target': 0.3},
            ]
        elif current_stage == 2:
            criteria = [
                {'name': '4+ weeks of data', 'met': weeks_of_data >= 4, 'current': round(weeks_of_data, 1), 'target': 4},
                {'name': 'Positive patterns', 'met': positive_patterns, 'current': positive_patterns, 'target': True},
                {'name': 'Confidence > 0.55', 'met': confidence > 0.55, 'current': round(confidence, 2), 'target': 0.55},
            ]
        elif current_stage == 3:
            criteria = [
                {'name': '3+ synergies', 'met': synergy_count >= 3, 'current': synergy_count, 'target': 3},
                {'name': '8+ weeks of data', 'met': weeks_of_data >= 8, 'current': round(weeks_of_data, 1), 'target': 8},
                {'name': 'Confidence > 0.75', 'met': confidence > 0.75, 'current': round(confidence, 2), 'target': 0.75},
            ]
        elif current_stage == 4:
            criteria = [
                {'name': '12+ weeks of data', 'met': weeks_of_data >= 12, 'current': round(weeks_of_data, 1), 'target': 12},
                {'name': 'Stability > 0.85', 'met': stability_score > 0.85, 'current': round(stability_score, 2), 'target': 0.85},
                {'name': 'Confidence > 0.9', 'met': confidence > 0.9, 'current': round(confidence, 2), 'target': 0.9},
            ]

        met_count = sum(1 for c in criteria if c['met'])
        progress = met_count / len(criteria) if criteria else 0

        return {
            'stage': current_stage,
            'name': STAGE_NAMES[current_stage],
            'next_stage': next_stage,
            'next_name': STAGE_NAMES[next_stage],
            'progress': round(progress, 2),
            'criteria': criteria,
            'at_max': False,
        }
```

- [ ] **Step 5: Run evolution engine tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_evolution_engine.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add evolution_engine.py tests/test_evolution_engine.py
git commit -m "feat(profile): add EvolutionEngine with stages, confidence, synergies"
```

---

## Chunk 4: Core Orchestrator

### Task 5: UserProfileEngine — Orchestrator

**Files:**
- Create: `user_profile_engine.py`
- Create: `tests/test_profile_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_profile_engine.py
"""Tests for UserProfileEngine orchestrator."""

import pytest
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestUserProfileEngine:
    """Test the main orchestrator."""

    @pytest.fixture
    def engine(self, tmp_path):
        from user_profile_engine import UserProfileEngine
        eng = UserProfileEngine(db_path=str(tmp_path / 'test.db'))
        yield eng
        eng.close()

    def test_log_event_and_get_profile(self, engine):
        engine.log_event('user1', 'emotional', 'emotion_classified',
                         {'emotion': 'joy', 'family': 'Joy', 'confidence': 0.9},
                         'nanogpt', 0.9)
        profile = engine.get_profile('user1')
        assert profile is not None
        assert profile['user_id'] == 'user1'

    def test_process_updates_fingerprint(self, engine):
        for i in range(10):
            engine.log_event('user1', 'emotional', 'emotion_classified',
                             {'emotion': 'joy', 'family': 'Joy', 'confidence': 0.9},
                             'nanogpt', 0.9)
        engine.process('user1')
        profile = engine.get_profile('user1')
        fingerprint = json.loads(profile['fingerprint_json']) if profile.get('fingerprint_json') else {}
        assert 'emotional' in fingerprint

    def test_log_event_noop_when_closed(self, engine):
        engine.close()
        # Should not raise
        engine.log_event('user1', 'emotional', 'e1', {}, 'nanogpt')

    def test_get_snapshot_returns_current(self, engine):
        engine.log_event('user1', 'emotional', 'emotion_classified',
                         {'emotion': 'joy', 'family': 'Joy'}, 'nanogpt')
        engine.process('user1')
        snapshot = engine.get_snapshot('user1')
        assert 'fingerprint' in snapshot
        assert 'stage' in snapshot
        assert 'confidence' in snapshot

    def test_evolution_stage_starts_at_nascent(self, engine):
        engine.log_event('user1', 'emotional', 'e1', {}, 'nanogpt')
        engine.process('user1')
        profile = engine.get_profile('user1')
        assert profile['evolution_stage'] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Implement UserProfileEngine**

```python
# user_profile_engine.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - User Profile Engine
===============================================================================
Central orchestrator for the genetic fingerprint system. Manages event logging,
domain processing, fingerprint synthesis, and evolution tracking.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from biometrics, emotions, therapy sessions
===============================================================================
"""

import time
import json
import logging
import threading
from typing import Optional, Dict, List, Any

from profile_db import ProfileDB, VALID_DOMAINS, VALID_SOURCES
from domain_processors import get_all_processors
from evolution_engine import EvolutionEngine, STAGE_NAMES

logger = logging.getLogger(__name__)


# Biometric event types that need rate limiting (1/min/domain/user)
BIOMETRIC_EVENT_TYPES = {'hr_reading', 'hrv_reading', 'eda_reading', 'breathing_reading'}


class UserProfileEngine:
    """Central orchestrator for the genetic fingerprint system.

    Handles:
    - Event logging with rate limiting for biometric events
    - Domain processing via 8 isolated processors
    - Fingerprint synthesis
    - Evolution stage management
    - Snapshot scheduling
    """

    def __init__(self, db_path: str = 'data/profile.db'):
        self.db = ProfileDB(db_path)
        self.evolution = EvolutionEngine(self.db)
        self.processors = {p.domain: p for p in get_all_processors()}
        self._closed = False

        # Rate limiting for biometric events: {(user_id, domain): last_timestamp}
        self._rate_limit_cache = {}
        self._rate_limit_lock = threading.Lock()

        # Consecutive stage criteria met counter per user
        self._consecutive_met = {}

        logger.info("UserProfileEngine initialized with %d domain processors", len(self.processors))

    def log_event(self, user_id: str, domain: str, event_type: str,
                  payload: dict, source: str = 'nanogpt',
                  confidence: float = None) -> Optional[int]:
        """Log an event to the profile event store.

        Rate-limits biometric events to 1 per minute per domain per user.
        Safe to call even when engine is closed (no-op).
        """
        if self._closed:
            return None

        # Rate limit biometric events
        if event_type in BIOMETRIC_EVENT_TYPES:
            cache_key = (user_id, domain)
            now = time.time()
            with self._rate_limit_lock:
                last_time = self._rate_limit_cache.get(cache_key, 0)
                if now - last_time < 60:
                    return None  # Skip — too soon
                self._rate_limit_cache[cache_key] = now

        try:
            # Ensure profile exists
            self.db.get_or_create_profile(user_id)
            event_id = self.db.log_event(user_id, domain, event_type, payload, source, confidence)
            return event_id
        except Exception as e:
            logger.error("Failed to log event for %s: %s", user_id, e)
            return None

    def process(self, user_id: str) -> Dict:
        """Process all domains for a user and update fingerprint.

        1. Run each domain processor on its events
        2. Compute overall confidence
        3. Evaluate evolution stage
        4. Update profile in DB

        Returns: updated fingerprint dict
        """
        profile = self.db.get_or_create_profile(user_id)
        fingerprint = {}
        domain_meta = {}

        for domain, processor in self.processors.items():
            events = self.db.get_events(user_id, domain=domain, limit=10000)
            # Events come newest-first; processors expect oldest-first
            events.reverse()
            domain_score = processor.compute(events)
            fingerprint[domain] = domain_score

            # Collect metadata for confidence computation
            latest_time = self.db.get_latest_event_time(user_id, domain)
            sources = self.db.get_domain_sources(user_id, domain)
            domain_meta[domain] = {
                'latest_event_time': latest_time,
                'source_count': len(sources),
            }

        # Compute confidence
        confidence = self.evolution.compute_overall_confidence(fingerprint, domain_meta)

        # Evaluate stage
        total_events = sum(s.get('event_count', 0) for s in fingerprint.values())
        domains_with_events = sum(1 for s in fingerprint.values() if s.get('event_count', 0) > 0)

        created_at = profile.get('created_at', time.time())
        weeks_of_data = (time.time() - created_at) / (7 * 86400)

        # Check for positive patterns (simplified: recovery rate > 0.1 in emotional domain)
        emotional = fingerprint.get('emotional', {})
        emotional_metrics = emotional.get('metrics', {})
        positive_patterns = emotional_metrics.get('recovery_rate', 0) > 0.1

        synergies = self.db.get_synergies(user_id)
        synergy_count = len(synergies)

        # Stability: inverse of emotional volatility
        volatility = emotional_metrics.get('volatility', 1.0)
        stability_score = 1.0 - volatility

        current_stage = profile.get('evolution_stage', 1)
        consecutive = self._consecutive_met.get(user_id, 0)

        stage_result = self.evolution.evaluate_stage(
            current_stage=current_stage,
            confidence=confidence,
            total_events=total_events,
            domains_with_events=domains_with_events,
            weeks_of_data=weeks_of_data,
            positive_patterns=positive_patterns,
            synergy_count=synergy_count,
            stability_score=stability_score,
            consecutive_met=consecutive,
        )

        # Track consecutive met
        if stage_result['new_stage'] > current_stage and not stage_result['changed']:
            # Criteria met but consecutive not reached yet
            self._consecutive_met[user_id] = consecutive + 1
        elif stage_result['changed']:
            self._consecutive_met[user_id] = 0
        else:
            # Criteria not met — check if they would be met ignoring consecutive
            test_result = self.evolution.evaluate_stage(
                current_stage=current_stage, confidence=confidence,
                total_events=total_events, domains_with_events=domains_with_events,
                weeks_of_data=weeks_of_data, positive_patterns=positive_patterns,
                synergy_count=synergy_count, stability_score=stability_score,
                consecutive_met=3  # force check
            )
            if test_result['new_stage'] > current_stage:
                self._consecutive_met[user_id] = consecutive + 1
            else:
                self._consecutive_met[user_id] = 0

        new_stage = stage_result['new_stage']
        evolution_count = profile.get('evolution_count', 0) + 1

        # Update DB
        self.db.update_profile(
            user_id,
            fingerprint=fingerprint,
            confidence=confidence,
            evolution_stage=new_stage,
            evolution_count=evolution_count,
        )

        # Stage change snapshot
        if stage_result['changed']:
            self.db.save_snapshot(user_id, 'stage_change', fingerprint, new_stage, confidence)
            logger.info("User %s evolved to stage %d (%s)", user_id, new_stage, stage_result['stage_name'])

        return fingerprint

    def get_profile(self, user_id: str) -> Optional[Dict]:
        """Get the current profile for a user."""
        return self.db.get_or_create_profile(user_id)

    def get_snapshot(self, user_id: str) -> Dict:
        """Get current fingerprint snapshot."""
        profile = self.db.get_or_create_profile(user_id)
        fingerprint = json.loads(profile['fingerprint_json']) if profile.get('fingerprint_json') else {}
        return {
            'user_id': user_id,
            'fingerprint': fingerprint,
            'stage': profile.get('evolution_stage', 1),
            'stage_name': STAGE_NAMES.get(profile.get('evolution_stage', 1), 'Nascent'),
            'confidence': profile.get('confidence', 0.0),
            'evolution_count': profile.get('evolution_count', 0),
            'last_evolved': profile.get('last_evolved'),
        }

    def get_evolution(self, user_id: str) -> Dict:
        """Get full evolution timeline."""
        profile = self.db.get_or_create_profile(user_id)
        snapshots = self.db.get_snapshots(user_id)
        return {
            'user_id': user_id,
            'current_stage': profile.get('evolution_stage', 1),
            'current_stage_name': STAGE_NAMES.get(profile.get('evolution_stage', 1), 'Nascent'),
            'confidence': profile.get('confidence', 0.0),
            'evolution_count': profile.get('evolution_count', 0),
            'snapshots': snapshots,
        }

    def get_stage_progress(self, user_id: str) -> Dict:
        """Get progress toward next evolution stage."""
        profile = self.db.get_or_create_profile(user_id)
        fingerprint = json.loads(profile['fingerprint_json']) if profile.get('fingerprint_json') else {}

        total_events = sum(d.get('event_count', 0) for d in fingerprint.values())
        domains_with_events = sum(1 for d in fingerprint.values() if d.get('event_count', 0) > 0)
        created_at = profile.get('created_at', time.time())
        weeks_of_data = (time.time() - created_at) / (7 * 86400)

        emotional = fingerprint.get('emotional', {}).get('metrics', {})
        positive_patterns = emotional.get('recovery_rate', 0) > 0.1
        stability_score = 1.0 - emotional.get('volatility', 1.0)

        synergies = self.db.get_synergies(user_id)

        return self.evolution.get_stage_progress(
            current_stage=profile.get('evolution_stage', 1),
            confidence=profile.get('confidence', 0.0),
            total_events=total_events,
            domains_with_events=domains_with_events,
            weeks_of_data=weeks_of_data,
            positive_patterns=positive_patterns,
            synergy_count=len(synergies),
            stability_score=stability_score,
        )

    def delete_user(self, user_id: str):
        """Purge all data for a user."""
        self.db.delete_user(user_id)
        self._consecutive_met.pop(user_id, None)
        with self._rate_limit_lock:
            keys_to_remove = [k for k in self._rate_limit_cache if k[0] == user_id]
            for k in keys_to_remove:
                del self._rate_limit_cache[k]
        logger.info("Deleted all profile data for user %s", user_id)

    def close(self):
        """Shutdown the engine."""
        self._closed = True
        self.db.close()
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_engine.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add user_profile_engine.py tests/test_profile_engine.py
git commit -m "feat(profile): add UserProfileEngine orchestrator"
```

---

## Chunk 5: API Layer

### Task 6: Profile API Blueprint

**Files:**
- Create: `profile_api.py`
- Create: `tests/test_profile_api.py`

- [ ] **Step 1: Write failing tests for API endpoints**

```python
# tests/test_profile_api.py
"""Tests for profile API endpoints."""

import pytest
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def app(tmp_path):
    from flask import Flask
    from profile_api import create_profile_blueprint
    from user_profile_engine import UserProfileEngine

    app = Flask(__name__)
    engine = UserProfileEngine(db_path=str(tmp_path / 'test.db'))
    bp = create_profile_blueprint(engine)
    app.register_blueprint(bp)
    app.config['TESTING'] = True

    # Set service keys for testing
    os.environ['PROFILE_SERVICE_KEY_BACKEND'] = 'test-backend-key'
    os.environ['PROFILE_SERVICE_KEY_FRONTEND'] = 'test-frontend-key'

    yield app
    engine.close()


@pytest.fixture
def client(app):
    return app.test_client()


class TestProfileHealthEndpoint:

    def test_health_check(self, client):
        resp = client.get('/api/profile/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'healthy'


class TestIngestEndpoint:

    def test_ingest_requires_service_key(self, client):
        resp = client.post('/api/profile/ingest', json={
            'user_id': 'u1', 'domain': 'emotional',
            'event_type': 'emotion_classified', 'payload': {}
        })
        assert resp.status_code == 401

    def test_ingest_with_valid_key(self, client):
        resp = client.post('/api/profile/ingest',
            headers={'X-Service-Key': 'test-backend-key'},
            json={
                'user_id': 'u1', 'domain': 'emotional',
                'event_type': 'emotion_classified',
                'payload': {'emotion': 'joy'},
                'source': 'backend'
            })
        assert resp.status_code == 200


class TestSnapshotEndpoint:

    def test_get_snapshot(self, client):
        # Ingest some data first
        client.post('/api/profile/ingest',
            headers={'X-Service-Key': 'test-backend-key'},
            json={'user_id': 'u1', 'domain': 'emotional',
                  'event_type': 'emotion_classified',
                  'payload': {'emotion': 'joy', 'family': 'Joy'},
                  'source': 'backend'})
        resp = client.get('/api/profile/u1/snapshot')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'fingerprint' in data
        assert 'stage' in data


class TestDeleteEndpoint:

    def test_delete_user(self, client):
        client.post('/api/profile/ingest',
            headers={'X-Service-Key': 'test-backend-key'},
            json={'user_id': 'u1', 'domain': 'emotional',
                  'event_type': 'e1', 'payload': {}, 'source': 'backend'})
        resp = client.delete('/api/profile/u1')
        assert resp.status_code == 200


class TestFingerprintSyncEndpoint:

    def test_sync_returns_fingerprint(self, client):
        client.post('/api/profile/ingest',
            headers={'X-Service-Key': 'test-backend-key'},
            json={'user_id': 'u1', 'domain': 'emotional',
                  'event_type': 'emotion_classified',
                  'payload': {'emotion': 'joy', 'family': 'Joy'},
                  'source': 'backend'})
        resp = client.post('/api/v1/users/u1/genetic-fingerprint/sync',
                           json={'user_id': 'u1'})
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'fingerprint' in data
        assert 'evolution_stage' in data
```

- [ ] **Step 2: Implement profile_api.py**

```python
# profile_api.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Profile API Blueprint
===============================================================================
Flask Blueprint providing REST endpoints for user profiles, genetic fingerprints,
evolution tracking, and ecosystem integration.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- Frontend GeneticFingerprintService.ts
- Ecosystem services via ingest webhook
===============================================================================
"""

import os
import json
import time
import logging
from functools import wraps

from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

# Valid service keys from environment
SERVICE_KEYS = {}


def _load_service_keys():
    """Load service API keys from environment."""
    global SERVICE_KEYS
    SERVICE_KEYS = {}
    for name in ('BACKEND', 'FRONTEND', 'MASTER'):
        key = os.environ.get(f'PROFILE_SERVICE_KEY_{name}')
        if key:
            SERVICE_KEYS[key] = name.lower()


def require_service_key(f):
    """Decorator requiring valid X-Service-Key header."""
    @wraps(f)
    def decorated(*args, **kwargs):
        _load_service_keys()
        key = request.headers.get('X-Service-Key')
        if not key or key not in SERVICE_KEYS:
            return jsonify({'error': 'Invalid or missing service key'}), 401
        request.service_name = SERVICE_KEYS[key]
        return f(*args, **kwargs)
    return decorated


def create_profile_blueprint(engine) -> Blueprint:
    """Create Flask Blueprint with all profile endpoints.

    Args:
        engine: UserProfileEngine instance
    """
    bp = Blueprint('profile', __name__)

    # --- Health ---

    @bp.route('/api/profile/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy',
            'service': 'profile-engine',
            'timestamp': time.time(),
        })

    # --- Ingest webhook ---

    @bp.route('/api/profile/ingest', methods=['POST'])
    @require_service_key
    def ingest():
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body required'}), 400

        user_id = data.get('user_id')
        domain = data.get('domain')
        event_type = data.get('event_type')
        payload = data.get('payload', {})
        source = data.get('source', request.service_name)
        confidence = data.get('confidence')

        if not all([user_id, domain, event_type]):
            return jsonify({'error': 'user_id, domain, and event_type required'}), 400

        event_id = engine.log_event(user_id, domain, event_type, payload, source, confidence)
        return jsonify({'event_id': event_id, 'status': 'logged'})

    # --- Profile endpoints ---

    @bp.route('/api/profile/<user_id>/snapshot', methods=['GET'])
    def snapshot(user_id):
        engine.process(user_id)
        return jsonify(engine.get_snapshot(user_id))

    @bp.route('/api/profile/<user_id>/evolution', methods=['GET'])
    def evolution(user_id):
        return jsonify(engine.get_evolution(user_id))

    @bp.route('/api/profile/<user_id>/evolution/stage', methods=['GET'])
    def evolution_stage(user_id):
        return jsonify(engine.get_stage_progress(user_id))

    @bp.route('/api/profile/<user_id>/domain/<domain>', methods=['GET'])
    def domain_detail(user_id, domain):
        engine.process(user_id)
        profile = engine.get_profile(user_id)
        fingerprint = json.loads(profile['fingerprint_json']) if profile.get('fingerprint_json') else {}
        domain_data = fingerprint.get(domain, {})
        events = engine.db.get_events(user_id, domain=domain, limit=50)
        return jsonify({'domain': domain, 'data': domain_data, 'recent_events': events})

    @bp.route('/api/profile/<user_id>/synergies', methods=['GET'])
    def synergies(user_id):
        return jsonify({'synergies': engine.db.get_synergies(user_id)})

    @bp.route('/api/profile/<user_id>/events', methods=['GET'])
    def events(user_id):
        domain = request.args.get('domain')
        start_time = request.args.get('start_time', type=float)
        end_time = request.args.get('end_time', type=float)
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        events = engine.db.get_events(user_id, domain, start_time, end_time, min(limit, 1000), offset)
        return jsonify({'events': events, 'count': len(events)})

    @bp.route('/api/profile/<user_id>/export', methods=['GET'])
    def export(user_id):
        profile = engine.get_profile(user_id)
        snapshots = engine.db.get_snapshots(user_id)
        synergies = engine.db.get_synergies(user_id)
        event_count = engine.db.get_event_count(user_id)
        return jsonify({
            'profile': profile,
            'snapshots': snapshots,
            'synergies': synergies,
            'event_count': event_count,
        })

    @bp.route('/api/profile/<user_id>', methods=['DELETE'])
    def delete(user_id):
        engine.delete_user(user_id)
        return jsonify({'status': 'deleted', 'user_id': user_id})

    # --- Ecosystem service endpoints ---

    @bp.route('/api/profile/<user_id>/predictions', methods=['GET'])
    def predictions(user_id):
        engine.process(user_id)
        profile = engine.get_profile(user_id)
        fingerprint = json.loads(profile['fingerprint_json']) if profile.get('fingerprint_json') else {}
        preds = _compute_predictions(fingerprint, profile)
        return jsonify(preds)

    @bp.route('/api/profile/<user_id>/context', methods=['GET'])
    @require_service_key
    def context(user_id):
        profile = engine.get_profile(user_id)
        fingerprint = json.loads(profile['fingerprint_json']) if profile.get('fingerprint_json') else {}
        summary = {}
        for domain, data in fingerprint.items():
            summary[domain] = {
                'score': data.get('score', 0),
                'event_count': data.get('event_count', 0),
            }
        return jsonify({
            'user_id': user_id,
            'stage': profile.get('evolution_stage', 1),
            'confidence': profile.get('confidence', 0.0),
            'domains': summary,
        })

    @bp.route('/api/profile/<user_id>/domain/<domain>/score', methods=['GET'])
    @require_service_key
    def domain_score(user_id, domain):
        profile = engine.get_profile(user_id)
        fingerprint = json.loads(profile['fingerprint_json']) if profile.get('fingerprint_json') else {}
        domain_data = fingerprint.get(domain, {})
        return jsonify({
            'domain': domain,
            'score': domain_data.get('score', 0.0),
            'event_count': domain_data.get('event_count', 0),
        })

    # --- Frontend-compatible endpoints ---

    @bp.route('/api/v1/users/<user_id>/genetic-fingerprint/sync', methods=['POST'])
    def fingerprint_sync(user_id):
        engine.process(user_id)
        snapshot = engine.get_snapshot(user_id)
        synergies = engine.db.get_synergies(user_id)
        return jsonify({
            'fingerprint': snapshot['fingerprint'],
            'confidence': snapshot['confidence'],
            'evolution_stage': snapshot['stage'],
            'stage_name': snapshot['stage_name'],
            'evolution_count': snapshot['evolution_count'],
            'synergies': synergies,
        })

    @bp.route('/api/v1/users/<user_id>/insights', methods=['POST'])
    def insights(user_id):
        engine.process(user_id)
        profile = engine.get_profile(user_id)
        fingerprint = json.loads(profile['fingerprint_json']) if profile.get('fingerprint_json') else {}
        insights_list = _generate_insights(fingerprint, profile)
        return jsonify({'insights': insights_list})

    @bp.route('/api/v1/cognitive/theory-of-mind', methods=['POST'])
    def theory_of_mind():
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default')
        engine.process(user_id)
        profile = engine.get_profile(user_id)
        fingerprint = json.loads(profile['fingerprint_json']) if profile.get('fingerprint_json') else {}
        big_five = _compute_big_five(fingerprint)
        return jsonify({'user_id': user_id, 'big_five': big_five})

    return bp


def _generate_insights(fingerprint: dict, profile: dict) -> list:
    """Generate AI insights from fingerprint data."""
    insights = []

    emotional = fingerprint.get('emotional', {}).get('metrics', {})
    if emotional.get('dominant_emotion'):
        insights.append({
            'category': 'Emotional',
            'insight': f"Your dominant emotion is {emotional['dominant_emotion']} with an emotional range of {emotional.get('emotional_range', 0)} distinct emotions.",
            'confidence': 0.8,
            'actionable': emotional.get('volatility', 0) > 0.5,
            'action': 'Consider mindfulness exercises to improve emotional stability.' if emotional.get('volatility', 0) > 0.5 else None,
        })

    biometric = fingerprint.get('biometric', {}).get('metrics', {})
    if biometric.get('resting_hr'):
        insights.append({
            'category': 'Biometric',
            'insight': f"Your resting heart rate baseline is {biometric['resting_hr']} BPM with stress ratio of {biometric.get('stress_ratio', 0):.0%}.",
            'confidence': 0.7,
            'actionable': biometric.get('stress_ratio', 0) > 0.3,
            'action': 'Your stress levels are elevated. Try breathing exercises.' if biometric.get('stress_ratio', 0) > 0.3 else None,
        })

    temporal = fingerprint.get('temporal', {}).get('metrics', {})
    if temporal.get('chronotype'):
        insights.append({
            'category': 'Temporal',
            'insight': f"Your chronotype appears to be {temporal['chronotype']} with peak activity hours at {temporal.get('peak_hours', [])}.",
            'confidence': 0.6,
        })

    return insights


def _compute_big_five(fingerprint: dict) -> dict:
    """Compute Big Five personality traits from DNA domains."""
    behavioral = fingerprint.get('behavioral', {}).get('metrics', {})
    emotional = fingerprint.get('emotional', {}).get('metrics', {})
    social = fingerprint.get('social', {}).get('metrics', {})
    linguistic = fingerprint.get('linguistic', {}).get('metrics', {})

    return {
        'openness': {
            'score': round(min(1.0, emotional.get('emotional_range', 0) / 15), 2),
            'sources': ['emotional_range', 'novelty_seeking'],
        },
        'conscientiousness': {
            'score': round(behavioral.get('completion_rate', 0.5), 2),
            'sources': ['completion_rate', 'engagement_streak'],
        },
        'extraversion': {
            'score': round(min(1.0, social.get('daily_interaction_rate', 0) / 10), 2),
            'sources': ['interaction_frequency', 'group_affinity'],
        },
        'agreeableness': {
            'score': round(behavioral.get('intervention_response_rate', 0.5), 2),
            'sources': ['feedback_receptivity', 'communication_style'],
        },
        'neuroticism': {
            'score': round(emotional.get('volatility', 0.5), 2),
            'sources': ['stress_response', 'emotional_volatility'],
        },
    }
```

- [ ] **Step 3: Run API tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_api.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add profile_api.py tests/test_profile_api.py
git commit -m "feat(profile): add Profile API Blueprint with all endpoints"
```

---

## Chunk 6: Integration with Existing Modules

### Task 7: Wire Profile Engine into emotion_api_server.py

**Files:**
- Modify: `emotion_api_server.py`

- [ ] **Step 1: Add profile engine initialization**

Near the top of `emotion_api_server.py`, after existing imports, add:

```python
# Profile engine integration
try:
    from user_profile_engine import UserProfileEngine
    from profile_api import create_profile_blueprint
    HAS_PROFILE_ENGINE = True
except ImportError:
    HAS_PROFILE_ENGINE = False

profile_engine = None
```

- [ ] **Step 2: Initialize profile engine after Flask app creation**

After the Flask app is created (after `app = Flask(__name__)`), add:

```python
# Initialize profile engine
if HAS_PROFILE_ENGINE:
    try:
        profile_engine = UserProfileEngine(db_path='data/profile.db')
        profile_bp = create_profile_blueprint(profile_engine)
        app.register_blueprint(profile_bp)
        print("✓ Profile engine initialized")
    except Exception as e:
        print(f"⚠ Profile engine failed to initialize: {e}")
        profile_engine = None
```

- [ ] **Step 3: Add log_event calls to /api/emotion/analyze endpoint**

Inside the analyze endpoint handler, after the emotion result is computed, add:

```python
# Log to profile engine
if profile_engine:
    user_id = data.get('user_id', 'default')
    profile_engine.log_event(user_id, 'emotional', 'emotion_classified', {
        'emotion': result.get('emotion', 'neutral'),
        'family': result.get('family', 'Neutral'),
        'confidence': result.get('confidence', 0.5),
    }, 'nanogpt', result.get('confidence'))
    # Linguistic DNA from input text
    if text:
        profile_engine.log_event(user_id, 'linguistic', 'text_analyzed', {
            'text': text[:500],  # Truncate for storage
            'word_count': len(text.split()),
        }, 'nanogpt')
    # Temporal DNA
    profile_engine.log_event(user_id, 'temporal', 'session_started', {
        'endpoint': 'analyze',
    }, 'nanogpt')
```

- [ ] **Step 4: Add log_event calls to transition/record endpoint**

Inside the transition record handler, add:

```python
if profile_engine:
    profile_engine.log_event(user_id, 'behavioral', 'transition_recorded', {
        'emotion': emotion,
        'family': family,
    }, 'nanogpt', confidence)
```

- [ ] **Step 5: Add log_event calls to RuView biometric endpoints**

Inside RuView biometric handlers, add:

```python
if profile_engine:
    profile_engine.log_event(user_id, 'biometric', 'hr_reading', {
        'hr': biometrics.get('heart_rate'),
        'hrv': biometrics.get('hrv'),
        'eda': biometrics.get('eda'),
    }, 'nanogpt')
```

- [ ] **Step 6: Test the integration**

Run: `cd /Users/bel/quantara-nanoGPT && python -c "from emotion_api_server import app; print('Server imports OK')" 2>&1`
Expected: "Server imports OK" (no import errors)

- [ ] **Step 7: Commit**

```bash
git add emotion_api_server.py
git commit -m "feat(profile): integrate profile engine into emotion API server"
```

### Task 8: Wire into emotion_transition_tracker.py

**Files:**
- Modify: `emotion_transition_tracker.py`

- [ ] **Step 1: Add optional profile engine parameter**

Add `profile_engine=None` parameter to `EmotionTransitionTracker.__init__()` and store it:

```python
self.profile_engine = profile_engine
```

- [ ] **Step 2: Log events in record() method**

After the existing record logic, add:

```python
if self.profile_engine:
    self.profile_engine.log_event(user_id, 'emotional', 'transition_recorded', {
        'emotion': emotion,
        'family': family,
        'confidence': confidence,
    }, 'nanogpt', confidence)
```

- [ ] **Step 3: Log pattern detection results**

In `detect_patterns()`, after patterns are computed, add:

```python
if self.profile_engine and patterns:
    for pattern in patterns:
        self.profile_engine.log_event(user_id, 'emotional', 'pattern_detected', {
            'pattern_type': pattern.get('type', 'unknown'),
            'severity': pattern.get('severity', 'info'),
        }, 'nanogpt')
```

- [ ] **Step 4: Run existing tracker tests to verify nothing breaks**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_transition_tracker.py -v`
Expected: All existing tests PASS

- [ ] **Step 5: Commit**

```bash
git add emotion_transition_tracker.py
git commit -m "feat(profile): wire profile engine into emotion transition tracker"
```

---

## Chunk 7: Full Integration Test

### Task 9: End-to-End Integration Test

**Files:**
- Create: `tests/test_profile_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_profile_integration.py
"""End-to-end integration test for the profile engine."""

import pytest
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestProfileIntegration:
    """Test full flow: log events → process → fingerprint → evolution."""

    @pytest.fixture
    def engine(self, tmp_path):
        from user_profile_engine import UserProfileEngine
        eng = UserProfileEngine(db_path=str(tmp_path / 'integration.db'))
        yield eng
        eng.close()

    def test_full_lifecycle(self, engine):
        """Simulate a user going from Nascent through Awareness."""
        user_id = 'integration_user'

        # Log diverse events across multiple domains
        for i in range(20):
            engine.log_event(user_id, 'emotional', 'emotion_classified', {
                'emotion': ['joy', 'sadness', 'calm', 'anger'][i % 4],
                'family': ['Joy', 'Sadness', 'Calm', 'Anger'][i % 4],
                'confidence': 0.8,
            }, 'nanogpt', 0.8)

        for i in range(15):
            engine.log_event(user_id, 'biometric', 'biometric_reading', {
                'hr': 70 + i, 'hrv': 55 + i, 'eda': 3.0,
            }, 'nanogpt')

        for i in range(10):
            engine.log_event(user_id, 'behavioral', 'session_completed', {
                'technique': 'breathing',
                'response_positive': True,
            }, 'nanogpt')

        for i in range(10):
            engine.log_event(user_id, 'linguistic', 'text_analyzed', {
                'text': f'I am feeling good today test sentence {i}',
            }, 'nanogpt')

        # Process
        fingerprint = engine.process(user_id)

        # Verify all logged domains appear in fingerprint
        assert 'emotional' in fingerprint
        assert 'biometric' in fingerprint
        assert 'behavioral' in fingerprint
        assert 'linguistic' in fingerprint

        # Verify emotional metrics
        assert fingerprint['emotional']['metrics']['dominant_emotion'] in ['joy', 'sadness', 'calm', 'anger']
        assert fingerprint['emotional']['event_count'] == 20

        # Verify profile was created
        profile = engine.get_profile(user_id)
        assert profile['evolution_stage'] == 1  # Still Nascent (need 50+ events across 3+ domains)
        assert profile['confidence'] > 0.0

        # Verify snapshot works
        snapshot = engine.get_snapshot(user_id)
        assert snapshot['stage'] == 1
        assert snapshot['fingerprint'] == fingerprint

    def test_api_integration(self, engine, tmp_path):
        """Test API endpoints work with real engine."""
        from flask import Flask
        from profile_api import create_profile_blueprint

        app = Flask(__name__)
        os.environ['PROFILE_SERVICE_KEY_BACKEND'] = 'test-key'
        bp = create_profile_blueprint(engine)
        app.register_blueprint(bp)

        with app.test_client() as client:
            # Ingest via API
            resp = client.post('/api/profile/ingest',
                headers={'X-Service-Key': 'test-key'},
                json={
                    'user_id': 'api_user', 'domain': 'emotional',
                    'event_type': 'emotion_classified',
                    'payload': {'emotion': 'joy', 'family': 'Joy'},
                    'source': 'backend'
                })
            assert resp.status_code == 200

            # Get snapshot via API
            resp = client.get('/api/profile/api_user/snapshot')
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['stage'] == 1

            # Sync fingerprint (frontend-compatible)
            resp = client.post('/api/v1/users/api_user/genetic-fingerprint/sync',
                               json={'user_id': 'api_user'})
            assert resp.status_code == 200
            data = resp.get_json()
            assert 'fingerprint' in data
            assert 'evolution_stage' in data

            # Delete user
            resp = client.delete('/api/profile/api_user')
            assert resp.status_code == 200
```

- [ ] **Step 2: Run integration tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_integration.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_db.py tests/test_domain_processors.py tests/test_evolution_engine.py tests/test_profile_engine.py tests/test_profile_api.py tests/test_profile_integration.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_profile_integration.py
git commit -m "feat(profile): add end-to-end integration tests"
```

- [ ] **Step 5: Final commit — update requirements.txt if needed**

Check if scipy is already in requirements (it is). No changes needed.

```bash
git log --oneline -10  # Verify all commits
```

---

## Chunk 8: Missing Spec Features — Sync Worker, Retention, WebSocket, Predictions

### Task 10: ProfileSyncWorker — Ecosystem Polling

**Files:**
- Create: `profile_sync_worker.py`
- Create: `tests/test_profile_sync.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_profile_sync.py
"""Tests for ProfileSyncWorker ecosystem polling."""

import pytest
import os
import sys
import time
import threading
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestProfileSyncWorker:

    @pytest.fixture
    def engine(self, tmp_path):
        from user_profile_engine import UserProfileEngine
        eng = UserProfileEngine(db_path=str(tmp_path / 'test.db'))
        yield eng
        eng.close()

    def test_worker_starts_and_stops(self, engine):
        from profile_sync_worker import ProfileSyncWorker
        worker = ProfileSyncWorker(engine, poll_interval=1)
        worker.start()
        assert worker.is_alive()
        worker.stop()
        worker.join(timeout=3)
        assert not worker.is_alive()

    def test_backoff_increases_on_failure(self, engine):
        from profile_sync_worker import ProfileSyncWorker
        worker = ProfileSyncWorker(engine, poll_interval=5)
        assert worker._current_interval == 5
        worker._on_poll_failure()
        assert worker._current_interval == 10
        worker._on_poll_failure()
        assert worker._current_interval == 20

    def test_backoff_caps_at_60_min(self, engine):
        from profile_sync_worker import ProfileSyncWorker
        worker = ProfileSyncWorker(engine, poll_interval=5)
        for _ in range(20):
            worker._on_poll_failure()
        assert worker._current_interval <= 3600

    def test_backoff_resets_on_success(self, engine):
        from profile_sync_worker import ProfileSyncWorker
        worker = ProfileSyncWorker(engine, poll_interval=5)
        worker._on_poll_failure()
        worker._on_poll_failure()
        assert worker._current_interval > 5
        worker._on_poll_success()
        assert worker._current_interval == 5
```

- [ ] **Step 2: Implement ProfileSyncWorker**

```python
# profile_sync_worker.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Profile Sync Worker
===============================================================================
Background thread that polls Quantara Backend and Master APIs for new data
and ingests it into the profile engine.

Integrates with:
- Neural Workflow AI Engine
- Quantara Backend engines
- Quantara Master workflows
===============================================================================
"""

import time
import logging
import threading
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class ProfileSyncWorker(threading.Thread):
    """Background thread polling ecosystem services for profile data."""

    def __init__(self, engine, poll_interval: int = 300,
                 backend_url: str = None, master_url: str = None):
        super().__init__(daemon=True, name='ProfileSyncWorker')
        self.engine = engine
        self._base_interval = poll_interval
        self._current_interval = poll_interval
        self._max_interval = 3600  # 60 min cap
        self._running = False
        self._stop_event = threading.Event()

        self.backend_url = backend_url or 'http://localhost:3001'
        self.master_url = master_url or 'http://localhost:3002'

    def start(self):
        self._running = True
        super().start()

    def stop(self):
        self._running = False
        self._stop_event.set()

    def run(self):
        logger.info("ProfileSyncWorker started (interval=%ds)", self._base_interval)
        while self._running:
            try:
                self._poll_backend()
                self._poll_master()
                self._on_poll_success()
            except Exception as e:
                logger.error("Sync poll failed: %s", e)
                self._on_poll_failure()
            self._stop_event.wait(timeout=self._current_interval)
            if self._stop_event.is_set():
                break
        logger.info("ProfileSyncWorker stopped")

    def _poll_backend(self):
        """Pull emotion/biometric/stress data from Quantara Backend."""
        try:
            resp = requests.get(f'{self.backend_url}/api/health', timeout=5)
            if resp.status_code != 200:
                return
            # Pull recent emotion data if available
            resp = requests.get(f'{self.backend_url}/api/emotion/recent', timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get('items', []):
                    self.engine.log_event(
                        item.get('user_id', 'default'),
                        item.get('domain', 'emotional'),
                        item.get('event_type', 'backend_sync'),
                        item.get('payload', {}),
                        'backend',
                        item.get('confidence')
                    )
        except requests.RequestException:
            raise  # Let run() handle it

    def _poll_master(self):
        """Pull workflow/case data from Quantara Master."""
        try:
            resp = requests.get(f'{self.master_url}/api/v1/health', timeout=5)
            if resp.status_code != 200:
                return
            resp = requests.get(f'{self.master_url}/api/v1/workflows/recent', timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get('items', []):
                    self.engine.log_event(
                        item.get('user_id', 'default'),
                        'behavioral',
                        'workflow_action',
                        item.get('payload', {}),
                        'master',
                        item.get('confidence')
                    )
        except requests.RequestException:
            raise

    def _on_poll_failure(self):
        self._current_interval = min(self._current_interval * 2, self._max_interval)
        logger.warning("Backoff increased to %ds", self._current_interval)

    def _on_poll_success(self):
        self._current_interval = self._base_interval

    def is_alive(self):
        return self._running and super().is_alive()
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_sync.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add profile_sync_worker.py tests/test_profile_sync.py
git commit -m "feat(profile): add ProfileSyncWorker for ecosystem polling"
```

### Task 11: Event Retention & Aggregation

**Files:**
- Create: `profile_retention.py`
- Create: `tests/test_profile_retention.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_profile_retention.py
"""Tests for event retention and aggregation."""

import pytest
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRetention:

    @pytest.fixture
    def db(self, tmp_path):
        from profile_db import ProfileDB
        db = ProfileDB(str(tmp_path / 'test.db'))
        yield db
        db.close()

    def test_aggregate_old_events(self, db):
        from profile_retention import RetentionManager
        mgr = RetentionManager(db)
        # Log events with old timestamps (40 days ago)
        old_time = time.time() - 40 * 86400
        for i in range(100):
            db.log_event('user1', 'biometric', 'hr_reading',
                         {'hr': 70 + i % 10}, 'nanogpt',
                         timestamp=old_time + i * 60)
        # Run aggregation
        result = mgr.run_aggregation('user1')
        assert result['aggregated'] > 0

    def test_storage_ceiling_triggers_early_aggregation(self, db):
        from profile_retention import RetentionManager
        mgr = RetentionManager(db, ceiling_per_user=50)  # Low ceiling for testing
        for i in range(60):
            db.log_event('user1', 'emotional', 'e1', {'i': i}, 'nanogpt')
        result = mgr.enforce_ceiling('user1')
        count = db.get_event_count('user1')
        assert count <= 50
```

- [ ] **Step 2: Implement RetentionManager**

```python
# profile_retention.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Profile Event Retention
===============================================================================
Tiered event retention: raw → hourly → daily → weekly aggregation.
Enforces per-user storage ceiling.

Integrates with:
- Neural Workflow AI Engine
- Profile Database
===============================================================================
"""

import time
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class RetentionManager:
    """Manages event retention tiers and storage ceilings."""

    def __init__(self, db, ceiling_per_user: int = 500000):
        self.db = db
        self.ceiling_per_user = ceiling_per_user

    def run_aggregation(self, user_id: str) -> Dict:
        """Aggregate old events into summaries per retention tiers."""
        now = time.time()
        aggregated = 0

        # Tier 1: Events 30-90 days old → hourly summaries
        cutoff_30d = now - 30 * 86400
        cutoff_90d = now - 90 * 86400
        aggregated += self._aggregate_range(user_id, cutoff_90d, cutoff_30d, 3600, 'hourly')

        # Tier 2: Events 90-180 days old → daily summaries
        cutoff_180d = now - 180 * 86400
        aggregated += self._aggregate_range(user_id, cutoff_180d, cutoff_90d, 86400, 'daily')

        # Tier 3: Events >180 days old → weekly summaries
        aggregated += self._aggregate_range(user_id, 0, cutoff_180d, 7 * 86400, 'weekly')

        return {'aggregated': aggregated, 'user_id': user_id}

    def _aggregate_range(self, user_id: str, start: float, end: float,
                         window_seconds: int, summary_type: str) -> int:
        """Aggregate events in a time range into windows."""
        conn = self.db._read_conn()
        rows = conn.execute(
            "SELECT * FROM events WHERE user_id = ? AND timestamp >= ? AND timestamp < ? "
            "AND event_type NOT LIKE '%_summary' ORDER BY timestamp",
            (user_id, start, end)
        ).fetchall()
        conn.close()

        if not rows:
            return 0

        events = [dict(r) for r in rows]
        # Group by domain + time window
        windows = {}
        for event in events:
            domain = event['domain']
            window_key = int(event['timestamp'] // window_seconds)
            key = (domain, window_key)
            if key not in windows:
                windows[key] = []
            windows[key].append(event)

        aggregated = 0
        for (domain, window_key), window_events in windows.items():
            if len(window_events) <= 1:
                continue
            # Create summary event
            window_start = window_key * window_seconds
            summary_payload = {
                'type': summary_type,
                'count': len(window_events),
                'window_start': window_start,
                'window_end': window_start + window_seconds,
            }
            self.db.log_event(user_id, domain, f'{summary_type}_summary',
                              summary_payload, 'nanogpt', timestamp=window_start)
            # Delete original events in this window
            event_ids = [e['event_id'] for e in window_events]
            for eid in event_ids:
                self.db._enqueue_write(
                    "DELETE FROM events WHERE event_id = ?", (eid,), wait=True
                )
            aggregated += len(event_ids)

        return aggregated

    def enforce_ceiling(self, user_id: str) -> Dict:
        """If user exceeds event ceiling, trigger early aggregation."""
        count = self.db.get_event_count(user_id)
        if count <= self.ceiling_per_user:
            return {'trimmed': False, 'count': count}

        # Aggressive aggregation: aggregate everything older than 7 days
        cutoff = time.time() - 7 * 86400
        self._aggregate_range(user_id, 0, cutoff, 3600, 'hourly')

        new_count = self.db.get_event_count(user_id)
        return {'trimmed': True, 'before': count, 'after': new_count}
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_retention.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add profile_retention.py tests/test_profile_retention.py
git commit -m "feat(profile): add RetentionManager for event aggregation and ceiling"
```

### Task 12: WebSocket Events & Predictions Endpoint

**Files:**
- Modify: `profile_api.py` — add predictions helper
- Modify: `user_profile_engine.py` — emit WebSocket events, call synergy detection

- [ ] **Step 1: Add _compute_predictions to profile_api.py**

Add after `_compute_big_five`:

```python
def _compute_predictions(fingerprint: dict, profile: dict) -> dict:
    """Compute next-state predictions from fingerprint trends."""
    emotional = fingerprint.get('emotional', {}).get('metrics', {})
    biometric = fingerprint.get('biometric', {}).get('metrics', {})
    temporal = fingerprint.get('temporal', {}).get('metrics', {})

    predictions = {
        'predicted_dominant_family': emotional.get('dominant_family', 'Neutral'),
        'expected_stress_level': biometric.get('stress_ratio', 0.0),
        'optimal_intervention_hours': temporal.get('peak_hours', []),
        'risk_factors': [],
        'window_hours': 48,
    }

    # Risk factors
    if emotional.get('volatility', 0) > 0.6:
        predictions['risk_factors'].append('High emotional volatility — monitor for spirals')
    if biometric.get('stress_ratio', 0) > 0.4:
        predictions['risk_factors'].append('Elevated stress ratio — consider proactive calming exercises')
    if emotional.get('recovery_rate', 1.0) < 0.05:
        predictions['risk_factors'].append('Low recovery rate — emotional support may be needed')

    return predictions
```

- [ ] **Step 2: Add WebSocket emission to UserProfileEngine.process()**

After the stage change snapshot in `process()`, add:

```python
# Emit WebSocket events if socketio is available
if hasattr(self, '_socketio') and self._socketio:
    try:
        self._socketio.emit('profile:updated', {
            'user_id': user_id,
            'domains_changed': list(fingerprint.keys()),
            'confidence': confidence,
        })
        if stage_result['changed']:
            self._socketio.emit('profile:stage-change', {
                'user_id': user_id,
                'old_stage': current_stage,
                'new_stage': new_stage,
                'criteria_met': stage_result['reason'],
            })
    except Exception as e:
        logger.warning("WebSocket emit failed: %s", e)
```

Add a method to wire socketio:

```python
def set_socketio(self, socketio):
    """Wire socket.io instance for real-time events."""
    self._socketio = socketio
```

- [ ] **Step 3: Add synergy detection call to process()**

After computing the fingerprint in `process()`, add:

```python
# Run synergy detection (weekly, but check on every process for simplicity)
try:
    daily_scores = self._get_daily_domain_scores(user_id)
    detected = self.evolution.detect_synergies(user_id, daily_scores)
    for syn in detected:
        self.db.save_synergy(user_id, syn['domain_a'], syn['domain_b'],
                             syn['correlation'], syn['insight'])
except Exception as e:
    logger.warning("Synergy detection failed for %s: %s", user_id, e)
```

Add the helper:

```python
def _get_daily_domain_scores(self, user_id: str) -> dict:
    """Get daily aggregate scores per domain over last 28 days."""
    daily_scores = {}
    now = time.time()
    for domain in self.processors:
        scores = []
        for day_offset in range(28):
            day_start = now - (day_offset + 1) * 86400
            day_end = now - day_offset * 86400
            events = self.db.get_events(user_id, domain=domain,
                                        start_time=day_start, end_time=day_end, limit=10000)
            if events:
                events.reverse()
                result = self.processors[domain].compute(events)
                scores.append(result['score'])
            else:
                scores.append(0.0)
        scores.reverse()  # oldest first
        daily_scores[domain] = scores
    return daily_scores
```

- [ ] **Step 4: Wire socketio in emotion_api_server.py**

After profile engine init, add:

```python
if HAS_PROFILE_ENGINE and profile_engine:
    try:
        from flask_socketio import SocketIO
        socketio = SocketIO(app, cors_allowed_origins="*")
        profile_engine.set_socketio(socketio)
    except ImportError:
        pass
```

- [ ] **Step 5: Add test for predictions endpoint**

Append to `tests/test_profile_api.py`:

```python
class TestPredictionsEndpoint:

    def test_predictions_returns_data(self, client):
        client.post('/api/profile/ingest',
            headers={'X-Service-Key': 'test-backend-key'},
            json={'user_id': 'u1', 'domain': 'emotional',
                  'event_type': 'emotion_classified',
                  'payload': {'emotion': 'joy', 'family': 'Joy'},
                  'source': 'backend'})
        resp = client.get('/api/profile/u1/predictions')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'predicted_dominant_family' in data
        assert 'risk_factors' in data
```

- [ ] **Step 6: Run all tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_api.py tests/test_profile_engine.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add profile_api.py user_profile_engine.py emotion_api_server.py tests/test_profile_api.py
git commit -m "feat(profile): add predictions endpoint, WebSocket events, synergy detection"
```

---

## Chunk 9: Remaining Integrations

### Task 13: Wire into wifi_calibration.py, auto_retrain.py, external_context.py

**Files:**
- Modify: `wifi_calibration.py`
- Modify: `auto_retrain.py`
- Modify: `external_context.py`

- [ ] **Step 1: Add profile engine to wifi_calibration.py**

In `PersonalCalibrationBuffer.__init__()`, add optional `profile_engine=None` param. In the method that adds readings, add:

```python
if self.profile_engine:
    self.profile_engine.log_event(self.profile_id, 'biometric', 'hr_reading', {
        'hr': heart_rate, 'hrv': hrv, 'eda': eda,
    }, 'nanogpt')
```

- [ ] **Step 2: Add profile engine to auto_retrain.py**

In `DriftDetector` or `RetrainWorker`, when drift is detected or retraining triggers, add:

```python
if profile_engine:
    profile_engine.log_event(user_id, 'biometric', 'model_drift_detected', {
        'drift_score': drift_score,
        'retrain_triggered': True,
    }, 'nanogpt')
```

- [ ] **Step 3: Add temporal enrichment to external_context.py**

When external context is fetched, if profile engine is available:

```python
if profile_engine:
    profile_engine.log_event(user_id, 'temporal', 'time_correlation', {
        'hour': datetime.now().hour,
        'day_of_week': datetime.now().strftime('%A'),
        'weather': context.get('weather'),
    }, 'nanogpt')
```

- [ ] **Step 4: Run existing tests to verify nothing breaks**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/ -v --ignore=tests/test_profile_integration.py`
Expected: All existing tests PASS

- [ ] **Step 5: Commit**

```bash
git add wifi_calibration.py auto_retrain.py external_context.py
git commit -m "feat(profile): wire profile engine into WiFi calibration, auto-retrain, external context"
```

### Task 14: Snapshot Scheduling

**Files:**
- Modify: `user_profile_engine.py` — add snapshot scheduler

- [ ] **Step 1: Add snapshot scheduling to UserProfileEngine**

Add a method that checks and creates missed snapshots, called from `process()`:

```python
def _check_snapshots(self, user_id: str, fingerprint: dict, stage: int, confidence: float):
    """Check for and create any missed scheduled snapshots."""
    missed = self.evolution.check_missed_snapshots(user_id)
    for snapshot_type in missed:
        self.db.save_snapshot(user_id, snapshot_type, fingerprint, stage, confidence)
        logger.info("Created catch-up %s snapshot for %s", snapshot_type, user_id)
```

Call it at the end of `process()`:

```python
self._check_snapshots(user_id, fingerprint, new_stage, confidence)
```

- [ ] **Step 2: Test snapshot catch-up**

Add to `tests/test_profile_engine.py`:

```python
def test_catch_up_snapshots(self, engine):
    """Verify missed snapshots are created on process."""
    # Create profile with old created_at (2 weeks ago)
    engine.log_event('user1', 'emotional', 'e1',
                     {'emotion': 'joy', 'family': 'Joy'}, 'nanogpt')
    engine.db._enqueue_write(
        "UPDATE profiles SET created_at = ? WHERE user_id = ?",
        (time.time() - 15 * 86400, 'user1'), wait=True
    )
    engine.process('user1')
    snapshots = engine.db.get_snapshots('user1')
    # Should have at least a weekly catch-up snapshot
    assert any(s['snapshot_type'] == 'weekly' for s in snapshots)
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_engine.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add user_profile_engine.py tests/test_profile_engine.py
git commit -m "feat(profile): add snapshot scheduling with catch-up on startup"
```

---

## Chunk 10: Final Integration Test & Cleanup

### Task 15: Comprehensive Final Test

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_db.py tests/test_domain_processors.py tests/test_evolution_engine.py tests/test_profile_engine.py tests/test_profile_api.py tests/test_profile_integration.py tests/test_profile_sync.py tests/test_profile_retention.py -v`
Expected: All tests PASS

- [ ] **Step 2: Verify server starts cleanly**

Run: `cd /Users/bel/quantara-nanoGPT && python -c "from emotion_api_server import app; print('Server imports OK')" 2>&1`
Expected: No import errors

- [ ] **Step 3: Final commit**

```bash
git log --oneline -15  # Verify all commits in order
```

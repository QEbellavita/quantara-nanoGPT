"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Profile Database Tests
===============================================================================
Comprehensive test suite for ProfileDB: schema validation, event CRUD,
profile lifecycle, snapshot operations, and synergy management.

Integrates with:
- Neural Workflow AI Engine
- User Profile Engine
- Evolution Engine
- Domain Processors
===============================================================================
"""

import json
import os
import sys
import time
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profile_db import (
    ProfileDB,
    SCHEMA_VERSION,
    VALID_DOMAINS,
    VALID_SOURCES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Provide a fresh ProfileDB instance and close it after the test."""
    pdb = ProfileDB(db_path=str(tmp_path / "test_profile.db"))
    yield pdb
    pdb.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestProfileDBSchema:
    """Verify that all four tables are created and WAL mode is active."""

    def test_tables_created(self, db):
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = sorted(r[0] for r in cur.fetchall())
        conn.close()

        assert "events" in tables
        assert "profiles" in tables
        assert "snapshots" in tables
        assert "synergies" in tables

    def test_wal_mode_enabled(self, db):
        import sqlite3
        conn = sqlite3.connect(db.db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()

        assert mode.lower() == "wal"

    def test_schema_version_constant(self):
        assert SCHEMA_VERSION == 1

    def test_valid_domains_is_tuple(self):
        assert isinstance(VALID_DOMAINS, tuple)
        assert "emotion" in VALID_DOMAINS

    def test_valid_sources_is_tuple(self):
        assert isinstance(VALID_SOURCES, tuple)
        assert "api" in VALID_SOURCES


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestEventOperations:
    """Test event CRUD: log, get with filters, counts, isolation."""

    def test_log_event_returns_id(self, db):
        eid = db.log_event("u1", "emotion", "detection", payload={"joy": 0.9})
        assert isinstance(eid, int)
        assert eid >= 1

    def test_log_and_retrieve_event(self, db):
        db.log_event("u1", "emotion", "detection", payload={"joy": 0.9}, source="api")
        events = db.get_events("u1")
        assert len(events) == 1
        assert events[0]["domain"] == "emotion"
        assert events[0]["source"] == "api"

    def test_get_events_filters_by_domain(self, db):
        db.log_event("u1", "emotion", "detection")
        db.log_event("u1", "biometric", "heartrate")
        db.log_event("u1", "emotion", "sentiment")

        emotion_events = db.get_events("u1", domain="emotion")
        assert len(emotion_events) == 2
        for e in emotion_events:
            assert e["domain"] == "emotion"

    def test_get_events_filters_by_time(self, db):
        now = time.time()
        db.log_event("u1", "emotion", "a", timestamp=now - 100)
        db.log_event("u1", "emotion", "b", timestamp=now - 50)
        db.log_event("u1", "emotion", "c", timestamp=now)

        events = db.get_events("u1", start_time=now - 60, end_time=now - 10)
        assert len(events) == 1
        assert events[0]["event_type"] == "b"

    def test_get_events_pagination(self, db):
        for i in range(5):
            db.log_event("u1", "emotion", f"evt_{i}", timestamp=time.time() + i)

        page = db.get_events("u1", limit=2, offset=0)
        assert len(page) == 2

        page2 = db.get_events("u1", limit=2, offset=2)
        assert len(page2) == 2

    def test_event_count(self, db):
        db.log_event("u1", "emotion", "a")
        db.log_event("u1", "emotion", "b")
        db.log_event("u1", "biometric", "c")

        assert db.get_event_count("u1") == 3
        assert db.get_event_count("u1", domain="emotion") == 2

    def test_domain_event_counts(self, db):
        db.log_event("u1", "emotion", "a")
        db.log_event("u1", "emotion", "b")
        db.log_event("u1", "biometric", "c")

        counts = db.get_domain_event_counts("u1")
        assert counts["emotion"] == 2
        assert counts["biometric"] == 1

    def test_domain_sources(self, db):
        db.log_event("u1", "emotion", "a", source="api")
        db.log_event("u1", "emotion", "b", source="websocket")
        db.log_event("u1", "emotion", "c", source="api")

        sources = db.get_domain_sources("u1", "emotion")
        assert set(sources) == {"api", "websocket"}

    def test_latest_event_time(self, db):
        now = time.time()
        db.log_event("u1", "emotion", "a", timestamp=now - 10)
        db.log_event("u1", "emotion", "b", timestamp=now)

        latest = db.get_latest_event_time("u1")
        assert latest == pytest.approx(now, abs=0.01)

    def test_latest_event_time_by_domain(self, db):
        now = time.time()
        db.log_event("u1", "emotion", "a", timestamp=now - 20)
        db.log_event("u1", "biometric", "b", timestamp=now)

        latest = db.get_latest_event_time("u1", domain="emotion")
        assert latest == pytest.approx(now - 20, abs=0.01)

    def test_user_isolation(self, db):
        db.log_event("u1", "emotion", "a")
        db.log_event("u2", "emotion", "b")

        assert db.get_event_count("u1") == 1
        assert db.get_event_count("u2") == 1
        assert len(db.get_events("u1")) == 1


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class TestProfileOperations:
    """Test profile get_or_create, update, and delete (purge all tables)."""

    def test_get_or_create_new_profile(self, db):
        profile = db.get_or_create_profile("u1")
        assert profile["user_id"] == "u1"
        assert profile["schema_version"] == SCHEMA_VERSION
        assert profile["evolution_stage"] == 1

    def test_get_or_create_existing_profile(self, db):
        p1 = db.get_or_create_profile("u1")
        p2 = db.get_or_create_profile("u1")
        assert p1["created_at"] == p2["created_at"]

    def test_update_profile(self, db):
        db.get_or_create_profile("u1")
        db.update_profile("u1", confidence=0.85, evolution_stage=3)

        profile = db.get_or_create_profile("u1")
        assert profile["confidence"] == pytest.approx(0.85)
        assert profile["evolution_stage"] == 3

    def test_update_profile_ignores_invalid_fields(self, db):
        db.get_or_create_profile("u1")
        # Should not raise even with unknown fields
        db.update_profile("u1", nonexistent_field="bad", confidence=0.5)
        profile = db.get_or_create_profile("u1")
        assert profile["confidence"] == pytest.approx(0.5)

    def test_delete_user_purges_all_tables(self, db):
        db.get_or_create_profile("u1")
        db.log_event("u1", "emotion", "a")
        db.save_snapshot("u1", snapshot_type="evolution")
        db.save_synergy("u1", "emotion", "biometric", 0.7)

        db.delete_user("u1")

        assert db.get_event_count("u1") == 0
        assert db.get_snapshots("u1") == []
        assert db.get_synergies("u1") == []

        # Profile should also be gone - get_or_create makes a new one
        profile = db.get_or_create_profile("u1")
        # It's a fresh profile, so evolution_count should be 0
        assert profile["evolution_count"] == 0


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

class TestSnapshotOperations:
    """Test snapshot save, get, and type filtering."""

    def test_save_snapshot_returns_id(self, db):
        sid = db.save_snapshot("u1", snapshot_type="evolution", stage=2, confidence=0.8)
        assert isinstance(sid, int)
        assert sid >= 1

    def test_save_and_get_snapshot(self, db):
        fingerprint = json.dumps({"trait": "curious"})
        db.save_snapshot(
            "u1",
            snapshot_type="evolution",
            fingerprint_json=fingerprint,
            stage=2,
            confidence=0.75,
        )

        snaps = db.get_snapshots("u1")
        assert len(snaps) == 1
        assert snaps[0]["snapshot_type"] == "evolution"
        assert snaps[0]["stage"] == 2
        assert json.loads(snaps[0]["fingerprint_json"])["trait"] == "curious"

    def test_get_snapshots_filter_by_type(self, db):
        db.save_snapshot("u1", snapshot_type="evolution")
        db.save_snapshot("u1", snapshot_type="milestone")
        db.save_snapshot("u1", snapshot_type="evolution")

        evolutions = db.get_snapshots("u1", snapshot_type="evolution")
        assert len(evolutions) == 2

        milestones = db.get_snapshots("u1", snapshot_type="milestone")
        assert len(milestones) == 1

    def test_get_last_snapshot_time(self, db):
        now = time.time()
        db.save_snapshot("u1", timestamp=now - 100)
        db.save_snapshot("u1", timestamp=now)

        latest = db.get_last_snapshot_time("u1")
        assert latest == pytest.approx(now, abs=0.01)

    def test_get_last_snapshot_time_no_snapshots(self, db):
        assert db.get_last_snapshot_time("u_none") is None


# ---------------------------------------------------------------------------
# Synergies
# ---------------------------------------------------------------------------

class TestSynergyOperations:
    """Test synergy save (upsert), get, and delete."""

    def test_save_synergy_returns_id(self, db):
        sid = db.save_synergy("u1", "emotion", "biometric", 0.82, insight="correlated")
        assert isinstance(sid, int)
        assert sid >= 1

    def test_save_and_get_synergy(self, db):
        db.save_synergy("u1", "emotion", "biometric", 0.82, insight="correlated")

        synergies = db.get_synergies("u1")
        assert len(synergies) == 1
        assert synergies[0]["domain_a"] == "emotion"
        assert synergies[0]["domain_b"] == "biometric"
        assert synergies[0]["correlation"] == pytest.approx(0.82)

    def test_save_synergy_upserts(self, db):
        db.save_synergy("u1", "emotion", "biometric", 0.5, insight="weak")
        db.save_synergy("u1", "emotion", "biometric", 0.9, insight="strong")

        synergies = db.get_synergies("u1")
        assert len(synergies) == 1
        assert synergies[0]["correlation"] == pytest.approx(0.9)
        assert synergies[0]["insight"] == "strong"

    def test_delete_synergy(self, db):
        sid = db.save_synergy("u1", "emotion", "biometric", 0.7)
        db.delete_synergy(sid)

        assert db.get_synergies("u1") == []

    def test_synergies_ordered_by_correlation(self, db):
        db.save_synergy("u1", "emotion", "biometric", 0.5)
        db.save_synergy("u1", "social", "cognitive", 0.9)
        db.save_synergy("u1", "behavioral", "environmental", 0.7)

        synergies = db.get_synergies("u1")
        correlations = [s["correlation"] for s in synergies]
        assert correlations == sorted(correlations, reverse=True)


# ---------------------------------------------------------------------------
# Writer thread
# ---------------------------------------------------------------------------

class TestWriterThread:
    """Verify the single-writer thread processes writes correctly."""

    def test_concurrent_writes(self, db):
        """Multiple threads can enqueue writes without data loss."""
        errors = []

        def write_events(user_id, count):
            try:
                for i in range(count):
                    db.log_event(user_id, "emotion", f"evt_{i}", wait=True)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_events, args=(f"u{t}", 20))
            for t in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        for t in range(5):
            assert db.get_event_count(f"u{t}") == 20

    def test_close_flushes_pending(self, tmp_path):
        """close() should drain any remaining writes."""
        pdb = ProfileDB(db_path=str(tmp_path / "flush_test.db"))
        pdb.log_event("u1", "emotion", "a", wait=False)
        pdb.log_event("u1", "emotion", "b", wait=False)
        pdb.close()

        # Verify via raw connection after close
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "flush_test.db"))
        cnt = conn.execute("SELECT COUNT(*) FROM events WHERE user_id = 'u1'").fetchone()[0]
        conn.close()
        assert cnt == 2

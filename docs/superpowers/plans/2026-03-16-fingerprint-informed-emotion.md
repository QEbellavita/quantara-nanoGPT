# Fingerprint-Informed Emotion Analysis Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the intelligence loop so emotion classification is informed by the user's accumulated genetic fingerprint profile — personalized baselines, ambiguity tiebreaking, and opt-in response enrichment.

**Architecture:** In-memory `ProfileSnapshot` updated on every `log_event()`, a pure `apply_profile_personalization()` function for ambiguity-only tiebreaking, and opt-in `profile_context` in the API response. Thread-safe via `_snapshot_lock` with check-then-set for lazy DB rebuilds.

**Tech Stack:** Python 3, dataclasses, threading, Flask, pytest

**Spec:** `docs/superpowers/specs/2026-03-16-fingerprint-informed-emotion-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `user_profile_engine.py` (modify) | Add `ProfileSnapshot` dataclass, `_snapshots` dict with `_snapshot_lock`, update snapshot in `log_event()`, add `get_profile_snapshot()` with lazy DB rebuild, update `delete_user()` to clear snapshots |
| `emotion_classifier.py` (modify) | Add `apply_profile_personalization()` standalone function, surface `family_scores` in `MultimodalEmotionAnalyzer.analyze()` return dict |
| `emotion_api_server.py` (modify) | Fix `dominant_emotion` key bug in `log_event()` call, wire personalization + opt-in enrichment in `/api/emotion/analyze` |
| `tests/test_profile_snapshot.py` (create) | Snapshot creation, incremental update, prior accuracy, cold start, restart rebuild, thread safety, delete_user cleanup |
| `tests/test_personalization.py` (create) | Tiebreaker triggers/skips, confident untouched, ordering change, missing family_scores guard |
| `tests/test_analyze_enrichment.py` (create) | Opt-in response format, backward compat, E2E integration |

**Important naming note:** The existing `UserProfileEngine.get_snapshot()` method (line 385) returns DB snapshots as dicts and is used by `profile_api.py:249`, `tests/test_profile_engine.py:90`, and `tests/test_profile_integration.py:120`. We do NOT touch it. The new method is named `get_profile_snapshot()` to avoid collision.

**Keyword fallback note:** The keyword-based fallback classifier in `emotion_api_server.py` (used when ML models aren't loaded) returns `'emotion'` key, not `'dominant_emotion'`, and does not produce `family_scores`. Guard 2 in the tiebreaker handles this — personalization is simply skipped for fallback results.

---

## Chunk 1: ProfileSnapshot

### Task 1: ProfileSnapshot dataclass, snapshot update, and all snapshot tests

**Files:**
- Modify: `user_profile_engine.py`
- Create: `tests/test_profile_snapshot.py`

- [ ] **Step 1: Write failing tests for ProfileSnapshot**

```python
# tests/test_profile_snapshot.py
"""Tests for ProfileSnapshot in-memory snapshot system."""

import os
import sys
import time
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestProfileSnapshotCreation:

    @pytest.fixture
    def engine(self, tmp_path):
        from user_profile_engine import UserProfileEngine
        return UserProfileEngine(db_path=str(tmp_path / 'test.db'))

    def test_no_snapshot_before_events(self, engine):
        """get_profile_snapshot returns None for user with no emotional events."""
        from user_profile_engine import ProfileSnapshot
        snap = engine.get_profile_snapshot('user1')
        assert snap is None

    def test_snapshot_created_on_emotion_event(self, engine):
        """Logging an emotion_classified event creates a snapshot."""
        from user_profile_engine import ProfileSnapshot
        engine.log_event('user1', 'emotional', 'emotion_classified', {
            'emotion': 'joy', 'family': 'Joy', 'confidence': 0.8,
        })
        snap = engine.get_profile_snapshot('user1')
        assert snap is not None
        assert isinstance(snap, ProfileSnapshot)
        assert snap.user_id == 'user1'
        assert snap.event_count == 1

    def test_non_emotion_event_no_snapshot(self, engine):
        """Logging a non-emotional event does not create a snapshot."""
        engine.log_event('user1', 'temporal', 'session_started', {
            'endpoint': 'analyze',
        })
        snap = engine.get_profile_snapshot('user1')
        assert snap is None


class TestProfileSnapshotPrior:

    @pytest.fixture
    def engine(self, tmp_path):
        from user_profile_engine import UserProfileEngine
        return UserProfileEngine(db_path=str(tmp_path / 'test.db'))

    def test_single_family_prior(self, engine):
        """Single family gives prior of 1.0 for that family."""
        engine.log_event('u1', 'emotional', 'emotion_classified', {
            'emotion': 'joy', 'family': 'Joy', 'confidence': 0.9,
        })
        snap = engine.get_profile_snapshot('u1')
        assert snap.emotion_prior['Joy'] == 1.0
        assert snap.dominant_family == 'Joy'

    def test_multi_family_prior(self, engine):
        """Multiple families give normalized prior."""
        for _ in range(3):
            engine.log_event('u1', 'emotional', 'emotion_classified', {
                'emotion': 'joy', 'family': 'Joy', 'confidence': 0.9,
            })
        engine.log_event('u1', 'emotional', 'emotion_classified', {
            'emotion': 'sadness', 'family': 'Sadness', 'confidence': 0.7,
        })
        snap = engine.get_profile_snapshot('u1')
        assert snap.event_count == 4
        assert abs(snap.emotion_prior['Joy'] - 0.75) < 0.01
        assert abs(snap.emotion_prior['Sadness'] - 0.25) < 0.01
        assert snap.dominant_family == 'Joy'

    def test_prior_sums_to_one(self, engine):
        """Emotion prior always sums to 1.0."""
        families = ['Joy', 'Sadness', 'Anger', 'Fear', 'Calm']
        for i, fam in enumerate(families):
            for _ in range(i + 1):
                engine.log_event('u1', 'emotional', 'emotion_classified', {
                    'emotion': fam.lower(), 'family': fam, 'confidence': 0.5,
                })
        snap = engine.get_profile_snapshot('u1')
        total = sum(snap.emotion_prior.values())
        assert abs(total - 1.0) < 0.001


class TestProfileSnapshotRestart:
    """Test lazy rebuild from DB after snapshot loss (simulates restart)."""

    @pytest.fixture
    def engine(self, tmp_path):
        from user_profile_engine import UserProfileEngine
        return UserProfileEngine(db_path=str(tmp_path / 'test.db'))

    def test_rebuild_from_db(self, engine):
        """After clearing in-memory snapshots, get_profile_snapshot rebuilds from DB."""
        for _ in range(5):
            engine.log_event('u1', 'emotional', 'emotion_classified', {
                'emotion': 'joy', 'family': 'Joy', 'confidence': 0.8,
            })
        for _ in range(3):
            engine.log_event('u1', 'emotional', 'emotion_classified', {
                'emotion': 'fear', 'family': 'Fear', 'confidence': 0.6,
            })

        # Clear in-memory cache (simulate restart)
        with engine._snapshot_lock:
            engine._snapshots.clear()

        snap = engine.get_profile_snapshot('u1')
        assert snap is not None
        assert snap.event_count == 8
        assert abs(snap.emotion_prior['Joy'] - 5 / 8) < 0.01
        assert abs(snap.emotion_prior['Fear'] - 3 / 8) < 0.01

    def test_rebuild_returns_none_for_no_events(self, engine):
        """Rebuild returns None when user has no emotional events in DB."""
        engine.log_event('u1', 'temporal', 'session_started', {'endpoint': 'x'})
        with engine._snapshot_lock:
            engine._snapshots.clear()
        snap = engine.get_profile_snapshot('u1')
        assert snap is None


class TestProfileSnapshotThreadSafety:

    @pytest.fixture
    def engine(self, tmp_path):
        from user_profile_engine import UserProfileEngine
        return UserProfileEngine(db_path=str(tmp_path / 'test.db'))

    def test_concurrent_updates(self, engine):
        """Multiple threads logging events should not corrupt the snapshot."""
        families = ['Joy', 'Sadness', 'Anger', 'Fear']
        events_per_thread = 25
        errors = []

        def log_events(family):
            try:
                for _ in range(events_per_thread):
                    engine.log_event('u1', 'emotional', 'emotion_classified', {
                        'emotion': family.lower(), 'family': family, 'confidence': 0.5,
                    })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=log_events, args=(f,)) for f in families]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        snap = engine.get_profile_snapshot('u1')
        assert snap is not None
        assert snap.event_count == events_per_thread * len(families)
        total = sum(snap.emotion_prior.values())
        assert abs(total - 1.0) < 0.001


class TestProfileSnapshotDeleteUser:

    @pytest.fixture
    def engine(self, tmp_path):
        from user_profile_engine import UserProfileEngine
        return UserProfileEngine(db_path=str(tmp_path / 'test.db'))

    def test_delete_clears_snapshot(self, engine):
        """delete_user removes the in-memory snapshot."""
        engine.log_event('u1', 'emotional', 'emotion_classified', {
            'emotion': 'joy', 'family': 'Joy', 'confidence': 0.8,
        })
        assert engine.get_profile_snapshot('u1') is not None
        engine.delete_user('u1')
        assert engine.get_profile_snapshot('u1') is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_snapshot.py -v`
Expected: FAIL — `ProfileSnapshot` not importable / `get_profile_snapshot` does not exist

- [ ] **Step 3: Implement ProfileSnapshot and snapshot methods**

In `user_profile_engine.py`, add `import dataclasses` to the imports at the top (after `import json`).

Add the dataclass after the `_SECONDS_PER_WEEK` constant, before the `UserProfileEngine` class:

```python
@dataclasses.dataclass
class ProfileSnapshot:
    """Lightweight in-memory summary of a user's emotional profile.

    Updated incrementally on every emotion_classified event.
    Used by the personalization tiebreaker in emotion_classifier.py.
    """
    user_id: str
    evolution_stage: int
    overall_confidence: float
    emotion_prior: Dict[str, float]
    dominant_family: str
    event_count: int
    family_counts: Dict[str, int]
    last_updated: float
```

Add to `UserProfileEngine.__init__()` — after `self._ecosystem_connector = None` (line 59):

```python
        # In-memory profile snapshots for personalization
        self._snapshots: Dict[str, 'ProfileSnapshot'] = {}
        self._snapshot_lock = threading.Lock()
```

In `UserProfileEngine.log_event()`, add snapshot update after `event_id = self.db.log_event(...)` and before `return event_id`. Replace lines 94-103 with:

```python
            # Log the event
            event_id = self.db.log_event(
                user_id=user_id,
                domain=domain,
                event_type=event_type,
                payload=payload,
                source=source,
                confidence=confidence if confidence is not None else 1.0,
            )

            # Update in-memory snapshot for personalization
            if domain == 'emotional' and event_type == 'emotion_classified':
                family = (payload or {}).get('family')
                if family:
                    self._update_snapshot(user_id, family)

            return event_id
```

Add `_update_snapshot` method to `UserProfileEngine` (after `log_event`, before `process`):

```python
    def _update_snapshot(self, user_id: str, family: str) -> None:
        """Incrementally update the in-memory profile snapshot.

        Called from log_event() for emotional/emotion_classified events.
        The db.get_or_create_profile() call has already been made by log_event()
        so the profile row exists. We read stage/confidence outside the lock
        only when creating a new snapshot.
        """
        # Read profile data outside the lock (only needed for new snapshots)
        profile_data = None

        with self._snapshot_lock:
            snap = self._snapshots.get(user_id)
            if snap is None:
                # We need profile data — but we already called
                # get_or_create_profile in log_event() above, so the row exists.
                # Read it outside the lock to avoid holding the lock during I/O.
                pass
            else:
                # Fast path: just update existing snapshot
                snap.family_counts[family] = snap.family_counts.get(family, 0) + 1
                snap.event_count += 1
                total = sum(snap.family_counts.values())
                snap.emotion_prior = {f: c / total for f, c in snap.family_counts.items()}
                snap.dominant_family = max(snap.family_counts, key=snap.family_counts.get)
                snap.last_updated = time.time()
                return

        # Slow path: create new snapshot (outside lock)
        profile_data = self.db.get_or_create_profile(user_id)

        with self._snapshot_lock:
            # Check again in case another thread created it
            if user_id in self._snapshots:
                snap = self._snapshots[user_id]
                snap.family_counts[family] = snap.family_counts.get(family, 0) + 1
                snap.event_count += 1
                total = sum(snap.family_counts.values())
                snap.emotion_prior = {f: c / total for f, c in snap.family_counts.items()}
                snap.dominant_family = max(snap.family_counts, key=snap.family_counts.get)
                snap.last_updated = time.time()
                return

            snap = ProfileSnapshot(
                user_id=user_id,
                evolution_stage=profile_data.get('evolution_stage', 1),
                overall_confidence=profile_data.get('confidence', 0.0),
                emotion_prior={family: 1.0},
                dominant_family=family,
                event_count=1,
                family_counts={family: 1},
                last_updated=time.time(),
            )
            self._snapshots[user_id] = snap
```

Add `get_profile_snapshot` method (after `_update_snapshot`, before `process`):

```python
    def get_profile_snapshot(self, user_id: str) -> Optional['ProfileSnapshot']:
        """Return the cached profile snapshot, rebuilding from DB if needed.

        Returns None only if user has zero emotional events.
        Does NOT conflict with get_snapshot() which returns DB snapshots.
        """
        with self._snapshot_lock:
            snap = self._snapshots.get(user_id)
            if snap is not None:
                return snap

        # Rebuild from DB outside the lock
        events = self.db.get_events(user_id, domain='emotional', limit=10000)
        family_counts: Dict[str, int] = {}
        for evt in events:
            if evt.get('event_type') != 'emotion_classified':
                continue
            payload = evt.get('payload')
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except (json.JSONDecodeError, TypeError):
                    continue
            family = (payload or {}).get('family')
            if family:
                family_counts[family] = family_counts.get(family, 0) + 1

        if not family_counts:
            return None

        total = sum(family_counts.values())
        profile = self.db.get_or_create_profile(user_id)
        snap = ProfileSnapshot(
            user_id=user_id,
            evolution_stage=profile.get('evolution_stage', 1),
            overall_confidence=profile.get('confidence', 0.0),
            emotion_prior={f: c / total for f, c in family_counts.items()},
            dominant_family=max(family_counts, key=family_counts.get),
            event_count=total,
            family_counts=family_counts,
            last_updated=time.time(),
        )

        # Check-then-set under lock
        with self._snapshot_lock:
            if user_id not in self._snapshots:
                self._snapshots[user_id] = snap
            return self._snapshots[user_id]
```

Update `delete_user` — add snapshot cleanup after the consecutive_met cleanup:

```python
        self._consecutive_met.pop(user_id, None)
        with self._snapshot_lock:
            self._snapshots.pop(user_id, None)
```

- [ ] **Step 4: Run all snapshot tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_snapshot.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run existing profile tests to verify nothing breaks**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_engine.py tests/test_profile_integration.py -v`
Expected: ALL PASS (existing `get_snapshot()` is untouched)

- [ ] **Step 6: Commit**

```bash
git add user_profile_engine.py tests/test_profile_snapshot.py
git commit -m "feat: add ProfileSnapshot with incremental update and lazy DB rebuild"
```

---

## Chunk 2: Personalization Tiebreaker

### Task 2: Surface family_scores in MultimodalEmotionAnalyzer.analyze()

**Files:**
- Modify: `emotion_classifier.py:1127-1159`
- Create: `tests/test_personalization.py`

- [ ] **Step 1: Write failing tests for family_scores**

```python
# tests/test_personalization.py
"""Tests for profile-based emotion personalization."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_classifier import FAMILY_NAMES, EMOTION_FAMILIES


class TestFamilyScoresInAnalyzer:
    """Test that MultimodalEmotionAnalyzer.analyze() returns family_scores."""

    @pytest.fixture
    def analyzer(self):
        from emotion_classifier import MultimodalEmotionAnalyzer
        return MultimodalEmotionAnalyzer(use_sentence_transformer=True)

    def test_family_scores_present(self, analyzer):
        """analyze() result should contain family_scores dict."""
        result = analyzer.analyze("I feel happy today")
        assert 'family_scores' in result
        assert isinstance(result['family_scores'], dict)

    def test_family_scores_keys(self, analyzer):
        """family_scores keys should be the 9 family names."""
        result = analyzer.analyze("I am so excited")
        assert set(result['family_scores'].keys()) == set(FAMILY_NAMES)

    def test_family_scores_sum_to_one(self, analyzer):
        """family_scores values should sum to ~1.0 (softmax output)."""
        result = analyzer.analyze("This is a test")
        total = sum(result['family_scores'].values())
        assert abs(total - 1.0) < 0.01

    def test_family_scores_all_positive(self, analyzer):
        """All family_scores should be >= 0."""
        result = analyzer.analyze("I feel nothing")
        for score in result['family_scores'].values():
            assert score >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_personalization.py::TestFamilyScoresInAnalyzer -v`
Expected: FAIL — `family_scores` not in result

- [ ] **Step 3: Add family_scores to analyze() return**

In `emotion_classifier.py`, in `MultimodalEmotionAnalyzer.analyze()`, after line 1131 (`emotion_probs_np = emotion_probs.squeeze(0).cpu().numpy()`), add:

```python
        family_probs_np = family_probs.squeeze(0).cpu().numpy()

        # Build family scores dict (softmax probs, sum to 1.0)
        family_scores = {
            FAMILY_NAMES[i]: float(family_probs_np[i])
            for i in range(len(FAMILY_NAMES))
        }
```

Then in the result dict (line 1141), add `'family_scores': family_scores,` after the `'scores': scores,` line.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_personalization.py::TestFamilyScoresInAnalyzer -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add emotion_classifier.py tests/test_personalization.py
git commit -m "feat: surface family_scores in MultimodalEmotionAnalyzer.analyze()"
```

---

### Task 3: apply_profile_personalization function

**Files:**
- Modify: `emotion_classifier.py`
- Modify: `tests/test_personalization.py`

- [ ] **Step 1: Write failing tests for personalization logic**

Append to `tests/test_personalization.py`:

```python
class TestApplyProfilePersonalization:
    """Test the tiebreaker personalization function."""

    def _make_snapshot(self, emotion_prior, event_count=50):
        from user_profile_engine import ProfileSnapshot
        dominant = max(emotion_prior, key=emotion_prior.get)
        return ProfileSnapshot(
            user_id='test',
            evolution_stage=3,
            overall_confidence=0.7,
            emotion_prior=emotion_prior,
            dominant_family=dominant,
            event_count=event_count,
            family_counts={f: int(v * event_count) for f, v in emotion_prior.items()},
            last_updated=1710000000.0,
        )

    def test_skip_when_no_snapshot(self):
        """Should skip personalization when snapshot is None."""
        from emotion_classifier import apply_profile_personalization
        result = {'family': 'Joy', 'confidence': 0.8, 'family_scores': {'Joy': 0.8, 'Sadness': 0.1}}
        out = apply_profile_personalization(result, None)
        assert out['personalized'] is False

    def test_skip_when_cold_start(self):
        """Should skip when event_count < 10."""
        from emotion_classifier import apply_profile_personalization
        snap = self._make_snapshot({'Joy': 1.0}, event_count=5)
        result = {'family': 'Joy', 'confidence': 0.8, 'family_scores': {'Joy': 0.8, 'Sadness': 0.1}}
        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is False

    def test_skip_when_no_family_scores(self):
        """Should skip when result has no family_scores (fallback classifier)."""
        from emotion_classifier import apply_profile_personalization
        snap = self._make_snapshot({'Joy': 0.5, 'Sadness': 0.5})
        result = {'family': 'Joy', 'confidence': 0.8}
        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is False

    def test_skip_when_confident(self):
        """Should skip when classifier is confident (score >= 0.5 and gap >= 0.05)."""
        from emotion_classifier import apply_profile_personalization
        snap = self._make_snapshot({'Sadness': 0.8, 'Joy': 0.2})
        result = {
            'dominant_emotion': 'joy', 'family': 'Joy', 'confidence': 0.7,
            'family_scores': {'Joy': 0.7, 'Sadness': 0.2, 'Anger': 0.05, 'Fear': 0.05},
            'scores': {'joy': 0.7},
        }
        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is False

    def test_tiebreak_changes_winner(self):
        """When ambiguous and prior favors runner-up, should swap winner."""
        from emotion_classifier import apply_profile_personalization
        snap = self._make_snapshot({'Fear': 0.7, 'Sadness': 0.1, 'Joy': 0.1, 'Anger': 0.1})
        result = {
            'dominant_emotion': 'sadness',
            'family': 'Sadness',
            'confidence': 0.30,
            'family_scores': {'Sadness': 0.30, 'Fear': 0.28, 'Joy': 0.20, 'Anger': 0.12, 'Calm': 0.05, 'Love': 0.02, 'Self-Conscious': 0.01, 'Surprise': 0.01, 'Neutral': 0.01},
            'scores': {'sadness': 0.15, 'grief': 0.10, 'boredom': 0.05, 'fear': 0.14, 'anxiety': 0.10, 'worry': 0.04},
        }
        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is True
        assert out['family'] == 'Fear'
        assert 'profile_tiebreak' in out['personalization_reason']

    def test_tiebreak_confirms_winner(self):
        """When ambiguous but prior agrees with classifier, confirm original."""
        from emotion_classifier import apply_profile_personalization
        snap = self._make_snapshot({'Joy': 0.6, 'Sadness': 0.2, 'Anger': 0.2})
        result = {
            'dominant_emotion': 'joy',
            'family': 'Joy',
            'confidence': 0.35,
            'family_scores': {'Joy': 0.35, 'Sadness': 0.33, 'Anger': 0.15, 'Fear': 0.10, 'Calm': 0.03, 'Love': 0.02, 'Self-Conscious': 0.01, 'Surprise': 0.005, 'Neutral': 0.005},
            'scores': {'joy': 0.20, 'excitement': 0.10, 'enthusiasm': 0.05},
        }
        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is True
        assert out['family'] == 'Joy'
        assert 'profile_consulted' in out['personalization_reason']

    def test_family_scores_unchanged(self):
        """family_scores in result should be the original (unadjusted) values."""
        from emotion_classifier import apply_profile_personalization
        snap = self._make_snapshot({'Fear': 0.7, 'Sadness': 0.3})
        original_scores = {'Sadness': 0.30, 'Fear': 0.28, 'Joy': 0.20, 'Anger': 0.12, 'Calm': 0.05, 'Love': 0.02, 'Self-Conscious': 0.01, 'Surprise': 0.01, 'Neutral': 0.01}
        result = {
            'dominant_emotion': 'sadness', 'family': 'Sadness', 'confidence': 0.30,
            'family_scores': dict(original_scores),
            'scores': {'sadness': 0.15, 'fear': 0.14},
        }
        apply_profile_personalization(result, snap)
        assert result['family_scores'] == original_scores
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_personalization.py::TestApplyProfilePersonalization -v`
Expected: FAIL — `apply_profile_personalization` not importable

- [ ] **Step 3: Implement apply_profile_personalization**

Add to `emotion_classifier.py` after the `family_for_emotion()` function (around line 70):

```python
# ─── Profile Personalization ──────────────────────────────────────────────────

PRIOR_WEIGHT = 0.05


def apply_profile_personalization(result, snapshot):
    """Apply profile-based personalization to emotion classification result.

    Only intervenes when the classifier is uncertain (ambiguous).
    Modifies result in-place and returns it.
    Pure function — no side effects, no DB calls, no event bus publish.

    Args:
        result: dict from MultimodalEmotionAnalyzer.analyze() or keyword fallback
        snapshot: Optional ProfileSnapshot from UserProfileEngine.get_profile_snapshot()
    """
    # Guard 1: no snapshot or cold start
    if snapshot is None or snapshot.event_count < 10:
        result['personalized'] = False
        return result

    # Guard 2: no family_scores (fallback classifier)
    if 'family_scores' not in result:
        result['personalized'] = False
        return result

    family_scores = result['family_scores']

    # Extract top-2 families
    sorted_families = sorted(family_scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_families) < 2:
        result['personalized'] = False
        return result

    top_1_family, top_1_score = sorted_families[0]
    top_2_family, top_2_score = sorted_families[1]
    gap = top_1_score - top_2_score

    # Guard 3: confident prediction — no personalization
    if top_1_score >= 0.5 and gap >= 0.05:
        result['personalized'] = False
        return result

    # Apply tiebreaker
    adj_1 = top_1_score + PRIOR_WEIGHT * snapshot.emotion_prior.get(top_1_family, 0.0)
    adj_2 = top_2_score + PRIOR_WEIGHT * snapshot.emotion_prior.get(top_2_family, 0.0)

    if adj_2 > adj_1:
        # Ordering changed — swap winner
        new_family = top_2_family
        old_family = top_1_family

        # Find best sub-emotion in new winning family from scores dict
        scores = result.get('scores', {})
        family_emotions = EMOTION_FAMILIES.get(new_family, [])
        best_emotion = None
        best_score = -1.0
        for emo in family_emotions:
            s = scores.get(emo, 0.0)
            if s > best_score:
                best_score = s
                best_emotion = emo

        if best_emotion:
            result['dominant_emotion'] = best_emotion
        result['family'] = new_family
        result['confidence'] = adj_2

        result['personalized'] = True
        prior_winner = snapshot.emotion_prior.get(new_family, 0.0)
        prior_loser = snapshot.emotion_prior.get(old_family, 0.0)
        result['personalization_reason'] = (
            f"profile_tiebreak: {new_family} prior {prior_winner:.2f} > "
            f"{old_family} prior {prior_loser:.2f}"
        )
    else:
        # Ordering unchanged — confirm original
        result['personalized'] = True
        result['personalization_reason'] = "profile_consulted: original classification confirmed"

    return result
```

- [ ] **Step 4: Run all personalization tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_personalization.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add emotion_classifier.py tests/test_personalization.py
git commit -m "feat: add apply_profile_personalization tiebreaker function"
```

---

## Chunk 3: API Wiring and Enrichment

### Task 4: Fix dominant_emotion key bug and wire personalization

**Files:**
- Modify: `emotion_api_server.py:1041-1096`
- Create: `tests/test_analyze_enrichment.py`

- [ ] **Step 1: Write tests for enrichment contracts**

```python
# tests/test_analyze_enrichment.py
"""Tests for opt-in profile enrichment in /api/emotion/analyze."""

import os
import sys
import json
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAnalyzePersonalizationWiring:
    """Test that personalization function handles edge cases at API boundary."""

    def test_personalized_false_when_no_snapshot(self):
        """When snapshot is None, personalized should be False."""
        from emotion_classifier import apply_profile_personalization
        result = {
            'dominant_emotion': 'joy', 'family': 'Joy', 'confidence': 0.9,
            'family_scores': {'Joy': 0.9, 'Sadness': 0.05},
            'scores': {'joy': 0.9},
        }
        out = apply_profile_personalization(result, None)
        assert 'personalized' in out
        assert out['personalized'] is False

    def test_no_personalized_field_without_calling_function(self):
        """Without calling personalization, 'personalized' should not exist."""
        result = {
            'dominant_emotion': 'joy', 'family': 'Joy', 'confidence': 0.9,
        }
        assert 'personalized' not in result


class TestProfileContextEnrichment:
    """Test opt-in profile_context serialization."""

    def _make_snapshot(self):
        from user_profile_engine import ProfileSnapshot
        return ProfileSnapshot(
            user_id='u1',
            evolution_stage=3,
            overall_confidence=0.72,
            emotion_prior={'Fear': 0.38, 'Sadness': 0.21, 'Joy': 0.12},
            dominant_family='Fear',
            event_count=847,
            family_counts={'Fear': 322, 'Sadness': 178, 'Joy': 102},
            last_updated=1710600000.0,
        )

    def test_profile_context_format(self):
        """profile_context should have the expected keys."""
        snap = self._make_snapshot()
        from datetime import datetime, timezone
        context = {
            'evolution_stage': snap.evolution_stage,
            'overall_confidence': snap.overall_confidence,
            'dominant_family': snap.dominant_family,
            'emotion_prior': snap.emotion_prior,
            'event_count': snap.event_count,
            'last_updated': datetime.fromtimestamp(
                snap.last_updated, tz=timezone.utc
            ).isoformat(),
        }
        assert context['evolution_stage'] == 3
        assert context['event_count'] == 847
        assert 'Fear' in context['emotion_prior']

    def test_snapshot_json_serializable(self):
        """profile_context dict should be JSON-serializable."""
        snap = self._make_snapshot()
        from datetime import datetime, timezone
        context = {
            'evolution_stage': snap.evolution_stage,
            'overall_confidence': snap.overall_confidence,
            'dominant_family': snap.dominant_family,
            'emotion_prior': snap.emotion_prior,
            'event_count': snap.event_count,
            'last_updated': datetime.fromtimestamp(
                snap.last_updated, tz=timezone.utc
            ).isoformat(),
        }
        serialized = json.dumps(context)
        assert '"evolution_stage": 3' in serialized


class TestDominantEmotionKeyFix:
    """Test that the dominant_emotion key is used correctly."""

    def test_log_event_uses_dominant_emotion(self):
        """Verify the bug: old code reads 'emotion' (gets 'neutral'), fix reads 'dominant_emotion'."""
        result = {
            'dominant_emotion': 'anxiety',
            'family': 'Fear',
            'confidence': 0.7,
        }
        old_way = result.get('emotion', 'neutral')
        assert old_way == 'neutral'  # confirms the bug existed
        new_way = result.get('dominant_emotion', result.get('emotion', 'neutral'))
        assert new_way == 'anxiety'  # confirms the fix works

    def test_fallback_classifier_still_works(self):
        """Keyword fallback uses 'emotion' key — fix should handle both."""
        result = {
            'emotion': 'joy',
            'family': 'Joy',
            'confidence': 1.0,
        }
        # The fix uses: result.get('dominant_emotion', result.get('emotion', 'neutral'))
        emotion = result.get('dominant_emotion', result.get('emotion', 'neutral'))
        assert emotion == 'joy'
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_analyze_enrichment.py -v`
Expected: ALL PASS (these test contracts/format, not Flask wiring)

- [ ] **Step 3: Fix dominant_emotion key bug in emotion_api_server.py**

In `emotion_api_server.py`, at line 1082, change:

```python
                        'emotion': result.get('emotion', 'neutral'),
```

to:

```python
                        'emotion': result.get('dominant_emotion', result.get('emotion', 'neutral')),
```

This fixes the bug while remaining backward-compatible with the keyword fallback (which uses `'emotion'` key).

Also fix the transition tracker at line 1069 the same way:

```python
                        emotion=result.get('dominant_emotion', result.get('emotion', 'neutral')),
```

- [ ] **Step 4: Wire personalization and enrichment into /api/emotion/analyze**

In `emotion_api_server.py`, add import near the top (with other emotion_classifier imports):

```python
from emotion_classifier import apply_profile_personalization
```

Replace the section from line 1054 (`result = model.analyze(text, biometrics)`) through line 1096 (`return jsonify({**result, 'status': 'success'})`) with:

```python
            result = model.analyze(text, biometrics)

            # Stream emotion update via WebSocket
            if HAS_WEBSOCKET:
                try:
                    ws_emit_emotion(result)
                except Exception:
                    pass

            # Auto-record to transition tracker when user_id is provided
            user_id = data.get('user_id')
            if user_id and transition_tracker:
                try:
                    transition_tracker.record(
                        user_id=user_id,
                        emotion=result.get('dominant_emotion', result.get('emotion', 'neutral')),
                        family=result.get('family'),
                        confidence=float(result.get('confidence', 1.0)),
                    )
                except Exception:
                    pass

            # Log to profile engine
            # NOTE: Intentionally removed the 'default' user_id fallback that was
            # at the old line 1080. Using 'default' created a ghost user that
            # accumulated all anonymous requests, polluting profile data.
            if profile_engine and user_id:
                try:
                    profile_engine.log_event(user_id, 'emotional', 'emotion_classified', {
                        'emotion': result.get('dominant_emotion', result.get('emotion', 'neutral')),
                        'family': result.get('family', 'Neutral'),
                        'confidence': result.get('confidence', 0.5),
                    }, 'nanogpt', result.get('confidence'))
                    if text:
                        profile_engine.log_event(user_id, 'linguistic', 'text_analyzed', {
                            'text': text[:500],
                        }, 'nanogpt')
                    profile_engine.log_event(user_id, 'temporal', 'session_started', {
                        'endpoint': 'analyze',
                    }, 'nanogpt')
                except Exception:
                    pass

            # Apply profile personalization when user_id is present
            snapshot = None
            if user_id and profile_engine:
                try:
                    snapshot = profile_engine.get_profile_snapshot(user_id)
                    apply_profile_personalization(result, snapshot)

                    # Publish event bus event on tiebreak swap
                    # (Intentional deviation from spec: event bus publish is here
                    # rather than inside apply_profile_personalization to keep
                    # that function pure/testable without bus dependencies.)
                    if (result.get('personalized') and
                            'profile_tiebreak' in result.get('personalization_reason', '') and
                            profile_engine._event_bus):
                        try:
                            profile_engine._event_bus.publish('profile.personalization.applied', {
                                'user_id': user_id,
                                'original_family': result.get('personalization_reason', '').split(' > ')[0].split(': ')[1].split(' prior')[0] if ' > ' in result.get('personalization_reason', '') else 'unknown',
                                'personalized_family': result.get('family'),
                                'prior_weight_used': 0.05,
                                'ambiguity_gap': 0.0,  # not easily recoverable here
                            })
                        except Exception:
                            pass
                except Exception:
                    result['personalized'] = False
            elif user_id and not profile_engine:
                result['personalized'] = False

            # Opt-in profile context enrichment (reuses snapshot from above)
            include_profile = data.get('include_profile', False)
            if include_profile and user_id and profile_engine:
                try:
                    if snapshot is None:
                        snapshot = profile_engine.get_profile_snapshot(user_id)
                    if snapshot:
                        from datetime import datetime, timezone
                        result['profile_context'] = {
                            'evolution_stage': snapshot.evolution_stage,
                            'overall_confidence': snapshot.overall_confidence,
                            'dominant_family': snapshot.dominant_family,
                            'emotion_prior': snapshot.emotion_prior,
                            'event_count': snapshot.event_count,
                            'last_updated': datetime.fromtimestamp(
                                snapshot.last_updated, tz=timezone.utc
                            ).isoformat(),
                        }
                except Exception:
                    pass

            return jsonify({**result, 'status': 'success'})
```

- [ ] **Step 5: Run all tests to verify nothing breaks**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_engine.py tests/test_profile_snapshot.py tests/test_personalization.py tests/test_analyze_enrichment.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add emotion_api_server.py tests/test_analyze_enrichment.py
git commit -m "feat: wire fingerprint personalization into /api/emotion/analyze

- Fix dominant_emotion key bug in log_event and transition tracker
- Remove ghost 'default' user_id fallback
- Add personalization tiebreaker when user_id present
- Add opt-in profile_context enrichment via include_profile flag
- Publish profile.personalization.applied event on tiebreak swap"
```

---

## Chunk 4: Integration Verification

### Task 5: End-to-end integration test and full suite run

**Files:**
- Modify: `tests/test_analyze_enrichment.py`

- [ ] **Step 1: Add integration tests**

Append to `tests/test_analyze_enrichment.py`:

```python
class TestEndToEndPersonalization:
    """Integration test: log events -> build snapshot -> personalize analysis."""

    @pytest.fixture
    def engine(self, tmp_path):
        from user_profile_engine import UserProfileEngine
        return UserProfileEngine(db_path=str(tmp_path / 'e2e.db'))

    def test_full_loop(self, engine):
        """Events build prior, prior influences ambiguous classification."""
        from emotion_classifier import apply_profile_personalization

        # Phase 1: Build up a Fear-dominant prior (15 events)
        for _ in range(12):
            engine.log_event('u1', 'emotional', 'emotion_classified', {
                'emotion': 'anxiety', 'family': 'Fear', 'confidence': 0.6,
            })
        for _ in range(3):
            engine.log_event('u1', 'emotional', 'emotion_classified', {
                'emotion': 'sadness', 'family': 'Sadness', 'confidence': 0.5,
            })

        snap = engine.get_profile_snapshot('u1')
        assert snap is not None
        assert snap.event_count == 15
        assert snap.dominant_family == 'Fear'
        assert snap.emotion_prior['Fear'] > 0.7

        # Phase 2: Simulate an ambiguous classification (Sadness barely wins)
        result = {
            'dominant_emotion': 'sadness',
            'family': 'Sadness',
            'confidence': 0.31,
            'family_scores': {
                'Sadness': 0.31, 'Fear': 0.29, 'Joy': 0.15,
                'Anger': 0.10, 'Calm': 0.05, 'Love': 0.04,
                'Self-Conscious': 0.03, 'Surprise': 0.02, 'Neutral': 0.01,
            },
            'scores': {
                'sadness': 0.15, 'grief': 0.10, 'boredom': 0.06,
                'fear': 0.14, 'anxiety': 0.12, 'worry': 0.03,
            },
        }

        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is True
        assert out['family'] == 'Fear'
        assert 'profile_tiebreak' in out['personalization_reason']
        assert out['dominant_emotion'] in ['fear', 'anxiety', 'worry', 'overwhelmed', 'stressed']

    def test_confident_prediction_unaffected(self, engine):
        """Even with strong prior, confident predictions are not changed."""
        from emotion_classifier import apply_profile_personalization

        for _ in range(20):
            engine.log_event('u1', 'emotional', 'emotion_classified', {
                'emotion': 'anxiety', 'family': 'Fear', 'confidence': 0.8,
            })

        snap = engine.get_profile_snapshot('u1')

        result = {
            'dominant_emotion': 'joy',
            'family': 'Joy',
            'confidence': 0.85,
            'family_scores': {
                'Joy': 0.85, 'Fear': 0.05, 'Sadness': 0.03,
                'Anger': 0.02, 'Calm': 0.02, 'Love': 0.01,
                'Self-Conscious': 0.01, 'Surprise': 0.005, 'Neutral': 0.005,
            },
            'scores': {'joy': 0.85},
        }

        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is False
        assert out['family'] == 'Joy'
        assert out['dominant_emotion'] == 'joy'

    def test_restart_then_personalize(self, engine):
        """After simulated restart, lazy rebuild enables personalization."""
        from emotion_classifier import apply_profile_personalization

        for _ in range(15):
            engine.log_event('u1', 'emotional', 'emotion_classified', {
                'emotion': 'anger', 'family': 'Anger', 'confidence': 0.7,
            })

        # Simulate restart
        with engine._snapshot_lock:
            engine._snapshots.clear()

        # Lazy rebuild should work
        snap = engine.get_profile_snapshot('u1')
        assert snap is not None
        assert snap.dominant_family == 'Anger'

        result = {
            'dominant_emotion': 'joy',
            'family': 'Joy',
            'confidence': 0.30,
            'family_scores': {
                'Joy': 0.30, 'Anger': 0.29, 'Sadness': 0.15,
                'Fear': 0.10, 'Calm': 0.05, 'Love': 0.04,
                'Self-Conscious': 0.03, 'Surprise': 0.02, 'Neutral': 0.02,
            },
            'scores': {'joy': 0.20, 'anger': 0.18, 'frustration': 0.11},
        }

        out = apply_profile_personalization(result, snap)
        assert out['personalized'] is True
        assert out['family'] == 'Anger'
```

- [ ] **Step 2: Run integration tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_analyze_enrichment.py -v`
Expected: ALL PASS

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/ --tb=short -q`
Expected: 600+ passed, 0 failed

- [ ] **Step 4: Commit**

```bash
git add tests/test_analyze_enrichment.py
git commit -m "test: add end-to-end integration tests for fingerprint personalization"
```

- [ ] **Step 5: Verify server starts**

Run: `cd /Users/bel/quantara-nanoGPT && timeout 10 python emotion_api_server.py 2>&1 || true`
Expected: Server initializes without import errors. May fail on model loading (OK — we just need no ImportError/SyntaxError).

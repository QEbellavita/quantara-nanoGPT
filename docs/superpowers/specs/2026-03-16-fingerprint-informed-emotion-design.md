# Fingerprint-Informed Emotion Analysis — Design Spec

**Date:** 2026-03-16
**Goal:** Close the intelligence loop by feeding the user's accumulated genetic fingerprint profile back into emotion classification, so analysis improves with usage.

**Connected to:**
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.

---

## Problem

The current flow is one-directional: emotion analysis results get logged *into* the profile engine via `log_event()`, but the profile's accumulated knowledge (fingerprint, baselines, evolution stage) is never fed *back* to inform the analysis. The classifier treats every user identically regardless of history.

**Pre-existing bug:** The API server logs `result.get('emotion', 'neutral')` but `MultimodalEmotionAnalyzer.analyze()` returns the key as `dominant_emotion`. This means the profile engine has been accumulating `'neutral'` instead of actual emotions. This spec fixes this bug as part of the integration work.

## Solution

Three layered capabilities, each independently useful:

1. **Profile Snapshot** — lightweight in-memory summary updated on every `log_event()`, zero-latency read
2. **Ambiguity Tiebreaker** — personalized classification only when the classifier is uncertain
3. **Opt-in Enriched Response** — downstream consumers can request profile context alongside the emotion result

---

## Component 1: ProfileSnapshot

### Data Structure

```python
@dataclasses.dataclass
class ProfileSnapshot:
    user_id: str
    evolution_stage: int          # 1-5
    overall_confidence: float     # 0.0-1.0
    emotion_prior: Dict[str, float]  # family -> frequency ratio (normalized, sums to 1.0)
    dominant_family: str          # most frequent family
    event_count: int              # total emotional events seen
    family_counts: Dict[str, int] # raw counts per family (internal, for incremental updates)
    last_updated: float           # timestamp
```

### Storage

- `UserProfileEngine._snapshots: Dict[str, ProfileSnapshot]` — in-memory dict, ~200 bytes per user
- No SQLite persistence — snapshots rebuild on server restart (see Restart Rebuild below)

### Thread Safety

Access to `_snapshots` is protected by `UserProfileEngine._snapshot_lock` (a `threading.Lock`). The lock is held during the read-modify-write cycle in `log_event()` and during `get_snapshot()`. This is cheap (nanosecond-scale lock on a dict lookup) and prevents data races under Flask's threaded server or gunicorn with threads.

### Update Logic

In `UserProfileEngine.log_event()`, after the existing event logging:

1. If the event domain is `emotional` and event type is `emotion_classified`:
   - Acquire `_snapshot_lock`
   - If no snapshot exists for this user, create one with defaults:
     - `evolution_stage` from `db.get_or_create_profile(user_id)`
     - Empty `emotion_prior`, `event_count = 0`
   - Extract `family` from event data
   - Increment `family_counts[family]`
   - Recompute `emotion_prior` as normalized `family_counts` (each value = count / total, sums to 1.0)
   - Update `dominant_family` to the argmax
   - Increment `event_count`
   - Set `last_updated` to current time
   - Release `_snapshot_lock`

### Restart Rebuild

On `get_snapshot()`, if no snapshot exists for the requested user:

1. Release `_snapshot_lock` (or don't acquire it yet)
2. Query `ProfileDB.get_events(user_id, domain='emotional', limit=10000)` — outside the lock
3. Count families across all `emotion_classified` events to build `family_counts`
4. Compute `emotion_prior` and `dominant_family` from counts
5. Acquire `_snapshot_lock`, check if another thread already inserted a snapshot for this user
6. If not yet cached: insert the new `ProfileSnapshot`
7. Release lock, return snapshot

This check-then-set pattern ensures the DB query (potentially slow for 10K rows) never blocks snapshot operations for other users.

### Public API

```python
def get_snapshot(self, user_id: str) -> Optional[ProfileSnapshot]:
    """Return the cached profile snapshot, rebuilding from DB if needed.
    Returns None only if user has zero emotional events."""
```

---

## Component 2: Ambiguity Tiebreaker

### Function Signature

```python
def apply_profile_personalization(
    result: dict,
    snapshot: Optional[ProfileSnapshot],
) -> dict:
    """Apply profile-based personalization to emotion classification result.

    Modifies result in-place and returns it.
    Adds 'personalized' (bool) and 'personalization_reason' (str) keys.
    """
```

### Logic

```
1. Guard: skip if snapshot is None or snapshot.event_count < 10
   -> result['personalized'] = False, return

2. Guard: skip if 'family_scores' not in result (fallback classifier)
   -> result['personalized'] = False, return

3. Extract top-2 candidates from result['family_scores']:
   - top_1_family, top_1_score = highest scoring family
   - top_2_family, top_2_score = second highest

4. Check ambiguity condition:
   - top_1_score < 0.5  OR  (top_1_score - top_2_score) < 0.05
   - If neither: result['personalized'] = False, return

5. Apply tiebreaker:
   - PRIOR_WEIGHT = 0.05
   - For each of top-2 candidates:
     adjusted_score = original_score + PRIOR_WEIGHT * snapshot.emotion_prior.get(family, 0.0)
   - If the ordering changes (top-2 overtakes top-1 after adjustment):
     - Swap winner family in result['family']
     - Look up the top sub-emotion within the new winner family from result['scores']
     - Update result['dominant_emotion'] to that sub-emotion
     - Update result['confidence'] to the new family's adjusted score
     - Publish event: profile_engine.event_bus.publish('profile.personalization.applied', ...)
   - result['personalized'] = True
   - result['personalization_reason'] = f"profile_tiebreak: {winner} prior {prior:.2f} > {loser} prior {prior:.2f}"

6. If ambiguity detected but ordering didn't change:
   - result['personalized'] = True
   - result['personalization_reason'] = "profile_consulted: original classification confirmed"

7. family_scores in the response always reflects the ORIGINAL classifier output (unadjusted).
   The adjusted scores are internal to the tiebreaker decision only.
```

### Constraints

- `PRIOR_WEIGHT = 0.05` — constant, not configurable
- Minimum 10 events before activation (cold start protection)
- Never overrides confident predictions (confidence >= 0.5 AND gap >= 0.05)
- The function is pure except for the optional event bus publish (fire-and-forget, non-blocking)
- `family_scores` in the response is always the raw classifier output — not adjusted

### Family Score Availability

The `MultimodalEmotionAnalyzer.analyze()` method currently returns `dominant_emotion`, `family`, `confidence`, `is_fallback`, and `scores`. For the tiebreaker to work, it also needs family-level scores.

**Change:** `MultimodalEmotionAnalyzer.analyze()` will include a `family_scores` dict in its return value — a mapping of family name to float confidence. This is the `family_probs` tensor (already computed via softmax at `emotion_classifier.py:531`/`746`) converted to a dict keyed by `FAMILY_NAMES`. These scores sum to 1.0.

### Key Name Fix

The canonical key for the detected emotion is `dominant_emotion` (as returned by `MultimodalEmotionAnalyzer.analyze()`). The API server's `log_event()` call currently reads `result.get('emotion', 'neutral')` which always falls back to `'neutral'`. This will be fixed to use `result.get('dominant_emotion', 'neutral')` so that snapshots accumulate correct priors.

---

## Component 3: Opt-in Enriched Response

### Request Format

```json
{
  "text": "I can't stop worrying about tomorrow",
  "user_id": "user_123",
  "include_profile": true
}
```

### Response Format (when include_profile=true and user_id present)

```json
{
  "dominant_emotion": "anxiety",
  "family": "Fear",
  "confidence": 0.47,
  "family_scores": {"Fear": 0.47, "Sadness": 0.42, "Anger": 0.05},
  "personalized": true,
  "personalization_reason": "profile_tiebreak: Fear prior 0.38 > Sadness prior 0.21",
  "status": "success",
  "profile_context": {
    "evolution_stage": 3,
    "overall_confidence": 0.72,
    "dominant_family": "Fear",
    "emotion_prior": {"Fear": 0.38, "Sadness": 0.21, "Anger": 0.15, "Joy": 0.12},
    "event_count": 847,
    "last_updated": "2026-03-16T10:42:00Z"
  }
}
```

### Backward Compatibility

| Condition | `personalized` field | `profile_context` field |
|-----------|---------------------|------------------------|
| No `user_id` | absent | absent |
| `user_id` present, `profile_engine` is None | `personalized: false` | absent |
| `user_id` present, `profile_engine` available | present (bool) | absent |
| `user_id` + `include_profile: true` + `profile_engine` available | present (bool) | present (dict) |

The `personalized` and `personalization_reason` fields are included when `user_id` is provided and `profile_engine` is available. When `profile_engine` is `None` (not initialized), `personalized` is set to `false` and `personalization_reason` is omitted. The heavier `profile_context` block requires explicit opt-in.

---

## Event Bus Integration

When a personalization tiebreak changes the winning emotion (step 5 ordering change), publish an event to the profile event bus:

```python
bus.publish('profile.personalization.applied', {
    'user_id': user_id,
    'original_family': original_family,
    'personalized_family': new_family,
    'prior_weight_used': PRIOR_WEIGHT,
    'ambiguity_gap': gap,
})
```

This allows downstream subscribers (AlertEngine, IntelligencePublisher) to react to personalization events. The publish is fire-and-forget — failure does not affect the analysis response.

When personalization is consulted but the ordering doesn't change, no event is published.

---

## File Changes

| File | Change | Lines (est.) |
|------|--------|-------------|
| `user_profile_engine.py` | Add `ProfileSnapshot` dataclass, `_snapshots` dict with lock, update in `log_event()`, `get_snapshot()` with lazy DB rebuild | ~80 |
| `emotion_classifier.py` | Add `apply_profile_personalization()` function, surface `family_scores` from `family_probs` in `analyze()` return | ~60 |
| `emotion_api_server.py` | In `/api/emotion/analyze`: fix `dominant_emotion` key bug, fetch snapshot, call personalization, conditionally attach `profile_context` | ~30 |
| `tests/test_profile_snapshot.py` | New — snapshot creation, update on log_event, prior accuracy, cold start, restart rebuild, thread safety | ~100 |
| `tests/test_personalization.py` | New — tiebreaker triggers/skips, confident untouched, ordering change, missing family_scores guard, event bus publish | ~100 |
| `tests/test_analyze_enrichment.py` | New — opt-in format, backward compat, missing user_id, profile_engine=None | ~80 |

**No changes to:** `profile_db.py`, `profile_event_bus.py`, `ecosystem_connector.py`, `alert_engine.py`, `intelligence_publisher.py`.

---

## Edge Cases

1. **Server restart:** Snapshots are rebuilt lazily from DB on first `get_snapshot()` call per user. Accurate priors from historical events, one-time cost.
2. **New user (< 10 events):** Personalization skipped, `personalized: false`.
3. **Fallback classifier (keyword-based):** No `family_scores` in result. Guard in step 2 skips personalization, `personalized: false`.
4. **All families have equal prior:** Tiebreaker nudges are equal, original classification stands.
5. **Single-family user:** Prior heavily weighted to one family. PRIOR_WEIGHT of 0.05 limits the nudge — can only shift scores by 0.05 max, preventing over-reinforcement.
6. **`profile_engine` is None:** Can happen if DB init fails. `personalized` set to `false`, no `profile_context`, analysis proceeds normally.
7. **`family_scores` response values:** Always reflect the original classifier output (unadjusted). Adjusted scores are internal to the tiebreaker decision only, so `family_scores` always sums to 1.0.
8. **Concurrent requests for same user:** Protected by `_snapshot_lock`. Lock contention is minimal — DB queries for lazy rebuild run outside the lock using a check-then-set pattern. Only dict read/write happens under the lock.

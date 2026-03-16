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
    emotion_prior: Dict[str, float]  # family -> frequency ratio (normalized)
    dominant_family: str          # most frequent family
    event_count: int              # total emotional events seen
    family_counts: Dict[str, int] # raw counts per family (internal, for incremental updates)
    last_updated: float           # timestamp
```

### Storage

- `UserProfileEngine._snapshots: Dict[str, ProfileSnapshot]` — in-memory dict, ~200 bytes per user
- No SQLite persistence — snapshots are ephemeral and rebuild from events on server restart (lazy, on first `log_event()` or `get_snapshot()` call)

### Update Logic

In `UserProfileEngine.log_event()`, after the existing event logging:

1. If the event domain is `emotional` and event type is `emotion_classified`:
   - Extract `family` from event data
   - Increment `family_counts[family]`
   - Recompute `emotion_prior` as normalized `family_counts`
   - Update `dominant_family` to the argmax
   - Increment `event_count`
   - Set `last_updated` to current time
2. If no snapshot exists for this user, create one with defaults:
   - `evolution_stage` from `db.get_or_create_profile(user_id)`
   - Empty `emotion_prior`, `event_count = 0`

### Public API

```python
def get_snapshot(self, user_id: str) -> Optional[ProfileSnapshot]:
    """Return the cached profile snapshot, or None if no data yet."""
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

2. Extract top-2 candidates from result:
   - Need result to contain 'family_scores' or 'all_scores' with family-level confidences
   - top_1_family, top_1_score = highest scoring family
   - top_2_family, top_2_score = second highest

3. Check ambiguity condition:
   - top_1_score < 0.5  OR  (top_1_score - top_2_score) < 0.05
   - If neither: result['personalized'] = False, return

4. Apply tiebreaker:
   - PRIOR_WEIGHT = 0.05
   - For each of top-2 candidates:
     adjusted_score = original_score + PRIOR_WEIGHT * snapshot.emotion_prior.get(family, 0.0)
   - If the ordering changes (top-2 overtakes top-1 after adjustment):
     - Swap winner: result['emotion'] = new winner's top emotion
     - result['family'] = new winner family
     - result['confidence'] = new winner's adjusted score (renormalized)
   - result['personalized'] = True
   - result['personalization_reason'] = f"profile_tiebreak: {winner} prior {prior:.2f} > {loser} prior {prior:.2f}"

5. If ambiguity detected but ordering didn't change:
   - result['personalized'] = True
   - result['personalization_reason'] = "profile_consulted: original classification confirmed"
```

### Constraints

- `PRIOR_WEIGHT = 0.05` — constant, not configurable
- Minimum 10 events before activation (cold start protection)
- Never overrides confident predictions (confidence >= 0.5 AND gap >= 0.05)
- The function is pure — no side effects, no DB calls, no network

### Family Score Availability

The `MultimodalEmotionAnalyzer.analyze()` method currently returns `emotion`, `family`, `confidence`, and `is_fallback`. For the tiebreaker to work, it also needs family-level scores.

**Change:** `MultimodalEmotionAnalyzer.analyze()` will include a `family_scores` dict in its return value — a mapping of family name to confidence score. This data is already computed internally during classification (the family classifier produces logits for all 9 families); it just needs to be surfaced in the response.

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
  "emotion": "anxiety",
  "family": "Fear",
  "confidence": 0.47,
  "family_scores": {"Fear": 0.47, "Sadness": 0.42, "Anger": 0.05, ...},
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
| `user_id` present, no `include_profile` | present (bool) | absent |
| `user_id` + `include_profile: true` | present (bool) | present (dict) |

The `personalized` and `personalization_reason` fields are always included when `user_id` is provided — they're cheap and useful for debugging. The heavier `profile_context` block requires explicit opt-in.

---

## File Changes

| File | Change | Lines (est.) |
|------|--------|-------------|
| `user_profile_engine.py` | Add `ProfileSnapshot` dataclass, `_snapshots` dict, update in `log_event()`, `get_snapshot()` method | ~60 |
| `emotion_classifier.py` | Add `apply_profile_personalization()` function, surface `family_scores` in `analyze()` return | ~50 |
| `emotion_api_server.py` | In `/api/emotion/analyze`: fetch snapshot, call personalization, conditionally attach `profile_context` | ~25 |
| `tests/test_profile_snapshot.py` | New — snapshot creation, update on log_event, prior accuracy, cold start | ~80 |
| `tests/test_personalization.py` | New — tiebreaker triggers/skips, confident untouched, ordering change | ~90 |
| `tests/test_analyze_enrichment.py` | New — opt-in format, backward compat, missing user_id | ~70 |

**No changes to:** `profile_db.py`, `profile_event_bus.py`, `ecosystem_connector.py`, `alert_engine.py`, `intelligence_publisher.py`, or any other existing module.

---

## Edge Cases

1. **Server restart:** Snapshots are lost. Rebuilt lazily on first `log_event()` per user. Until then, `get_snapshot()` returns `None` and personalization is skipped.
2. **New user (< 10 events):** Personalization skipped, `personalized: false`.
3. **Fallback classifier (keyword-based):** No `family_scores` available. Personalization skipped, result unchanged.
4. **All families have equal prior:** Tiebreaker has no effect, original classification stands.
5. **Single-family user:** Prior heavily weighted to one family. PRIOR_WEIGHT of 0.05 limits the nudge — can only shift scores by 0.05 max, preventing over-reinforcement.

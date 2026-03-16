# EmotionGPT Coordinator — Design Spec

**Date:** 2026-03-16
**Goal:** Unify the scattered emotion analysis subsystems (classifier, transitions, websocket, auto-retrain) under a single coordinator that also monitors personalization swap rates to emit retrain signals.

**Connected to:**
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.

---

## Problem

The 6 "Emotion GPT enhancements" (attention fusion, transitions, websocket streaming, auto-retrain, personalization, evaluation) are scattered across 4+ files with no unified entry point. The API server manually assembles the pipeline in the analyze endpoint. There is also no mechanism to detect when the classifier is underperforming and retraining is needed — the personalization system compensates silently.

## Solution

A thin `EmotionGPT` coordinator class that:
1. Holds references to all emotion subsystems (facade pattern)
2. Exposes `analyze_with_context()` as a unified pipeline entry point
3. Monitors personalization swap rates via MetricsCollector and publishes `retrain.recommended` events when the classifier shows signs of drift

---

## Component 1: EmotionGPT Class

### Data Structure

```python
class EmotionGPT:
    """Coordinator for the emotion analysis pipeline.

    Unifies: analyzer, transitions, auto-retrain.
    Subscribes to event bus for retrain signal monitoring.
    """
    analyzer: MultimodalEmotionAnalyzer   # or keyword fallback model
    transitions: EmotionTransitionTracker  # or None
    auto_retrain: AutoRetrainManager       # or None
    profile_engine: UserProfileEngine      # or None
    _metrics: MetricsCollector             # or None
    _bus: ProfileEventBus                  # or None
    _check_interval: int                   # seconds between retrain checks (default 300)
    _swap_windows: List[float]             # rolling window of swap rates
    _last_total: float                     # counter snapshot from last check
    _last_swapped: float                   # counter snapshot from last check
    _timer: threading.Timer               # self-repeating daemon timer
    _stopped: bool                        # flag to prevent rescheduling after shutdown
```

### Constructor

```python
def __init__(
    self,
    analyzer,
    transition_tracker=None,
    auto_retrain_manager=None,
    profile_engine=None,
    metrics=None,
    bus=None,
    check_interval=300,
):
```

All parameters except `analyzer` are optional. The coordinator degrades gracefully — if transitions is None, `analyze_with_context()` skips transition recording. If metrics is None, retrain monitoring is disabled. If profile_engine is None, personalization is skipped.

### Startup

On init, if metrics and bus are both available, start the retrain monitoring timer:
```python
self._stopped = False
if self._metrics and self._bus:
    self._start_retrain_monitor()
```

### Shutdown

```python
def shutdown(self):
    """Cancel the retrain monitoring timer and prevent rescheduling."""
    self._stopped = True
    if self._timer:
        self._timer.cancel()
```

---

## Component 2: Retrain Signal Monitoring

### How It Works

A self-repeating `threading.Timer` fires every `check_interval` seconds (default 5 minutes). Each check runs inside a try/finally to guarantee rescheduling:

```python
def _check_retrain_signal(self):
    try:
        # ... evaluation logic ...
    except Exception:
        logger.exception("Retrain check failed")
    finally:
        if not self._stopped:
            self._schedule_next_check()
```

### Evaluation Logic (inside the try block):

1. **Minimum sample guard (evaluated first, before any division):** Read `personalization.requests` and `personalization.swapped` from MetricsCollector. Compute `window_total = total - last_total`. If `window_total < 20`, skip — not enough data.
2. Compute **window swap rate:** `window_swapped / window_total` where `window_swapped = swapped - last_swapped`
3. Store the window rate in `_swap_windows` (keep last 3 = 15 minutes)
4. Evaluate triggers (only one event per check — threshold takes priority):
   - **Immediate threshold:** If window swap rate > 0.20 (20%), publish `retrain.recommended` with `trigger='threshold'`
   - **Trend detection (only if threshold not triggered):** If 3 consecutive windows show increasing rates AND the latest > 0.10 (10%), publish `retrain.recommended` with `trigger='trend'`
5. Save current counter values as `_last_total` / `_last_swapped` for next window

### Event Payload

```python
bus.publish('retrain.recommended', {
    'trigger': 'threshold' | 'trend',
    'swap_rate': 0.23,
    'window_rates': [0.12, 0.18, 0.23],
    'window_seconds': 300,
    'total_requests': 567,
    'total_swapped': 130,
    'recommendation': 'Classifier swap rate exceeds threshold. Retraining recommended.',
})
```

### Timer Lifecycle

- Timer is a daemon thread (`timer.daemon = True`) — dies with the server
- Timer reschedules itself in the `finally` block of each check
- `shutdown()` sets `_stopped = True` and cancels the pending timer — the `_stopped` flag prevents rescheduling from a currently executing check
- If metrics or bus is None, no timer is created

---

## Component 3: Unified Pipeline

### analyze_with_context()

```python
def analyze_with_context(
    self,
    text: str,
    biometrics: dict = None,
    pose: dict = None,
    user_id: str = None,
    include_profile: bool = False,
    return_embedding: bool = False,
) -> dict:
    """Unified emotion analysis pipeline.

    1. Classify via analyzer (forwarding all parameters)
    2. Apply personalization (if user_id and profile engine available)
    3. Record transition (if user_id and transition tracker available)
    4. Return enriched result
    """
```

**Pipeline steps:**

1. `result = self.analyzer.analyze(text, biometrics, pose=pose, return_embedding=return_embedding)` — core classification, forwarding all analyzer parameters
2. If `user_id` and `self.profile_engine`:
   - `snapshot = self.profile_engine.get_profile_snapshot(user_id)`
   - `apply_profile_personalization(result, snapshot)`
3. If `user_id` and `self.transitions`:
   - `self.transitions.record(user_id, result.get('dominant_emotion', result.get('emotion', 'neutral')), ...)`
4. If `include_profile` and snapshot available:
   - Attach `profile_context` to result (same format as the API endpoint)
5. Return result

**Relationship to API endpoint:** The API endpoint continues to work as-is. `analyze_with_context()` is an alternative entry point for programmatic consumers. The API endpoint can optionally be refactored to delegate to it later, but that is not part of this spec.

---

## File Changes

| File | Change | Lines (est.) |
|------|--------|-------------|
| `emotion_gpt.py` (create) | EmotionGPT class — facade refs, analyze_with_context(), retrain signal with timer | ~160 |
| `emotion_api_server.py` (modify) | Create EmotionGPT in create_app() after subsystem init, wire all refs | ~15 |
| `tests/test_emotion_gpt.py` (create) | Facade access, pipeline, threshold trigger, trend trigger, minimum sample guard, timer lifecycle, shutdown, graceful degradation | ~200 |

**No changes to:** `emotion_classifier.py`, `emotion_transition_tracker.py`, `auto_retrain.py`, `profile_event_bus.py`, `user_profile_engine.py`, `metrics_collector.py`.

---

## Edge Cases

1. **No metrics/bus:** Retrain monitoring disabled. `analyze_with_context()` still works for classification + transitions.
2. **No transition tracker:** Pipeline skips transition recording. No error.
3. **No auto-retrain manager:** Coordinator holds None reference. Retrain signal is still published (consumers decide what to do).
4. **No profile engine:** Personalization skipped in `analyze_with_context()`. No error.
5. **Server restart:** Counters reset to 0. `_last_total`/`_last_swapped` start at 0. First window may be inaccurate but self-corrects on second window.
6. **Low traffic:** Minimum sample guard (20 requests per window) prevents false positives. Guard is checked before division to avoid divide-by-zero.
7. **Timer drift:** `threading.Timer` is not perfectly periodic but 5-minute granularity makes drift irrelevant.
8. **Retrain signal deduplication:** Only one `retrain.recommended` event per check cycle — threshold takes priority over trend. Consumers (auto-retrain) should implement their own cooldown (the existing auto-retrain system already has one).
9. **Check callback exception:** try/finally ensures timer always reschedules even if the check logic raises.
10. **Shutdown during check:** `_stopped` flag prevents rescheduling from the finally block of a currently executing check.

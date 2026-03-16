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
    _metrics: MetricsCollector             # or None
    _bus: ProfileEventBus                  # or None
    _check_interval: int                   # seconds between retrain checks (default 300)
    _swap_windows: List[float]             # rolling window of swap rates
    _last_total: float                     # counter snapshot from last check
    _last_swapped: float                   # counter snapshot from last check
    _timer: threading.Timer               # self-repeating daemon timer
```

### Constructor

```python
def __init__(
    self,
    analyzer,
    transition_tracker=None,
    auto_retrain_manager=None,
    metrics=None,
    bus=None,
    check_interval=300,
):
```

All parameters except `analyzer` are optional. The coordinator degrades gracefully — if transitions is None, `analyze_with_context()` skips transition recording. If metrics is None, retrain monitoring is disabled.

### Startup

On init, if metrics and bus are both available, start the retrain monitoring timer:
```python
if self._metrics and self._bus:
    self._start_retrain_monitor()
```

### Shutdown

```python
def shutdown(self):
    """Cancel the retrain monitoring timer."""
    if self._timer:
        self._timer.cancel()
```

---

## Component 2: Retrain Signal Monitoring

### How It Works

A self-repeating `threading.Timer` fires every `check_interval` seconds (default 5 minutes). Each check:

1. Read current `personalization.requests` and `personalization.swapped` counters from MetricsCollector
2. Compute the **window swap rate** for this interval: `(swapped - last_swapped) / (total - last_total)` — this gives the rate for the last 5 minutes, not lifetime
3. Store the window rate in `_swap_windows` (keep last 3 = 15 minutes)
4. Evaluate triggers:
   - **Immediate threshold:** If window swap rate > 0.20 (20%), publish `retrain.recommended`
   - **Trend detection:** If 3 consecutive windows show increasing rates AND the latest > 0.10 (10%), publish `retrain.recommended`
5. Save current counter values as `_last_total` / `_last_swapped` for next window

### Minimum Sample Guard

If fewer than 20 new personalization requests occurred in the window, skip evaluation — not enough data to draw conclusions.

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
- Timer reschedules itself at the end of each check
- `shutdown()` cancels the pending timer
- If metrics or bus is None, no timer is created

---

## Component 3: Unified Pipeline

### analyze_with_context()

```python
def analyze_with_context(
    self,
    text: str,
    biometrics: dict = None,
    user_id: str = None,
    include_profile: bool = False,
) -> dict:
    """Unified emotion analysis pipeline.

    1. Classify via analyzer
    2. Apply personalization (if user_id and profile engine available)
    3. Record transition (if user_id and transition tracker available)
    4. Return enriched result
    """
```

**Pipeline steps:**

1. `result = self.analyzer.analyze(text, biometrics)` — core classification
2. If `user_id` and personalization is available (profile engine accessible):
   - Fetch profile snapshot
   - Call `apply_profile_personalization(result, snapshot)`
3. If `user_id` and `self.transitions`:
   - `self.transitions.record(user_id, result['dominant_emotion'], ...)`
4. If `include_profile` and snapshot available:
   - Attach `profile_context` to result
5. Return result

**Important:** This method accesses the profile engine through the bus's ecosystem or directly if wired. The coordinator accepts an optional `profile_engine` reference for this purpose.

**Relationship to API endpoint:** The API endpoint continues to work as-is. `analyze_with_context()` is an alternative entry point for programmatic consumers. The API endpoint can optionally be refactored to delegate to it later, but that is not part of this spec.

---

## File Changes

| File | Change | Lines (est.) |
|------|--------|-------------|
| `emotion_gpt.py` (create) | EmotionGPT class — facade refs, analyze_with_context(), retrain signal with timer | ~150 |
| `emotion_api_server.py` (modify) | Create EmotionGPT in create_app() after subsystem init, wire all refs | ~15 |
| `tests/test_emotion_gpt.py` (create) | Facade access, pipeline, threshold trigger, trend trigger, minimum sample guard, timer lifecycle, graceful degradation | ~180 |

**No changes to:** `emotion_classifier.py`, `emotion_transition_tracker.py`, `auto_retrain.py`, `profile_event_bus.py`, `user_profile_engine.py`, `metrics_collector.py`.

---

## Edge Cases

1. **No metrics/bus:** Retrain monitoring disabled. `analyze_with_context()` still works without personalization.
2. **No transition tracker:** Pipeline skips step 3. No error.
3. **No auto-retrain manager:** Coordinator holds None reference. Retrain signal is still published (consumers decide what to do).
4. **Server restart:** Counters reset to 0. First check window has no baseline — uses 0 as `_last_total`/`_last_swapped`. First window rate may be inaccurate but self-corrects on second window.
5. **Low traffic:** Minimum sample guard (20 requests per window) prevents false positives during quiet periods.
6. **Timer drift:** `threading.Timer` is not perfectly periodic but 5-minute granularity makes drift irrelevant.
7. **Retrain signal spam:** The threshold trigger can fire every 5 minutes. Consumers (auto-retrain) should implement their own cooldown (the existing auto-retrain system already has one).
8. **Profile engine access:** `analyze_with_context()` accepts an optional `profile_engine` parameter. If not provided, personalization is skipped.

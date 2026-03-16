# Cross-Service Observability Metrics — Design Spec

**Date:** 2026-03-16
**Goal:** Add in-memory metrics collection across all subsystems and enrich the existing health endpoint so operators can monitor bus throughput, dead letter depth, personalization hit rates, classifier performance, and profile cache efficiency.

**Connected to:**
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.

---

## Problem

The event bus, ecosystem connector, intelligence publisher, alert engine, and personalization system are all wired and running but invisible. There are no metrics, no health indicators, and no way to know if the system is performing well without reading logs. The existing `/api/emotion/status` endpoint only reports model availability.

## Solution

A single `MetricsCollector` class that aggregates atomic counters and gauges from all subsystems. No new dependencies, no background threads, no polling. Each subsystem increments counters at the relevant code point. The existing `/api/emotion/status` endpoint is enriched with a `metrics` block.

---

## Component 1: MetricsCollector

### Data Structure

```python
class MetricsCollector:
    """Passive in-memory metrics aggregator.

    Thread-safe counters and gauges, no external dependencies.
    """
    _counters: Dict[str, float]   # monotonically increasing (or summing)
    _gauges: Dict[str, float]     # point-in-time values
    _lock: threading.Lock
    _started_at: float            # server start timestamp
```

### Operations

```python
def increment(self, name: str, amount: float = 1.0) -> None:
    """Atomically increment a counter."""

def set_gauge(self, name: str, value: float) -> None:
    """Set a gauge to a point-in-time value."""

def get_counter(self, name: str) -> float:
    """Read a single counter value (0.0 if not set)."""

def get_gauge(self, name: str) -> float:
    """Read a single gauge value (0.0 if not set)."""

def get_all(self) -> dict:
    """Return all metrics as a JSON-serializable dict."""
    # Returns:
    # {
    #   'server_started_at': ISO timestamp,
    #   'metrics_at': ISO timestamp,
    #   'uptime_seconds': float,
    #   'counters': {name: value, ...},
    #   'gauges': {name: value, ...},
    # }
```

### Thread Safety

All operations are protected by a single `threading.Lock`. The lock scope is minimal (dict read/write only). No I/O under the lock.

### Subsystem Wiring

Each subsystem receives the collector via a `set_metrics(collector)` method or constructor parameter. If collector is None, all metrics calls are silently skipped — metrics are fully optional.

---

## Component 2: Metrics Points (16 total)

### Event Bus (profile_event_bus.py)

| Metric | Type | Where |
|--------|------|-------|
| `bus.publish_count` | counter | `ProfileEventBus.publish()` — after successful dispatch |
| `bus.subscriber_count` | gauge | `ProfileEventBus.subscribe()` / `unsubscribe()` — set to current len |

### Ecosystem Connector (ecosystem_connector.py)

| Metric | Type | Where |
|--------|------|-------|
| `connector.delivery_count` | counter | After successful outbound webhook delivery |
| `connector.delivery_failures` | counter | After failed delivery (before dead letter) |
| `connector.dead_letter_count` | gauge | After `_store_dead_letter()` and `replay_dead_letters()` — set to current count |

### Personalization (emotion_api_server.py analyze endpoint)

| Metric | Type | Where |
|--------|------|-------|
| `personalization.requests` | counter | Every time `apply_profile_personalization()` is called |
| `personalization.skipped` | counter | When `result['personalized'] is False` |
| `personalization.consulted` | counter | When `result['personalized'] is True` and reason contains `profile_consulted` |
| `personalization.swapped` | counter | When `result['personalized'] is True` and reason contains `profile_tiebreak` |

### Emotion Classifier (emotion_classifier.py / emotion_api_server.py)

| Metric | Type | Where |
|--------|------|-------|
| `classifier.requests` | counter | In `/api/emotion/analyze` after `model.analyze()` returns |
| `classifier.fallback_count` | counter | When `result.get('is_fallback')` is True, or when keyword fallback path is used |
| `classifier.confidence_sum` | counter | Accumulate `result.get('confidence', 0)` — divide by requests for avg |

### Profile Engine (user_profile_engine.py)

| Metric | Type | Where |
|--------|------|-------|
| `profile.active_snapshots` | gauge | After `_update_snapshot()` — set to `len(self._snapshots)` |
| `profile.events_logged` | counter | In `log_event()` after successful DB write |
| `profile.cache_hits` | counter | In `get_profile_snapshot()` when snapshot found in `_snapshots` |
| `profile.cache_rebuilds` | counter | In `get_profile_snapshot()` when rebuilt from DB |

---

## Component 3: Health Endpoint Enrichment

### Current Response (unchanged fields)

```json
{
  "status": "online",
  "model": "quantara-emotion-gpt",
  ...existing fields...
}
```

### Added Field

```json
{
  ...existing fields...,
  "metrics": {
    "server_started_at": "2026-03-16T12:00:00+00:00",
    "metrics_at": "2026-03-16T14:32:15+00:00",
    "uptime_seconds": 9135.2,
    "counters": {
      "bus.publish_count": 1423,
      "connector.delivery_count": 892,
      "connector.delivery_failures": 3,
      "personalization.requests": 567,
      "personalization.skipped": 412,
      "personalization.consulted": 155,
      "personalization.swapped": 23,
      "classifier.requests": 890,
      "classifier.fallback_count": 12,
      "classifier.confidence_sum": 623.4,
      "profile.events_logged": 2341,
      "profile.cache_hits": 540,
      "profile.cache_rebuilds": 27
    },
    "gauges": {
      "bus.subscriber_count": 4,
      "connector.dead_letter_count": 1,
      "profile.active_snapshots": 43
    }
  }
}
```

When `metrics_collector` is None (not initialized), the `metrics` key is omitted entirely. Fully backward compatible.

---

## File Changes

| File | Change | Lines (est.) |
|------|--------|-------------|
| `metrics_collector.py` (create) | MetricsCollector class — counters, gauges, lock, get_all() | ~70 |
| `profile_event_bus.py` (modify) | Add `set_metrics()`, increment bus counters in publish/subscribe | ~10 |
| `ecosystem_connector.py` (modify) | Add `set_metrics()`, increment delivery/failure counters, set dead_letter gauge | ~15 |
| `emotion_classifier.py` (modify) | Add `set_metrics()` to MultimodalEmotionAnalyzer, increment classifier counters in analyze() | ~10 |
| `user_profile_engine.py` (modify) | Add `set_metrics()`, increment profile counters in log_event/get_profile_snapshot | ~15 |
| `emotion_api_server.py` (modify) | Create MetricsCollector on startup, wire to subsystems, increment personalization counters in analyze, enrich /api/emotion/status | ~30 |
| `tests/test_metrics_collector.py` (create) | Counter/gauge ops, thread safety, get_all format, no-op when None | ~80 |
| `tests/test_metrics_integration.py` (create) | Verify subsystems increment metrics after operations | ~100 |

**No changes to:** `profile_db.py`, `alert_engine.py`, `intelligence_publisher.py`, `websocket_router.py`.

---

## Edge Cases

1. **metrics_collector is None:** All subsystems check `if self._metrics:` before incrementing. No errors, no metrics.
2. **Server restart:** All counters reset to 0. `server_started_at` records when.
3. **High concurrency:** Single lock is fine — increment is a dict read/write, nanosecond-scale. No I/O under lock.
4. **confidence_sum overflow:** Float64 can hold ~10^308. At 1000 requests/sec for 100 years, we'd accumulate ~3*10^12. No overflow risk.
5. **Gauge staleness:** `connector.dead_letter_count` is set after each store/replay operation. Between operations it reflects the last known value, not real-time. Acceptable — the gauge updates on every write path.
6. **Memory:** 16 metrics = 16 dict entries. Negligible.

# Cross-Service Observability Metrics Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add passive in-memory metrics collection (19 counters/gauges) across all subsystems and enrich the existing health endpoint with a `metrics` block.

**Architecture:** Single `MetricsCollector` class with thread-safe counters and gauges. Each subsystem receives the collector via `set_metrics()` and increments counters at relevant code points. The existing `/api/emotion/status` endpoint gains a `metrics` key with all counters, gauges, and timestamps.

**Tech Stack:** Python 3, threading, Flask, pytest

**Spec:** `docs/superpowers/specs/2026-03-16-observability-metrics-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `metrics_collector.py` (create) | `MetricsCollector` class — counters, gauges, lock, get_all() |
| `profile_event_bus.py` (modify) | Add `set_metrics()`, increment `bus.publish_count`, `bus.publish_errors`, set `bus.subscriber_count` |
| `ecosystem_connector.py` (modify) | Add `set_metrics()`, increment delivery/failure/inbound counters, maintain `dead_letter_count` gauge |
| `user_profile_engine.py` (modify) | Add `set_metrics()`, increment `profile.events_logged`, `cache_hits`, `cache_rebuilds`, set `active_snapshots` |
| `emotion_api_server.py` (modify) | Create MetricsCollector on startup, wire to subsystems, increment classifier + personalization counters, enrich `/api/emotion/status` |
| `tests/test_metrics_collector.py` (create) | Counter/gauge ops, thread safety, get_all format |
| `tests/test_metrics_integration.py` (create) | Verify subsystems increment metrics after operations |

---

## Chunk 1: MetricsCollector Core

### Task 1: MetricsCollector class and tests

**Files:**
- Create: `metrics_collector.py`
- Create: `tests/test_metrics_collector.py`

- [ ] **Step 1: Write failing tests for MetricsCollector**

```python
# tests/test_metrics_collector.py
"""Tests for MetricsCollector — passive in-memory metrics aggregation."""

import os
import sys
import time
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCounterOperations:

    def test_increment_default(self):
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        m.increment('test.counter')
        assert m.get_counter('test.counter') == 1.0

    def test_increment_by_amount(self):
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        m.increment('test.counter', 5.0)
        assert m.get_counter('test.counter') == 5.0

    def test_increment_accumulates(self):
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        m.increment('c', 3.0)
        m.increment('c', 2.0)
        assert m.get_counter('c') == 5.0

    def test_get_counter_default_zero(self):
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        assert m.get_counter('nonexistent') == 0.0


class TestGaugeOperations:

    def test_set_gauge(self):
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        m.set_gauge('test.gauge', 42.0)
        assert m.get_gauge('test.gauge') == 42.0

    def test_gauge_overwrite(self):
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        m.set_gauge('g', 10.0)
        m.set_gauge('g', 20.0)
        assert m.get_gauge('g') == 20.0

    def test_get_gauge_default_zero(self):
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        assert m.get_gauge('nonexistent') == 0.0


class TestGetAll:

    def test_get_all_structure(self):
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        m.increment('a.b', 1)
        m.set_gauge('c.d', 5)
        result = m.get_all()
        assert 'server_started_at' in result
        assert 'metrics_at' in result
        assert 'uptime_seconds' in result
        assert 'counters' in result
        assert 'gauges' in result
        assert result['counters']['a.b'] == 1.0
        assert result['gauges']['c.d'] == 5.0

    def test_get_all_uptime(self):
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        time.sleep(0.1)
        result = m.get_all()
        assert result['uptime_seconds'] >= 0.1

    def test_get_all_json_serializable(self):
        import json
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        m.increment('x', 1)
        m.set_gauge('y', 2)
        serialized = json.dumps(m.get_all())
        assert '"x"' in serialized


class TestThreadSafety:

    def test_concurrent_increments(self):
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        errors = []
        per_thread = 1000

        def inc():
            try:
                for _ in range(per_thread):
                    m.increment('concurrent')
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=inc) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert m.get_counter('concurrent') == 4 * per_thread
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_metrics_collector.py -v`
Expected: FAIL — `metrics_collector` not importable

- [ ] **Step 3: Implement MetricsCollector**

```python
# metrics_collector.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Metrics Collector
===============================================================================
Passive in-memory metrics aggregation for cross-service observability.

Thread-safe counters and gauges with no external dependencies.
Wired into subsystems via set_metrics() — fully optional.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
===============================================================================
"""

import threading
import time
from datetime import datetime, timezone
from typing import Dict


class MetricsCollector:
    """Passive in-memory metrics aggregator.

    Thread-safe counters and gauges, no external dependencies.
    Each subsystem receives this via set_metrics() and increments
    counters at relevant code points.
    """

    def __init__(self):
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._started_at = time.time()

    def increment(self, name: str, amount: float = 1.0) -> None:
        """Atomically increment a counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + amount

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge to a point-in-time value."""
        with self._lock:
            self._gauges[name] = value

    def get_counter(self, name: str) -> float:
        """Read a single counter value (0.0 if not set)."""
        with self._lock:
            return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> float:
        """Read a single gauge value (0.0 if not set)."""
        with self._lock:
            return self._gauges.get(name, 0.0)

    def get_all(self) -> dict:
        """Return all metrics as a JSON-serializable dict."""
        now = time.time()
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
        return {
            'server_started_at': datetime.fromtimestamp(
                self._started_at, tz=timezone.utc
            ).isoformat(),
            'metrics_at': datetime.fromtimestamp(
                now, tz=timezone.utc
            ).isoformat(),
            'uptime_seconds': round(now - self._started_at, 2),
            'counters': counters,
            'gauges': gauges,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_metrics_collector.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add metrics_collector.py tests/test_metrics_collector.py
git commit -m "feat: add MetricsCollector for cross-service observability"
```

---

## Chunk 2: Subsystem Instrumentation

### Task 2: Instrument event bus

**Files:**
- Modify: `profile_event_bus.py`
- Create: `tests/test_metrics_integration.py`

- [ ] **Step 1: Write failing tests for bus metrics**

```python
# tests/test_metrics_integration.py
"""Tests for metrics instrumentation across subsystems."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEventBusMetrics:

    @pytest.fixture
    def bus_with_metrics(self):
        from profile_event_bus import ProfileEventBus
        from metrics_collector import MetricsCollector
        bus = ProfileEventBus()
        metrics = MetricsCollector()
        bus.set_metrics(metrics)
        return bus, metrics

    def test_publish_increments_counter(self, bus_with_metrics):
        bus, metrics = bus_with_metrics
        bus.publish('test.topic', {'data': 1})
        bus.publish('test.topic', {'data': 2})
        assert metrics.get_counter('bus.publish_count') == 2

    def test_subscriber_count_gauge(self, bus_with_metrics):
        bus, metrics = bus_with_metrics
        sub_id = bus.subscribe('test.*', lambda t, p: None)
        assert metrics.get_gauge('bus.subscriber_count') == 1
        bus.unsubscribe(sub_id)
        assert metrics.get_gauge('bus.subscriber_count') == 0

    def test_publish_error_increments(self, bus_with_metrics):
        bus, metrics = bus_with_metrics
        def bad_callback(topic, payload):
            raise ValueError("boom")
        bus.subscribe('err.*', bad_callback)
        bus.publish('err.test', {'x': 1})
        assert metrics.get_counter('bus.publish_errors') == 1
        assert metrics.get_counter('bus.publish_count') == 1

    def test_no_metrics_no_error(self):
        """Bus works fine without metrics."""
        from profile_event_bus import ProfileEventBus
        bus = ProfileEventBus()
        bus.publish('test', {'x': 1})  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_metrics_integration.py::TestEventBusMetrics -v`
Expected: FAIL — `set_metrics` not found

- [ ] **Step 3: Instrument profile_event_bus.py**

Add `self._metrics = None` to `ProfileEventBus.__init__()`.

Add method:
```python
    def set_metrics(self, collector) -> None:
        """Wire metrics collector for observability."""
        self._metrics = collector
```

In `subscribe()` — after the subscriber is added to `self._subscribers`, add:
```python
        if self._metrics:
            self._metrics.set_gauge('bus.subscriber_count', len(self._subscribers))
```

In `unsubscribe()` — after the subscriber is removed, add:
```python
        if self._metrics:
            self._metrics.set_gauge('bus.subscriber_count', len(self._subscribers))
```

In `publish()` — after `if not self._running: return` (line 162), before the for loop, add:
```python
        if self._metrics:
            self._metrics.increment('bus.publish_count')
```

In `publish()` — in the `except Exception` block (line 174), add:
```python
                    if self._metrics:
                        self._metrics.increment('bus.publish_errors')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_metrics_integration.py::TestEventBusMetrics tests/test_event_bus.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add profile_event_bus.py tests/test_metrics_integration.py
git commit -m "feat: instrument event bus with publish/error/subscriber metrics"
```

---

### Task 3: Instrument ecosystem connector

**Files:**
- Modify: `ecosystem_connector.py`
- Modify: `tests/test_metrics_integration.py`

- [ ] **Step 1: Write failing tests for connector metrics**

Append to `tests/test_metrics_integration.py`:

```python
class TestEcosystemConnectorMetrics:

    @pytest.fixture
    def connector_with_metrics(self, tmp_path):
        from ecosystem_connector import EcosystemConnector
        from profile_event_bus import ProfileEventBus
        from metrics_collector import MetricsCollector
        bus = ProfileEventBus()
        connector = EcosystemConnector(bus, db_path=str(tmp_path / 'test.db'))
        metrics = MetricsCollector()
        connector.set_metrics(metrics)
        return connector, metrics

    def test_dead_letter_on_missing_user_id(self, connector_with_metrics):
        connector, metrics = connector_with_metrics
        connector.route_inbound({'domain': 'emotional', 'event_type': 'test'})
        assert metrics.get_counter('connector.inbound_dead_letters') == 1

    def test_dead_letter_gauge_increments(self, connector_with_metrics):
        connector, metrics = connector_with_metrics
        connector.route_inbound({'domain': 'emotional'})
        assert metrics.get_gauge('connector.dead_letter_count') >= 1

    def test_no_metrics_no_error(self, tmp_path):
        """Connector works fine without metrics."""
        from ecosystem_connector import EcosystemConnector
        from profile_event_bus import ProfileEventBus
        bus = ProfileEventBus()
        connector = EcosystemConnector(bus, db_path=str(tmp_path / 'c.db'))
        connector.route_inbound({'domain': 'x'})  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_metrics_integration.py::TestEcosystemConnectorMetrics -v`
Expected: FAIL — `set_metrics` not found

- [ ] **Step 3: Instrument ecosystem_connector.py**

Add `self._metrics = None` to `EcosystemConnector.__init__()`.

Add method:
```python
    def set_metrics(self, collector) -> None:
        """Wire metrics collector for observability."""
        self._metrics = collector
```

In `route_inbound()` — after `self._store_dead_letter(event_data, error_msg)` (line 102), add:
```python
            if self._metrics:
                self._metrics.increment('connector.inbound_dead_letters')
```

In `_store_dead_letter()` — at the end, after the INSERT, add:
```python
        if self._metrics:
            self._metrics.set_gauge('connector.dead_letter_count',
                                    self.get_dead_letter_count())
```

Note: `_store_dead_letter` does a DB write anyway, so one extra `SELECT COUNT(*)` is acceptable here. This keeps the gauge accurate.

In `deliver_intelligence()` — after a successful delivery, add:
```python
            if self._metrics:
                self._metrics.increment('connector.delivery_count')
```

In `deliver_intelligence()` — in the failure/exception path, add:
```python
            if self._metrics:
                self._metrics.increment('connector.delivery_failures')
```

In `replay_dead_letters()` — at the end of the method, add:
```python
        if self._metrics:
            self._metrics.set_gauge('connector.dead_letter_count',
                                    self.get_dead_letter_count())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_metrics_integration.py::TestEcosystemConnectorMetrics tests/test_ecosystem_connector.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add ecosystem_connector.py tests/test_metrics_integration.py
git commit -m "feat: instrument ecosystem connector with delivery/dead-letter metrics"
```

---

### Task 4: Instrument profile engine

**Files:**
- Modify: `user_profile_engine.py`
- Modify: `tests/test_metrics_integration.py`

- [ ] **Step 1: Write failing tests for profile metrics**

Append to `tests/test_metrics_integration.py`:

```python
class TestProfileEngineMetrics:

    @pytest.fixture
    def engine_with_metrics(self, tmp_path):
        from user_profile_engine import UserProfileEngine
        from metrics_collector import MetricsCollector
        engine = UserProfileEngine(db_path=str(tmp_path / 'test.db'))
        metrics = MetricsCollector()
        engine.set_metrics(metrics)
        return engine, metrics

    def test_events_logged_counter(self, engine_with_metrics):
        engine, metrics = engine_with_metrics
        engine.log_event('u1', 'emotional', 'emotion_classified', {
            'emotion': 'joy', 'family': 'Joy', 'confidence': 0.8,
        })
        assert metrics.get_counter('profile.events_logged') == 1

    def test_active_snapshots_gauge(self, engine_with_metrics):
        engine, metrics = engine_with_metrics
        engine.log_event('u1', 'emotional', 'emotion_classified', {
            'emotion': 'joy', 'family': 'Joy', 'confidence': 0.8,
        })
        assert metrics.get_gauge('profile.active_snapshots') == 1

    def test_cache_hit_counter(self, engine_with_metrics):
        engine, metrics = engine_with_metrics
        engine.log_event('u1', 'emotional', 'emotion_classified', {
            'emotion': 'joy', 'family': 'Joy', 'confidence': 0.8,
        })
        # First call — already in cache from log_event
        engine.get_profile_snapshot('u1')
        assert metrics.get_counter('profile.cache_hits') == 1

    def test_cache_rebuild_counter(self, engine_with_metrics):
        engine, metrics = engine_with_metrics
        engine.log_event('u1', 'emotional', 'emotion_classified', {
            'emotion': 'joy', 'family': 'Joy', 'confidence': 0.8,
        })
        # Clear cache, force rebuild
        with engine._snapshot_lock:
            engine._snapshots.clear()
        engine.get_profile_snapshot('u1')
        assert metrics.get_counter('profile.cache_rebuilds') == 1

    def test_no_metrics_no_error(self, tmp_path):
        from user_profile_engine import UserProfileEngine
        engine = UserProfileEngine(db_path=str(tmp_path / 'x.db'))
        engine.log_event('u1', 'emotional', 'emotion_classified', {
            'emotion': 'joy', 'family': 'Joy', 'confidence': 0.8,
        })
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_metrics_integration.py::TestProfileEngineMetrics -v`
Expected: FAIL — `set_metrics` not found

- [ ] **Step 3: Instrument user_profile_engine.py**

Add `self._metrics = None` to `UserProfileEngine.__init__()` (after `self._snapshot_lock`).

Add method:
```python
    def set_metrics(self, collector) -> None:
        """Wire metrics collector for observability."""
        self._metrics = collector
```

In `log_event()` — after `event_id = self.db.log_event(...)` and before the snapshot update block, add:
```python
            if self._metrics:
                self._metrics.increment('profile.events_logged')
```

In `_update_snapshot()` — at the end of the fast path (after updating existing snapshot), add:
```python
                if self._metrics:
                    self._metrics.set_gauge('profile.active_snapshots', len(self._snapshots))
```

In `_update_snapshot()` — at the end of the slow path (after creating new snapshot), add:
```python
            if self._metrics:
                self._metrics.set_gauge('profile.active_snapshots', len(self._snapshots))
```

In `get_profile_snapshot()` — when snapshot found in cache (inside the first `with self._snapshot_lock` block), add:
```python
            if snap is not None:
                if self._metrics:
                    self._metrics.increment('profile.cache_hits')
                return snap
```

In `get_profile_snapshot()` — inside `if user_id not in self._snapshots` (the check-then-set branch), add:
```python
                self._snapshots[user_id] = snap
                if self._metrics:
                    self._metrics.increment('profile.cache_rebuilds')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_metrics_integration.py::TestProfileEngineMetrics tests/test_profile_engine.py tests/test_profile_snapshot.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add user_profile_engine.py tests/test_metrics_integration.py
git commit -m "feat: instrument profile engine with event/cache/snapshot metrics"
```

---

## Chunk 3: API Wiring and Health Enrichment

### Task 5: Wire metrics into API server and enrich status endpoint

**Files:**
- Modify: `emotion_api_server.py`
- Modify: `tests/test_metrics_integration.py`

- [ ] **Step 1: Write tests for classifier and personalization metrics**

Append to `tests/test_metrics_integration.py`:

```python
class TestClassifierMetrics:
    """Test classifier metrics are incremented correctly."""

    def test_classifier_counter_pattern(self):
        """Verify metrics increment logic works in isolation."""
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        # Simulate what the API server would do
        result = {'confidence': 0.85, 'is_fallback': False}
        m.increment('classifier.requests')
        m.increment('classifier.confidence_sum', result.get('confidence', 0))
        if result.get('is_fallback'):
            m.increment('classifier.fallback_count')
        assert m.get_counter('classifier.requests') == 1
        assert m.get_counter('classifier.confidence_sum') == 0.85
        assert m.get_counter('classifier.fallback_count') == 0.0

    def test_fallback_counted(self):
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        result = {'confidence': 1.0, 'is_fallback': True}
        m.increment('classifier.requests')
        if result.get('is_fallback'):
            m.increment('classifier.fallback_count')
        assert m.get_counter('classifier.fallback_count') == 1


class TestPersonalizationMetrics:

    def test_personalization_counters(self):
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        # Simulate skipped
        m.increment('personalization.requests')
        m.increment('personalization.skipped')
        assert m.get_counter('personalization.requests') == 1
        assert m.get_counter('personalization.skipped') == 1

    def test_personalization_swapped(self):
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        m.increment('personalization.requests')
        reason = 'profile_tiebreak: Fear prior 0.38 > Sadness prior 0.21'
        if 'profile_tiebreak' in reason:
            m.increment('personalization.swapped')
        assert m.get_counter('personalization.swapped') == 1


class TestStatusEndpointMetrics:
    """Test that get_all() produces the format expected by /api/emotion/status."""

    def test_metrics_block_structure(self):
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
        m.increment('classifier.requests', 10)
        m.set_gauge('bus.subscriber_count', 3)
        result = m.get_all()
        assert 'server_started_at' in result
        assert 'uptime_seconds' in result
        assert result['counters']['classifier.requests'] == 10
        assert result['gauges']['bus.subscriber_count'] == 3
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_metrics_integration.py -v`
Expected: ALL PASS (these test patterns, not wiring)

- [ ] **Step 3: Wire MetricsCollector into emotion_api_server.py**

Add import near the top:
```python
from metrics_collector import MetricsCollector
```

In `create_app()`, after profile_engine initialization and before the route definitions, add:
```python
    # Initialize metrics collector
    metrics_collector = MetricsCollector()

    # Wire metrics to subsystems
    if profile_engine:
        profile_engine.set_metrics(metrics_collector)
    if profile_engine and profile_engine._event_bus:
        profile_engine._event_bus.set_metrics(metrics_collector)
    if profile_engine and profile_engine._ecosystem_connector:
        profile_engine._ecosystem_connector.set_metrics(metrics_collector)
```

In the `/api/emotion/analyze` endpoint, after `result = model.analyze(text, biometrics)`, add:
```python
            # Classifier metrics
            metrics_collector.increment('classifier.requests')
            metrics_collector.increment('classifier.confidence_sum',
                                       float(result.get('confidence', 0)))
            if result.get('is_fallback'):
                metrics_collector.increment('classifier.fallback_count')
```

After the personalization block (where `apply_profile_personalization` is called), add:
```python
            # Personalization metrics
            if user_id and profile_engine:
                metrics_collector.increment('personalization.requests')
                if result.get('personalized'):
                    reason = result.get('personalization_reason', '')
                    if 'profile_tiebreak' in reason:
                        metrics_collector.increment('personalization.swapped')
                    else:
                        metrics_collector.increment('personalization.consulted')
                else:
                    metrics_collector.increment('personalization.skipped')
```

In the `/api/emotion/status` endpoint, add `metrics` to the response dict:
```python
    @app.route('/api/emotion/status', methods=['GET'])
    def status():
        """Health check endpoint"""
        resp = {
            'status': 'online' if model else 'degraded',
            'model': 'quantara-emotion-gpt',
            'device': model.device if model else 'none',
            'timestamp': datetime.now().isoformat(),
            'version': '2.1.0',
            'taxonomy': '32-emotion / 9-family',
            'external_context': {
                'available': HAS_EXTERNAL_CONTEXT,
                'weather': True,
                'nutrition': bool(os.environ.get('NUTRITIONIX_APP_ID')),
                'sentiment': bool(os.environ.get('NLPCLOUD_API_KEY')),
            }
        }
        if metrics_collector:
            resp['metrics'] = metrics_collector.get_all()
        return jsonify(resp)
```

- [ ] **Step 4: Run all tests to verify nothing breaks**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_metrics_collector.py tests/test_metrics_integration.py tests/test_profile_engine.py tests/test_profile_snapshot.py tests/test_personalization.py tests/test_analyze_enrichment.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add emotion_api_server.py tests/test_metrics_integration.py
git commit -m "feat: wire MetricsCollector into API server, enrich /api/emotion/status

- Classifier metrics: requests, fallback_count, confidence_sum
- Personalization metrics: requests, skipped, consulted, swapped
- Health endpoint returns metrics block with all counters/gauges"
```

---

## Chunk 4: Full Suite Verification

### Task 6: Full test suite and server verification

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/ --tb=short -q`
Expected: 650+ passed, 0 failed

- [ ] **Step 2: Verify server starts**

Run: `cd /Users/bel/quantara-nanoGPT && timeout 10 python emotion_api_server.py 2>&1 || true`
Expected: No ImportError/SyntaxError

- [ ] **Step 3: Commit if any remaining changes**

```bash
git status
# If clean, nothing to commit
```

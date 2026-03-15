# Next-Level Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the profile engine into a bidirectional intelligence hub with an event bus, ecosystem ingestion, intelligence feedback loops, proactive alerts, and WebSocket smart streaming.

**Architecture:** In-process ProfileEventBus with topic-based pub/sub. Sync subscribers for fast in-memory work, async subscribers with worker threads for CPU/IO work. Debounced process() triggers (30s debounce, 20-event threshold, 5min periodic). Services push events via existing `/api/profile/ingest` webhook; intelligence flows back via outbound webhooks.

**Tech Stack:** Python 3, Flask, flask-socketio, SQLite3, scipy, threading, queue, requests, numpy (cosine similarity)

**Spec:** `docs/superpowers/specs/2026-03-15-next-level-integration-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `profile_event_bus.py` | ProfileEventBus pub/sub, TopicMatcher wildcards, sync/async subscriber modes |
| `ecosystem_connector.py` | EcosystemConnector inbound routing + multi-domain publishing, OutboundWebhook retry logic, service registration, dead letter SQLite persistence |
| `alert_engine.py` | AlertEngine bus subscriber, ReactiveDetector (6 pattern types), PredictiveDetector (signature matching, trend extrapolation, temporal patterns), AlertThrottler (4hr cooldown, severity suppression) |
| `intelligence_publisher.py` | IntelligencePublisher orchestrator, TherapyPersonalizer, CoachingPersonalizer, WorkflowPrioritizer, therapy_audit_log table |
| `websocket_router.py` | WebSocketRouter per-connection subscriptions, ScreenSubscriptionMap, BatchBuffer for backgrounded apps |
| `process_scheduler.py` | ProcessScheduler — debounced/count/periodic triggers for process() |
| `user_profile_engine.py` (modify) | Wire event bus, remove set_socketio, publish events after process(), add set_event_bus() |
| `profile_api.py` (modify) | Add service registration endpoint, wire EcosystemConnector to ingest |
| `emotion_api_server.py` (modify) | Initialize bus + all components on startup, register socket.io namespaces |
| `profile_db.py` (modify) | Add dead_letter_events table, therapy_audit_log table |
| `tests/test_event_bus.py` | Bus pub/sub, wildcards, sync/async modes |
| `tests/test_ecosystem_connector.py` | Multi-domain routing, dead letter, service registration |
| `tests/test_alert_engine.py` | Reactive patterns, predictive signatures, throttling |
| `tests/test_intelligence_publisher.py` | Therapy/coaching/workflow personalization, tiered autonomy |
| `tests/test_websocket_router.py` | Screen subscriptions, batching, namespace routing |
| `tests/test_process_scheduler.py` | Debounce, count trigger, periodic |
| `tests/test_next_level_integration.py` | End-to-end: event in → bus → process → intelligence out → WebSocket |

---

## Chunk 1: Event Bus

### Task 1: ProfileEventBus — Core Pub/Sub with Sync/Async Modes

**Files:**
- Create: `profile_event_bus.py`
- Create: `tests/test_event_bus.py`

- [ ] **Step 1: Write failing tests for TopicMatcher**

```python
# tests/test_event_bus.py
"""Tests for ProfileEventBus pub/sub system."""

import pytest
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTopicMatcher:

    def test_exact_match(self):
        from profile_event_bus import TopicMatcher
        assert TopicMatcher.matches('event.emotional', 'event.emotional')

    def test_no_match(self):
        from profile_event_bus import TopicMatcher
        assert not TopicMatcher.matches('event.emotional', 'event.biometric')

    def test_wildcard_match(self):
        from profile_event_bus import TopicMatcher
        assert TopicMatcher.matches('event.*', 'event.emotional')
        assert TopicMatcher.matches('event.*', 'event.biometric')

    def test_wildcard_no_match_different_prefix(self):
        from profile_event_bus import TopicMatcher
        assert not TopicMatcher.matches('event.*', 'profile.updated')

    def test_double_wildcard(self):
        from profile_event_bus import TopicMatcher
        assert TopicMatcher.matches('profile.domain.*', 'profile.domain.emotional')

    def test_full_wildcard(self):
        from profile_event_bus import TopicMatcher
        assert TopicMatcher.matches('*', 'event.emotional')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_event_bus.py::TestTopicMatcher -v`
Expected: FAIL

- [ ] **Step 3: Write failing tests for bus publish/subscribe**

```python
class TestProfileEventBus:

    @pytest.fixture
    def bus(self):
        from profile_event_bus import ProfileEventBus
        b = ProfileEventBus()
        yield b
        b.shutdown()

    def test_sync_subscriber_receives_event(self, bus):
        received = []
        bus.subscribe('event.emotional', lambda topic, payload: received.append(payload), mode='sync')
        bus.publish('event.emotional', {'emotion': 'joy'})
        assert len(received) == 1
        assert received[0]['emotion'] == 'joy'

    def test_wildcard_subscriber(self, bus):
        received = []
        bus.subscribe('event.*', lambda t, p: received.append(t), mode='sync')
        bus.publish('event.emotional', {})
        bus.publish('event.biometric', {})
        bus.publish('profile.updated', {})
        assert len(received) == 2

    def test_async_subscriber_receives_event(self, bus):
        received = []
        event = threading.Event()
        def cb(topic, payload):
            received.append(payload)
            event.set()
        bus.subscribe('event.emotional', cb, mode='async')
        bus.publish('event.emotional', {'emotion': 'joy'})
        event.wait(timeout=2)
        assert len(received) == 1

    def test_unsubscribe(self, bus):
        received = []
        sub_id = bus.subscribe('event.*', lambda t, p: received.append(p), mode='sync')
        bus.publish('event.emotional', {})
        assert len(received) == 1
        bus.unsubscribe(sub_id)
        bus.publish('event.emotional', {})
        assert len(received) == 1

    def test_subscriber_exception_doesnt_break_others(self, bus):
        received = []
        bus.subscribe('event.emotional', lambda t, p: 1/0, mode='sync')  # Raises
        bus.subscribe('event.emotional', lambda t, p: received.append(p), mode='sync')
        bus.publish('event.emotional', {'ok': True})
        assert len(received) == 1

    def test_multiple_subscribers_same_topic(self, bus):
        counts = {'a': 0, 'b': 0}
        bus.subscribe('event.emotional', lambda t, p: counts.update(a=counts['a']+1), mode='sync')
        bus.subscribe('event.emotional', lambda t, p: counts.update(b=counts['b']+1), mode='sync')
        bus.publish('event.emotional', {})
        assert counts['a'] == 1
        assert counts['b'] == 1
```

- [ ] **Step 4: Implement ProfileEventBus**

```python
# profile_event_bus.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Profile Event Bus
===============================================================================
In-process topic-based publish/subscribe dispatcher for the profile engine.
Supports sync (in-thread) and async (worker thread) subscriber modes.
Wildcard topic matching via glob-style patterns.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- User Profile Engine
- Alert Engine
- Intelligence Publisher
- WebSocket Router
===============================================================================
"""

import logging
import threading
import queue
import uuid
import fnmatch
from typing import Callable, Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class TopicMatcher:
    """Glob-style topic pattern matching."""

    @staticmethod
    def matches(pattern: str, topic: str) -> bool:
        return fnmatch.fnmatch(topic, pattern)


class ProfileEventBus:
    """In-process topic-based pub/sub dispatcher.

    Subscribers register as sync (called in publisher thread, must be <1ms)
    or async (enqueued to a worker thread).
    """

    def __init__(self):
        self._subscribers: Dict[str, dict] = {}  # sub_id -> {pattern, callback, mode}
        self._lock = threading.Lock()
        self._async_workers: Dict[str, threading.Thread] = {}
        self._async_queues: Dict[str, queue.Queue] = {}
        self._running = True

    def subscribe(self, topic_pattern: str, callback: Callable,
                  mode: str = 'sync') -> str:
        """Register a subscriber.

        Args:
            topic_pattern: Topic pattern with optional wildcards (e.g., 'event.*')
            callback: Function(topic, payload) to call on match
            mode: 'sync' (in-thread) or 'async' (worker thread)

        Returns:
            Subscription ID for unsubscribing
        """
        sub_id = str(uuid.uuid4())
        sub = {
            'pattern': topic_pattern,
            'callback': callback,
            'mode': mode,
        }

        if mode == 'async':
            q = queue.Queue()
            self._async_queues[sub_id] = q
            worker = threading.Thread(
                target=self._async_worker_loop,
                args=(sub_id, callback, q),
                daemon=True,
                name=f'bus-async-{sub_id[:8]}'
            )
            self._async_workers[sub_id] = worker
            worker.start()

        with self._lock:
            self._subscribers[sub_id] = sub
        return sub_id

    def unsubscribe(self, sub_id: str):
        """Remove a subscription."""
        with self._lock:
            self._subscribers.pop(sub_id, None)
        if sub_id in self._async_queues:
            self._async_queues[sub_id].put(None)  # Poison pill
            self._async_workers.pop(sub_id, None)
            self._async_queues.pop(sub_id, None)

    def publish(self, topic: str, payload: Any):
        """Publish an event to all matching subscribers.

        Sync subscribers are called directly. Async subscribers have
        the event enqueued to their worker thread.
        """
        with self._lock:
            subscribers = list(self._subscribers.items())

        for sub_id, sub in subscribers:
            if not TopicMatcher.matches(sub['pattern'], topic):
                continue
            if sub['mode'] == 'sync':
                try:
                    sub['callback'](topic, payload)
                except Exception as e:
                    logger.error("Sync subscriber %s failed on %s: %s", sub_id[:8], topic, e)
            elif sub['mode'] == 'async':
                q = self._async_queues.get(sub_id)
                if q:
                    q.put((topic, payload))

    def _async_worker_loop(self, sub_id: str, callback: Callable, q: queue.Queue):
        """Worker thread for async subscribers."""
        while self._running:
            try:
                item = q.get(timeout=1.0)
                if item is None:
                    break
                topic, payload = item
                try:
                    callback(topic, payload)
                except Exception as e:
                    logger.error("Async subscriber %s failed on %s: %s", sub_id[:8], topic, e)
            except queue.Empty:
                continue

    def shutdown(self):
        """Stop all async workers."""
        self._running = False
        for q in self._async_queues.values():
            q.put(None)
        for worker in self._async_workers.values():
            worker.join(timeout=2.0)
        self._async_workers.clear()
        self._async_queues.clear()
```

- [ ] **Step 5: Run all bus tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_event_bus.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add profile_event_bus.py tests/test_event_bus.py
git commit -m "feat(integration): add ProfileEventBus with sync/async pub/sub"
```

---

## Chunk 2: Process Scheduler

### Task 2: ProcessScheduler — Debounced/Count/Periodic Triggers

**Files:**
- Create: `process_scheduler.py`
- Create: `tests/test_process_scheduler.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_process_scheduler.py
"""Tests for ProcessScheduler — debounced, count, periodic triggers."""

import pytest
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestProcessScheduler:

    @pytest.fixture
    def processed_users(self):
        return []

    def test_debounce_triggers_after_delay(self, processed_users):
        from process_scheduler import ProcessScheduler
        scheduler = ProcessScheduler(
            process_fn=lambda uid: processed_users.append(uid),
            debounce_seconds=0.2, count_threshold=100, periodic_seconds=600
        )
        scheduler.start()
        scheduler.notify_event('user1')
        time.sleep(0.1)
        assert 'user1' not in processed_users  # Not yet
        time.sleep(0.3)
        assert 'user1' in processed_users
        scheduler.stop()

    def test_debounce_resets_on_new_event(self, processed_users):
        from process_scheduler import ProcessScheduler
        scheduler = ProcessScheduler(
            process_fn=lambda uid: processed_users.append(uid),
            debounce_seconds=0.3, count_threshold=100, periodic_seconds=600
        )
        scheduler.start()
        scheduler.notify_event('user1')
        time.sleep(0.1)
        scheduler.notify_event('user1')  # Reset debounce
        time.sleep(0.1)
        assert 'user1' not in processed_users  # Still waiting
        time.sleep(0.4)
        assert processed_users.count('user1') == 1  # Only once
        scheduler.stop()

    def test_count_threshold_triggers_immediately(self, processed_users):
        from process_scheduler import ProcessScheduler
        scheduler = ProcessScheduler(
            process_fn=lambda uid: processed_users.append(uid),
            debounce_seconds=60, count_threshold=5, periodic_seconds=600
        )
        scheduler.start()
        for _ in range(5):
            scheduler.notify_event('user1')
        time.sleep(0.2)
        assert 'user1' in processed_users
        scheduler.stop()

    def test_different_users_independent(self, processed_users):
        from process_scheduler import ProcessScheduler
        scheduler = ProcessScheduler(
            process_fn=lambda uid: processed_users.append(uid),
            debounce_seconds=0.2, count_threshold=100, periodic_seconds=600
        )
        scheduler.start()
        scheduler.notify_event('user1')
        scheduler.notify_event('user2')
        time.sleep(0.4)
        assert 'user1' in processed_users
        assert 'user2' in processed_users
        scheduler.stop()
```

- [ ] **Step 2: Implement ProcessScheduler**

```python
# process_scheduler.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Process Scheduler
===============================================================================
Debounced, count-triggered, and periodic scheduling for profile processing.
Prevents processing on every inbound event while maintaining responsiveness.

Integrates with:
- Neural Workflow AI Engine
- User Profile Engine
- Profile Event Bus
===============================================================================
"""

import time
import logging
import threading
from typing import Callable, Dict, Set

logger = logging.getLogger(__name__)


class ProcessScheduler:
    """Schedules process() calls with debounce, count, and periodic triggers.

    - Debounce: Wait N seconds after last event before processing
    - Count: Process immediately when N events accumulate for a user
    - Periodic: Process all active users every N seconds regardless
    """

    def __init__(self, process_fn: Callable[[str], None],
                 debounce_seconds: float = 30.0,
                 count_threshold: int = 20,
                 periodic_seconds: float = 300.0):
        self.process_fn = process_fn
        self.debounce_seconds = debounce_seconds
        self.count_threshold = count_threshold
        self.periodic_seconds = periodic_seconds

        self._pending: Dict[str, dict] = {}  # user_id -> {count, last_event_time}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._debounce_thread = None
        self._periodic_thread = None

    def start(self):
        """Start scheduler threads."""
        self._stop_event.clear()
        self._debounce_thread = threading.Thread(
            target=self._debounce_loop, daemon=True, name='ProcessScheduler-debounce'
        )
        self._periodic_thread = threading.Thread(
            target=self._periodic_loop, daemon=True, name='ProcessScheduler-periodic'
        )
        self._debounce_thread.start()
        self._periodic_thread.start()

    def stop(self):
        """Stop scheduler threads."""
        self._stop_event.set()
        if self._debounce_thread:
            self._debounce_thread.join(timeout=3)
        if self._periodic_thread:
            self._periodic_thread.join(timeout=3)

    def notify_event(self, user_id: str):
        """Notify that a new event arrived for a user."""
        with self._lock:
            if user_id not in self._pending:
                self._pending[user_id] = {'count': 0, 'last_event_time': 0}
            self._pending[user_id]['count'] += 1
            self._pending[user_id]['last_event_time'] = time.time()

            # Count threshold — process immediately
            if self._pending[user_id]['count'] >= self.count_threshold:
                self._process_user(user_id)

    def _debounce_loop(self):
        """Check for users whose debounce timer has expired."""
        while not self._stop_event.is_set():
            now = time.time()
            to_process = []
            with self._lock:
                for user_id, info in list(self._pending.items()):
                    elapsed = now - info['last_event_time']
                    if elapsed >= self.debounce_seconds and info['count'] > 0:
                        to_process.append(user_id)
            for user_id in to_process:
                self._process_user(user_id)
            self._stop_event.wait(timeout=0.1)

    def _periodic_loop(self):
        """Process all active users periodically."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.periodic_seconds)
            if self._stop_event.is_set():
                break
            with self._lock:
                users = list(self._pending.keys())
            for user_id in users:
                self._process_user(user_id)

    def _process_user(self, user_id: str):
        """Run process() for a user and reset their counter."""
        with self._lock:
            if user_id in self._pending:
                self._pending[user_id]['count'] = 0
        try:
            self.process_fn(user_id)
        except Exception as e:
            logger.error("ProcessScheduler failed for %s: %s", user_id, e)
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_process_scheduler.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add process_scheduler.py tests/test_process_scheduler.py
git commit -m "feat(integration): add ProcessScheduler with debounce/count/periodic triggers"
```

---

## Chunk 3: Ecosystem Connector

### Task 3: EcosystemConnector — Multi-Domain Routing, Dead Letter, Service Registration

**Files:**
- Create: `ecosystem_connector.py`
- Create: `tests/test_ecosystem_connector.py`
- Modify: `profile_db.py` — add dead_letter_events table

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ecosystem_connector.py
"""Tests for EcosystemConnector — routing, dead letter, service registration."""

import pytest
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEcosystemConnector:

    @pytest.fixture
    def bus(self):
        from profile_event_bus import ProfileEventBus
        b = ProfileEventBus()
        yield b
        b.shutdown()

    @pytest.fixture
    def connector(self, bus, tmp_path):
        from ecosystem_connector import EcosystemConnector
        from profile_db import ProfileDB
        db = ProfileDB(str(tmp_path / 'test.db'))
        c = EcosystemConnector(bus, db)
        yield c
        db.close()

    def test_single_domain_event_publishes_to_bus(self, connector, bus):
        received = []
        bus.subscribe('event.emotional', lambda t, p: received.append(p), mode='sync')
        connector.route_inbound({
            'user_id': 'u1', 'domain': 'emotional',
            'event_type': 'emotion_classified', 'payload': {'emotion': 'joy'},
            'source': 'backend'
        })
        assert len(received) == 1

    def test_multi_domain_event_publishes_multiple(self, connector, bus):
        received = []
        bus.subscribe('event.*', lambda t, p: received.append(t), mode='sync')
        connector.route_inbound({
            'user_id': 'u1', 'domain': 'biometric,temporal',
            'event_type': 'health_reading', 'payload': {'hr': 72},
            'source': 'frontend'
        })
        assert 'event.biometric' in received
        assert 'event.temporal' in received

    def test_dead_letter_on_failure(self, connector):
        # Force a failure by passing invalid data
        connector.route_inbound({'invalid': True})
        dead = connector.get_dead_letter_count()
        assert dead >= 1

    def test_service_registration(self, connector):
        connector.register_service('backend', 'http://localhost:3001/api/intelligence/notify')
        services = connector.get_registered_services()
        assert 'backend' in services

    def test_service_deregistration_after_failures(self, connector):
        connector.register_service('backend', 'http://localhost:3001/api/intelligence/notify')
        for _ in range(5):
            connector.record_delivery_failure('backend')
        services = connector.get_registered_services()
        assert 'backend' not in services
```

- [ ] **Step 2: Add dead_letter_events table to profile_db.py**

In `ProfileDB._init_db()`, add to the CREATE TABLE block:

```python
CREATE TABLE IF NOT EXISTS dead_letter_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    payload TEXT NOT NULL,
    error TEXT,
    retries INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS therapy_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    old_weight REAL,
    new_weight REAL,
    reason TEXT,
    timestamp REAL NOT NULL
);
```

- [ ] **Step 3: Implement EcosystemConnector**

```python
# ecosystem_connector.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Ecosystem Connector
===============================================================================
Manages inbound event routing (multi-domain publishing to bus) and outbound
intelligence delivery (webhook retry with degradation). Persists dead letter
events to SQLite.

Integrates with:
- Neural Workflow AI Engine
- Profile Event Bus
- All ecosystem services (Backend, Frontend, Master)
===============================================================================
"""

import os
import json
import time
import logging
import threading
import requests
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


MULTI_DOMAIN_MAP = {
    'biometric,temporal': ['biometric', 'temporal'],
    'biometric,emotional': ['biometric', 'emotional'],
    'linguistic,social,aspirational': ['linguistic', 'social', 'aspirational'],
    'emotional,behavioral': ['emotional', 'behavioral'],
    'social,temporal': ['social', 'temporal'],
    'behavioral,cognitive': ['behavioral', 'cognitive'],
}


class EcosystemConnector:
    """Routes inbound events to bus topics and delivers outbound intelligence."""

    def __init__(self, bus, db):
        self.bus = bus
        self.db = db
        self._services: Dict[str, dict] = {}  # name -> {url, failures}
        self._lock = threading.Lock()

        allowed_hosts = os.environ.get('PROFILE_ALLOWED_WEBHOOK_HOSTS', '')
        self._allowed_hosts = set(h.strip() for h in allowed_hosts.split(',') if h.strip()) if allowed_hosts else None

    def route_inbound(self, event_data: dict):
        """Route an inbound event to the appropriate bus topics.

        Multi-domain events (domain='biometric,temporal') publish to each topic separately.
        """
        try:
            user_id = event_data.get('user_id')
            domain = event_data.get('domain', '')
            event_type = event_data.get('event_type', '')
            payload = event_data.get('payload', {})
            source = event_data.get('source', 'unknown')
            confidence = event_data.get('confidence')

            if not user_id or not domain:
                self._store_dead_letter(event_data, 'Missing user_id or domain')
                return

            # Resolve multi-domain
            domains = MULTI_DOMAIN_MAP.get(domain, [domain])

            for d in domains:
                topic = f'event.{d}'
                self.bus.publish(topic, {
                    'user_id': user_id,
                    'domain': d,
                    'event_type': event_type,
                    'payload': payload,
                    'source': source,
                    'confidence': confidence,
                    'timestamp': time.time(),
                })

        except Exception as e:
            logger.error("Failed to route inbound event: %s", e)
            self._store_dead_letter(event_data, str(e))

    def deliver_intelligence(self, intelligence_type: str, payload: dict):
        """Deliver intelligence to all registered services."""
        payload['schema_version'] = 1

        with self._lock:
            services = list(self._services.items())

        for name, info in services:
            self._deliver_to_service(name, info['url'], intelligence_type, payload)

    def _deliver_to_service(self, name: str, url: str, intelligence_type: str, payload: dict):
        """Deliver with 3 retries and exponential backoff."""
        delays = [1, 2, 4]
        for attempt, delay in enumerate(delays):
            try:
                resp = requests.post(url, json={
                    'intelligence_type': intelligence_type,
                    **payload,
                }, timeout=5)
                if resp.status_code < 400:
                    self._reset_failures(name)
                    return
            except requests.RequestException:
                pass
            if attempt < len(delays) - 1:
                time.sleep(delay)

        self.record_delivery_failure(name)

    def register_service(self, name: str, url: str) -> bool:
        """Register a service for intelligence delivery."""
        if self._allowed_hosts:
            from urllib.parse import urlparse
            host = urlparse(url).hostname
            if host not in self._allowed_hosts:
                logger.warning("Rejected service registration for %s: host %s not in allowlist", name, host)
                return False

        with self._lock:
            self._services[name] = {'url': url, 'failures': 0}
        logger.info("Registered service %s at %s", name, url)
        return True

    def get_registered_services(self) -> Dict[str, str]:
        with self._lock:
            return {name: info['url'] for name, info in self._services.items()}

    def record_delivery_failure(self, name: str):
        with self._lock:
            if name in self._services:
                self._services[name]['failures'] += 1
                if self._services[name]['failures'] >= 5:
                    logger.warning("Deregistering service %s after 5 consecutive failures", name)
                    del self._services[name]

    def _reset_failures(self, name: str):
        with self._lock:
            if name in self._services:
                self._services[name]['failures'] = 0

    def _store_dead_letter(self, event_data: dict, error: str):
        try:
            self.db._enqueue_write(
                "INSERT INTO dead_letter_events (timestamp, payload, error, retries) VALUES (?, ?, ?, 0)",
                (time.time(), json.dumps(event_data), error),
                wait=False
            )
        except Exception as e:
            logger.error("Failed to store dead letter: %s", e)

    def get_dead_letter_count(self) -> int:
        conn = self.db._read_conn()
        try:
            row = conn.execute("SELECT COUNT(*) FROM dead_letter_events").fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
        finally:
            conn.close()

    def replay_dead_letters(self):
        """Replay pending dead letter events."""
        conn = self.db._read_conn()
        rows = conn.execute("SELECT id, payload FROM dead_letter_events WHERE retries < 3 ORDER BY id").fetchall()
        conn.close()
        for row in rows:
            try:
                event_data = json.loads(row[1])
                self.route_inbound(event_data)
                self.db._enqueue_write("DELETE FROM dead_letter_events WHERE id = ?", (row[0],), wait=False)
            except Exception:
                self.db._enqueue_write(
                    "UPDATE dead_letter_events SET retries = retries + 1 WHERE id = ?",
                    (row[0],), wait=False
                )
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_ecosystem_connector.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add ecosystem_connector.py tests/test_ecosystem_connector.py profile_db.py
git commit -m "feat(integration): add EcosystemConnector with multi-domain routing and dead letter"
```

---

## Chunk 4: Alert Engine

### Task 4: AlertEngine — Reactive + Predictive + Throttling

**Files:**
- Create: `alert_engine.py`
- Create: `tests/test_alert_engine.py`

- [ ] **Step 1: Write failing tests for ReactiveDetector**

```python
# tests/test_alert_engine.py
"""Tests for AlertEngine — reactive patterns, predictive signatures, throttling."""

import pytest
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestReactiveDetector:

    @pytest.fixture
    def detector(self):
        from alert_engine import ReactiveDetector
        return ReactiveDetector()

    def test_emotional_spiral_detected(self, detector):
        now = time.time()
        events = [
            {'domain': 'emotional', 'payload': {'family': 'Sadness'}, 'timestamp': now - 60 * i}
            for i in range(5)
        ]
        alerts = detector.check(events, 'user1')
        types = [a['alert_type'] for a in alerts]
        assert 'emotional_spiral' in types

    def test_no_spiral_with_positive_emotions(self, detector):
        now = time.time()
        events = [
            {'domain': 'emotional', 'payload': {'family': 'Joy'}, 'timestamp': now - 60 * i}
            for i in range(5)
        ]
        alerts = detector.check(events, 'user1')
        types = [a['alert_type'] for a in alerts]
        assert 'emotional_spiral' not in types

    def test_rapid_cycling_detected(self, detector):
        now = time.time()
        families = ['Joy', 'Sadness', 'Anger', 'Fear', 'Joy', 'Calm', 'Sadness', 'Anger', 'Joy']
        events = [
            {'domain': 'emotional', 'payload': {'family': f}, 'timestamp': now - 5 * i}
            for i, f in enumerate(families)
        ]
        alerts = detector.check(events, 'user1')
        types = [a['alert_type'] for a in alerts]
        assert 'rapid_cycling' in types

    def test_sustained_stress_detected(self, detector):
        now = time.time()
        events = [
            {'domain': 'biometric', 'payload': {'stress_ratio': 0.6}, 'timestamp': now - 60 * i}
            for i in range(35)
        ]
        alerts = detector.check(events, 'user1')
        types = [a['alert_type'] for a in alerts]
        assert 'sustained_stress' in types

    def test_recovery_detected(self, detector):
        now = time.time()
        events = [
            {'domain': 'emotional', 'payload': {'family': 'Sadness'}, 'timestamp': now - 120},
            {'domain': 'emotional', 'payload': {'family': 'Sadness'}, 'timestamp': now - 90},
            {'domain': 'emotional', 'payload': {'family': 'Sadness'}, 'timestamp': now - 60},
            {'domain': 'emotional', 'payload': {'family': 'Joy'}, 'timestamp': now - 30},
            {'domain': 'emotional', 'payload': {'family': 'Joy'}, 'timestamp': now},
        ]
        alerts = detector.check(events, 'user1')
        types = [a['alert_type'] for a in alerts]
        assert 'recovery_detected' in types


class TestAlertThrottler:

    @pytest.fixture
    def throttler(self):
        from alert_engine import AlertThrottler
        return AlertThrottler(cooldown_seconds=1)  # 1s for testing

    def test_first_alert_passes(self, throttler):
        assert throttler.should_fire('user1', 'emotional_spiral', 'reactive', 'high')

    def test_same_alert_throttled(self, throttler):
        throttler.should_fire('user1', 'emotional_spiral', 'reactive', 'high')
        assert not throttler.should_fire('user1', 'emotional_spiral', 'reactive', 'high')

    def test_different_method_not_throttled(self, throttler):
        throttler.should_fire('user1', 'emotional_spiral', 'reactive', 'high')
        assert throttler.should_fire('user1', 'emotional_spiral', 'predictive', 'high')

    def test_low_suppressed_by_active_high(self, throttler):
        throttler.should_fire('user1', 'emotional_spiral', 'reactive', 'high')
        assert not throttler.should_fire('user1', 'engagement_drop', 'reactive', 'low')

    def test_cooldown_expires(self, throttler):
        throttler.should_fire('user1', 'emotional_spiral', 'reactive', 'high')
        time.sleep(1.2)
        assert throttler.should_fire('user1', 'emotional_spiral', 'reactive', 'high')

    def test_stage1_only_positive(self, throttler):
        assert not throttler.should_fire('user1', 'emotional_spiral', 'reactive', 'high', user_stage=1)
        assert throttler.should_fire('user1', 'recovery_detected', 'reactive', 'positive', user_stage=1)
```

- [ ] **Step 2: Implement alert_engine.py**

```python
# alert_engine.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Alert Engine
===============================================================================
Reactive (pattern-based) and predictive (fingerprint-based) alert system.
Subscribes to event bus, detects concerning patterns, and publishes alerts.
Alert fatigue prevention via throttling.

Integrates with:
- Neural Workflow AI Engine
- Profile Event Bus
- User Profile Engine
- WebSocket Router
===============================================================================
"""

import time
import json
import math
import logging
import threading
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

NEGATIVE_FAMILIES = {'Sadness', 'Anger', 'Fear', 'Self-Conscious'}
POSITIVE_FAMILIES = {'Joy', 'Love', 'Calm'}


class ReactiveDetector:
    """Pattern-based alert detection from recent events."""

    def check(self, events: List[Dict], user_id: str) -> List[Dict]:
        """Check events for known concerning patterns. Returns list of alert dicts."""
        alerts = []
        now = time.time()

        emotional_events = [e for e in events if e.get('domain') == 'emotional']
        biometric_events = [e for e in events if e.get('domain') == 'biometric']

        # Emotional Spiral: 5+ negative emotions in 2 hours
        recent_emotional = [e for e in emotional_events if now - e.get('timestamp', 0) < 7200]
        negative_count = sum(
            1 for e in recent_emotional
            if e.get('payload', {}).get('family') in NEGATIVE_FAMILIES
        )
        if negative_count >= 5:
            alerts.append(self._make_alert(
                user_id, 'emotional_spiral', 'reactive', 'high',
                'Sustained negative emotional pattern detected.',
                'Try a breathing exercise or reach out to someone you trust.'
            ))

        # Rapid Cycling: 8+ emotion changes in 1 hour
        recent_1h = [e for e in emotional_events if now - e.get('timestamp', 0) < 3600]
        if len(recent_1h) >= 2:
            changes = sum(
                1 for i in range(1, len(recent_1h))
                if recent_1h[i].get('payload', {}).get('family') != recent_1h[i-1].get('payload', {}).get('family')
            )
            if changes >= 8:
                alerts.append(self._make_alert(
                    user_id, 'rapid_cycling', 'reactive', 'medium',
                    'Rapid emotional changes detected.',
                    'Take a moment to ground yourself with a mindfulness exercise.'
                ))

        # Sustained Stress: stress_ratio > 0.5 for 30+ minutes
        recent_bio = [e for e in biometric_events if now - e.get('timestamp', 0) < 3600]
        stress_readings = [
            e for e in recent_bio
            if e.get('payload', {}).get('stress_ratio', 0) > 0.5
        ]
        if len(stress_readings) >= 30:
            earliest = min(e.get('timestamp', now) for e in stress_readings)
            if now - earliest >= 1800:
                alerts.append(self._make_alert(
                    user_id, 'sustained_stress', 'reactive', 'high',
                    'Elevated stress levels sustained for over 30 minutes.',
                    'Consider a calming activity or short walk.'
                ))

        # Recovery: sustained negative → positive transition
        if len(recent_emotional) >= 4:
            families = [e.get('payload', {}).get('family', '') for e in recent_emotional]
            # Check if recent 2 are positive and preceding 2 are negative
            if len(families) >= 4:
                recent_2 = families[-2:]
                prev_2 = families[-4:-2]
                if all(f in NEGATIVE_FAMILIES for f in prev_2) and all(f in POSITIVE_FAMILIES for f in recent_2):
                    alerts.append(self._make_alert(
                        user_id, 'recovery_detected', 'reactive', 'positive',
                        'Positive emotional recovery detected!',
                        'Great progress — keep doing what works for you.'
                    ))

        return alerts

    def _make_alert(self, user_id, alert_type, method, severity, message, action, confidence=0.8):
        return {
            'user_id': user_id,
            'alert_type': alert_type,
            'detection_method': method,
            'severity': severity,
            'message': message,
            'recommended_action': action,
            'confidence': confidence,
            'timestamp': time.time(),
        }


class PredictiveDetector:
    """Fingerprint-based predictive alerts using signature matching."""

    def __init__(self):
        self._signatures: Dict[str, List[List[float]]] = defaultdict(list)  # user_id -> [signature_vectors]

    def store_signature(self, user_id: str, fingerprint: dict):
        """Store a signature from a reactive alert event."""
        vector = self._extract_vector(fingerprint)
        if vector is None:
            return
        sigs = self._signatures[user_id]
        sigs.append(vector)
        if len(sigs) > 3:
            sigs.pop(0)  # Sliding window of 3

    def check(self, user_id: str, fingerprint: dict) -> List[Dict]:
        """Check current fingerprint against stored signatures."""
        alerts = []
        sigs = self._signatures.get(user_id, [])
        if len(sigs) < 3:
            return []  # Cold start — need 3+ signatures

        current = self._extract_vector(fingerprint)
        if current is None:
            return []

        for sig in sigs:
            similarity = self._cosine_similarity(current, sig)
            if similarity > 0.8:
                alerts.append({
                    'user_id': user_id,
                    'alert_type': 'predicted_pattern_match',
                    'detection_method': 'predictive',
                    'severity': 'medium',
                    'message': 'Current patterns match a previously concerning state.',
                    'recommended_action': 'Consider proactive self-care.',
                    'confidence': round(similarity, 2),
                    'timestamp': time.time(),
                })
                break  # One match is enough

        return alerts

    def _extract_vector(self, fingerprint: dict) -> Optional[List[float]]:
        """Extract 6-dimension signature vector from fingerprint."""
        try:
            emotional = fingerprint.get('emotional', {}).get('metrics', {})
            biometric = fingerprint.get('biometric', {}).get('metrics', {})
            behavioral = fingerprint.get('behavioral', {}).get('metrics', {})
            temporal = fingerprint.get('temporal', {}).get('metrics', {})

            # Normalize resting_hr to 0-1 range (40-120 BPM)
            hr = biometric.get('resting_hr', 70)
            hr_normalized = max(0, min(1, (hr - 40) / 80))

            # Peak hours variance
            peak_hours = temporal.get('peak_hours', [12])
            if len(peak_hours) >= 2:
                mean_h = sum(peak_hours) / len(peak_hours)
                variance = sum((h - mean_h)**2 for h in peak_hours) / len(peak_hours)
                peak_var = min(1, variance / 50)
            else:
                peak_var = 0.0

            return [
                emotional.get('volatility', 0),
                1.0 if emotional.get('dominant_family') in NEGATIVE_FAMILIES else 0.0,
                biometric.get('stress_ratio', 0),
                hr_normalized,
                behavioral.get('completion_rate', 0.5),
                peak_var,
            ]
        except Exception:
            return None

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x**2 for x in a)) or 1
        mag_b = math.sqrt(sum(x**2 for x in b)) or 1
        return dot / (mag_a * mag_b)


class AlertThrottler:
    """Prevents alert fatigue with cooldowns and severity suppression."""

    def __init__(self, cooldown_seconds: float = 14400):  # 4 hours default
        self.cooldown_seconds = cooldown_seconds
        self._last_fired: Dict[str, float] = {}  # (user, type, method) -> timestamp
        self._active_high: Dict[str, float] = {}  # user -> timestamp of last high severity

    def should_fire(self, user_id: str, alert_type: str, method: str,
                    severity: str, user_stage: int = 2) -> bool:
        """Check if this alert should fire given throttling rules."""
        now = time.time()

        # Stage 1: only positive alerts
        if user_stage <= 1 and severity != 'positive':
            return False

        # Low suppressed by active high
        if severity == 'low':
            last_high = self._active_high.get(user_id, 0)
            if now - last_high < self.cooldown_seconds:
                return False

        # Cooldown per type+method
        key = f"{user_id}:{alert_type}:{method}"
        last = self._last_fired.get(key, 0)
        if now - last < self.cooldown_seconds:
            return False

        # Fire it
        self._last_fired[key] = now
        if severity == 'high':
            self._active_high[user_id] = now
        return True


class AlertEngine:
    """Main alert engine — subscribes to bus, runs detectors, publishes alerts."""

    def __init__(self, bus, db=None, cooldown_seconds: float = 14400):
        self.bus = bus
        self.db = db
        self.reactive = ReactiveDetector()
        self.predictive = PredictiveDetector()
        self.throttler = AlertThrottler(cooldown_seconds)
        self._recent_events: Dict[str, List[Dict]] = defaultdict(list)  # user_id -> events
        self._lock = threading.Lock()
        self._max_recent = 500  # Max recent events per user

        # Subscribe to all events
        bus.subscribe('event.*', self._on_event, mode='async')
        bus.subscribe('alert.reactive', self._on_reactive_alert, mode='sync')

    def _on_event(self, topic: str, payload: dict):
        """Buffer recent events and check for patterns."""
        user_id = payload.get('user_id', 'default')
        with self._lock:
            self._recent_events[user_id].append(payload)
            if len(self._recent_events[user_id]) > self._max_recent:
                self._recent_events[user_id] = self._recent_events[user_id][-self._max_recent:]

        # Run reactive detection
        with self._lock:
            events = list(self._recent_events[user_id])
        alerts = self.reactive.check(events, user_id)
        for alert in alerts:
            if self.throttler.should_fire(
                user_id, alert['alert_type'], alert['detection_method'], alert['severity']
            ):
                self.bus.publish('alert.reactive', alert)

    def _on_reactive_alert(self, topic: str, payload: dict):
        """When a reactive alert fires, store signature for predictive use."""
        user_id = payload.get('user_id')
        if user_id and self.db:
            try:
                profile = self.db.get_or_create_profile(user_id)
                fingerprint = json.loads(profile.get('fingerprint_json', '{}')) if profile.get('fingerprint_json') else {}
                if fingerprint:
                    self.predictive.store_signature(user_id, fingerprint)
            except Exception as e:
                logger.warning("Failed to store predictive signature: %s", e)

    def check_predictive(self, user_id: str, fingerprint: dict) -> List[Dict]:
        """Run predictive checks (called from process cycle)."""
        alerts = self.predictive.check(user_id, fingerprint)
        fired = []
        for alert in alerts:
            if alert['confidence'] > 0.7 and self.throttler.should_fire(
                user_id, alert['alert_type'], 'predictive', alert['severity']
            ):
                self.bus.publish('alert.predictive', alert)
                fired.append(alert)
        return fired
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_alert_engine.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add alert_engine.py tests/test_alert_engine.py
git commit -m "feat(integration): add AlertEngine with reactive/predictive detection and throttling"
```

---

## Chunk 5: Intelligence Publisher

### Task 5: IntelligencePublisher — Therapy, Coaching, Workflow Personalization

**Files:**
- Create: `intelligence_publisher.py`
- Create: `tests/test_intelligence_publisher.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_intelligence_publisher.py
"""Tests for IntelligencePublisher — personalization computation and delivery."""

import pytest
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTherapyPersonalizer:

    def test_computes_technique_scores(self):
        from intelligence_publisher import TherapyPersonalizer
        p = TherapyPersonalizer()
        fingerprint = {
            'behavioral': {'metrics': {
                'completion_rate': 0.8,
                'intervention_response_rate': 0.7,
            }},
            'cognitive': {'metrics': {
                'technique_usage': {'breathing': 10, 'journaling': 5},
                'technique_retention_rate': 0.85,
            }},
        }
        result = p.compute(fingerprint, stage=3)
        assert 'technique_scores' in result
        assert result['mode'] == 'active'

    def test_advisory_mode_for_low_stages(self):
        from intelligence_publisher import TherapyPersonalizer
        p = TherapyPersonalizer()
        result = p.compute({'behavioral': {'metrics': {}}, 'cognitive': {'metrics': {}}}, stage=1)
        assert result['mode'] == 'advisory'


class TestCoachingPersonalizer:

    def test_computes_coaching_profile(self):
        from intelligence_publisher import CoachingPersonalizer
        p = CoachingPersonalizer()
        fingerprint = {
            'linguistic': {'metrics': {
                'avg_sentence_length': 15.0,
                'vocabulary_complexity': 0.7,
                'dominant_tone': 'reflective',
            }},
            'social': {'metrics': {'daily_interaction_rate': 5.0}},
            'aspirational': {'metrics': {'core_values': ['growth', 'health']}},
        }
        result = p.compute(fingerprint, stage=4)
        assert 'preferred_tone' in result
        assert 'vocabulary_level' in result
        assert result['mode'] == 'active'


class TestWorkflowPrioritizer:

    def test_computes_readiness_score(self):
        from intelligence_publisher import WorkflowPrioritizer
        p = WorkflowPrioritizer()
        fingerprint = {
            'emotional': {'metrics': {
                'volatility': 0.3,
                'dominant_family': 'Joy',
            }},
            'behavioral': {'metrics': {
                'completion_rate': 0.8,
                'engagement_streak': 5,
            }},
        }
        result = p.compute(fingerprint, stage=3)
        assert 'emotional_readiness_score' in result
        assert 0 <= result['emotional_readiness_score'] <= 1


class TestIntelligencePublisher:

    @pytest.fixture
    def bus(self):
        from profile_event_bus import ProfileEventBus
        b = ProfileEventBus()
        yield b
        b.shutdown()

    def test_publishes_intelligence_on_profile_update(self, bus):
        from intelligence_publisher import IntelligencePublisher
        published = []
        bus.subscribe('intelligence.*', lambda t, p: published.append(t), mode='sync')
        publisher = IntelligencePublisher(bus)
        fingerprint = {
            'emotional': {'metrics': {'volatility': 0.3, 'dominant_family': 'Joy'}},
            'behavioral': {'metrics': {'completion_rate': 0.8, 'engagement_streak': 5}},
            'cognitive': {'metrics': {'technique_retention_rate': 0.8}},
            'linguistic': {'metrics': {'avg_sentence_length': 12}},
            'social': {'metrics': {'daily_interaction_rate': 3}},
            'aspirational': {'metrics': {}},
            'biometric': {'metrics': {'resting_hr': 68, 'hrv_baseline': 55}},
        }
        publisher.publish_for_user('user1', fingerprint, stage=3, confidence=0.7)
        assert 'intelligence.therapy' in published
        assert 'intelligence.coaching' in published
        assert 'intelligence.workflow' in published
```

- [ ] **Step 2: Implement intelligence_publisher.py**

```python
# intelligence_publisher.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Intelligence Publisher
===============================================================================
Computes and publishes personalization data after each profile process() cycle.
Tiered autonomy: advisory at stages 1-2, active at stages 3-5.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- Profile Event Bus
- Therapy Transition Engine
- AI Coach (Backend)
- Workflow Engine (Master)
===============================================================================
"""

import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

POSITIVE_FAMILIES = {'Joy', 'Love', 'Calm', 'Surprise'}
NEGATIVE_FAMILIES = {'Sadness', 'Anger', 'Fear', 'Self-Conscious'}


def _get_mode(stage: int) -> str:
    return 'active' if stage >= 3 else 'advisory'


class TherapyPersonalizer:
    """Computes therapy personalization from Behavioral + Cognitive DNA."""

    def compute(self, fingerprint: dict, stage: int) -> dict:
        behavioral = fingerprint.get('behavioral', {}).get('metrics', {})
        cognitive = fingerprint.get('cognitive', {}).get('metrics', {})

        technique_scores = {}
        usage = cognitive.get('technique_usage', {})
        retention = cognitive.get('technique_retention_rate', 0.5)
        for technique, count in usage.items():
            technique_scores[technique] = round(min(1.0, count / 20) * retention, 4)

        return {
            'intelligence_type': 'therapy_personalization',
            'mode': _get_mode(stage),
            'technique_scores': technique_scores,
            'completion_rate': behavioral.get('completion_rate', 0.5),
            'intervention_response_rate': behavioral.get('intervention_response_rate', 0.5),
            'preferred_step_type': 'cognitive' if retention > 0.7 else 'calming',
            'weight_change_bound': 0.3,  # ±30% max
        }


class CoachingPersonalizer:
    """Computes coaching personalization from Linguistic + Social + Aspirational DNA."""

    def compute(self, fingerprint: dict, stage: int) -> dict:
        linguistic = fingerprint.get('linguistic', {}).get('metrics', {})
        social = fingerprint.get('social', {}).get('metrics', {})
        aspirational = fingerprint.get('aspirational', {}).get('metrics', {})

        avg_len = linguistic.get('avg_sentence_length', 10)
        complexity = linguistic.get('vocabulary_complexity', 0.5)

        return {
            'intelligence_type': 'coaching_personalization',
            'mode': _get_mode(stage),
            'preferred_tone': 'direct' if avg_len < 10 else 'supportive',
            'vocabulary_level': 'complex' if complexity > 0.6 else 'simple',
            'response_depth': 'detailed' if avg_len > 15 else 'brief',
            'dominant_tone': linguistic.get('dominant_tone', 'neutral'),
            'active_goals': aspirational.get('core_values', []),
            'communication_style': 'direct' if avg_len < 10 else 'indirect',
        }


class WorkflowPrioritizer:
    """Computes workflow prioritization from Emotional + Behavioral DNA."""

    def compute(self, fingerprint: dict, stage: int) -> dict:
        emotional = fingerprint.get('emotional', {}).get('metrics', {})
        behavioral = fingerprint.get('behavioral', {}).get('metrics', {})

        volatility = emotional.get('volatility', 0.5)
        dominant = emotional.get('dominant_family', 'Neutral')
        valence = 1.0 if dominant in POSITIVE_FAMILIES else (0.3 if dominant in NEGATIVE_FAMILIES else 0.5)
        stability = 1.0 - volatility

        completion = behavioral.get('completion_rate', 0.5)
        streak = behavioral.get('engagement_streak', 0)
        momentum = min(1.0, (completion * 0.6) + (min(streak, 10) / 10 * 0.4))

        readiness = round((stability * 0.4 + valence * 0.3 + momentum * 0.3), 4)

        return {
            'intelligence_type': 'workflow_prioritization',
            'mode': _get_mode(stage),
            'emotional_readiness_score': readiness,
            'engagement_momentum': round(momentum, 4),
            'evolution_stage': stage,
        }


class IntelligencePublisher:
    """Orchestrates intelligence computation and bus publication."""

    def __init__(self, bus, connector=None):
        self.bus = bus
        self.connector = connector
        self.therapy = TherapyPersonalizer()
        self.coaching = CoachingPersonalizer()
        self.workflow = WorkflowPrioritizer()

    def publish_for_user(self, user_id: str, fingerprint: dict,
                         stage: int, confidence: float):
        """Compute and publish all intelligence types for a user."""
        base = {'user_id': user_id, 'stage': stage, 'confidence': confidence, 'timestamp': time.time()}

        # Therapy
        therapy = {**base, **self.therapy.compute(fingerprint, stage)}
        self.bus.publish('intelligence.therapy', therapy)

        # Coaching
        coaching = {**base, **self.coaching.compute(fingerprint, stage)}
        self.bus.publish('intelligence.coaching', coaching)

        # Workflow
        workflow = {**base, **self.workflow.compute(fingerprint, stage)}
        self.bus.publish('intelligence.workflow', workflow)

        # Calibration (all stages)
        biometric = fingerprint.get('biometric', {}).get('metrics', {})
        if biometric:
            calibration = {
                **base,
                'intelligence_type': 'calibration',
                'resting_hr': biometric.get('resting_hr'),
                'hrv_baseline': biometric.get('hrv_baseline'),
                'eda_baseline': biometric.get('eda_baseline'),
            }
            self.bus.publish('intelligence.calibration', calibration)

        # Deliver via outbound webhooks if connector available
        if self.connector:
            for payload in [therapy, coaching, workflow]:
                try:
                    self.connector.deliver_intelligence(payload['intelligence_type'], payload)
                except Exception as e:
                    logger.warning("Intelligence delivery failed: %s", e)
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_intelligence_publisher.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add intelligence_publisher.py tests/test_intelligence_publisher.py
git commit -m "feat(integration): add IntelligencePublisher with therapy/coaching/workflow personalization"
```

---

## Chunk 6: WebSocket Router

### Task 6: WebSocketRouter — Smart Streaming with Screen Subscriptions

**Files:**
- Create: `websocket_router.py`
- Create: `tests/test_websocket_router.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_websocket_router.py
"""Tests for WebSocketRouter — screen subscriptions, batching, namespace routing."""

import pytest
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestScreenSubscriptionMap:

    def test_dna_strands_screen(self):
        from websocket_router import ScreenSubscriptionMap
        topics = ScreenSubscriptionMap.get_topics('dna_strands')
        assert 'profile.domain.*' in topics

    def test_evolution_screen(self):
        from websocket_router import ScreenSubscriptionMap
        topics = ScreenSubscriptionMap.get_topics('evolution_timeline')
        assert 'profile.stage.*' in topics

    def test_backgrounded_returns_empty(self):
        from websocket_router import ScreenSubscriptionMap
        topics = ScreenSubscriptionMap.get_topics('backgrounded')
        assert len(topics) == 0

    def test_unknown_screen_returns_default(self):
        from websocket_router import ScreenSubscriptionMap
        topics = ScreenSubscriptionMap.get_topics('unknown_screen')
        assert len(topics) > 0  # Returns default set


class TestBatchBuffer:

    def test_buffers_events(self):
        from websocket_router import BatchBuffer
        buf = BatchBuffer(max_size=100)
        buf.add('user1', {'topic': 'profile.updated', 'data': {}})
        buf.add('user1', {'topic': 'alert.reactive', 'data': {}})
        events = buf.flush('user1')
        assert len(events) == 2

    def test_max_size_fifo(self):
        from websocket_router import BatchBuffer
        buf = BatchBuffer(max_size=3)
        for i in range(5):
            buf.add('user1', {'i': i})
        events = buf.flush('user1')
        assert len(events) == 3
        assert events[0]['i'] == 2  # Oldest kept

    def test_flush_clears_buffer(self):
        from websocket_router import BatchBuffer
        buf = BatchBuffer()
        buf.add('user1', {'data': 1})
        buf.flush('user1')
        assert len(buf.flush('user1')) == 0

    def test_expires_after_timeout(self):
        from websocket_router import BatchBuffer
        buf = BatchBuffer(max_size=100, expire_seconds=0.2)
        buf.add('user1', {'data': 1})
        time.sleep(0.3)
        buf.cleanup()
        events = buf.flush('user1')
        assert len(events) == 0


class TestWebSocketRouter:

    @pytest.fixture
    def bus(self):
        from profile_event_bus import ProfileEventBus
        b = ProfileEventBus()
        yield b
        b.shutdown()

    def test_routes_event_to_matching_screen(self, bus):
        from websocket_router import WebSocketRouter
        router = WebSocketRouter(bus)
        emitted = []
        router._emit = lambda sid, event, data, namespace: emitted.append((event, data))

        router.connect('sid1', 'user1', 'dna_strands')
        bus.publish('profile.domain.emotional', {'user_id': 'user1', 'score': 0.8})
        time.sleep(0.2)
        assert len(emitted) >= 1

    def test_screen_change_updates_subscriptions(self, bus):
        from websocket_router import WebSocketRouter
        router = WebSocketRouter(bus)
        router.connect('sid1', 'user1', 'dna_strands')
        router.change_screen('sid1', 'evolution_timeline')
        conn = router._connections.get('sid1')
        assert conn['screen'] == 'evolution_timeline'

    def test_disconnect_cleans_up(self, bus):
        from websocket_router import WebSocketRouter
        router = WebSocketRouter(bus)
        router.connect('sid1', 'user1', 'dna_strands')
        router.disconnect('sid1')
        assert 'sid1' not in router._connections
```

- [ ] **Step 2: Implement websocket_router.py**

```python
# websocket_router.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - WebSocket Router
===============================================================================
Smart streaming to frontend via socket.io. Routes bus events to connected
clients based on their active screen. Buffers events for backgrounded apps.

Integrates with:
- Neural Workflow AI Engine (Phases 4-5)
- Profile Event Bus
- Frontend dashboards (3D Fingerprint, DNA Strands, Evolution)
===============================================================================
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict

from profile_event_bus import TopicMatcher

logger = logging.getLogger(__name__)


SCREEN_TOPICS = {
    'dna_strands': ['profile.domain.*'],
    'evolution_timeline': ['profile.stage.*', 'profile.snapshot.*'],
    '3d_fingerprint': ['profile.updated'],
    'coaching_session': ['intelligence.coaching', 'alert.*'],
    'therapist_dashboard': ['profile.updated', 'alert.*', 'intelligence.therapy'],
    'backgrounded': [],
}

TOPIC_TO_NAMESPACE = {
    'profile': '/profile',
    'alert': '/alerts',
    'intelligence': '/intelligence',
}

DEFAULT_TOPICS = ['profile.updated', 'alert.*']


class ScreenSubscriptionMap:
    """Maps frontend screens to bus topic patterns."""

    @staticmethod
    def get_topics(screen: str) -> List[str]:
        return SCREEN_TOPICS.get(screen, DEFAULT_TOPICS)


class BatchBuffer:
    """Buffers events for backgrounded apps."""

    def __init__(self, max_size: int = 100, expire_seconds: float = 300):
        self.max_size = max_size
        self.expire_seconds = expire_seconds
        self._buffers: Dict[str, dict] = {}  # user_id -> {events, created_at}
        self._lock = threading.Lock()

    def add(self, user_id: str, event: dict):
        with self._lock:
            if user_id not in self._buffers:
                self._buffers[user_id] = {'events': [], 'created_at': time.time()}
            buf = self._buffers[user_id]
            buf['events'].append(event)
            if len(buf['events']) > self.max_size:
                buf['events'] = buf['events'][-self.max_size:]

    def flush(self, user_id: str) -> List[dict]:
        with self._lock:
            if user_id in self._buffers:
                events = self._buffers[user_id]['events']
                del self._buffers[user_id]
                return events
            return []

    def cleanup(self):
        """Remove expired buffers."""
        now = time.time()
        with self._lock:
            expired = [
                uid for uid, buf in self._buffers.items()
                if now - buf['created_at'] > self.expire_seconds
            ]
            for uid in expired:
                del self._buffers[uid]


class WebSocketRouter:
    """Routes bus events to connected frontend clients based on active screen."""

    def __init__(self, bus, socketio=None):
        self.bus = bus
        self.socketio = socketio
        self._connections: Dict[str, dict] = {}  # sid -> {user_id, screen, topics}
        self._user_sids: Dict[str, set] = defaultdict(set)  # user_id -> {sids}
        self._batch_buffer = BatchBuffer()
        self._lock = threading.Lock()

        # Subscribe to all topics we might route
        bus.subscribe('profile.*', self._on_bus_event, mode='async')
        bus.subscribe('alert.*', self._on_bus_event, mode='async')
        bus.subscribe('intelligence.*', self._on_bus_event, mode='async')

    def connect(self, sid: str, user_id: str, screen: str):
        """Register a new client connection."""
        topics = ScreenSubscriptionMap.get_topics(screen)
        with self._lock:
            self._connections[sid] = {
                'user_id': user_id,
                'screen': screen,
                'topics': topics,
            }
            self._user_sids[user_id].add(sid)

        # Deliver any buffered events
        buffered = self._batch_buffer.flush(user_id)
        if buffered:
            self._emit(sid, 'profile:batch-update', {'events': buffered}, '/profile')

        logger.info("Client %s connected for user %s on screen %s", sid, user_id, screen)

    def change_screen(self, sid: str, screen: str):
        """Update a client's active screen."""
        with self._lock:
            if sid in self._connections:
                self._connections[sid]['screen'] = screen
                self._connections[sid]['topics'] = ScreenSubscriptionMap.get_topics(screen)

    def disconnect(self, sid: str):
        """Clean up a disconnected client."""
        with self._lock:
            conn = self._connections.pop(sid, None)
            if conn:
                user_id = conn['user_id']
                self._user_sids[user_id].discard(sid)
                if not self._user_sids[user_id]:
                    del self._user_sids[user_id]

    def _on_bus_event(self, topic: str, payload: dict):
        """Route a bus event to matching connected clients."""
        user_id = payload.get('user_id')
        if not user_id:
            return

        with self._lock:
            sids = list(self._user_sids.get(user_id, set()))

        if not sids:
            # No active connections — buffer for later
            self._batch_buffer.add(user_id, {'topic': topic, 'data': payload, 'timestamp': time.time()})
            return

        namespace = self._get_namespace(topic)
        event_name = topic.replace('.', ':')

        for sid in sids:
            with self._lock:
                conn = self._connections.get(sid)
            if not conn:
                continue
            # Check if this topic matches the client's screen subscriptions
            for pattern in conn.get('topics', []):
                if TopicMatcher.matches(pattern, topic):
                    self._emit(sid, event_name, payload, namespace)
                    break

    def _get_namespace(self, topic: str) -> str:
        prefix = topic.split('.')[0]
        return TOPIC_TO_NAMESPACE.get(prefix, '/profile')

    def _emit(self, sid: str, event: str, data: dict, namespace: str):
        """Emit to a specific client. Override for testing."""
        if self.socketio:
            try:
                self.socketio.emit(event, data, room=sid, namespace=namespace)
            except Exception as e:
                logger.warning("WebSocket emit failed for %s: %s", sid, e)
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_websocket_router.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add websocket_router.py tests/test_websocket_router.py
git commit -m "feat(integration): add WebSocketRouter with smart screen-based streaming"
```

---

## Chunk 7: Wire Everything Together

### Task 7: Integrate Bus into UserProfileEngine + API Server

**Files:**
- Modify: `user_profile_engine.py`
- Modify: `profile_api.py`
- Modify: `emotion_api_server.py`

- [ ] **Step 1: Add set_event_bus() to UserProfileEngine, remove set_socketio()**

In `user_profile_engine.py`:
- Add `self._event_bus = None` and `self._intelligence_publisher = None` and `self._alert_engine = None` to `__init__`
- Add method:
```python
def set_event_bus(self, bus, connector=None):
    """Wire the event bus for real-time intelligence and alerts."""
    self._event_bus = bus
    from intelligence_publisher import IntelligencePublisher
    from alert_engine import AlertEngine
    self._intelligence_publisher = IntelligencePublisher(bus, connector)
    self._alert_engine = AlertEngine(bus, self.db)
```
- Remove `set_socketio()` method and any `self._socketio` references
- At the end of `process()`, after updating the profile, add:
```python
if self._event_bus:
    self._event_bus.publish('profile.updated', {
        'user_id': user_id, 'confidence': confidence,
        'domains_changed': list(fingerprint.keys()),
    })
    for domain, data in fingerprint.items():
        self._event_bus.publish(f'profile.domain.{domain}', {
            'user_id': user_id, 'domain': domain, 'data': data,
        })
    if stage_result['changed']:
        self._event_bus.publish('profile.stage.changed', {
            'user_id': user_id, 'old_stage': current_stage,
            'new_stage': new_stage, 'reason': stage_result['reason'],
        })
    # Publish intelligence
    if self._intelligence_publisher:
        self._intelligence_publisher.publish_for_user(user_id, fingerprint, new_stage, confidence)
    # Run predictive alerts
    if self._alert_engine:
        self._alert_engine.check_predictive(user_id, fingerprint)
```

- [ ] **Step 2: Add service registration endpoint to profile_api.py**

Add to the blueprint:
```python
@bp.route('/api/profile/services/register', methods=['POST'])
@require_service_key
def register_service():
    data = request.get_json()
    name = data.get('name')
    url = data.get('url')
    if not name or not url:
        return jsonify({'error': 'name and url required'}), 400
    if hasattr(engine, '_ecosystem_connector') and engine._ecosystem_connector:
        success = engine._ecosystem_connector.register_service(name, url)
        if not success:
            return jsonify({'error': 'URL not in allowlist'}), 403
        return jsonify({'status': 'registered', 'name': name})
    return jsonify({'error': 'Connector not initialized'}), 503
```

- [ ] **Step 3: Update emotion_api_server.py startup**

After profile engine initialization, add:
```python
if HAS_PROFILE_ENGINE and profile_engine:
    try:
        from profile_event_bus import ProfileEventBus
        from ecosystem_connector import EcosystemConnector
        from websocket_router import WebSocketRouter
        from process_scheduler import ProcessScheduler

        event_bus = ProfileEventBus()
        ecosystem_connector = EcosystemConnector(event_bus, profile_engine.db)
        profile_engine._ecosystem_connector = ecosystem_connector
        profile_engine.set_event_bus(event_bus, ecosystem_connector)

        # Process scheduler
        process_scheduler = ProcessScheduler(
            process_fn=profile_engine.process,
            debounce_seconds=30, count_threshold=20, periodic_seconds=300
        )
        process_scheduler.start()

        # WebSocket router
        try:
            from flask_socketio import SocketIO
            socketio = SocketIO(app, cors_allowed_origins="*")
            ws_router = WebSocketRouter(event_bus, socketio)

            @socketio.on('connect', namespace='/profile')
            def on_connect():
                from flask import request as req
                user_id = req.args.get('user_id', 'default')
                screen = req.args.get('screen', '3d_fingerprint')
                ws_router.connect(req.sid, user_id, screen)

            @socketio.on('screen:changed', namespace='/profile')
            def on_screen_changed(data):
                from flask import request as req
                ws_router.change_screen(req.sid, data.get('screen', ''))

            @socketio.on('disconnect', namespace='/profile')
            def on_disconnect():
                from flask import request as req
                ws_router.disconnect(req.sid)

            print("✓ WebSocket router initialized")
        except ImportError:
            print("⚠ flask-socketio not available, WebSocket disabled")

        print("✓ Event bus + ecosystem connector initialized")
    except Exception as e:
        print(f"⚠ Event bus failed to initialize: {e}")
```

- [ ] **Step 4: Run existing profile tests to verify nothing breaks**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_profile_engine.py tests/test_profile_api.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add user_profile_engine.py profile_api.py emotion_api_server.py
git commit -m "feat(integration): wire event bus, intelligence, alerts, and WebSocket into engine"
```

---

## Chunk 8: End-to-End Integration Test

### Task 8: Full Pipeline Integration Test

**Files:**
- Create: `tests/test_next_level_integration.py`

- [ ] **Step 1: Write E2E test**

```python
# tests/test_next_level_integration.py
"""End-to-end test: event in → bus → process → intelligence out → alerts."""

import pytest
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNextLevelIntegration:

    @pytest.fixture
    def system(self, tmp_path):
        from profile_event_bus import ProfileEventBus
        from user_profile_engine import UserProfileEngine
        from ecosystem_connector import EcosystemConnector
        from process_scheduler import ProcessScheduler

        bus = ProfileEventBus()
        engine = UserProfileEngine(db_path=str(tmp_path / 'test.db'))
        connector = EcosystemConnector(bus, engine.db)
        engine.set_event_bus(bus, connector)

        scheduler = ProcessScheduler(
            process_fn=engine.process,
            debounce_seconds=0.2, count_threshold=100, periodic_seconds=600
        )
        scheduler.start()

        yield {'bus': bus, 'engine': engine, 'connector': connector, 'scheduler': scheduler}

        scheduler.stop()
        bus.shutdown()
        engine.close()

    def test_event_flows_through_bus_to_profile(self, system):
        """Inbound event → bus → process → fingerprint updated."""
        published = []
        system['bus'].subscribe('profile.updated', lambda t, p: published.append(p), mode='sync')

        # Route an event through the connector
        system['connector'].route_inbound({
            'user_id': 'test_user', 'domain': 'emotional',
            'event_type': 'emotion_classified',
            'payload': {'emotion': 'joy', 'family': 'Joy', 'confidence': 0.9},
            'source': 'backend'
        })

        # Wait for debounce to trigger process
        time.sleep(0.5)

        # Profile should exist and be updated
        profile = system['engine'].get_profile('test_user')
        assert profile is not None

    def test_intelligence_published_after_process(self, system):
        """Process → intelligence events published to bus."""
        intelligence = []
        system['bus'].subscribe('intelligence.*', lambda t, p: intelligence.append(t), mode='sync')

        # Log events and process directly
        for i in range(5):
            system['engine'].log_event('test_user', 'emotional', 'emotion_classified',
                                        {'emotion': 'joy', 'family': 'Joy'}, 'nanogpt')
        system['engine'].process('test_user')

        time.sleep(0.3)
        assert 'intelligence.therapy' in intelligence
        assert 'intelligence.coaching' in intelligence

    def test_multi_domain_event_routing(self, system):
        """Multi-domain event publishes to multiple bus topics."""
        topics = []
        system['bus'].subscribe('event.*', lambda t, p: topics.append(t), mode='sync')

        system['connector'].route_inbound({
            'user_id': 'test_user', 'domain': 'biometric,temporal',
            'event_type': 'health_reading',
            'payload': {'hr': 72, 'sleep_hours': 7},
            'source': 'frontend'
        })

        assert 'event.biometric' in topics
        assert 'event.temporal' in topics

    def test_alert_fires_on_negative_spiral(self, system):
        """5+ negative emotions → reactive alert published."""
        alerts = []
        system['bus'].subscribe('alert.reactive', lambda t, p: alerts.append(p), mode='sync')

        for i in range(6):
            system['engine'].log_event('test_user', 'emotional', 'emotion_classified',
                                        {'emotion': 'sadness', 'family': 'Sadness'},
                                        'nanogpt')

        # AlertEngine processes events asynchronously
        time.sleep(0.5)

        # Check if spiral was detected
        spiral_alerts = [a for a in alerts if a.get('alert_type') == 'emotional_spiral']
        assert len(spiral_alerts) >= 1

    def test_full_api_with_bus(self, system, tmp_path):
        """API endpoints work with bus wired in."""
        from flask import Flask
        from profile_api import create_profile_blueprint

        app = Flask(__name__)
        os.environ['PROFILE_SERVICE_KEY_BACKEND'] = 'test-key'
        bp = create_profile_blueprint(system['engine'])
        app.register_blueprint(bp)

        with app.test_client() as client:
            # Ingest
            resp = client.post('/api/profile/ingest',
                headers={'X-Service-Key': 'test-key'},
                json={'user_id': 'api_user', 'domain': 'emotional',
                      'event_type': 'emotion_classified',
                      'payload': {'emotion': 'calm', 'family': 'Calm'},
                      'source': 'backend'})
            assert resp.status_code == 200

            # Snapshot triggers process which publishes to bus
            resp = client.get('/api/profile/api_user/snapshot')
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['stage'] == 1
```

- [ ] **Step 2: Run E2E tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_next_level_integration.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_event_bus.py tests/test_process_scheduler.py tests/test_ecosystem_connector.py tests/test_alert_engine.py tests/test_intelligence_publisher.py tests/test_websocket_router.py tests/test_next_level_integration.py tests/test_profile_db.py tests/test_domain_processors.py tests/test_evolution_engine.py tests/test_profile_engine.py tests/test_profile_api.py tests/test_profile_integration.py -q`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_next_level_integration.py
git commit -m "feat(integration): add end-to-end integration tests for event bus pipeline"
```

- [ ] **Step 5: Verify server starts**

Run: `cd /Users/bel/quantara-nanoGPT && python -c "from emotion_api_server import app; print('OK')"`
Expected: OK with bus + connector + scheduler initialized

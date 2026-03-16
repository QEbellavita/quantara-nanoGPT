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

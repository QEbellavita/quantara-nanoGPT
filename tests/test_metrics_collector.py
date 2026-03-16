"""Tests for MetricsCollector — passive in-memory metrics aggregation."""

import os
import sys
import time
import threading
import json

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

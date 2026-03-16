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
        from profile_event_bus import ProfileEventBus
        bus = ProfileEventBus()
        bus.publish('test', {'x': 1})

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


class TestClassifierMetrics:

    def test_classifier_counter_pattern(self):
        from metrics_collector import MetricsCollector
        m = MetricsCollector()
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

"""
Tests for EmotionGPT coordinator — facade creation, retrain monitoring,
timer lifecycle, and exception resilience.

Connected to:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics_collector import MetricsCollector
from profile_event_bus import ProfileEventBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analyzer(result=None):
    """Return a mock analyzer whose .analyze() returns *result*."""
    a = MagicMock()
    a.analyze.return_value = result or {'emotion': 'joy', 'confidence': 0.9}
    return a


def _make_gpt(**overrides):
    """Build an EmotionGPT with sensible defaults.  check_interval=9999 so
    the timer never fires during a test run."""
    from emotion_gpt import EmotionGPT
    defaults = dict(
        analyzer=_make_analyzer(),
        check_interval=9999,
    )
    defaults.update(overrides)
    return EmotionGPT(**defaults)


# ---------------------------------------------------------------------------
# TestEmotionGPTFacade
# ---------------------------------------------------------------------------

class TestEmotionGPTFacade:
    def test_create_with_analyzer_only(self):
        from emotion_gpt import EmotionGPT
        analyzer = _make_analyzer()
        gpt = EmotionGPT(analyzer=analyzer)
        try:
            assert gpt.analyzer is analyzer
            assert gpt.transitions is None
            assert gpt.auto_retrain is None
            assert gpt.profile_engine is None
            assert gpt._timer is None  # no metrics/bus => no monitor
        finally:
            gpt.shutdown()

    def test_create_with_all_subsystems(self):
        metrics = MetricsCollector()
        bus = ProfileEventBus()
        tracker = MagicMock()
        retrain = MagicMock()
        profile = MagicMock()

        gpt = _make_gpt(
            metrics=metrics,
            bus=bus,
            transition_tracker=tracker,
            auto_retrain_manager=retrain,
            profile_engine=profile,
        )
        try:
            assert gpt.transitions is tracker
            assert gpt.auto_retrain is retrain
            assert gpt.profile_engine is profile
            assert gpt._timer is not None  # monitor started
        finally:
            gpt.shutdown()
            bus.shutdown()


# ---------------------------------------------------------------------------
# TestRetrainThreshold
# ---------------------------------------------------------------------------

class TestRetrainThreshold:
    def test_threshold_trigger(self):
        """Swap rate > 20% must publish retrain.recommended with trigger='threshold'."""
        metrics = MetricsCollector()
        bus = ProfileEventBus()
        published = []
        bus.subscribe('retrain.*', lambda t, p: published.append((t, p)))

        gpt = _make_gpt(metrics=metrics, bus=bus)
        try:
            # Simulate 100 requests, 25 swapped => 25% > 20% threshold
            metrics.increment('personalization.requests', 100)
            metrics.increment('personalization.swapped', 25)
            gpt._check_retrain_signal()

            assert len(published) == 1
            topic, payload = published[0]
            assert topic == 'retrain.recommended'
            assert payload['trigger'] == 'threshold'
            assert payload['swap_rate'] == 0.25
        finally:
            gpt.shutdown()
            bus.shutdown()

    def test_below_threshold_no_trigger(self):
        """Swap rate <= 20% with insufficient windows must NOT publish."""
        metrics = MetricsCollector()
        bus = ProfileEventBus()
        published = []
        bus.subscribe('retrain.*', lambda t, p: published.append((t, p)))

        gpt = _make_gpt(metrics=metrics, bus=bus)
        try:
            # 100 requests, 10 swapped => 10% — below threshold
            metrics.increment('personalization.requests', 100)
            metrics.increment('personalization.swapped', 10)
            gpt._check_retrain_signal()

            assert len(published) == 0
        finally:
            gpt.shutdown()
            bus.shutdown()


# ---------------------------------------------------------------------------
# TestRetrainTrend
# ---------------------------------------------------------------------------

class TestRetrainTrend:
    def test_trend_trigger(self):
        """Three consecutive rising windows (all > 10%) must trigger trend event."""
        metrics = MetricsCollector()
        bus = ProfileEventBus()
        published = []
        bus.subscribe('retrain.*', lambda t, p: published.append((t, p)))

        gpt = _make_gpt(metrics=metrics, bus=bus)
        try:
            # Window 1: 100 requests, 12 swapped => 12%
            metrics.increment('personalization.requests', 100)
            metrics.increment('personalization.swapped', 12)
            gpt._check_retrain_signal()
            assert len(published) == 0

            # Window 2: another 100 requests, 15 swapped => 15%
            metrics.increment('personalization.requests', 100)
            metrics.increment('personalization.swapped', 15)
            gpt._check_retrain_signal()
            assert len(published) == 0

            # Window 3: another 100 requests, 18 swapped => 18%
            # All below 20% threshold but trending up: 12% -> 15% -> 18%
            metrics.increment('personalization.requests', 100)
            metrics.increment('personalization.swapped', 18)
            gpt._check_retrain_signal()

            assert len(published) == 1
            topic, payload = published[0]
            assert topic == 'retrain.recommended'
            assert payload['trigger'] == 'trend'
            assert payload['window_rates'] == [0.12, 0.15, 0.18]
        finally:
            gpt.shutdown()
            bus.shutdown()


# ---------------------------------------------------------------------------
# TestMinimumSampleGuard
# ---------------------------------------------------------------------------

class TestMinimumSampleGuard:
    def test_skip_when_too_few_requests(self):
        """Fewer than 20 requests in a window must be silently skipped."""
        metrics = MetricsCollector()
        bus = ProfileEventBus()
        published = []
        bus.subscribe('retrain.*', lambda t, p: published.append((t, p)))

        gpt = _make_gpt(metrics=metrics, bus=bus)
        try:
            # Only 10 requests — below _MIN_WINDOW_SAMPLES (20)
            metrics.increment('personalization.requests', 10)
            metrics.increment('personalization.swapped', 10)
            gpt._check_retrain_signal()

            assert len(published) == 0
            assert len(gpt._swap_windows) == 0  # window not recorded
        finally:
            gpt.shutdown()
            bus.shutdown()


# ---------------------------------------------------------------------------
# TestTimerLifecycle
# ---------------------------------------------------------------------------

class TestTimerLifecycle:
    def test_starts_with_metrics_and_bus(self):
        metrics = MetricsCollector()
        bus = ProfileEventBus()
        gpt = _make_gpt(metrics=metrics, bus=bus)
        try:
            assert gpt._timer is not None
            assert gpt._timer.is_alive()
        finally:
            gpt.shutdown()
            bus.shutdown()

    def test_no_timer_without_metrics(self):
        gpt = _make_gpt()
        try:
            assert gpt._timer is None
        finally:
            gpt.shutdown()

    def test_shutdown_stops_timer(self):
        metrics = MetricsCollector()
        bus = ProfileEventBus()
        gpt = _make_gpt(metrics=metrics, bus=bus)
        timer = gpt._timer
        gpt.shutdown()
        try:
            assert gpt._stopped is True
            # Timer should be cancelled (not alive after a moment)
            timer.join(timeout=1.0)
            assert not timer.is_alive()
        finally:
            bus.shutdown()

    def test_check_survives_exception(self):
        """If metrics.get_counter raises, the timer should still reschedule."""
        metrics = MagicMock()
        metrics.get_counter.side_effect = RuntimeError("boom")
        bus = MagicMock()

        gpt = _make_gpt(metrics=metrics, bus=bus)
        try:
            # Should not raise
            gpt._check_retrain_signal()
            # Bus.publish should NOT have been called (error before that)
            bus.publish.assert_not_called()
        finally:
            gpt.shutdown()

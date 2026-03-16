# EmotionGPT Coordinator Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify emotion analysis subsystems under a single EmotionGPT coordinator with a retrain signal that monitors personalization swap rates to detect classifier drift.

**Architecture:** Thin facade class holding references to analyzer, transition tracker, auto-retrain manager, profile engine. Self-repeating timer monitors swap rate metrics and publishes `retrain.recommended` events to the bus. `analyze_with_context()` provides a unified pipeline entry point.

**Tech Stack:** Python 3, threading, pytest

**Spec:** `docs/superpowers/specs/2026-03-16-emotion-gpt-coordinator-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `emotion_gpt.py` (create) | `EmotionGPT` class — facade refs, `analyze_with_context()` pipeline, retrain signal monitoring with timer |
| `emotion_api_server.py` (modify) | Create `EmotionGPT` in `create_app()` after all subsystems initialized |
| `tests/test_emotion_gpt.py` (create) | Facade access, pipeline tests, retrain threshold/trend triggers, timer lifecycle, graceful degradation |

---

## Chunk 1: EmotionGPT Core + Retrain Signal

### Task 1: EmotionGPT class with retrain monitoring

**Files:**
- Create: `emotion_gpt.py`
- Create: `tests/test_emotion_gpt.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_emotion_gpt.py
"""Tests for EmotionGPT coordinator."""

import os
import sys
import time
import threading
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEmotionGPTFacade:
    """Test that the coordinator holds and exposes subsystem references."""

    def test_create_with_analyzer_only(self):
        from emotion_gpt import EmotionGPT
        analyzer = MagicMock()
        gpt = EmotionGPT(analyzer)
        assert gpt.analyzer is analyzer
        assert gpt.transitions is None
        assert gpt.auto_retrain is None
        assert gpt.profile_engine is None
        gpt.shutdown()

    def test_create_with_all_subsystems(self):
        from emotion_gpt import EmotionGPT
        analyzer = MagicMock()
        transitions = MagicMock()
        retrain = MagicMock()
        profile = MagicMock()
        gpt = EmotionGPT(
            analyzer, transition_tracker=transitions,
            auto_retrain_manager=retrain, profile_engine=profile,
        )
        assert gpt.transitions is transitions
        assert gpt.auto_retrain is retrain
        assert gpt.profile_engine is profile
        gpt.shutdown()


class TestRetrainThreshold:
    """Test that high swap rates trigger retrain.recommended."""

    def test_threshold_trigger(self):
        from emotion_gpt import EmotionGPT
        from metrics_collector import MetricsCollector
        from profile_event_bus import ProfileEventBus

        metrics = MetricsCollector()
        bus = ProfileEventBus()
        analyzer = MagicMock()
        published = []
        bus.subscribe('retrain.*', lambda t, p: published.append((t, p)))

        gpt = EmotionGPT(analyzer, metrics=metrics, bus=bus, check_interval=9999)

        # Simulate: 100 requests, 25 swapped (25% > 20% threshold)
        metrics.increment('personalization.requests', 100)
        metrics.increment('personalization.swapped', 25)

        gpt._check_retrain_signal()

        assert len(published) == 1
        assert published[0][0] == 'retrain.recommended'
        assert published[0][1]['trigger'] == 'threshold'
        assert published[0][1]['swap_rate'] > 0.20
        gpt.shutdown()

    def test_below_threshold_no_trigger(self):
        from emotion_gpt import EmotionGPT
        from metrics_collector import MetricsCollector
        from profile_event_bus import ProfileEventBus

        metrics = MetricsCollector()
        bus = ProfileEventBus()
        published = []
        bus.subscribe('retrain.*', lambda t, p: published.append((t, p)))

        gpt = EmotionGPT(MagicMock(), metrics=metrics, bus=bus, check_interval=9999)

        # 100 requests, 10 swapped (10% < 20%)
        metrics.increment('personalization.requests', 100)
        metrics.increment('personalization.swapped', 10)

        gpt._check_retrain_signal()

        assert len(published) == 0
        gpt.shutdown()


class TestRetrainTrend:
    """Test that increasing swap rates over 3 windows trigger trend signal."""

    def test_trend_trigger(self):
        from emotion_gpt import EmotionGPT
        from metrics_collector import MetricsCollector
        from profile_event_bus import ProfileEventBus

        metrics = MetricsCollector()
        bus = ProfileEventBus()
        published = []
        bus.subscribe('retrain.*', lambda t, p: published.append((t, p)))

        gpt = EmotionGPT(MagicMock(), metrics=metrics, bus=bus, check_interval=9999)

        # Window 1: 5% swap rate (50 requests, ~3 swapped)
        metrics.increment('personalization.requests', 50)
        metrics.increment('personalization.swapped', 3)
        gpt._check_retrain_signal()
        assert len(published) == 0

        # Window 2: 8% swap rate (50 new requests, ~4 swapped)
        metrics.increment('personalization.requests', 50)
        metrics.increment('personalization.swapped', 4)
        gpt._check_retrain_signal()
        assert len(published) == 0

        # Window 3: 12% swap rate (50 new requests, ~6 swapped) — increasing + >10%
        metrics.increment('personalization.requests', 50)
        metrics.increment('personalization.swapped', 6)
        gpt._check_retrain_signal()
        assert len(published) == 1
        assert published[0][1]['trigger'] == 'trend'
        gpt.shutdown()


class TestMinimumSampleGuard:

    def test_skip_when_too_few_requests(self):
        from emotion_gpt import EmotionGPT
        from metrics_collector import MetricsCollector
        from profile_event_bus import ProfileEventBus

        metrics = MetricsCollector()
        bus = ProfileEventBus()
        published = []
        bus.subscribe('retrain.*', lambda t, p: published.append((t, p)))

        gpt = EmotionGPT(MagicMock(), metrics=metrics, bus=bus, check_interval=9999)

        # Only 10 requests (< 20 minimum)
        metrics.increment('personalization.requests', 10)
        metrics.increment('personalization.swapped', 8)  # 80% swap rate but too few

        gpt._check_retrain_signal()
        assert len(published) == 0
        gpt.shutdown()


class TestTimerLifecycle:

    def test_timer_starts_with_metrics_and_bus(self):
        from emotion_gpt import EmotionGPT
        from metrics_collector import MetricsCollector
        from profile_event_bus import ProfileEventBus

        gpt = EmotionGPT(
            MagicMock(), metrics=MetricsCollector(),
            bus=ProfileEventBus(), check_interval=9999,
        )
        assert gpt._timer is not None
        gpt.shutdown()

    def test_no_timer_without_metrics(self):
        from emotion_gpt import EmotionGPT
        gpt = EmotionGPT(MagicMock())
        assert gpt._timer is None
        gpt.shutdown()

    def test_shutdown_stops_timer(self):
        from emotion_gpt import EmotionGPT
        from metrics_collector import MetricsCollector
        from profile_event_bus import ProfileEventBus

        gpt = EmotionGPT(
            MagicMock(), metrics=MetricsCollector(),
            bus=ProfileEventBus(), check_interval=9999,
        )
        gpt.shutdown()
        assert gpt._stopped is True

    def test_check_survives_exception(self):
        """Timer reschedules even if check logic raises."""
        from emotion_gpt import EmotionGPT
        from metrics_collector import MetricsCollector
        from profile_event_bus import ProfileEventBus

        metrics = MetricsCollector()
        bus = ProfileEventBus()
        gpt = EmotionGPT(MagicMock(), metrics=metrics, bus=bus, check_interval=9999)

        # Corrupt metrics to cause an error
        with patch.object(metrics, 'get_counter', side_effect=RuntimeError("boom")):
            gpt._check_retrain_signal()  # should not raise

        # Timer should still be rescheduled (not stopped)
        assert not gpt._stopped
        gpt.shutdown()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_emotion_gpt.py -v`
Expected: FAIL — `emotion_gpt` not importable

- [ ] **Step 3: Implement EmotionGPT**

```python
# emotion_gpt.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - EmotionGPT Coordinator
===============================================================================
Unified facade for the emotion analysis pipeline.

Holds references to analyzer, transition tracker, auto-retrain, and
profile engine. Monitors personalization swap rates via MetricsCollector
and publishes retrain.recommended events when the classifier drifts.

Integrates with:
- Neural Workflow AI Engine (Phases 1-5)
- ML Training & Prediction Systems
- Backend APIs (cases, workflows, analytics)
- All Dashboard Data Integration
- Real-time data from customer service, distribution, etc.
===============================================================================
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Retrain signal constants
_THRESHOLD_SWAP_RATE = 0.20    # immediate trigger: >20% swap rate
_TREND_MIN_RATE = 0.10         # trend trigger: latest window >10%
_MIN_WINDOW_SAMPLES = 20       # minimum requests per window
_MAX_WINDOWS = 3               # rolling window depth


class EmotionGPT:
    """Coordinator for the emotion analysis pipeline.

    Unifies: analyzer, transitions, auto-retrain, profile engine.
    Monitors personalization swap rates and publishes retrain signals.
    """

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
        self.analyzer = analyzer
        self.transitions = transition_tracker
        self.auto_retrain = auto_retrain_manager
        self.profile_engine = profile_engine
        self._metrics = metrics
        self._bus = bus
        self._check_interval = check_interval
        self._swap_windows = []
        self._last_total = 0.0
        self._last_swapped = 0.0
        self._timer = None
        self._stopped = False

        if self._metrics and self._bus:
            self._start_retrain_monitor()

    # ─── Retrain Signal Monitoring ────────────────────────────────────────

    def _start_retrain_monitor(self):
        """Start the self-repeating retrain check timer."""
        self._schedule_next_check()

    def _schedule_next_check(self):
        """Schedule the next retrain check as a daemon timer."""
        if self._stopped:
            return
        self._timer = threading.Timer(self._check_interval, self._check_retrain_signal)
        self._timer.daemon = True
        self._timer.start()

    def _check_retrain_signal(self):
        """Evaluate whether retraining is recommended based on swap rates."""
        try:
            total = self._metrics.get_counter('personalization.requests')
            swapped = self._metrics.get_counter('personalization.swapped')

            window_total = total - self._last_total
            window_swapped = swapped - self._last_swapped

            # Save snapshots for next window
            self._last_total = total
            self._last_swapped = swapped

            # Minimum sample guard (before division)
            if window_total < _MIN_WINDOW_SAMPLES:
                return

            window_rate = window_swapped / window_total
            self._swap_windows.append(window_rate)
            if len(self._swap_windows) > _MAX_WINDOWS:
                self._swap_windows.pop(0)

            # Build evidence payload
            evidence = {
                'swap_rate': round(window_rate, 4),
                'window_rates': [round(r, 4) for r in self._swap_windows],
                'window_seconds': self._check_interval,
                'total_requests': int(total),
                'total_swapped': int(swapped),
            }

            # Immediate threshold (takes priority)
            if window_rate > _THRESHOLD_SWAP_RATE:
                evidence['trigger'] = 'threshold'
                evidence['recommendation'] = (
                    f'Classifier swap rate {window_rate:.1%} exceeds '
                    f'{_THRESHOLD_SWAP_RATE:.0%} threshold. Retraining recommended.'
                )
                self._bus.publish('retrain.recommended', evidence)
                return

            # Trend detection (only if threshold not triggered)
            if len(self._swap_windows) == _MAX_WINDOWS:
                w = self._swap_windows
                if w[0] < w[1] < w[2] and w[2] > _TREND_MIN_RATE:
                    evidence['trigger'] = 'trend'
                    evidence['recommendation'] = (
                        f'Classifier swap rate trending up '
                        f'({w[0]:.1%} → {w[1]:.1%} → {w[2]:.1%}). '
                        f'Retraining recommended.'
                    )
                    self._bus.publish('retrain.recommended', evidence)

        except Exception:
            logger.exception("EmotionGPT: retrain check failed")
        finally:
            if not self._stopped:
                self._schedule_next_check()

    # ─── Unified Pipeline ─────────────────────────────────────────────────

    def analyze_with_context(
        self,
        text,
        biometrics=None,
        pose=None,
        user_id=None,
        include_profile=False,
        return_embedding=False,
    ):
        """Unified emotion analysis pipeline.

        1. Classify via analyzer (forwarding all parameters)
        2. Apply personalization (if user_id and profile engine available)
        3. Record transition (if user_id and transition tracker available)
        4. Return enriched result
        """
        # Step 1: Core classification
        result = self.analyzer.analyze(
            text, biometrics, pose=pose, return_embedding=return_embedding,
        )

        # Step 2: Personalization
        snapshot = None
        if user_id and self.profile_engine:
            try:
                from emotion_classifier import apply_profile_personalization
                snapshot = self.profile_engine.get_profile_snapshot(user_id)
                apply_profile_personalization(result, snapshot)
            except Exception:
                logger.exception("EmotionGPT: personalization failed for %s", user_id)

        # Step 3: Transition recording
        if user_id and self.transitions:
            try:
                self.transitions.record(
                    user_id=user_id,
                    emotion=result.get('dominant_emotion', result.get('emotion', 'neutral')),
                    family=result.get('family'),
                    confidence=float(result.get('confidence', 1.0)),
                )
            except Exception:
                logger.exception("EmotionGPT: transition recording failed for %s", user_id)

        # Step 4: Profile context enrichment
        if include_profile and snapshot:
            try:
                result['profile_context'] = {
                    'evolution_stage': snapshot.evolution_stage,
                    'overall_confidence': snapshot.overall_confidence,
                    'dominant_family': snapshot.dominant_family,
                    'emotion_prior': snapshot.emotion_prior,
                    'event_count': snapshot.event_count,
                    'last_updated': datetime.fromtimestamp(
                        snapshot.last_updated, tz=timezone.utc
                    ).isoformat(),
                }
            except Exception:
                logger.exception("EmotionGPT: profile enrichment failed")

        return result

    # ─── Lifecycle ────────────────────────────────────────────────────────

    def shutdown(self):
        """Cancel the retrain monitoring timer and prevent rescheduling."""
        self._stopped = True
        if self._timer:
            self._timer.cancel()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_emotion_gpt.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add emotion_gpt.py tests/test_emotion_gpt.py
git commit -m "feat: add EmotionGPT coordinator with retrain signal monitoring"
```

---

## Chunk 2: Pipeline Tests + API Wiring

### Task 2: analyze_with_context pipeline tests

**Files:**
- Modify: `tests/test_emotion_gpt.py`

- [ ] **Step 1: Append pipeline tests**

Append to `tests/test_emotion_gpt.py`:

```python
class TestAnalyzeWithContext:
    """Test the unified analysis pipeline."""

    def test_basic_classification(self):
        from emotion_gpt import EmotionGPT
        analyzer = MagicMock()
        analyzer.analyze.return_value = {
            'dominant_emotion': 'joy', 'family': 'Joy',
            'confidence': 0.9, 'is_fallback': False,
        }
        gpt = EmotionGPT(analyzer)
        result = gpt.analyze_with_context("I feel great")
        analyzer.analyze.assert_called_once_with(
            "I feel great", None, pose=None, return_embedding=False,
        )
        assert result['dominant_emotion'] == 'joy'
        gpt.shutdown()

    def test_forwards_pose_and_embedding(self):
        from emotion_gpt import EmotionGPT
        analyzer = MagicMock()
        analyzer.analyze.return_value = {'dominant_emotion': 'calm', 'family': 'Calm', 'confidence': 0.7}
        gpt = EmotionGPT(analyzer)
        gpt.analyze_with_context("test", pose={'head': 0.5}, return_embedding=True)
        analyzer.analyze.assert_called_once_with(
            "test", None, pose={'head': 0.5}, return_embedding=True,
        )
        gpt.shutdown()

    def test_records_transition(self):
        from emotion_gpt import EmotionGPT
        analyzer = MagicMock()
        analyzer.analyze.return_value = {
            'dominant_emotion': 'sadness', 'family': 'Sadness', 'confidence': 0.8,
        }
        tracker = MagicMock()
        gpt = EmotionGPT(analyzer, transition_tracker=tracker)
        gpt.analyze_with_context("I feel sad", user_id='u1')
        tracker.record.assert_called_once_with(
            user_id='u1', emotion='sadness', family='Sadness', confidence=0.8,
        )
        gpt.shutdown()

    def test_skips_transition_without_user_id(self):
        from emotion_gpt import EmotionGPT
        analyzer = MagicMock()
        analyzer.analyze.return_value = {'dominant_emotion': 'joy', 'family': 'Joy', 'confidence': 0.9}
        tracker = MagicMock()
        gpt = EmotionGPT(analyzer, transition_tracker=tracker)
        gpt.analyze_with_context("I feel great")
        tracker.record.assert_not_called()
        gpt.shutdown()

    def test_skips_personalization_without_profile_engine(self):
        from emotion_gpt import EmotionGPT
        analyzer = MagicMock()
        analyzer.analyze.return_value = {
            'dominant_emotion': 'joy', 'family': 'Joy', 'confidence': 0.9,
            'family_scores': {'Joy': 0.9},
        }
        gpt = EmotionGPT(analyzer)
        result = gpt.analyze_with_context("test", user_id='u1')
        assert 'personalized' not in result
        gpt.shutdown()

    def test_personalization_with_profile_engine(self):
        from emotion_gpt import EmotionGPT
        from user_profile_engine import ProfileSnapshot
        analyzer = MagicMock()
        analyzer.analyze.return_value = {
            'dominant_emotion': 'joy', 'family': 'Joy', 'confidence': 0.4,
            'family_scores': {'Joy': 0.4, 'Sadness': 0.38},
            'scores': {'joy': 0.3},
        }
        profile_engine = MagicMock()
        profile_engine.get_profile_snapshot.return_value = ProfileSnapshot(
            user_id='u1', evolution_stage=2, overall_confidence=0.5,
            emotion_prior={'Joy': 0.5, 'Sadness': 0.5},
            dominant_family='Joy', event_count=50,
            family_counts={'Joy': 25, 'Sadness': 25},
            last_updated=time.time(),
        )
        gpt = EmotionGPT(analyzer, profile_engine=profile_engine)
        result = gpt.analyze_with_context("test", user_id='u1')
        assert 'personalized' in result
        gpt.shutdown()

    def test_include_profile_enrichment(self):
        from emotion_gpt import EmotionGPT
        from user_profile_engine import ProfileSnapshot
        analyzer = MagicMock()
        analyzer.analyze.return_value = {
            'dominant_emotion': 'joy', 'family': 'Joy', 'confidence': 0.9,
        }
        profile_engine = MagicMock()
        profile_engine.get_profile_snapshot.return_value = ProfileSnapshot(
            user_id='u1', evolution_stage=3, overall_confidence=0.7,
            emotion_prior={'Joy': 0.6}, dominant_family='Joy',
            event_count=100, family_counts={'Joy': 60},
            last_updated=time.time(),
        )
        gpt = EmotionGPT(analyzer, profile_engine=profile_engine)
        result = gpt.analyze_with_context("test", user_id='u1', include_profile=True)
        assert 'profile_context' in result
        assert result['profile_context']['evolution_stage'] == 3
        gpt.shutdown()

    def test_graceful_degradation_analyzer_only(self):
        """Pipeline works with just an analyzer — no transitions, no profile."""
        from emotion_gpt import EmotionGPT
        analyzer = MagicMock()
        analyzer.analyze.return_value = {'dominant_emotion': 'neutral', 'family': 'Neutral', 'confidence': 0.5}
        gpt = EmotionGPT(analyzer)
        result = gpt.analyze_with_context("test", user_id='u1', include_profile=True)
        assert result['dominant_emotion'] == 'neutral'
        assert 'profile_context' not in result
        gpt.shutdown()
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_emotion_gpt.py -v`
Expected: ALL PASS (implementation already covers pipeline)

- [ ] **Step 3: Commit**

```bash
git add tests/test_emotion_gpt.py
git commit -m "test: add analyze_with_context pipeline tests for EmotionGPT"
```

---

### Task 3: Wire EmotionGPT into emotion_api_server.py

**Files:**
- Modify: `emotion_api_server.py`

- [ ] **Step 1: Add import near the top** (around line 76):

```python
from emotion_gpt import EmotionGPT
```

- [ ] **Step 2: Create EmotionGPT after metrics wiring** (after line 970, after `profile_engine._ecosystem_connector.set_metrics(metrics_collector)`):

```python
    # Initialize EmotionGPT coordinator
    emotion_gpt = None
    try:
        emotion_gpt = EmotionGPT(
            analyzer=model,
            transition_tracker=transition_tracker,
            auto_retrain_manager=auto_retrain_manager if 'auto_retrain_manager' in dir() else None,
            profile_engine=profile_engine,
            metrics=metrics_collector,
            bus=profile_engine._event_bus if profile_engine and profile_engine._event_bus else None,
        )
        print("✓ EmotionGPT coordinator initialized")
    except Exception as e:
        print(f"⚠ EmotionGPT coordinator failed: {e}")
```

- [ ] **Step 3: Run existing tests to verify nothing breaks**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_emotion_gpt.py tests/test_personalization.py tests/test_analyze_enrichment.py tests/test_profile_engine.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add emotion_api_server.py
git commit -m "feat: wire EmotionGPT coordinator into API server startup"
```

---

## Chunk 3: Full Verification

### Task 4: Full test suite and server verification

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/ --tb=short -q`
Expected: 670+ passed, 0 failed

- [ ] **Step 2: Verify server starts**

Run: `cd /Users/bel/quantara-nanoGPT && timeout 10 python emotion_api_server.py 2>&1 || true`
Expected: No ImportError/SyntaxError. Should see "EmotionGPT coordinator initialized" in output.

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

_THRESHOLD_SWAP_RATE = 0.20
_TREND_MIN_RATE = 0.10
_MIN_WINDOW_SAMPLES = 20
_MAX_WINDOWS = 3


class EmotionGPT:
    """Coordinator for the emotion analysis pipeline."""

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

    def _start_retrain_monitor(self):
        self._schedule_next_check()

    def _schedule_next_check(self):
        if self._stopped:
            return
        self._timer = threading.Timer(self._check_interval, self._check_retrain_signal)
        self._timer.daemon = True
        self._timer.start()

    def _check_retrain_signal(self):
        try:
            total = self._metrics.get_counter('personalization.requests')
            swapped = self._metrics.get_counter('personalization.swapped')

            window_total = total - self._last_total
            window_swapped = swapped - self._last_swapped

            self._last_total = total
            self._last_swapped = swapped

            if window_total < _MIN_WINDOW_SAMPLES:
                return

            window_rate = window_swapped / window_total
            self._swap_windows.append(window_rate)
            if len(self._swap_windows) > _MAX_WINDOWS:
                self._swap_windows.pop(0)

            evidence = {
                'swap_rate': round(window_rate, 4),
                'window_rates': [round(r, 4) for r in self._swap_windows],
                'window_seconds': self._check_interval,
                'total_requests': int(total),
                'total_swapped': int(swapped),
            }

            if window_rate > _THRESHOLD_SWAP_RATE:
                evidence['trigger'] = 'threshold'
                evidence['recommendation'] = (
                    f'Classifier swap rate {window_rate:.1%} exceeds '
                    f'{_THRESHOLD_SWAP_RATE:.0%} threshold. Retraining recommended.'
                )
                self._bus.publish('retrain.recommended', evidence)
                return

            if len(self._swap_windows) == _MAX_WINDOWS:
                w = self._swap_windows
                if w[0] < w[1] < w[2] and w[2] > _TREND_MIN_RATE:
                    evidence['trigger'] = 'trend'
                    evidence['recommendation'] = (
                        f'Classifier swap rate trending up '
                        f'({w[0]:.1%} \u2192 {w[1]:.1%} \u2192 {w[2]:.1%}). '
                        f'Retraining recommended.'
                    )
                    self._bus.publish('retrain.recommended', evidence)

        except Exception:
            logger.exception("EmotionGPT: retrain check failed")
        finally:
            if not self._stopped:
                self._schedule_next_check()

    def analyze_with_context(
        self,
        text,
        biometrics=None,
        pose=None,
        user_id=None,
        include_profile=False,
        return_embedding=False,
    ):
        result = self.analyzer.analyze(
            text, biometrics, pose=pose, return_embedding=return_embedding,
        )

        snapshot = None
        if user_id and self.profile_engine:
            try:
                from emotion_classifier import apply_profile_personalization
                snapshot = self.profile_engine.get_profile_snapshot(user_id)
                apply_profile_personalization(result, snapshot)
            except Exception:
                logger.exception("EmotionGPT: personalization failed for %s", user_id)

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

    def shutdown(self):
        self._stopped = True
        if self._timer:
            self._timer.cancel()

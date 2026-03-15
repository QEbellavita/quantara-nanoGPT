"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Auto-Retraining Pipeline
===============================================================================
Monitors WiFi calibration model performance and triggers automatic retraining
when drift is detected or sufficient new data has accumulated.

Integrates with:
- Neural Workflow AI Engine (adaptive model lifecycle)
- ML Training & Prediction Systems (automated retraining)
- Biometric Integration Engine (calibration quality)
- Real-time Data (continuous monitoring)
- Dashboard Data Integration (retrain status reporting)

Components:
  DriftDetector        — KS-test based statistical drift detection
  ThresholdMonitor     — Sample-count triggers for retraining
  RetrainLog           — SQLite audit log of retrain events
  AutoRetrainManager   — Background daemon coordinating the pipeline

Safety:
  - Validation gate: new model must beat old model on held-out data
  - Cooldown periods: 1h after success, 4h after failure
  - Thread-safe hot-swap via threading.Lock
  - All retrain events logged to SQLite
===============================================================================
"""

import os
import copy
import time
import logging
import sqlite3
import threading
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DriftDetector — KS-test based statistical drift detection
# ═══════════════════════════════════════════════════════════════════════════════


class DriftDetector:
    """
    Detects distribution shift in prediction errors using the
    Kolmogorov-Smirnov two-sample test.

    Usage:
        1. Populate with baseline errors via add_error() + set_baseline()
        2. Continue adding new errors
        3. Call is_drifting() to check if current window differs from baseline

    Connected to:
    - Neural Workflow AI Engine (drift-aware model management)
    - ML Training & Prediction Systems (trigger signal)
    """

    def __init__(self, window_size: int = 50, p_threshold: float = 0.05):
        self.window_size = window_size
        self.p_threshold = p_threshold
        self._errors = deque(maxlen=window_size)
        self._baseline = None

    def add_error(self, error: float):
        """Add a prediction error to the rolling window."""
        self._errors.append(error)

    def set_baseline(self):
        """Snapshot current error window as the baseline distribution."""
        if len(self._errors) < 2:
            logger.warning("[DriftDetector] Not enough errors for baseline")
            return
        self._baseline = list(self._errors)

    def is_drifting(self) -> bool:
        """
        Test whether current errors differ significantly from baseline
        using the two-sample KS test (p < threshold => drift).
        """
        if self._baseline is None or len(self._errors) < 2:
            return False
        _, p_value = stats.ks_2samp(self._baseline, list(self._errors))
        return p_value < self.p_threshold

    def get_drift_score(self) -> dict:
        """Return drift analysis details."""
        if self._baseline is None or len(self._errors) < 2:
            return {
                'has_baseline': self._baseline is not None,
                'current_window_size': len(self._errors),
                'drifting': False,
                'p_value': None,
                'ks_statistic': None,
            }
        ks_stat, p_value = stats.ks_2samp(self._baseline, list(self._errors))
        return {
            'has_baseline': True,
            'current_window_size': len(self._errors),
            'baseline_size': len(self._baseline),
            'drifting': p_value < self.p_threshold,
            'p_value': float(p_value),
            'ks_statistic': float(ks_stat),
            'baseline_mean': float(np.mean(self._baseline)),
            'current_mean': float(np.mean(list(self._errors))),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ThresholdMonitor — Sample-count based retrain triggers
# ═══════════════════════════════════════════════════════════════════════════════


class ThresholdMonitor:
    """
    Triggers retraining based on accumulated sample counts.

    Schedule:
      - First retrain at `first_threshold` samples
      - Subsequent retrains every `subsequent_interval` samples after last retrain

    Connected to:
    - Neural Workflow AI Engine (scheduled model updates)
    - ML Training & Prediction Systems (data accumulation tracking)
    """

    def __init__(self, first_threshold: int = 20, subsequent_interval: int = 50):
        self.first_threshold = first_threshold
        self.subsequent_interval = subsequent_interval
        self._triggered_first = False
        self._last_retrain_count = 0

    def should_retrain(self, total_samples: int) -> bool:
        """Check if retraining should be triggered given total sample count."""
        if not self._triggered_first:
            if total_samples >= self.first_threshold:
                self._triggered_first = True
                return True
            return False

        # After first trigger, check interval from last retrain
        if self._last_retrain_count == 0:
            # First was triggered but not yet marked as retrained
            return False
        return (total_samples - self._last_retrain_count) >= self.subsequent_interval

    def mark_retrained(self, total_samples: int):
        """Record that retraining occurred at this sample count."""
        self._last_retrain_count = total_samples


# ═══════════════════════════════════════════════════════════════════════════════
# Validation gate
# ═══════════════════════════════════════════════════════════════════════════════


def validate_retrained_model(
    old_model: nn.Module,
    new_model: nn.Module,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
) -> dict:
    """
    Compare MAE of old vs new model on validation data.
    New model is accepted only if it achieves lower MAE.

    Connected to:
    - ML Training & Prediction Systems (model quality gate)
    - Neural Workflow AI Engine (safe model promotion)

    Returns:
        dict with old_mae, new_mae, accepted (bool), improvement (float)
    """
    old_model.eval()
    new_model.eval()

    with torch.no_grad():
        old_preds = old_model(val_inputs)
        new_preds = new_model(val_inputs)

    old_mae = float(torch.mean(torch.abs(old_preds - val_targets)))
    new_mae = float(torch.mean(torch.abs(new_preds - val_targets)))

    return {
        'old_mae': old_mae,
        'new_mae': new_mae,
        'accepted': new_mae < old_mae,
        'improvement': old_mae - new_mae,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RetrainLog — SQLite audit trail
# ═══════════════════════════════════════════════════════════════════════════════


class RetrainLog:
    """
    Persistent audit log of retrain events stored in SQLite.

    Connected to:
    - Dashboard Data Integration (retrain history display)
    - Neural Workflow AI Engine (model lifecycle tracking)
    """

    def __init__(self, db_path: str = 'data/retrain_log.db'):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.execute('PRAGMA busy_timeout=5000')
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS retrain_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                samples_used INTEGER,
                mae_before REAL,
                mae_after REAL,
                outcome TEXT NOT NULL,
                details TEXT
            )
        ''')
        self._conn.commit()

    def log(
        self,
        trigger_type: str,
        samples_used: int,
        mae_before: float,
        mae_after: float,
        outcome: str,
        details: str = None,
    ):
        """Record a retrain event."""
        self._conn.execute(
            '''INSERT INTO retrain_log
               (timestamp, trigger_type, samples_used, mae_before, mae_after, outcome, details)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (
                datetime.now().isoformat(),
                trigger_type,
                samples_used,
                mae_before,
                mae_after,
                outcome,
                details,
            ),
        )
        self._conn.commit()

    def get_log(self, limit: int = 50) -> list:
        """Retrieve recent retrain events as list of dicts."""
        cursor = self._conn.execute(
            'SELECT * FROM retrain_log ORDER BY id DESC LIMIT ?', (limit,)
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


# ═══════════════════════════════════════════════════════════════════════════════
# AutoRetrainManager — Background daemon
# ═══════════════════════════════════════════════════════════════════════════════


class AutoRetrainManager:
    """
    Coordinates automatic retraining of the WiFi calibration model.

    Monitors:
      1. ThresholdMonitor — sample count triggers
      2. DriftDetector — statistical distribution shift

    Safety:
      - Validation gate: new model must improve MAE
      - Cooldown: 1h after success, 4h after failure
      - Thread-safe hot-swap via threading.Lock
      - All events logged to SQLite

    Connected to:
    - Neural Workflow AI Engine (autonomous model lifecycle)
    - ML Training & Prediction Systems (fine-tuning pipeline)
    - Biometric Integration Engine (calibration quality)
    - Real-time Data (continuous monitoring loop)
    - Dashboard Data Integration (status reporting)
    """

    COOLDOWN_SUCCESS = 3600    # 1 hour
    COOLDOWN_FAILURE = 14400   # 4 hours

    def __init__(
        self,
        calibration_buffer,
        model,
        checkpoint_path: str = 'checkpoints/ruview_calibration.pt',
        socketio=None,
        profile_engine=None,
    ):
        self.buffer = calibration_buffer
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.socketio = socketio
        self.profile_engine = profile_engine

        self.threshold_monitor = ThresholdMonitor(first_threshold=20, subsequent_interval=50)
        self.drift_detector = DriftDetector(window_size=50)
        self.retrain_log = RetrainLog()

        self._model_lock = threading.Lock()
        self._running = False
        self._thread = None
        self._last_retrain_time = 0
        self._last_retrain_success = True

    def start(self, check_interval: int = 60):
        """Start the background monitoring daemon thread."""
        if self._running:
            logger.warning("[AutoRetrain] Already running")
            return
        self._running = True
        self._check_interval = check_interval
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name='auto-retrain-monitor',
        )
        self._thread.start()
        logger.info(f"[AutoRetrain] Started monitoring (interval={check_interval}s)")

    def stop(self):
        """Stop the background monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("[AutoRetrain] Stopped")

    def _monitor_loop(self):
        """Main loop: periodically check retrain triggers."""
        while self._running:
            try:
                self._check_triggers()
            except Exception as e:
                logger.error(f"[AutoRetrain] Monitor error: {e}")
            time.sleep(self._check_interval)

    def _check_triggers(self):
        """Check all retrain triggers and initiate if conditions met."""
        # Respect cooldown
        elapsed = time.time() - self._last_retrain_time
        cooldown = self.COOLDOWN_SUCCESS if self._last_retrain_success else self.COOLDOWN_FAILURE
        if self._last_retrain_time > 0 and elapsed < cooldown:
            return

        # Feed prediction errors to drift detector
        errors = self.buffer.get_prediction_errors(self.model)
        for err in errors:
            self.drift_detector.add_error(err)

        # Check threshold trigger
        total = self.buffer.total_samples_seen
        if self.threshold_monitor.should_retrain(total):
            logger.info(f"[AutoRetrain] Threshold trigger at {total} samples")
            self._do_retrain('threshold')
            return

        # Check drift trigger
        if self.drift_detector.is_drifting():
            logger.info("[AutoRetrain] Drift detected, triggering retrain")
            if self.profile_engine:
                try:
                    self.profile_engine.log_event('default', 'biometric', 'model_drift_detected', {
                        'drift_detected': True,
                    }, 'nanogpt')
                except Exception:
                    pass
            self._do_retrain('drift')
            return

    def _do_retrain(self, trigger: str):
        """Execute retraining with validation gate and hot-swap."""
        try:
            buffer_data = self.buffer.get_buffer_data()
            if len(buffer_data) < 5:
                logger.warning("[AutoRetrain] Not enough buffer data for retraining")
                return

            # Split into train/val (80/20)
            n = len(buffer_data)
            split = max(1, int(n * 0.8))
            train_data = buffer_data[:split]
            val_data = buffer_data[split:] if split < n else buffer_data[-2:]

            train_inputs = torch.cat([p[0] for p in train_data], dim=0)
            train_targets = torch.cat([p[1] for p in train_data], dim=0)
            val_inputs = torch.cat([p[0] for p in val_data], dim=0)
            val_targets = torch.cat([p[1] for p in val_data], dim=0)

            # Clone model for fine-tuning
            with self._model_lock:
                new_model = copy.deepcopy(self.model)

            # Fine-tune with early stopping
            optimizer = optim.Adam(new_model.parameters(), lr=1e-4)
            loss_fn = nn.MSELoss()
            new_model.train()

            best_val_loss = float('inf')
            patience_counter = 0
            patience = 5
            best_state = None

            for epoch in range(50):
                optimizer.zero_grad()
                preds = new_model(train_inputs)
                loss = loss_fn(preds, train_targets)
                loss.backward()
                optimizer.step()

                # Early stopping on validation loss
                new_model.eval()
                with torch.no_grad():
                    val_preds = new_model(val_inputs)
                    val_loss = loss_fn(val_preds, val_targets).item()
                new_model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = copy.deepcopy(new_model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            # Restore best state
            if best_state is not None:
                new_model.load_state_dict(best_state)
            new_model.eval()

            # Validation gate
            result = validate_retrained_model(self.model, new_model, val_inputs, val_targets)

            if result['accepted']:
                # Hot-swap model
                with self._model_lock:
                    self.model.load_state_dict(new_model.state_dict())
                    self.model.eval()

                # Save checkpoint
                ckpt_dir = os.path.dirname(self.checkpoint_path)
                if ckpt_dir:
                    os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(new_model.state_dict(), self.checkpoint_path)

                # Update monitors
                self.threshold_monitor.mark_retrained(self.buffer.total_samples_seen)
                self.drift_detector.set_baseline()
                self._last_retrain_time = time.time()
                self._last_retrain_success = True

                # Log
                self.retrain_log.log(
                    trigger_type=trigger,
                    samples_used=len(buffer_data),
                    mae_before=result['old_mae'],
                    mae_after=result['new_mae'],
                    outcome='accepted',
                    details=f"improvement={result['improvement']:.6f}",
                )

                logger.info(
                    f"[AutoRetrain] Retrain accepted: MAE {result['old_mae']:.4f} -> {result['new_mae']:.4f}"
                )

                # Emit WebSocket event
                if self.socketio:
                    try:
                        self.socketio.emit('retrain_complete', {
                            'trigger': trigger,
                            'old_mae': result['old_mae'],
                            'new_mae': result['new_mae'],
                            'improvement': result['improvement'],
                            'timestamp': datetime.now().isoformat(),
                        }, namespace='/biometrics')
                    except Exception:
                        pass

            else:
                self._last_retrain_time = time.time()
                self._last_retrain_success = False

                self.retrain_log.log(
                    trigger_type=trigger,
                    samples_used=len(buffer_data),
                    mae_before=result['old_mae'],
                    mae_after=result['new_mae'],
                    outcome='rejected',
                    details=f"no improvement (diff={result['improvement']:.6f})",
                )

                logger.info(
                    f"[AutoRetrain] Retrain rejected: MAE {result['old_mae']:.4f} vs {result['new_mae']:.4f}"
                )

        except Exception as e:
            logger.error(f"[AutoRetrain] Retrain failed: {e}")
            self._last_retrain_time = time.time()
            self._last_retrain_success = False

            try:
                self.retrain_log.log(
                    trigger_type=trigger,
                    samples_used=0,
                    mae_before=0,
                    mae_after=0,
                    outcome='error',
                    details=str(e),
                )
            except Exception:
                pass

    def manual_retrain(self) -> dict:
        """Trigger a manual retrain and return the result."""
        try:
            buffer_data = self.buffer.get_buffer_data()
            if len(buffer_data) < 5:
                return {'status': 'error', 'message': 'Not enough data (need at least 5 samples)'}

            self._do_retrain('manual')
            log_entries = self.retrain_log.get_log(limit=1)
            if log_entries:
                return {'status': 'ok', 'result': log_entries[0]}
            return {'status': 'ok', 'message': 'Retrain completed'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def get_status(self) -> dict:
        """Return current auto-retrain status for dashboard display."""
        drift_score = self.drift_detector.get_drift_score()
        return {
            'running': self._running,
            'drift': drift_score,
            'buffer_total_samples': self.buffer.total_samples_seen,
            'buffer_current_size': self.buffer.buffer_size,
            'last_retrain_time': (
                datetime.fromtimestamp(self._last_retrain_time).isoformat()
                if self._last_retrain_time > 0 else None
            ),
            'last_retrain_success': self._last_retrain_success,
            'threshold_monitor': {
                'first_threshold': self.threshold_monitor.first_threshold,
                'subsequent_interval': self.threshold_monitor.subsequent_interval,
                'triggered_first': self.threshold_monitor._triggered_first,
                'last_retrain_count': self.threshold_monitor._last_retrain_count,
            },
        }

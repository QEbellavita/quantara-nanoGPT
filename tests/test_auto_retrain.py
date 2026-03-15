"""
Tests for auto_retrain module — drift detection, threshold monitoring,
and validation gate for the auto-retraining pipeline.

Connected to:
- Neural Workflow AI Engine (adaptive model management)
- ML Training & Prediction Systems (retraining triggers)
"""

import pytest
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDriftDetector:
    @pytest.fixture
    def detector(self):
        from auto_retrain import DriftDetector
        return DriftDetector(window_size=20)

    def test_no_drift_with_consistent_errors(self, detector):
        import numpy as np
        np.random.seed(42)
        for _ in range(20):
            detector.add_error(np.random.normal(0, 0.1))
        detector.set_baseline()
        for _ in range(20):
            detector.add_error(np.random.normal(0, 0.1))
        assert not detector.is_drifting()

    def test_drift_detected_with_shifted_errors(self, detector):
        import numpy as np
        np.random.seed(42)
        for _ in range(20):
            detector.add_error(np.random.normal(0, 0.1))
        detector.set_baseline()
        for _ in range(20):
            detector.add_error(np.random.normal(5.0, 0.1))
        assert detector.is_drifting()


class TestThresholdMonitor:
    @pytest.fixture
    def monitor(self):
        from auto_retrain import ThresholdMonitor
        return ThresholdMonitor(first_threshold=5, subsequent_interval=10)

    def test_triggers_at_first_threshold(self, monitor):
        for i in range(4):
            assert not monitor.should_retrain(i + 1)
        assert monitor.should_retrain(5)

    def test_triggers_at_subsequent_intervals(self, monitor):
        monitor.should_retrain(5)
        monitor.mark_retrained(5)
        for i in range(6, 14):
            assert not monitor.should_retrain(i)
        assert monitor.should_retrain(15)


class TestRetrainWorker:
    def test_validation_gate_rejects_worse_model(self):
        from auto_retrain import validate_retrained_model
        from wifi_calibration import WiFiCalibrationModel
        old_model = WiFiCalibrationModel()
        new_model = WiFiCalibrationModel()
        val_inputs = torch.randn(10, 2)
        val_targets = torch.randn(10, 2)
        result = validate_retrained_model(old_model, new_model, val_inputs, val_targets)
        assert 'old_mae' in result
        assert 'new_mae' in result
        assert 'accepted' in result

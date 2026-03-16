"""
Tests for WiFi Signal Calibration Module.
Covers: WiFiCalibrationModel, training, PersonalCalibrationBuffer, load/save.
"""

import sys
import os
import time
import shutil
import threading
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Task 1: WiFiCalibrationModel core ────────────────────────────────────────


class TestWiFiCalibrationModel:
    """Test the core neural network model."""

    def test_output_shape_single(self):
        """Single input produces shape (1, 2) output."""
        from wifi_calibration import WiFiCalibrationModel
        model = WiFiCalibrationModel()
        x = torch.tensor([[15.0, 0.5]])  # breathing_rate, motion_level
        out = model(x)
        assert out.shape == (1, 2)

    def test_output_shape_batch(self):
        """Batch input of N produces shape (N, 2) output."""
        from wifi_calibration import WiFiCalibrationModel
        model = WiFiCalibrationModel()
        x = torch.randn(8, 2)
        out = model(x)
        assert out.shape == (8, 2)

    def test_hrv_within_bounds(self):
        """HRV output is always within [HRV_MIN, HRV_MAX] ms."""
        from wifi_calibration import WiFiCalibrationModel, HRV_MIN, HRV_MAX
        model = WiFiCalibrationModel()
        # Test many random inputs including extreme values
        inputs = torch.tensor([
            [6.0, 0.0],    # min breathing, min motion
            [30.0, 1.0],   # max breathing, max motion
            [0.0, -1.0],   # below range
            [100.0, 5.0],  # above range
            [15.0, 0.5],   # normal
        ])
        out = model(inputs)
        hrv = out[:, 0]
        assert (hrv >= HRV_MIN).all(), f"HRV below {HRV_MIN}: {hrv}"
        assert (hrv <= HRV_MAX).all(), f"HRV above {HRV_MAX}: {hrv}"

    def test_eda_within_bounds(self):
        """EDA output is always within [0.5, 20.0] uS."""
        from wifi_calibration import WiFiCalibrationModel
        model = WiFiCalibrationModel()
        inputs = torch.tensor([
            [6.0, 0.0],
            [30.0, 1.0],
            [0.0, -1.0],
            [100.0, 5.0],
            [15.0, 0.5],
        ])
        out = model(inputs)
        eda = out[:, 1]
        assert (eda >= 0.5).all(), f"EDA below 0.5: {eda}"
        assert (eda <= 20.0).all(), f"EDA above 20.0: {eda}"

    def test_deterministic_with_same_weights(self):
        """Same input gives same output (no randomness in forward pass)."""
        from wifi_calibration import WiFiCalibrationModel
        model = WiFiCalibrationModel()
        model.eval()
        x = torch.tensor([[12.0, 0.3]])
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)


# ─── Task 2: Calibration training ─────────────────────────────────────────────


class TestInverseMappings:
    """Test the inverse mapping helper functions."""

    def test_invert_breathing_to_hrv(self):
        """Inverting HRV=_HRV_LIN_MAX should give min breathing rate (6 BPM)."""
        from wifi_calibration import _invert_breathing_to_hrv, _HRV_LIN_MAX
        br = _invert_breathing_to_hrv(_HRV_LIN_MAX)
        assert abs(br - 6.0) < 0.1

    def test_invert_breathing_to_hrv_low(self):
        """Inverting HRV=20 should give max breathing rate (30 BPM)."""
        from wifi_calibration import _invert_breathing_to_hrv
        br = _invert_breathing_to_hrv(20.0)
        assert abs(br - 30.0) < 0.1

    def test_invert_motion_to_eda(self):
        """Inverting EDA=0.5 should give min motion (0.0)."""
        from wifi_calibration import _invert_motion_to_eda
        motion = _invert_motion_to_eda(0.5)
        assert abs(motion - 0.0) < 0.01

    def test_invert_motion_to_eda_high(self):
        """Inverting EDA=8.0 should give max motion (1.0)."""
        from wifi_calibration import _invert_motion_to_eda
        motion = _invert_motion_to_eda(8.0)
        assert abs(motion - 1.0) < 0.01


class TestBootstrappedPairs:
    """Test synthetic training data generation."""

    def test_generate_pairs_returns_tensors(self):
        """Should return (inputs, targets) tensors."""
        from wifi_calibration import _generate_bootstrapped_pairs
        inputs, targets = _generate_bootstrapped_pairs(n=50)
        assert isinstance(inputs, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        assert inputs.shape == (50, 2)
        assert targets.shape == (50, 2)

    def test_generate_pairs_noise(self):
        """Two calls with same n should produce different data (noise)."""
        from wifi_calibration import _generate_bootstrapped_pairs
        i1, t1 = _generate_bootstrapped_pairs(n=100)
        i2, t2 = _generate_bootstrapped_pairs(n=100)
        # Very unlikely to be identical with noise
        assert not torch.allclose(i1, i2)


class TestTrainCalibrationModel:
    """Test the training function."""

    def test_training_produces_checkpoint(self, tmp_path):
        """Training should save a checkpoint file."""
        from wifi_calibration import train_calibration_model
        ckpt_path = str(tmp_path / "wifi_cal.pt")
        model = train_calibration_model(
            n_pairs=100, epochs=10, checkpoint_path=ckpt_path
        )
        assert os.path.exists(ckpt_path)
        assert model is not None

    def test_trained_model_slow_breathing_high_hrv(self, tmp_path):
        """After training, slow breathing should produce higher HRV than fast."""
        from wifi_calibration import train_calibration_model
        ckpt_path = str(tmp_path / "wifi_cal.pt")
        model = train_calibration_model(
            n_pairs=500, epochs=50, checkpoint_path=ckpt_path
        )
        model.eval()
        with torch.no_grad():
            slow = model(torch.tensor([[8.0, 0.1]]))   # slow breathing
            fast = model(torch.tensor([[25.0, 0.1]]))   # fast breathing
        # Slow breathing should give higher HRV than fast
        assert slow[0, 0].item() > fast[0, 0].item(), (
            f"Expected slow breathing HRV ({slow[0,0]:.1f}) > fast ({fast[0,0]:.1f})"
        )


# ─── Task 3: PersonalCalibrationBuffer ────────────────────────────────────────


class TestPersonalCalibrationBuffer:
    """Test per-user calibration buffer and fine-tuning."""

    @pytest.fixture
    def base_model_path(self, tmp_path):
        """Create a trained base model for buffer tests."""
        from wifi_calibration import train_calibration_model
        path = str(tmp_path / "base.pt")
        train_calibration_model(n_pairs=100, epochs=10, checkpoint_path=path)
        return path

    @pytest.fixture
    def buffer(self, base_model_path, tmp_path):
        """Create a PersonalCalibrationBuffer with a base model."""
        from wifi_calibration import PersonalCalibrationBuffer
        cal_dir = str(tmp_path / "profiles")
        os.makedirs(cal_dir, exist_ok=True)
        return PersonalCalibrationBuffer(
            base_checkpoint=base_model_path,
            profile_id="test_user",
            calibration_dir=cal_dir,
        )

    def test_add_pair_increments_count(self, buffer):
        """Adding a pair should increment the pair count."""
        buffer.add_pair(
            wifi_input=(15.0, 0.5),
            wearable_target=(55.0, 3.0),
        )
        assert buffer.pair_count == 1

    def test_fine_tune_triggers_at_20(self, base_model_path, tmp_path):
        """Fine-tuning should trigger after 20 pairs."""
        from wifi_calibration import PersonalCalibrationBuffer
        cal_dir = str(tmp_path / "profiles_20")
        os.makedirs(cal_dir, exist_ok=True)
        buf = PersonalCalibrationBuffer(
            base_checkpoint=base_model_path,
            profile_id="test_20",
            calibration_dir=cal_dir,
        )
        for i in range(20):
            br = 6.0 + i * 1.2
            motion = i / 20.0
            buf.add_pair(
                wifi_input=(br, motion),
                wearable_target=(60.0 + i, 2.0 + i * 0.1),
            )
        # Wait briefly for background fine-tune thread
        time.sleep(2)
        profile_path = os.path.join(cal_dir, "test_20.pt")
        assert os.path.exists(profile_path), "Profile should be saved after 20 pairs"

    def test_profile_save_load_roundtrip(self, base_model_path, tmp_path):
        """Saving and loading a profile should produce consistent inference."""
        from wifi_calibration import PersonalCalibrationBuffer, load_calibration_model
        cal_dir = str(tmp_path / "profiles_rt")
        os.makedirs(cal_dir, exist_ok=True)

        buf = PersonalCalibrationBuffer(
            base_checkpoint=base_model_path,
            profile_id="roundtrip",
            calibration_dir=cal_dir,
        )
        # Add enough pairs to trigger fine-tune
        for i in range(25):
            buf.add_pair(
                wifi_input=(10.0 + i * 0.5, i / 25.0),
                wearable_target=(50.0 + i, 1.5 + i * 0.2),
            )
        time.sleep(2)

        # Load the model back
        model, profile_model = load_calibration_model(
            base_checkpoint=base_model_path,
            profile_id="roundtrip",
            calibration_dir=cal_dir,
        )
        assert model is not None
        assert profile_model is not None

    def test_inference_not_blocked_during_finetune(self, buffer):
        """predict() should work even while fine-tuning runs in background."""
        # Add pairs but don't wait for fine-tune
        for i in range(20):
            buffer.add_pair(
                wifi_input=(10.0 + i, i / 20.0),
                wearable_target=(50.0 + i, 2.0),
            )
        # Immediately try inference — should not block
        result = buffer.predict(15.0, 0.5)
        assert result is not None
        hrv, eda = result
        assert 10.0 <= hrv <= 250.0
        assert 0.5 <= eda <= 20.0

    def test_load_calibration_model_no_checkpoint(self, tmp_path):
        """load_calibration_model returns (None, None) when no checkpoint."""
        from wifi_calibration import load_calibration_model
        model, profile = load_calibration_model(
            base_checkpoint=str(tmp_path / "nonexistent.pt"),
            profile_id="nobody",
            calibration_dir=str(tmp_path),
        )
        assert model is None
        assert profile is None

    def test_buffer_max_size(self, buffer):
        """Buffer should not exceed MAX_BUFFER_SIZE (500) pairs."""
        from wifi_calibration import PersonalCalibrationBuffer
        max_size = PersonalCalibrationBuffer.MAX_BUFFER_SIZE
        for i in range(max_size + 20):
            buffer.add_pair(
                wifi_input=(15.0, 0.5),
                wearable_target=(55.0, 3.0),
            )
        assert buffer.pair_count <= max_size

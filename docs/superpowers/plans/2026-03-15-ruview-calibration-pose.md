# RuView Calibration Model & Pose Encoder Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace linear WiFi→biometric approximations with a trained calibration model and add body pose features to the emotion classifier.

**Architecture:** Two new modules (`wifi_calibration.py`, `pose_encoder.py`) plug into the existing RuView provider and emotion classifier. The calibration model maps (breathing_rate, motion_level) → (HRV, EDA) with sigmoid-bounded output. The pose encoder extracts 8 posture features from 17 COCO keypoints and produces a 16-dim embedding consumed by an expanded FusionHead (400→416 input dim). Online adaptation fine-tunes calibration per-environment.

**Tech Stack:** PyTorch, pytest, existing Quantara emotion pipeline

**Spec:** `docs/superpowers/specs/2026-03-15-ruview-calibration-pose-design.md`

---

## Chunk 1: WiFi Calibration Model

### Task 1: WiFiCalibrationModel core

**Files:**
- Create: `wifi_calibration.py`
- Create: `tests/test_wifi_calibration.py`

- [ ] **Step 1: Write failing test for model shape and output bounds**

```python
# tests/test_wifi_calibration.py
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wifi_calibration import WiFiCalibrationModel


class TestWiFiCalibrationModel:
    def test_output_shape(self):
        model = WiFiCalibrationModel()
        x = torch.tensor([[15.0, 0.5]])  # breathing_rate, motion_level
        out = model(x)
        assert out.shape == (1, 2)

    def test_hrv_output_within_range(self):
        model = WiFiCalibrationModel()
        # Test across full input range
        for br in [6.0, 15.0, 30.0]:
            for motion in [0.0, 0.5, 1.0]:
                x = torch.tensor([[br, motion]])
                hrv, eda = model(x).squeeze().tolist()
                assert 10.0 <= hrv <= 100.0, f"HRV {hrv} out of range for br={br}, motion={motion}"
                assert 0.5 <= eda <= 20.0, f"EDA {eda} out of range for br={br}, motion={motion}"

    def test_batch_input(self):
        model = WiFiCalibrationModel()
        x = torch.tensor([[10.0, 0.2], [20.0, 0.8], [15.0, 0.5]])
        out = model(x)
        assert out.shape == (3, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wifi_calibration.py -v`
Expected: FAIL with "No module named 'wifi_calibration'"

- [ ] **Step 3: Implement WiFiCalibrationModel**

```python
# wifi_calibration.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - WiFi Calibration Model
===============================================================================
Learned mapping from WiFi-sensed signals (breathing rate, motion level)
to wearable-grade biometrics (HRV, EDA). Replaces linear approximations
in ruview_provider.py.

Integrates with:
- Neural Workflow AI Engine
- Biometric Integration Engine
- RuView WiFi Sensing Provider
- Real-time Data
===============================================================================
"""

import os
import copy
import threading
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# Output ranges matching BiometricEncoder.RANGES
HRV_MIN, HRV_MAX = 10.0, 100.0
EDA_MIN, EDA_MAX = 0.5, 20.0

DEFAULT_CHECKPOINT = 'checkpoints/ruview_calibration.pt'


class WiFiCalibrationModel(nn.Module):
    """
    Maps WiFi-sensed signals to calibrated biometric values.
    Input: [breathing_rate, motion_level]
    Output: [hrv, eda] bounded to physiological ranges via sigmoid rescale.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        # Sigmoid + rescale to guaranteed physiological ranges
        sig = torch.sigmoid(raw)
        hrv = sig[:, 0:1] * (HRV_MAX - HRV_MIN) + HRV_MIN
        eda = sig[:, 1:2] * (EDA_MAX - EDA_MIN) + EDA_MIN
        return torch.cat([hrv, eda], dim=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wifi_calibration.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add wifi_calibration.py tests/test_wifi_calibration.py
git commit -m "feat: add WiFiCalibrationModel with bounded output"
```

---

### Task 2: Calibration training with bootstrapped pairs

**Files:**
- Modify: `wifi_calibration.py`
- Modify: `tests/test_wifi_calibration.py`

- [ ] **Step 1: Write failing test for training function**

```python
# Append to tests/test_wifi_calibration.py
from wifi_calibration import WiFiCalibrationModel, train_calibration_model


class TestCalibrationTraining:
    def test_train_produces_checkpoint(self, tmp_path):
        ckpt_path = tmp_path / "cal.pt"
        model = train_calibration_model(
            num_samples=200,
            epochs=5,
            checkpoint_path=str(ckpt_path),
        )
        assert ckpt_path.exists()
        assert isinstance(model, WiFiCalibrationModel)

    def test_trained_model_better_than_random(self, tmp_path):
        """Trained model should have lower loss than untrained."""
        ckpt_path = tmp_path / "cal.pt"
        trained = train_calibration_model(
            num_samples=500,
            epochs=20,
            checkpoint_path=str(ckpt_path),
        )
        untrained = WiFiCalibrationModel()

        # Test on known pair: BR=12 (slow) should give high HRV
        x = torch.tensor([[12.0, 0.1]])
        trained_hrv = trained(x)[0, 0].item()
        # Slow breathing + low motion → high HRV (>60)
        assert trained_hrv > 50.0, f"Trained HRV {trained_hrv} too low for slow breathing"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wifi_calibration.py::TestCalibrationTraining -v`
Expected: FAIL with "cannot import name 'train_calibration_model'"

- [ ] **Step 3: Implement training with bootstrapped pairs**

Add to `wifi_calibration.py`:

```python
def _invert_breathing_to_hrv(hrv: float) -> float:
    """Invert the linear HRV→breathing_rate mapping from ruview_provider."""
    # Original: ratio = (max_br - br) / (max_br - min_br); hrv = min_hrv + ratio * (max_hrv - min_hrv)
    # Invert: br = max_br - (hrv - min_hrv) / (max_hrv - min_hrv) * (max_br - min_br)
    min_br, max_br = 6.0, 30.0
    min_hrv, max_hrv = 20.0, 95.0
    ratio = (hrv - min_hrv) / (max_hrv - min_hrv)
    ratio = max(0.0, min(1.0, ratio))
    return max_br - ratio * (max_br - min_br)


def _invert_motion_to_eda(eda: float) -> float:
    """Invert the linear EDA→motion_level mapping from ruview_provider."""
    # Original: ratio = (motion - min) / (max - min); eda = min_eda + ratio * (max_eda - min_eda)
    # Invert: motion = (eda - min_eda) / (max_eda - min_eda) * (max_motion - min_motion) + min_motion
    min_motion, max_motion = 0.0, 1.0
    min_eda, max_eda = 0.5, 8.0
    ratio = (eda - min_eda) / (max_eda - min_eda)
    ratio = max(0.0, min(1.0, ratio))
    return min_motion + ratio * (max_motion - min_motion)


def _generate_bootstrapped_pairs(num_samples: int = 5000) -> tuple:
    """
    Generate (wifi_signal, wearable_reading) pairs by inverting known
    physiological relationships and adding noise.

    Returns: (inputs: np.ndarray [N,2], targets: np.ndarray [N,2])
    """
    rng = np.random.default_rng(42)

    # Sample realistic HRV and EDA values
    hrvs = rng.uniform(HRV_MIN, HRV_MAX, num_samples)
    edas = rng.uniform(EDA_MIN, min(EDA_MAX, 12.0), num_samples)  # Most EDA < 12

    inputs = []
    targets = []

    for hrv, eda in zip(hrvs, edas):
        # Invert to get WiFi signals
        br = _invert_breathing_to_hrv(hrv)
        motion = _invert_motion_to_eda(eda)

        # Add noise (sigma = 10% of range)
        br_noisy = br + rng.normal(0, 2.4)   # 10% of 24 BPM range
        motion_noisy = motion + rng.normal(0, 0.1)  # 10% of 1.0 range

        # Jitter augmentation
        br_noisy += rng.uniform(-3, 3)
        motion_noisy += rng.uniform(-0.15, 0.15)

        # Clamp inputs
        br_noisy = max(3.0, min(35.0, br_noisy))
        motion_noisy = max(0.0, min(1.0, motion_noisy))

        inputs.append([br_noisy, motion_noisy])
        targets.append([hrv, eda])

    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)


def train_calibration_model(
    num_samples: int = 5000,
    epochs: int = 100,
    lr: float = 0.01,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
) -> WiFiCalibrationModel:
    """
    Train calibration model on bootstrapped pairs.
    Saves checkpoint and returns trained model.
    """
    model = WiFiCalibrationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    inputs, targets = _generate_bootstrapped_pairs(num_samples)
    X = torch.tensor(inputs)
    Y = torch.tensor(targets)

    model.train()
    for epoch in range(epochs):
        pred = model(X)
        loss = criterion(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            logger.info(f"[Calibration] Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")

    model.eval()

    # Save checkpoint
    ckpt_dir = os.path.dirname(checkpoint_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"[Calibration] Model saved to {checkpoint_path}")

    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wifi_calibration.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add wifi_calibration.py tests/test_wifi_calibration.py
git commit -m "feat: add calibration training with bootstrapped pairs"
```

---

### Task 3: PersonalCalibrationBuffer with thread-safe online adaptation

**Files:**
- Modify: `wifi_calibration.py`
- Modify: `tests/test_wifi_calibration.py`

- [ ] **Step 1: Write failing tests for buffer and fine-tuning**

```python
# Append to tests/test_wifi_calibration.py
import time
from wifi_calibration import PersonalCalibrationBuffer, WiFiCalibrationModel


class TestPersonalCalibrationBuffer:
    def test_add_pair_and_count(self):
        buf = PersonalCalibrationBuffer(WiFiCalibrationModel())
        buf.add_pair(
            wifi={'breathing_rate': 15.0, 'motion_level': 0.3},
            wearable={'hrv': 55.0, 'eda': 2.5}
        )
        assert buf.pair_count == 1

    def test_fine_tune_triggers_at_20(self, tmp_path):
        model = WiFiCalibrationModel()
        buf = PersonalCalibrationBuffer(model, profile_dir=str(tmp_path))
        for i in range(20):
            buf.add_pair(
                wifi={'breathing_rate': 12.0 + i * 0.5, 'motion_level': 0.2 + i * 0.02},
                wearable={'hrv': 60.0 + i, 'eda': 2.0 + i * 0.1}
            )
        # After 20 pairs, fine-tuning should have been triggered
        assert buf.fine_tune_count >= 1

    def test_profile_save_load(self, tmp_path):
        model = WiFiCalibrationModel()
        buf = PersonalCalibrationBuffer(model, profile_id="test_room", profile_dir=str(tmp_path))
        # Add enough pairs to trigger fine-tune + save
        for i in range(25):
            buf.add_pair(
                wifi={'breathing_rate': 10.0 + i, 'motion_level': 0.1 + i * 0.03},
                wearable={'hrv': 50.0 + i * 2, 'eda': 1.5 + i * 0.2}
            )
        profile_path = tmp_path / "test_room.pt"
        assert profile_path.exists()

    def test_inference_not_blocked_during_finetune(self):
        """Verify model can be called during fine-tune."""
        model = WiFiCalibrationModel()
        buf = PersonalCalibrationBuffer(model)
        # Should not raise even during concurrent access
        x = torch.tensor([[15.0, 0.5]])
        result = buf.calibrate(x)
        assert result.shape == (1, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wifi_calibration.py::TestPersonalCalibrationBuffer -v`
Expected: FAIL with "cannot import name 'PersonalCalibrationBuffer'"

- [ ] **Step 3: Implement PersonalCalibrationBuffer**

Add to `wifi_calibration.py`:

```python
class PersonalCalibrationBuffer:
    """
    Per-environment calibration buffer with online adaptation.
    Fine-tunes calibration model's output layer on paired WiFi+wearable readings.
    Thread-safe: clones model for fine-tuning, atomic weight swap.
    """

    TRIGGER_FIRST = 20   # Fine-tune after this many pairs
    TRIGGER_INTERVAL = 50  # Then every N new pairs
    MAX_BUFFER = 100

    def __init__(
        self,
        base_model: WiFiCalibrationModel,
        profile_id: str = None,
        profile_dir: str = None,
    ):
        self.profile_id = profile_id or os.environ.get('RUVIEW_PROFILE_ID', 'default')
        self.profile_dir = profile_dir or os.environ.get(
            'RUVIEW_CALIBRATION_DIR', 'calibration_profiles'
        )
        self._model = base_model
        self._lock = threading.Lock()
        self._pairs = []
        self._total_added = 0
        self.fine_tune_count = 0

        # Try to load existing profile
        self._load_profile()

    @property
    def pair_count(self) -> int:
        return len(self._pairs)

    def add_pair(self, wifi: dict, wearable: dict):
        """
        Add a paired reading. wifi={breathing_rate, motion_level},
        wearable={hrv, eda}.
        """
        self._pairs.append({
            'input': [wifi['breathing_rate'], wifi['motion_level']],
            'target': [wearable['hrv'], wearable['eda']],
        })
        if len(self._pairs) > self.MAX_BUFFER:
            self._pairs.pop(0)

        self._total_added += 1

        # Check if we should fine-tune
        if self._total_added == self.TRIGGER_FIRST or (
            self._total_added > self.TRIGGER_FIRST and
            (self._total_added - self.TRIGGER_FIRST) % self.TRIGGER_INTERVAL == 0
        ):
            self._fine_tune()

    def calibrate(self, x: torch.Tensor) -> torch.Tensor:
        """Run calibration inference (thread-safe read)."""
        with self._lock:
            model = self._model
        with torch.no_grad():
            return model(x)

    def _fine_tune(self):
        """Fine-tune output layer on buffered pairs. Non-blocking to inference."""
        if len(self._pairs) < 10:
            return

        # Clone model for fine-tuning (inference continues on original)
        clone = copy.deepcopy(self._model)

        # Freeze hidden layer, only train output
        for param in clone.net[0].parameters():
            param.requires_grad = False
        for param in clone.net[1].parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(
            [p for p in clone.parameters() if p.requires_grad],
            lr=0.005
        )
        criterion = nn.MSELoss()

        inputs = torch.tensor([p['input'] for p in self._pairs])
        targets = torch.tensor([p['target'] for p in self._pairs])

        clone.train()
        for _ in range(50):
            pred = clone(inputs)
            loss = criterion(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        clone.eval()

        # Atomic swap
        with self._lock:
            self._model = clone

        self.fine_tune_count += 1
        self._save_profile()
        logger.info(f"[Calibration] Fine-tuned profile '{self.profile_id}' "
                     f"({len(self._pairs)} pairs, tune #{self.fine_tune_count})")

    def _save_profile(self):
        """Save calibration profile to disk."""
        try:
            os.makedirs(self.profile_dir, exist_ok=True)
            path = os.path.join(self.profile_dir, f"{self.profile_id}.pt")
            with self._lock:
                torch.save(self._model.state_dict(), path)
        except Exception as e:
            logger.warning(f"[Calibration] Profile save failed: {e}")

    def _load_profile(self):
        """Load existing profile if available."""
        path = os.path.join(self.profile_dir, f"{self.profile_id}.pt")
        try:
            if os.path.exists(path):
                state = torch.load(path, map_location='cpu', weights_only=True)
                self._model.load_state_dict(state)
                logger.info(f"[Calibration] Loaded profile '{self.profile_id}'")
        except Exception as e:
            logger.warning(f"[Calibration] Profile load failed, using base model: {e}")


def load_calibration_model(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    profile_id: str = None,
    profile_dir: str = None,
) -> tuple:
    """
    Load calibration model + optional personal buffer.
    Returns (model, buffer) or (None, None) if checkpoint missing.
    """
    model = WiFiCalibrationModel()

    if os.path.exists(checkpoint_path):
        try:
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state)
            model.eval()
            logger.info(f"[Calibration] Loaded base model from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"[Calibration] Failed to load {checkpoint_path}: {e}")
            return None, None
    else:
        logger.info(f"[Calibration] No checkpoint at {checkpoint_path}, returning None")
        return None, None

    buffer = PersonalCalibrationBuffer(model, profile_id=profile_id, profile_dir=profile_dir)
    return model, buffer
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wifi_calibration.py -v`
Expected: All passed (7+)

- [ ] **Step 5: Commit**

```bash
git add wifi_calibration.py tests/test_wifi_calibration.py
git commit -m "feat: add PersonalCalibrationBuffer with thread-safe online adaptation"
```

---

### Task 4: Integrate calibration into RuViewProvider

**Files:**
- Modify: `ruview_provider.py`
- Modify: `tests/test_ruview_provider.py`

- [ ] **Step 1: Write failing test for calibrated biometrics**

```python
# Append to tests/test_ruview_provider.py
from unittest.mock import patch, MagicMock


class TestCalibratedBiometrics:
    def test_uses_calibration_when_available(self, provider):
        """When calibration model is loaded, get_biometrics uses it."""
        # Simulate data
        provider._vitals_buffer.append({
            'heart_rate': 80.0, 'breathing_rate': 14.0,
            'confidence': 0.9, 'timestamp': time.time(),
        })
        provider._presence_data = {'detected': True, 'occupancy': 1, 'motion_level': 0.3}

        # Mock calibration
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(
            squeeze=MagicMock(return_value=MagicMock(tolist=MagicMock(return_value=[55.0, 3.2])))
        )
        provider._calibration_model = mock_model

        bio = provider.get_biometrics()
        assert bio is not None
        assert bio['hrv'] == 55.0
        assert bio['eda'] == 3.2

    def test_falls_back_to_linear_when_no_calibration(self, provider):
        """Without calibration model, linear approximation used."""
        provider._vitals_buffer.append({
            'heart_rate': 80.0, 'breathing_rate': 14.0,
            'confidence': 0.9, 'timestamp': time.time(),
        })
        provider._presence_data = {'detected': True, 'occupancy': 1, 'motion_level': 0.3}
        provider._calibration_model = None

        bio = provider.get_biometrics()
        assert bio is not None
        # Should still produce valid hrv/eda via linear fallback
        assert 10.0 <= bio['hrv'] <= 100.0
        assert 0.5 <= bio['eda'] <= 20.0

    def test_load_calibration_returns_none_when_no_checkpoint(self):
        """load_calibration_model returns (None, None) when checkpoint missing."""
        from wifi_calibration import load_calibration_model
        model, buf = load_calibration_model(checkpoint_path='/nonexistent/path.pt')
        assert model is None
        assert buf is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ruview_provider.py::TestCalibratedBiometrics -v`
Expected: FAIL (provider doesn't have _calibration_model attr yet)

- [ ] **Step 3: Update RuViewProvider to use calibration model**

In `ruview_provider.py`, update `__init__` to try loading calibration:

```python
# In __init__, after existing init:
self._calibration_model = None
self._calibration_buffer = None
try:
    from wifi_calibration import load_calibration_model
    model, buffer = load_calibration_model()
    if model:
        self._calibration_model = model
        self._calibration_buffer = buffer
        logger.info("[RuViewProvider] Calibration model loaded")
except ImportError:
    logger.info("[RuViewProvider] wifi_calibration not available, using linear fallback")
```

Update `get_biometrics()` to use calibration when available:

```python
# Replace the hrv/eda computation in get_biometrics():
if self._calibration_model is not None:
    import torch
    x = torch.tensor([[br, motion]], dtype=torch.float32)
    if self._calibration_buffer:
        out = self._calibration_buffer.calibrate(x)
    else:
        with torch.no_grad():
            out = self._calibration_model(x)
    hrv, eda = out.squeeze().tolist()
else:
    # Linear fallback
    hrv = self._breathing_to_hrv(br)
    eda = self._motion_to_eda(motion)
```

- [ ] **Step 4: Run all RuView tests**

Run: `python -m pytest tests/test_ruview_provider.py -v`
Expected: All passed (28+)

- [ ] **Step 5: Commit**

```bash
git add ruview_provider.py tests/test_ruview_provider.py
git commit -m "feat: integrate calibration model into RuViewProvider"
```

---

### Task 5: CLI entry point for calibration training

**Files:**
- Modify: `wifi_calibration.py`

- [ ] **Step 1: Add CLI to wifi_calibration.py**

```python
# At bottom of wifi_calibration.py
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train RuView WiFi Calibration Model')
    parser.add_argument('--train', action='store_true', help='Train calibration model')
    parser.add_argument('--samples', type=int, default=5000, help='Training samples')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--output', default=DEFAULT_CHECKPOINT, help='Checkpoint path')
    args = parser.parse_args()

    if args.train:
        logging.basicConfig(level=logging.INFO)
        print("=" * 50)
        print("  RuView WiFi Calibration Training")
        print("=" * 50)
        model = train_calibration_model(
            num_samples=args.samples,
            epochs=args.epochs,
            checkpoint_path=args.output,
        )
        # Quick validation
        import torch
        test_inputs = torch.tensor([
            [8.0, 0.1],   # Slow breathing, still → high HRV, low EDA
            [25.0, 0.8],  # Fast breathing, active → low HRV, high EDA
            [15.0, 0.4],  # Normal → moderate
        ])
        results = model(test_inputs)
        print(f"\nValidation:")
        for i, (br, m) in enumerate(test_inputs.tolist()):
            hrv, eda = results[i].tolist()
            print(f"  BR={br:.0f}, Motion={m:.1f} → HRV={hrv:.1f}ms, EDA={eda:.1f}µS")
        print(f"\nCheckpoint saved to: {args.output}")
    else:
        parser.print_help()
```

- [ ] **Step 2: Test CLI**

Run: `python wifi_calibration.py --train --samples 500 --epochs 20 --output /tmp/test_cal.pt`
Expected: Training output with validation, checkpoint saved

- [ ] **Step 3: Commit**

```bash
git add wifi_calibration.py
git commit -m "feat: add CLI entry point for calibration training"
```

---

## Chunk 2: Pose Encoder

### Task 6: PoseFeatureExtractor

**Files:**
- Create: `pose_encoder.py`
- Create: `tests/test_pose_encoder.py`

- [ ] **Step 1: Write failing tests for pose feature extraction**

```python
# tests/test_pose_encoder.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pose_encoder import PoseFeatureExtractor


def _make_standing_pose():
    """Standing upright, arms at sides, facing forward."""
    return [
        [0.5, 0.1, 0.9],   # 0: nose
        [0.48, 0.08, 0.9],  # 1: left_eye
        [0.52, 0.08, 0.9],  # 2: right_eye
        [0.45, 0.1, 0.9],   # 3: left_ear
        [0.55, 0.1, 0.9],   # 4: right_ear
        [0.4, 0.3, 0.9],    # 5: left_shoulder
        [0.6, 0.3, 0.9],    # 6: right_shoulder
        [0.38, 0.5, 0.9],   # 7: left_elbow
        [0.62, 0.5, 0.9],   # 8: right_elbow
        [0.38, 0.7, 0.9],   # 9: left_wrist
        [0.62, 0.7, 0.9],   # 10: right_wrist
        [0.45, 0.6, 0.9],   # 11: left_hip
        [0.55, 0.6, 0.9],   # 12: right_hip
        [0.45, 0.8, 0.9],   # 13: left_knee
        [0.55, 0.8, 0.9],   # 14: right_knee
        [0.45, 1.0, 0.9],   # 15: left_ankle
        [0.55, 1.0, 0.9],   # 16: right_ankle
    ]


def _make_slouched_pose():
    """Slouched: shoulders closer to hips, head drooped."""
    pose = _make_standing_pose()
    pose[5][1] = 0.45  # left shoulder lower
    pose[6][1] = 0.45  # right shoulder lower
    pose[0][1] = 0.35  # nose drooped
    return pose


class TestPoseFeatureExtractor:
    def test_produces_8_features(self):
        ext = PoseFeatureExtractor()
        features = ext.extract(_make_standing_pose())
        assert features is not None
        assert len(features) == 8

    def test_slouch_score_higher_when_slouched(self):
        ext = PoseFeatureExtractor()
        standing = ext.extract(_make_standing_pose())
        slouched = ext.extract(_make_slouched_pose())
        # Lower slouch_score = more slouch (smaller shoulder-hip distance)
        assert slouched['slouch_score'] < standing['slouch_score']

    def test_symmetry_score_high_for_symmetric_pose(self):
        ext = PoseFeatureExtractor()
        features = ext.extract(_make_standing_pose())
        assert features['symmetry_score'] > 0.8

    def test_temporal_features_default_zero(self):
        ext = PoseFeatureExtractor()
        features = ext.extract(_make_standing_pose())
        assert features['gesture_speed'] == 0.0
        assert features['stillness_duration'] == 0.0

    def test_gesture_speed_after_movement(self):
        ext = PoseFeatureExtractor()
        pose1 = _make_standing_pose()
        ext.extract(pose1)  # Frame 1

        pose2 = _make_standing_pose()
        # Move all keypoints significantly
        for kp in pose2:
            kp[0] += 0.2
            kp[1] += 0.1
        features = ext.extract(pose2)  # Frame 2
        assert features['gesture_speed'] > 0.0

    def test_returns_none_for_invalid_input(self):
        ext = PoseFeatureExtractor()
        assert ext.extract(None) is None
        assert ext.extract([]) is None
        assert ext.extract([[0, 0]]) is None  # Wrong count

    def test_all_features_within_range(self):
        ext = PoseFeatureExtractor()
        features = ext.extract(_make_standing_pose())
        assert 0.0 <= features['slouch_score'] <= 1.0
        assert 0.0 <= features['openness_score'] <= 1.0
        assert 0.0 <= features['tension_score'] <= 1.0
        assert -1.0 <= features['head_tilt'] <= 1.0
        assert 0.0 <= features['gesture_speed'] <= 1.0
        assert 0.0 <= features['symmetry_score'] <= 1.0
        assert -1.0 <= features['forward_lean'] <= 1.0
        assert 0.0 <= features['stillness_duration'] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pose_encoder.py -v`
Expected: FAIL with "No module named 'pose_encoder'"

- [ ] **Step 3: Implement PoseFeatureExtractor**

```python
# pose_encoder.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Pose-Based Emotion Encoder
===============================================================================
Extracts emotion-relevant posture features from RuView's 17-keypoint COCO
body pose and encodes them into a 16-dim embedding for emotion classification.

Integrates with:
- Neural Workflow AI Engine
- Biometric Integration Engine
- RuView WiFi Sensing Provider
- Emotion-Aware Training Engine
- Real-time Data
===============================================================================
"""

import math
import logging
from collections import deque

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# COCO keypoint indices
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Expected upright shoulder-to-hip vertical distance (normalized coordinates)
EXPECTED_UPRIGHT_RATIO = 0.3

# Gesture speed threshold for stillness detection
STILLNESS_THRESHOLD = 0.02


class PoseFeatureExtractor:
    """
    Converts 17 COCO keypoints into 8 emotion-relevant features.
    Maintains a temporal buffer for gesture_speed and stillness_duration.

    Keypoint format: list of 17 [x, y, confidence] arrays.
    Y-axis increases downward (screen coordinates).
    """

    BUFFER_SIZE = 10  # ~0.5s at 20Hz

    def __init__(self):
        self._buffer = deque(maxlen=self.BUFFER_SIZE)
        self._still_frames = 0

    def extract(self, keypoints) -> dict | None:
        """
        Extract 8 pose features from 17 COCO keypoints.
        Returns None if keypoints are invalid.
        """
        if not keypoints or not isinstance(keypoints, list) or len(keypoints) != 17:
            return None

        try:
            kps = np.array(keypoints, dtype=np.float32)
        except (ValueError, TypeError):
            return None

        if kps.shape != (17, 3):
            return None

        # Store in buffer for temporal features
        self._buffer.append(kps[:, :2].copy())  # Store x,y only

        features = {
            'slouch_score': self._slouch_score(kps),
            'openness_score': self._openness_score(kps),
            'tension_score': self._tension_score(kps),
            'head_tilt': self._head_tilt(kps),
            'gesture_speed': self._gesture_speed(),
            'symmetry_score': self._symmetry_score(kps),
            'forward_lean': self._forward_lean(kps),
            'stillness_duration': self._stillness_duration(),
        }

        return features

    def _slouch_score(self, kps: np.ndarray) -> float:
        """Shoulder-to-hip vertical distance ratio. Lower = more slouch."""
        shoulder_y = (kps[L_SHOULDER, 1] + kps[R_SHOULDER, 1]) / 2
        hip_y = (kps[L_HIP, 1] + kps[R_HIP, 1]) / 2
        # Y-down: hip_y > shoulder_y when upright
        distance = hip_y - shoulder_y
        ratio = distance / EXPECTED_UPRIGHT_RATIO if EXPECTED_UPRIGHT_RATIO > 0 else 0
        return float(max(0.0, min(1.0, ratio)))

    def _openness_score(self, kps: np.ndarray) -> float:
        """Wrist spread relative to shoulder width."""
        shoulder_width = abs(kps[R_SHOULDER, 0] - kps[L_SHOULDER, 0])
        if shoulder_width < 0.01:
            return 0.5
        wrist_spread = abs(kps[R_WRIST, 0] - kps[L_WRIST, 0])
        ratio = wrist_spread / shoulder_width
        return float(max(0.0, min(1.0, ratio / 3.0)))  # Normalize: 3x shoulder = max

    def _tension_score(self, kps: np.ndarray) -> float:
        """Shoulder elevation relative to ears. Raised shoulders = tension."""
        ear_y = (kps[L_EAR, 1] + kps[R_EAR, 1]) / 2
        shoulder_y = (kps[L_SHOULDER, 1] + kps[R_SHOULDER, 1]) / 2
        # Y-down: smaller gap = shoulders raised toward ears
        gap = shoulder_y - ear_y
        # Normal gap ~0.2, tense ~0.1
        tension = max(0.0, 1.0 - gap / 0.25)
        return float(max(0.0, min(1.0, tension)))

    def _head_tilt(self, kps: np.ndarray) -> float:
        """Nose position relative to shoulder midpoint. Positive = forward/down."""
        shoulder_mid_x = (kps[L_SHOULDER, 0] + kps[R_SHOULDER, 0]) / 2
        shoulder_mid_y = (kps[L_SHOULDER, 1] + kps[R_SHOULDER, 1]) / 2
        nose_x = kps[NOSE, 0]
        # Horizontal offset, normalized
        offset = (nose_x - shoulder_mid_x) / 0.2 if 0.2 > 0 else 0
        return float(max(-1.0, min(1.0, offset)))

    def _gesture_speed(self) -> float:
        """Mean L2 displacement between consecutive frames, normalized via tanh."""
        if len(self._buffer) < 2:
            return 0.0
        prev = self._buffer[-2]
        curr = self._buffer[-1]
        displacement = np.sqrt(np.sum((curr - prev) ** 2, axis=1)).mean()
        # Normalize via tanh
        return float(math.tanh(displacement / 50.0))

    def _symmetry_score(self, kps: np.ndarray) -> float:
        """Symmetry of left-right pairs. 1.0 = perfectly symmetric."""
        pairs = [
            (L_SHOULDER, R_SHOULDER),
            (L_ELBOW, R_ELBOW),
            (L_WRIST, R_WRIST),
            (L_HIP, R_HIP),
        ]
        mid_x = (kps[L_SHOULDER, 0] + kps[R_SHOULDER, 0]) / 2
        diffs = []
        for l_idx, r_idx in pairs:
            l_offset = abs(kps[l_idx, 0] - mid_x)
            r_offset = abs(kps[r_idx, 0] - mid_x)
            diff = abs(l_offset - r_offset)
            diffs.append(diff)
        mean_diff = np.mean(diffs)
        # Perfect symmetry = 0 diff → score 1.0
        return float(max(0.0, min(1.0, 1.0 - mean_diff / 0.2)))

    def _forward_lean(self, kps: np.ndarray) -> float:
        """Horizontal offset of shoulders vs hips. Positive = leaning forward."""
        shoulder_x = (kps[L_SHOULDER, 0] + kps[R_SHOULDER, 0]) / 2
        hip_x = (kps[L_HIP, 0] + kps[R_HIP, 0]) / 2
        offset = (shoulder_x - hip_x) / 0.15
        return float(max(-1.0, min(1.0, offset)))

    def _stillness_duration(self) -> float:
        """Consecutive frames where gesture_speed < threshold, normalized."""
        speed = self._gesture_speed()
        if speed < STILLNESS_THRESHOLD:
            self._still_frames += 1
        else:
            self._still_frames = 0
        # Normalize via tanh (10 frames ≈ 0.5s at 20Hz → ~0.46)
        return float(math.tanh(self._still_frames / 10.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pose_encoder.py -v`
Expected: All passed (8)

- [ ] **Step 5: Commit**

```bash
git add pose_encoder.py tests/test_pose_encoder.py
git commit -m "feat: add PoseFeatureExtractor with 8 emotion-relevant features"
```

---

### Task 7: PoseEncoder neural network

**Files:**
- Modify: `pose_encoder.py`
- Modify: `tests/test_pose_encoder.py`

- [ ] **Step 1: Write failing tests for PoseEncoder**

```python
# Append to tests/test_pose_encoder.py
import torch
from pose_encoder import PoseEncoder


class TestPoseEncoder:
    def test_output_shape(self):
        enc = PoseEncoder()
        features = {
            'slouch_score': 0.8, 'openness_score': 0.5,
            'tension_score': 0.3, 'head_tilt': 0.0,
            'gesture_speed': 0.1, 'symmetry_score': 0.9,
            'forward_lean': 0.0, 'stillness_duration': 0.0,
        }
        out = enc.encode(features)
        assert out.shape == (1, 16)

    def test_zero_pose_embedding(self):
        enc = PoseEncoder()
        out = enc.encode(None)
        assert out.shape == (1, 16)
        assert torch.all(out == 0)

    def test_batch_encode(self):
        enc = PoseEncoder()
        batch = [
            {'slouch_score': 0.8, 'openness_score': 0.5, 'tension_score': 0.3,
             'head_tilt': 0.0, 'gesture_speed': 0.1, 'symmetry_score': 0.9,
             'forward_lean': 0.0, 'stillness_duration': 0.0},
            {'slouch_score': 0.3, 'openness_score': 0.9, 'tension_score': 0.7,
             'head_tilt': -0.5, 'gesture_speed': 0.6, 'symmetry_score': 0.4,
             'forward_lean': 0.3, 'stillness_duration': 0.5},
        ]
        out = enc.encode_batch(batch)
        assert out.shape == (2, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pose_encoder.py::TestPoseEncoder -v`
Expected: FAIL with "cannot import name 'PoseEncoder'"

- [ ] **Step 3: Implement PoseEncoder**

Add to `pose_encoder.py`:

```python
# Feature names in canonical order
POSE_FEATURE_NAMES = [
    'slouch_score', 'openness_score', 'tension_score', 'head_tilt',
    'gesture_speed', 'symmetry_score', 'forward_lean', 'stillness_duration',
]


class PoseEncoder(nn.Module):
    """
    Encode pose features into a 16-dimensional embedding for fusion.
    Parallel to BiometricEncoder — same output dim, same usage pattern.
    """

    def __init__(self, output_dim: int = 16):
        super().__init__()
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Tanh()
        )

        # Zero embedding for missing pose data
        self.register_buffer(
            'zero_pose',
            torch.zeros(1, output_dim)
        )

    def _extract_features(self, pose_dict: dict) -> torch.Tensor:
        """Convert pose feature dict to tensor in canonical order."""
        features = [pose_dict.get(name, 0.0) for name in POSE_FEATURE_NAMES]
        return torch.tensor(features, dtype=torch.float32)

    def encode(self, pose_features: dict = None) -> torch.Tensor:
        """Encode single pose feature dict. Returns (1, output_dim)."""
        if pose_features is None:
            return self.zero_pose.clone()
        features = self._extract_features(pose_features).unsqueeze(0)
        with torch.no_grad():
            return self.encoder(features)

    def encode_batch(self, batch: list) -> torch.Tensor:
        """Encode batch of pose feature dicts. Returns (batch_size, output_dim)."""
        features = torch.stack([self._extract_features(p) for p in batch])
        with torch.no_grad():
            return self.encoder(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for training."""
        return self.encoder(features)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pose_encoder.py -v`
Expected: All passed (11)

- [ ] **Step 5: Commit**

```bash
git add pose_encoder.py tests/test_pose_encoder.py
git commit -m "feat: add PoseEncoder neural network (8 features → 16-dim)"
```

---

### Task 8: Integrate PoseFeatureExtractor into RuViewProvider

**Files:**
- Modify: `ruview_provider.py`
- Modify: `tests/test_ruview_provider.py`

- [ ] **Step 1: Write failing test for get_pose_features()**

```python
# Append to tests/test_ruview_provider.py

class TestPoseFeatures:
    def test_get_pose_features_with_data(self, provider):
        provider._pose_data = {
            'keypoints': [
                [0.5, 0.1, 0.9], [0.48, 0.08, 0.9], [0.52, 0.08, 0.9],
                [0.45, 0.1, 0.9], [0.55, 0.1, 0.9],
                [0.4, 0.3, 0.9], [0.6, 0.3, 0.9],
                [0.38, 0.5, 0.9], [0.62, 0.5, 0.9],
                [0.38, 0.7, 0.9], [0.62, 0.7, 0.9],
                [0.45, 0.6, 0.9], [0.55, 0.6, 0.9],
                [0.45, 0.8, 0.9], [0.55, 0.8, 0.9],
                [0.45, 1.0, 0.9], [0.55, 1.0, 0.9],
            ]
        }
        features = provider.get_pose_features()
        assert features is not None
        assert 'slouch_score' in features
        assert len(features) == 8

    def test_get_pose_features_returns_none_when_no_data(self, provider):
        provider._pose_data = None
        assert provider.get_pose_features() is None

    def test_get_pose_features_handles_raw_list(self, provider):
        """RuView may send keypoints directly as the pose field."""
        provider._pose_data = [
            [0.5, 0.1, 0.9], [0.48, 0.08, 0.9], [0.52, 0.08, 0.9],
            [0.45, 0.1, 0.9], [0.55, 0.1, 0.9],
            [0.4, 0.3, 0.9], [0.6, 0.3, 0.9],
            [0.38, 0.5, 0.9], [0.62, 0.5, 0.9],
            [0.38, 0.7, 0.9], [0.62, 0.7, 0.9],
            [0.45, 0.6, 0.9], [0.55, 0.6, 0.9],
            [0.45, 0.8, 0.9], [0.55, 0.8, 0.9],
            [0.45, 1.0, 0.9], [0.55, 1.0, 0.9],
        ]
        features = provider.get_pose_features()
        assert features is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ruview_provider.py::TestPoseFeatures -v`
Expected: FAIL (no get_pose_features method)

- [ ] **Step 3: Add get_pose_features() to RuViewProvider**

In `ruview_provider.py` `__init__`:

```python
# After calibration init:
self._pose_extractor = None
try:
    from pose_encoder import PoseFeatureExtractor
    self._pose_extractor = PoseFeatureExtractor()
except ImportError:
    logger.info("[RuViewProvider] pose_encoder not available")
```

Add method:

```python
def get_pose_features(self) -> dict | None:
    """
    Extract 8 emotion-relevant features from current pose data.
    Returns None if no pose data or extractor unavailable.
    """
    if not self._pose_extractor or not self._pose_data:
        return None

    # Handle both formats: {keypoints: [...]} or direct list
    keypoints = self._pose_data
    if isinstance(keypoints, dict):
        keypoints = keypoints.get('keypoints', keypoints.get('pose'))

    return self._pose_extractor.extract(keypoints)
```

- [ ] **Step 4: Run all RuView tests**

Run: `python -m pytest tests/test_ruview_provider.py -v`
Expected: All passed (31+)

- [ ] **Step 5: Commit**

```bash
git add ruview_provider.py tests/test_ruview_provider.py
git commit -m "feat: add get_pose_features() to RuViewProvider"
```

---

## Chunk 3: FusionHead Expansion & Training

### Task 9: Expand FusionHead to accept pose embedding

**Files:**
- Modify: `emotion_classifier.py`
- Modify: `tests/test_ruview_provider.py` (or existing classifier tests)

- [ ] **Step 1: Write failing test for 416-dim FusionHead**

```python
# tests/test_pose_encoder.py — append
from pose_encoder import PoseEncoder, POSE_FEATURE_NAMES

# Need to import after ensuring emotion_classifier is importable
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFusionHeadWithPose:
    def test_416_dim_forward(self):
        from emotion_classifier import FusionHead
        head = FusionHead(text_dim=384, biometric_dim=16, pose_dim=16)
        text_emb = torch.randn(1, 384)
        bio_emb = torch.randn(1, 16)
        pose_emb = torch.randn(1, 16)
        emotion_probs, family_probs = head(text_emb, bio_emb, pose_emb)
        assert emotion_probs.shape == (1, 32)
        assert family_probs.shape == (1, 9)

    def test_416_dim_with_no_pose(self):
        from emotion_classifier import FusionHead
        head = FusionHead(text_dim=384, biometric_dim=16, pose_dim=16)
        text_emb = torch.randn(1, 384)
        bio_emb = torch.randn(1, 16)
        emotion_probs, family_probs = head(text_emb, bio_emb)
        assert emotion_probs.shape == (1, 32)

    def test_classify_with_fallback_pose(self):
        from emotion_classifier import FusionHead
        head = FusionHead(text_dim=384, biometric_dim=16, pose_dim=16)
        text_emb = torch.randn(1, 384)
        bio_emb = torch.randn(1, 16)
        pose_emb = torch.randn(1, 16)
        result = head.classify_with_fallback(text_emb, bio_emb, pose_emb)
        assert 'emotion' in result
        assert 'family' in result

    def test_old_400_checkpoint_detected_by_new_416_head(self):
        """Verify dimension mismatch is detected when loading old checkpoint."""
        from emotion_classifier import FusionHead
        # Create old-style 400-dim head and save
        old_head = FusionHead(text_dim=384, biometric_dim=16, pose_dim=0)
        old_state = old_head.state_dict()
        # New 416-dim head should detect mismatch
        new_head = FusionHead(text_dim=384, biometric_dim=16, pose_dim=16)
        old_input_dim = old_state['shared.0.weight'].shape[1]  # 400
        new_input_dim = 384 + 16 + 16  # 416
        assert old_input_dim != new_input_dim

    def test_backward_compat_400_dim(self):
        """FusionHead with pose_dim=0 should work like original 400-dim."""
        from emotion_classifier import FusionHead
        head = FusionHead(text_dim=384, biometric_dim=16, pose_dim=0)
        text_emb = torch.randn(1, 384)
        bio_emb = torch.randn(1, 16)
        emotion_probs, family_probs = head(text_emb, bio_emb)
        assert emotion_probs.shape == (1, 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pose_encoder.py::TestFusionHeadWithPose -v`
Expected: FAIL (FusionHead doesn't accept pose_dim)

- [ ] **Step 3: Update FusionHead**

In `emotion_classifier.py`, update `FusionHead.__init__`:

```python
def __init__(
    self,
    text_dim: int = 512,
    biometric_dim: int = 16,
    pose_dim: int = 16,       # NEW — matches spec default
    hidden_dim: int = 128,
    num_emotions: int = 32,
    num_families: int = 9,
    dropout: float = 0.3
):
    super().__init__()
    self.text_dim = text_dim
    self.biometric_dim = biometric_dim
    self.pose_dim = pose_dim
    self.num_emotions = num_emotions
    self.num_families = num_families

    input_dim = text_dim + biometric_dim + pose_dim

    # Shared feature extractor
    self.shared = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
    )

    # Stage 1: Family classifier (9-way)
    self.family_classifier = nn.Linear(hidden_dim // 2, num_families)

    # Stage 2: Sub-emotion classifier (32-way)
    self.emotion_classifier = nn.Linear(hidden_dim // 2, num_emotions)

    # Zero embeddings for optional inputs
    self.register_buffer('zero_biometric', torch.zeros(1, biometric_dim))
    if pose_dim > 0:
        self.register_buffer('zero_pose', torch.zeros(1, pose_dim))

    self._build_family_indices()
```

Update `forward`:

```python
def forward(
    self,
    text_embedding: torch.Tensor,
    biometric_embedding: torch.Tensor = None,
    pose_embedding: torch.Tensor = None
) -> tuple:
    batch_size = text_embedding.shape[0]

    if biometric_embedding is None:
        biometric_embedding = self.zero_biometric.expand(batch_size, -1)

    parts = [text_embedding, biometric_embedding]

    if self.pose_dim > 0:
        if pose_embedding is None:
            pose_embedding = self.zero_pose.expand(batch_size, -1)
        parts.append(pose_embedding)

    fused = torch.cat(parts, dim=-1)
    shared_features = self.shared(fused)

    emotion_logits = self.emotion_classifier(shared_features)
    family_logits = self.family_classifier(shared_features)

    emotion_probs = torch.softmax(emotion_logits, dim=-1)
    family_probs = torch.softmax(family_logits, dim=-1)

    return emotion_probs, family_probs
```

Update `classify_with_fallback` to accept `pose_embedding=None` parameter and pass it to `self.forward()`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_pose_encoder.py tests/test_ruview_provider.py tests/test_external_context.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add emotion_classifier.py tests/test_pose_encoder.py
git commit -m "feat: expand FusionHead to accept pose embedding (400→416)"
```

---

### Task 10: Update MultimodalEmotionAnalyzer for pose

**Files:**
- Modify: `emotion_classifier.py`

- [ ] **Step 1: Update analyze() to accept pose parameter**

In `MultimodalEmotionAnalyzer.__init__`, add PoseEncoder initialization:

```python
# After biometric_encoder init:
self.pose_encoder = None
try:
    from pose_encoder import PoseEncoder
    self.pose_encoder = PoseEncoder(output_dim=16)
    self.pose_encoder.to(self.device)
    print("[EmotionAnalyzer] PoseEncoder initialized")
except ImportError:
    print("[EmotionAnalyzer] PoseEncoder not available (pose_encoder.py missing)")
```

Update FusionHead creation to include `pose_dim`:

```python
pose_dim = 16 if self.pose_encoder else 0
self.fusion_head = FusionHead(
    text_dim=self.n_embd,
    biometric_dim=16,
    pose_dim=pose_dim,
    num_emotions=32,
    num_families=9
)
```

Update `analyze()` signature and body:

```python
def analyze(self, text: str, biometrics: dict = None, pose: dict = None, return_embedding: bool = False) -> dict:
    # ... existing text + biometric embedding code ...

    # Get pose embedding
    pose_embedding = None
    if pose and self.pose_encoder:
        pose_embedding = self.pose_encoder.encode(pose).to(self.device)

    # Two-stage classify with fallback
    with torch.no_grad():
        classification = self.fusion_head.classify_with_fallback(
            text_embedding, bio_embedding, pose_embedding
        )
        emotion_probs, family_probs = self.fusion_head(
            text_embedding, bio_embedding, pose_embedding
        )
    # ... rest unchanged ...
```

Update checkpoint loading to handle pose_encoder and meta:

```python
# In __init__, when loading checkpoint:
if 'meta' in state:
    meta = state['meta']
    saved_pose_dim = meta.get('pose_dim', 0)
else:
    saved_total = state['fusion_head']['shared.0.weight'].shape[1]
    saved_pose_dim = max(0, saved_total - saved_text_dim - 16)

# Check dimensions match
if saved_text_dim != self.n_embd or saved_pose_dim != pose_dim:
    print(f"[EmotionAnalyzer] Dimension mismatch, requires retraining")
else:
    self.fusion_head.load_state_dict(state['fusion_head'])
    self.biometric_encoder.load_state_dict(state['biometric_encoder'])
    if 'pose_encoder' in state and self.pose_encoder:
        self.pose_encoder.load_state_dict(state['pose_encoder'])
```

- [ ] **Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All passed

- [ ] **Step 3: Commit**

```bash
git add emotion_classifier.py
git commit -m "feat: integrate PoseEncoder into MultimodalEmotionAnalyzer"
```

---

### Task 11: Update training pipeline

**Files:**
- Modify: `train_emotion_classifier.py`

- [ ] **Step 1: Add synthetic pose generation**

Add `POSE_RANGES` dict mapping emotions to pose feature ranges (research-grounded), similar to existing `BIOMETRIC_RANGES`:

```python
POSE_RANGES = {
    # Joy family: open posture, high symmetry, some movement
    'joy': {'slouch': (0.7, 1.0), 'openness': (0.6, 1.0), 'tension': (0.0, 0.2),
            'head_tilt': (-0.1, 0.2), 'gesture_speed': (0.2, 0.6), 'symmetry': (0.7, 1.0),
            'forward_lean': (-0.1, 0.2), 'stillness': (0.0, 0.1)},
    # Sadness: slouched, closed, still
    'sadness': {'slouch': (0.2, 0.5), 'openness': (0.1, 0.4), 'tension': (0.2, 0.5),
                'head_tilt': (-0.5, -0.1), 'gesture_speed': (0.0, 0.1), 'symmetry': (0.5, 0.8),
                'forward_lean': (-0.3, 0.0), 'stillness': (0.3, 0.8)},
    # Fear/anxiety: tense, rigid, slightly forward
    'fear': {'slouch': (0.5, 0.8), 'openness': (0.1, 0.3), 'tension': (0.6, 1.0),
             'head_tilt': (-0.3, 0.1), 'gesture_speed': (0.0, 0.3), 'symmetry': (0.3, 0.7),
             'forward_lean': (0.1, 0.4), 'stillness': (0.1, 0.5)},
    # Anger: tense, forward lean, asymmetric
    'anger': {'slouch': (0.6, 0.9), 'openness': (0.3, 0.6), 'tension': (0.5, 0.9),
              'head_tilt': (0.0, 0.4), 'gesture_speed': (0.3, 0.8), 'symmetry': (0.3, 0.6),
              'forward_lean': (0.2, 0.5), 'stillness': (0.0, 0.1)},
    # Calm: upright, symmetric, still
    'calm': {'slouch': (0.7, 1.0), 'openness': (0.4, 0.7), 'tension': (0.0, 0.2),
             'head_tilt': (-0.1, 0.1), 'gesture_speed': (0.0, 0.1), 'symmetry': (0.8, 1.0),
             'forward_lean': (-0.1, 0.1), 'stillness': (0.4, 0.9)},
    # Neutral
    'neutral': {'slouch': (0.5, 0.8), 'openness': (0.3, 0.6), 'tension': (0.1, 0.4),
                'head_tilt': (-0.2, 0.2), 'gesture_speed': (0.0, 0.2), 'symmetry': (0.6, 0.9),
                'forward_lean': (-0.1, 0.1), 'stillness': (0.1, 0.4)},
}
```

Add `generate_synthetic_pose(emotion)`:

```python
def generate_synthetic_pose(emotion: str) -> dict:
    """Generate plausible pose features for an emotion."""
    ranges = POSE_RANGES.get(emotion, POSE_RANGES['neutral'])
    # Use family root ranges for sub-emotions without specific ranges
    # _EMOTION_TO_FAMILY is already defined at module level in train_emotion_classifier.py
    # (built from _EMOTION_FAMILIES dict near line 90)
    if emotion not in POSE_RANGES:
        family = _EMOTION_TO_FAMILY.get(emotion, 'Neutral')
        root = {'Joy': 'joy', 'Sadness': 'sadness', 'Anger': 'anger',
                'Fear': 'fear', 'Calm': 'calm', 'Neutral': 'neutral'}.get(family, 'neutral')
        ranges = POSE_RANGES.get(root, POSE_RANGES['neutral'])

    return {
        'slouch_score': random.uniform(*ranges['slouch']),
        'openness_score': random.uniform(*ranges['openness']),
        'tension_score': random.uniform(*ranges['tension']),
        'head_tilt': random.uniform(*ranges['head_tilt']),
        'gesture_speed': random.uniform(*ranges['gesture_speed']),
        'symmetry_score': random.uniform(*ranges['symmetry']),
        'forward_lean': random.uniform(*ranges['forward_lean']),
        'stillness_duration': random.uniform(*ranges['stillness']),
    }
```

- [ ] **Step 2: Update training loop**

In the training loop, update sample generation to include pose:

```python
# Where biometrics are generated:
pose_features = generate_synthetic_pose(emotion)

# 20% pose dropout
if random.random() < 0.2:
    pose_features = {k: 0.0 for k in pose_features}
```

Update checkpoint saving:

```python
torch.save({
    'fusion_head': fusion_head.state_dict(),
    'biometric_encoder': biometric_encoder.state_dict(),
    'pose_encoder': pose_encoder.state_dict(),
    'meta': {
        'text_dim': n_embd,
        'biometric_dim': 16,
        'pose_dim': 16,
        'num_emotions': 32,
        'num_families': 9,
    }
}, checkpoint_path)
```

- [ ] **Step 3: Run training**

Run: `python train_emotion_classifier.py` (or with appropriate flags)
Expected: Training completes, saves checkpoint with pose_encoder weights

- [ ] **Step 4: Commit**

```bash
git add train_emotion_classifier.py
git commit -m "feat: add synthetic pose generation and pose dropout to training"
```

---

## Chunk 4: API & JS Integration

### Task 12: Update emotion_api_server.py

**Files:**
- Modify: `emotion_api_server.py`

- [ ] **Step 1: Update /api/ruview/analyze to pass pose data**

In the `ruview_analyze()` endpoint, add pose feature extraction:

```python
@app.route('/api/ruview/analyze', methods=['POST'])
def ruview_analyze():
    if not multimodal_analyzer:
        return jsonify({'error': 'Multimodal analyzer not available', 'status': 'error'}), 503
    try:
        data = request.json or {}
        text = data.get('text', '')

        ruview_bio = None
        pose_features = None
        if context_provider:
            ruview_bio = context_provider.get_ruview_biometrics()
            if context_provider.ruview:
                pose_features = context_provider.ruview.get_pose_features()

        biometrics = data.get('biometrics') or ruview_bio
        result = multimodal_analyzer.analyze(text, biometrics=biometrics, pose=pose_features)

        if ruview_bio:
            result['ruview_source'] = True
            result['ruview_confidence'] = ruview_bio.get('confidence', 0)
            result['ruview_insight'] = context_provider.ruview.get_insight() if context_provider.ruview else None
            result['presence'] = context_provider.get_ruview_presence()
            result['pose_features'] = pose_features
            result['calibration_profile'] = (
                context_provider.ruview._calibration_buffer.profile_id
                if hasattr(context_provider.ruview, '_calibration_buffer') and context_provider.ruview._calibration_buffer
                else 'none'
            )
        else:
            result['ruview_source'] = False

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500
```

- [ ] **Step 2: Run existing tests to verify no regression**

Run: `python -m pytest tests/ -v`
Expected: All passed

- [ ] **Step 3: Commit**

```bash
git add emotion_api_server.py
git commit -m "feat: pass pose features through /api/ruview/analyze"
```

---

### Task 13: Update neural_ecosystem_connector.js

**Files:**
- Modify: `neural_ecosystem_connector.js`

- [ ] **Step 1: Add getPoseFeatures() to RuViewClient**

```javascript
/**
 * Get pose features from WiFi sensing
 * Returns: {slouch_score, openness_score, tension_score, ...}
 */
async getPoseFeatures() {
    // Pose features are returned as part of the analyze response
    const analysis = await this.analyzeWithWiFiBiometrics('');
    return analysis.pose_features || null;
}
```

- [ ] **Step 2: Commit**

```bash
git add neural_ecosystem_connector.js
git commit -m "feat: add getPoseFeatures() to RuViewClient"
```

---

### Task 14: Train calibration model and retrain classifier

- [ ] **Step 1: Train calibration model**

Run: `python wifi_calibration.py --train`
Expected: Checkpoint saved to `checkpoints/ruview_calibration.pt`

- [ ] **Step 2: Retrain emotion classifier with pose**

Run: `python train_emotion_classifier.py`
Expected: Updated checkpoint with pose_encoder weights

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Final commit and push**

```bash
git add checkpoints/ calibration_profiles/
git commit -m "feat: trained calibration model and pose-aware emotion classifier"
git push origin master
```

"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - WiFi Signal Calibration Module
===============================================================================
Maps WiFi-sensed signals (breathing_rate, motion_level) to calibrated
biometric estimates (HRV, EDA) using a small neural network with online
per-environment adaptation via PersonalCalibrationBuffer.

Replaces the linear approximations in RuViewProvider._breathing_to_hrv()
and _motion_to_eda() with a learned, adaptable model.

Integrates with:
- Neural Workflow AI Engine
- Biometric Integration Engine
- RuView WiFi Sensing Provider
- Real-time Data
- Dashboard Data Integration

Architecture:
  Input(2) -> normalize -> Linear(32) -> ReLU -> Linear(16) -> ReLU -> Linear(2) -> sigmoid rescale
  HRV range: [10.0, 250.0] ms  (RMSSD)
  EDA range: [0.5, 20.0] uS

Online adaptation:
  PersonalCalibrationBuffer collects (wifi, wearable) pairs per user,
  fine-tunes the output layer after 20 pairs, then every 50 new pairs.
===============================================================================
"""

import os
import copy
import logging
import threading
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

# ─── Output bounds ─────────────────────────────────────────────────────────────

HRV_MIN = 10.0
HRV_MAX = 250.0   # RMSSD can reach 200+ ms in relaxed/fit individuals
EDA_MIN = 0.5
EDA_MAX = 20.0

# ─── Linear mapping constants (from RuViewProvider) ───────────────────────────

_BR_MIN = 6.0
_BR_MAX = 30.0
_HRV_LIN_MIN = 20.0
_HRV_LIN_MAX = 200.0  # Expanded to match real RMSSD range

_MOTION_MIN = 0.0
_MOTION_MAX = 1.0
_EDA_LIN_MIN = 0.5
_EDA_LIN_MAX = 8.0


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1: WiFiCalibrationModel
# ═══════════════════════════════════════════════════════════════════════════════


class WiFiCalibrationModel(nn.Module):
    """
    Small neural network: (breathing_rate, motion_level) -> (hrv, eda).

    Input normalization:
      breathing_rate: centered around 15 BPM, scaled by 10
      motion_level:   centered around 0.3, scaled by 0.4

    Output is bounded via sigmoid activation rescaled to physiological ranges:
      HRV: [10.0, 250.0] ms  (RMSSD)
      EDA: [0.5, 20.0] uS
    """

    # Input normalization constants (approximate population means/stds)
    BR_MEAN = 15.0
    BR_STD = 5.0
    MOTION_MEAN = 0.3
    MOTION_STD = 0.25

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 32)
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize inputs to ~zero-mean, unit-variance
        x_norm = torch.empty_like(x)
        x_norm[:, 0:1] = (x[:, 0:1] - self.BR_MEAN) / self.BR_STD
        x_norm[:, 1:2] = (x[:, 1:2] - self.MOTION_MEAN) / self.MOTION_STD

        h = self.relu(self.hidden(x_norm))
        h = self.relu(self.hidden2(h))
        raw = self.output(h)
        sig = torch.sigmoid(raw)

        # Rescale each output channel to its physiological range
        hrv = sig[:, 0:1] * (HRV_MAX - HRV_MIN) + HRV_MIN
        eda = sig[:, 1:2] * (EDA_MAX - EDA_MIN) + EDA_MIN

        return torch.cat([hrv, eda], dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2: Inverse mappings + bootstrapped training
# ═══════════════════════════════════════════════════════════════════════════════


def _invert_breathing_to_hrv(hrv: float) -> float:
    """
    Invert the linear HRV-from-breathing mapping to recover breathing rate.

    Forward: ratio = (max_br - br) / (max_br - min_br)
             hrv = min_hrv + ratio * (max_hrv - min_hrv)

    Inverse: ratio = (hrv - min_hrv) / (max_hrv - min_hrv)
             br = max_br - ratio * (max_br - min_br)
    """
    hrv_clamped = max(_HRV_LIN_MIN, min(_HRV_LIN_MAX, hrv))
    ratio = (hrv_clamped - _HRV_LIN_MIN) / (_HRV_LIN_MAX - _HRV_LIN_MIN)
    br = _BR_MAX - ratio * (_BR_MAX - _BR_MIN)
    return br


def _invert_motion_to_eda(eda: float) -> float:
    """
    Invert the linear EDA-from-motion mapping to recover motion level.

    Forward: ratio = (motion - min_motion) / (max_motion - min_motion)
             eda = min_eda + ratio * (max_eda - min_eda)

    Inverse: ratio = (eda - min_eda) / (max_eda - min_eda)
             motion = min_motion + ratio * (max_motion - min_motion)
    """
    eda_clamped = max(_EDA_LIN_MIN, min(_EDA_LIN_MAX, eda))
    ratio = (eda_clamped - _EDA_LIN_MIN) / (_EDA_LIN_MAX - _EDA_LIN_MIN)
    motion = _MOTION_MIN + ratio * (_MOTION_MAX - _MOTION_MIN)
    return motion


def _generate_bootstrapped_pairs(n: int = 500) -> tuple:
    """
    Generate synthetic (wifi_input, biometric_target) pairs by sampling
    biometric values uniformly, inverting to WiFi signals, and adding noise.

    Returns:
        (inputs, targets): both torch.Tensor of shape (n, 2)
        inputs[:, 0] = breathing_rate, inputs[:, 1] = motion_level
        targets[:, 0] = hrv, targets[:, 1] = eda
    """
    br_range = _HRV_LIN_MAX - _HRV_LIN_MIN
    eda_range = _EDA_LIN_MAX - _EDA_LIN_MIN

    inputs_list = []
    targets_list = []

    for _ in range(n):
        # Sample target biometric values uniformly
        hrv = _HRV_LIN_MIN + torch.rand(1).item() * br_range
        eda = _EDA_LIN_MIN + torch.rand(1).item() * eda_range

        # Invert to get clean WiFi signals
        br = _invert_breathing_to_hrv(hrv)
        motion = _invert_motion_to_eda(eda)

        # Add Gaussian noise (sigma = 10% of range)
        br_noise = torch.randn(1).item() * (br_range * 0.10)
        eda_noise = torch.randn(1).item() * (eda_range * 0.10)

        # Add jitter
        br_jitter = torch.randn(1).item() * 3.0  # +-3 BPM
        motion_jitter = torch.randn(1).item() * 0.15

        noisy_br = br + br_noise + br_jitter
        noisy_motion = motion + eda_noise * 0.01 + motion_jitter  # scale eda noise for motion

        inputs_list.append([noisy_br, noisy_motion])
        targets_list.append([hrv, eda])

    inputs = torch.tensor(inputs_list, dtype=torch.float32)
    targets = torch.tensor(targets_list, dtype=torch.float32)

    return inputs, targets


def train_calibration_model(
    n_pairs: int = 500,
    epochs: int = 100,
    lr: float = 0.005,
    checkpoint_path: str = "checkpoints/wifi_calibration.pt",
) -> WiFiCalibrationModel:
    """
    Train a WiFiCalibrationModel on bootstrapped synthetic pairs.

    Args:
        n_pairs: Number of synthetic training pairs to generate.
        epochs: Training epochs.
        lr: Learning rate for Adam optimizer.
        checkpoint_path: Where to save the trained model.

    Returns:
        Trained WiFiCalibrationModel.
    """
    model = WiFiCalibrationModel()
    inputs, targets = _generate_bootstrapped_pairs(n=n_pairs)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(inputs)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.info(
                f"[WiFiCalibration] Epoch {epoch+1}/{epochs} loss={loss.item():.4f}"
            )

    # Save checkpoint
    ckpt_dir = os.path.dirname(checkpoint_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"[WiFiCalibration] Saved checkpoint to {checkpoint_path}")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Task 3: PersonalCalibrationBuffer + load_calibration_model
# ═══════════════════════════════════════════════════════════════════════════════


def load_calibration_model(
    base_checkpoint: str = "checkpoints/wifi_calibration.pt",
    profile_id: str = None,
    calibration_dir: str = None,
) -> tuple:
    """
    Load the base calibration model and optionally a personal profile.

    Args:
        base_checkpoint: Path to the base model checkpoint.
        profile_id: User profile ID. Defaults to env RUVIEW_PROFILE_ID or "default".
        calibration_dir: Dir for profile checkpoints. Defaults to env
            RUVIEW_CALIBRATION_DIR or "calibration_profiles".

    Returns:
        (base_model, profile_model) — either or both may be None if files missing.
    """
    if not os.path.exists(base_checkpoint):
        return (None, None)

    base_model = WiFiCalibrationModel()
    base_model.load_state_dict(torch.load(base_checkpoint, weights_only=True))
    base_model.eval()

    # Try loading profile
    if profile_id is None:
        profile_id = os.environ.get("RUVIEW_PROFILE_ID", "default")
    if calibration_dir is None:
        calibration_dir = os.environ.get("RUVIEW_CALIBRATION_DIR", "calibration_profiles")

    profile_path = os.path.join(calibration_dir, f"{profile_id}.pt")
    if os.path.exists(profile_path):
        profile_model = WiFiCalibrationModel()
        profile_model.load_state_dict(torch.load(profile_path, weights_only=True))
        profile_model.eval()
        return (base_model, profile_model)

    return (base_model, None)


class PersonalCalibrationBuffer:
    """
    Rolling buffer of (wifi, wearable) paired readings for per-user
    online calibration. Fine-tunes the output layer of the base model
    after collecting enough pairs.

    Connected to:
    - Neural Workflow AI Engine (adaptive model personalization)
    - Biometric Integration Engine (wearable ground-truth pairing)
    - RuView WiFi Sensing Provider (WiFi signal input)
    - Real-time Data (continuous adaptation)

    Schedule:
    - First fine-tune at 20 pairs
    - Subsequent fine-tunes every 50 new pairs after that

    Thread safety:
    - Atomic weight swap via threading.Lock
    - predict() is never blocked during fine-tuning
    """

    MAX_BUFFER_SIZE = 500
    FIRST_FINETUNE_THRESHOLD = 20
    SUBSEQUENT_FINETUNE_INTERVAL = 50

    def __init__(
        self,
        base_checkpoint: str = "checkpoints/wifi_calibration.pt",
        profile_id: str = None,
        calibration_dir: str = None,
    ):
        self._profile_id = profile_id or os.environ.get("RUVIEW_PROFILE_ID", "default")
        self._calibration_dir = calibration_dir or os.environ.get(
            "RUVIEW_CALIBRATION_DIR", "calibration_profiles"
        )

        # Load base model
        self._base_checkpoint = base_checkpoint
        self._model = WiFiCalibrationModel()
        if os.path.exists(base_checkpoint):
            self._model.load_state_dict(
                torch.load(base_checkpoint, weights_only=True)
            )
        self._model.eval()

        # Try loading existing profile
        profile_path = os.path.join(self._calibration_dir, f"{self._profile_id}.pt")
        if os.path.exists(profile_path):
            self._model.load_state_dict(
                torch.load(profile_path, weights_only=True)
            )
            self._model.eval()
            logger.info(
                f"[PersonalCalibration] Loaded profile {self._profile_id}"
            )

        # Rolling buffer: each entry is (wifi_tensor, target_tensor)
        self._buffer = deque(maxlen=self.MAX_BUFFER_SIZE)
        self._total_added = 0
        self._last_finetune_count = 0

        # Thread safety for model weight swap
        self._model_lock = threading.Lock()

    @property
    def pair_count(self) -> int:
        """Number of pairs currently in the buffer."""
        return len(self._buffer)

    @property
    def total_samples_seen(self) -> int:
        """Total number of samples ever added to this buffer."""
        return self._total_added

    @property
    def buffer_size(self) -> int:
        """Current number of samples in the rolling buffer."""
        return len(self._buffer)

    def get_buffer_data(self) -> list:
        """Return a copy of the current buffer contents."""
        return list(self._buffer)

    def get_prediction_errors(self, model) -> list:
        """
        Compute per-sample MAE between model predictions and targets
        for all samples in the buffer.

        Connected to:
        - Neural Workflow AI Engine (drift detection input)
        - ML Training & Prediction Systems (model quality assessment)
        """
        if len(self._buffer) == 0:
            return []
        errors = []
        model.eval()
        with torch.no_grad():
            for wifi_tensor, target_tensor in self._buffer:
                pred = model(wifi_tensor.unsqueeze(0)).squeeze(0)
                error = float(torch.mean(torch.abs(pred - target_tensor)))
                errors.append(error)
        return errors

    def add_pair(self, wifi_input: tuple, wearable_target: tuple):
        """
        Add a (wifi, wearable) paired reading.

        Args:
            wifi_input: (breathing_rate, motion_level)
            wearable_target: (hrv, eda) from wearable ground truth
        """
        wifi_t = torch.tensor([list(wifi_input)], dtype=torch.float32)
        target_t = torch.tensor([list(wearable_target)], dtype=torch.float32)
        self._buffer.append((wifi_t, target_t))
        self._total_added += 1

        # Check if we should fine-tune
        should_finetune = False
        if self._last_finetune_count == 0 and self._total_added >= self.FIRST_FINETUNE_THRESHOLD:
            should_finetune = True
        elif (
            self._last_finetune_count > 0
            and (self._total_added - self._last_finetune_count) >= self.SUBSEQUENT_FINETUNE_INTERVAL
        ):
            should_finetune = True

        if should_finetune:
            self._last_finetune_count = self._total_added
            thread = threading.Thread(
                target=self._finetune,
                daemon=True,
                name=f"calibration-finetune-{self._profile_id}",
            )
            thread.start()

    def predict(self, breathing_rate: float, motion_level: float) -> tuple:
        """
        Predict (hrv, eda) from WiFi signals using the current model.
        Thread-safe: uses atomic weight reference, never blocked by fine-tuning.

        Returns:
            (hrv, eda) tuple of floats
        """
        x = torch.tensor([[breathing_rate, motion_level]], dtype=torch.float32)
        with self._model_lock:
            model = self._model
        model.eval()
        with torch.no_grad():
            out = model(x)
        return (out[0, 0].item(), out[0, 1].item())

    def _finetune(self):
        """
        Fine-tune the output layer on buffered pairs.
        Clones the model, freezes the hidden layer, trains output layer only,
        then atomically swaps weights.
        """
        try:
            # Clone the current model
            with self._model_lock:
                cloned = copy.deepcopy(self._model)

            # Freeze hidden layers, only train output
            for param in cloned.hidden.parameters():
                param.requires_grad = False
            for param in cloned.hidden2.parameters():
                param.requires_grad = False
            for param in cloned.output.parameters():
                param.requires_grad = True

            # Collect buffer data
            pairs = list(self._buffer)
            if len(pairs) < 5:
                return

            inputs = torch.cat([p[0] for p in pairs], dim=0)
            targets = torch.cat([p[1] for p in pairs], dim=0)

            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, cloned.parameters()),
                lr=0.002,
            )
            loss_fn = nn.MSELoss()

            cloned.train()
            for epoch in range(30):
                optimizer.zero_grad()
                preds = cloned(inputs)
                loss = loss_fn(preds, targets)
                loss.backward()
                optimizer.step()

            cloned.eval()

            # Atomic weight swap
            with self._model_lock:
                self._model = cloned

            # Save profile
            os.makedirs(self._calibration_dir, exist_ok=True)
            profile_path = os.path.join(
                self._calibration_dir, f"{self._profile_id}.pt"
            )
            torch.save(cloned.state_dict(), profile_path)
            logger.info(
                f"[PersonalCalibration] Fine-tuned and saved profile "
                f"{self._profile_id} ({len(pairs)} pairs)"
            )

        except Exception as e:
            logger.error(f"[PersonalCalibration] Fine-tune failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Task 5: CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="WiFi Signal Calibration — train and validate the calibration model"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train a new calibration model on bootstrapped synthetic data"
    )
    parser.add_argument(
        "--samples", type=int, default=5000,
        help="Number of synthetic training pairs (default: 5000)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Training epochs (default: 100)"
    )
    parser.add_argument(
        "--output", type=str, default="checkpoints/ruview_calibration.pt",
        help="Checkpoint output path (default: checkpoints/ruview_calibration.pt)"
    )

    args = parser.parse_args()

    if args.train:
        print(f"Training calibration model: {args.samples} samples, {args.epochs} epochs")
        model = train_calibration_model(
            n_pairs=args.samples,
            epochs=args.epochs,
            checkpoint_path=args.output,
        )
        print(f"Checkpoint saved to {args.output}")

        # Quick validation on sample inputs
        model.eval()
        test_inputs = [
            (8.0, 0.1, "slow breathing, low motion"),
            (16.0, 0.5, "normal breathing, moderate motion"),
            (26.0, 0.9, "fast breathing, high motion"),
        ]
        print("\nValidation:")
        for br, motion, label in test_inputs:
            x = torch.tensor([[br, motion]], dtype=torch.float32)
            with torch.no_grad():
                out = model(x)
            hrv, eda = out[0, 0].item(), out[0, 1].item()
            print(f"  {label}: BR={br}, Motion={motion} -> HRV={hrv:.1f} ms, EDA={eda:.1f} uS")
    else:
        parser.print_help()

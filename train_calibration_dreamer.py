"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Combined WESAD + DREAMER Calibration Training
===============================================================================
Extracts HR, HRV from DREAMER's 2-channel ECG (23 subjects, 18 trials each,
256 Hz) with valence/arousal labels, combines with WESAD data, and trains
an improved WiFi calibration model on a larger, more diverse dataset.

DREAMER provides:
- 2-channel ECG at 256 Hz → HR, HRV
- Valence (1-5), Arousal (1-5), Dominance (1-5) labels per trial
- 23 subjects × 18 film clips = 414 trials

Combined with WESAD (15 subjects, baseline/stress/amusement/meditation),
this gives a much richer training signal for the calibration model.

Usage:
  python train_calibration_dreamer.py
  python train_calibration_dreamer.py --epochs 300

Integrates with:
- Neural Workflow AI Engine
- Biometric Integration Engine
- RuView WiFi Sensing Provider
- Real-time Data
===============================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy import io as sio
from scipy import signal as scipy_signal
from scipy.stats import iqr
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wifi_calibration import WiFiCalibrationModel, HRV_MIN, HRV_MAX, EDA_MIN, EDA_MAX

logger = logging.getLogger(__name__)

DREAMER_PATH = Path('/Users/bel/quantara-nanoGPT/data/dreamer/DREAMER.mat')
ECG_FS = 256  # DREAMER ECG sampling rate


def extract_hr_hrv_from_ecg(ecg: np.ndarray, fs: int = 256) -> tuple:
    """
    Extract HR and HRV (RMSSD) from ECG signal.
    Uses R-peak detection on bandpass-filtered ECG.
    Returns (hr_bpm, hrv_rmssd_ms) or (None, None).
    """
    if len(ecg) < fs * 5:
        return None, None

    ecg = ecg.flatten()

    # Bandpass 5-15 Hz to isolate QRS complex
    nyq = fs / 2
    b, a = scipy_signal.butter(3, [5.0 / nyq, 15.0 / nyq], btype='band')
    filtered = scipy_signal.filtfilt(b, a, ecg)

    # Square to emphasize R-peaks
    squared = filtered ** 2

    # Find R-peaks
    min_distance = int(fs * 0.33)  # 180 BPM max
    height_thresh = np.percentile(squared, 70)
    peaks, _ = scipy_signal.find_peaks(squared, distance=min_distance, height=height_thresh)

    if len(peaks) < 3:
        return None, None

    # RR intervals in ms
    rr = np.diff(peaks) / fs * 1000
    rr = rr[(rr > 300) & (rr < 1500)]  # 40-200 BPM

    if len(rr) < 2:
        return None, None

    # Remove outliers
    rr_med = np.median(rr)
    rr_q = iqr(rr)
    rr = rr[(rr > rr_med - 2 * rr_q) & (rr < rr_med + 2 * rr_q)]

    if len(rr) < 2:
        return None, None

    hr = float(np.clip(60000.0 / np.mean(rr), 40, 180))
    hrv_rmssd = float(np.clip(np.sqrt(np.mean(np.diff(rr) ** 2)), 5, 250))

    return hr, hrv_rmssd


def extract_dreamer_data() -> list:
    """
    Extract HR, HRV, and emotion labels from all DREAMER subjects.
    Returns list of dicts with {hr, hrv, valence, arousal, dominance}.
    """
    if not DREAMER_PATH.exists():
        print(f"  DREAMER.mat not found at {DREAMER_PATH}")
        return []

    print(f"  Loading DREAMER.mat ({DREAMER_PATH.stat().st_size / 1024 / 1024:.0f} MB)...")
    data = sio.loadmat(str(DREAMER_PATH), simplify_cells=True)
    dreamer = data['DREAMER']
    subjects = dreamer['Data']

    print(f"  {len(subjects)} subjects, {dreamer['noOfVideoSequences']} trials each")

    all_windows = []

    for si, subject in enumerate(subjects):
        ecg_data = subject['ECG']
        valence_scores = subject['ScoreValence']
        arousal_scores = subject['ScoreArousal']
        dominance_scores = subject['ScoreDominance']

        stimuli = ecg_data['stimuli']
        trial_count = 0

        for ti in range(len(stimuli)):
            trial_ecg = stimuli[ti]  # (samples, 2) — 2-channel ECG

            if trial_ecg.ndim != 2 or trial_ecg.shape[1] < 2:
                continue

            # Use channel 0 (left ECG lead)
            ecg_ch = trial_ecg[:, 0].astype(np.float64)

            # Process in 30-second windows
            window_samples = 30 * ECG_FS
            step_samples = 15 * ECG_FS  # 50% overlap

            for start in range(0, len(ecg_ch) - window_samples, step_samples):
                window = ecg_ch[start:start + window_samples]
                hr, hrv = extract_hr_hrv_from_ecg(window, ECG_FS)

                if hr is None or hrv is None:
                    continue

                valence = int(valence_scores[ti]) if ti < len(valence_scores) else 3
                arousal = int(arousal_scores[ti]) if ti < len(arousal_scores) else 3

                # Map valence/arousal to approximate emotion state for EDA estimation
                # High arousal → higher EDA (sympathetic activation)
                # Low valence + high arousal → stress → highest EDA
                eda_estimate = 1.0 + (arousal - 1) * 1.5  # 1-7 µS based on arousal
                if valence <= 2 and arousal >= 4:
                    eda_estimate += 2.0  # Stress boost

                # Estimate breathing rate from HR/HRV relationship
                # Higher HRV (relaxation) → slower breathing
                # Lower HRV (stress) → faster breathing
                br_estimate = 20.0 - (hrv - 20) / (200 - 20) * 14.0  # Maps HRV 20-200 → BR 20-6
                br_estimate = float(np.clip(br_estimate, 6, 28))

                # Estimate motion from arousal (higher arousal → more motion)
                motion_estimate = (arousal - 1) / 4.0 * 0.5  # 0-0.5 range (seated film viewing)

                all_windows.append({
                    'hr': hr,
                    'hrv': hrv,
                    'eda': float(np.clip(eda_estimate, EDA_MIN, EDA_MAX)),
                    'breathing_rate': br_estimate,
                    'motion_level': float(motion_estimate),
                    'valence': valence,
                    'arousal': arousal,
                    'source': 'dreamer',
                })
                trial_count += 1

        print(f"    S{si+1:02d}: {trial_count} windows")

    return all_windows


def load_wesad_windows() -> list:
    """Load WESAD windows if train_calibration_wesad.py exists and WESAD is available."""
    try:
        from train_calibration_wesad import find_wesad_path, build_training_data
        wesad_path = find_wesad_path()
        if wesad_path:
            print(f"\n  Loading WESAD data...")
            X, Y = build_training_data(wesad_path)
            if X is not None:
                windows = []
                for i in range(len(X)):
                    windows.append({
                        'breathing_rate': float(X[i, 0]),
                        'motion_level': float(X[i, 1]),
                        'hrv': float(Y[i, 0]),
                        'eda': float(Y[i, 1]),
                        'source': 'wesad',
                    })
                return windows
    except Exception as e:
        print(f"  Could not load WESAD: {e}")
    return []


def train_combined(epochs: int = 300, lr: float = 0.003, output: str = 'checkpoints/ruview_calibration.pt'):
    """Train calibration model on combined DREAMER + WESAD data."""
    print("=" * 60)
    print("  COMBINED WESAD + DREAMER CALIBRATION TRAINING")
    print("=" * 60)

    # Extract DREAMER
    print(f"\n  Extracting DREAMER features...")
    dreamer_windows = extract_dreamer_data()
    print(f"  DREAMER: {len(dreamer_windows)} windows")

    # Load WESAD
    wesad_windows = load_wesad_windows()
    print(f"  WESAD: {len(wesad_windows)} windows")

    # Combine
    all_windows = dreamer_windows + wesad_windows
    if len(all_windows) < 50:
        print("  ERROR: Not enough data.")
        sys.exit(1)

    # Build tensors
    inputs = []
    targets = []
    for w in all_windows:
        inputs.append([w['breathing_rate'], w['motion_level']])
        targets.append([
            np.clip(w['hrv'], HRV_MIN, HRV_MAX),
            np.clip(w['eda'], EDA_MIN, EDA_MAX),
        ])

    X = torch.tensor(np.array(inputs, dtype=np.float32))
    Y = torch.tensor(np.array(targets, dtype=np.float32))

    print(f"\n  Combined dataset: {len(X)} samples")
    print(f"    DREAMER: {len(dreamer_windows)} ({len(dreamer_windows)/len(X)*100:.0f}%)")
    print(f"    WESAD:   {len(wesad_windows)} ({len(wesad_windows)/len(X)*100:.0f}%)")
    print(f"  Input range: BR=[{X[:,0].min():.1f}, {X[:,0].max():.1f}], Motion=[{X[:,1].min():.2f}, {X[:,1].max():.2f}]")
    print(f"  Target range: HRV=[{Y[:,0].min():.1f}, {Y[:,0].max():.1f}], EDA=[{Y[:,1].min():.1f}, {Y[:,1].max():.1f}]")

    # Augment with noise
    rng = np.random.default_rng(42)
    n = len(X)
    noise = torch.tensor(rng.normal(0, [1.5, 0.08], (n, 2)), dtype=torch.float32)
    X_aug = torch.cat([X, X + noise], dim=0)
    Y_aug = torch.cat([Y, Y], dim=0)
    perm = torch.randperm(len(X_aug))
    X_aug, Y_aug = X_aug[perm], Y_aug[perm]

    # Train/val split
    split = int(len(X_aug) * 0.9)
    X_train, X_val = X_aug[:split], X_aug[split:]
    Y_train, Y_val = Y_aug[:split], Y_aug[split:]

    # Weighted loss
    hrv_std = Y_train[:, 0].std()
    eda_std = Y_train[:, 1].std()
    hrv_w = 1.0 / (hrv_std ** 2 + 1e-6)
    eda_w = 1.0 / (eda_std ** 2 + 1e-6)
    total = hrv_w + eda_w
    hrv_w, eda_w = 2.0 * hrv_w / total, 2.0 * eda_w / total
    print(f"  Loss weights: HRV={hrv_w:.4f}, EDA={eda_w:.4f}")

    # Train
    model = WiFiCalibrationModel()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    print(f"\n  Training for {epochs} epochs...")
    best_val_loss = float('inf')
    best_state = None

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = ((pred[:, 0] - Y_train[:, 0]) ** 2).mean() * hrv_w + \
               ((pred[:, 1] - Y_train[:, 1]) ** 2).mean() * eda_w
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = ((val_pred[:, 0] - Y_val[:, 0]) ** 2).mean() * hrv_w + \
                       ((val_pred[:, 1] - Y_val[:, 1]) ** 2).mean() * eda_w

        scheduler.step(val_loss)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    Epoch {epoch+1}/{epochs} | train={loss.item():.4f} | val={val_loss.item():.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    torch.save(model.state_dict(), output)

    # Metrics
    with torch.no_grad():
        val_pred = model(X_val)
        hrv_mae = (val_pred[:, 0] - Y_val[:, 0]).abs().mean().item()
        eda_mae = (val_pred[:, 1] - Y_val[:, 1]).abs().mean().item()

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  HRV MAE: {hrv_mae:.1f} ms")
    print(f"  EDA MAE: {eda_mae:.1f} µS")
    print(f"  Checkpoint: {output}")

    # Spot checks
    print(f"\n  Spot checks:")
    tests = [
        ([8.0, 0.05], "Slow breathing, still → high HRV, low EDA"),
        ([14.0, 0.3], "Normal → moderate"),
        ([24.0, 0.7], "Fast breathing, active → low HRV, high EDA"),
    ]
    with torch.no_grad():
        for inp, desc in tests:
            x = torch.tensor([inp], dtype=torch.float32)
            hrv, eda = model(x).squeeze().tolist()
            print(f"    BR={inp[0]:5.1f}, Motion={inp[1]:.2f} → HRV={hrv:5.1f} ms, EDA={eda:4.1f} µS  ({desc})")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Train calibration from WESAD + DREAMER')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--output', default='checkpoints/ruview_calibration.pt')
    args = parser.parse_args()

    train_combined(epochs=args.epochs, lr=args.lr, output=args.output)

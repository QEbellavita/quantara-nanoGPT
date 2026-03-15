"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Calibration Model Training from WESAD Dataset
===============================================================================
Extracts real HR, HRV, EDA, and respiratory rate from the WESAD dataset
(15 subjects, chest ECG + wrist Empatica E4) and trains the WiFi calibration
model with ground-truth biometric pairings.

WESAD provides:
- Chest: ECG (700Hz) → HR, HRV ground truth
- Chest: Respiration (700Hz) → breathing rate ground truth
- Wrist: EDA (4Hz) → EDA ground truth
- Chest: EDA (700Hz) → EDA ground truth (higher quality)
- Labels: baseline(1), stress(2), amusement(3), meditation(4)

This script:
1. Extracts windowed (HR, HRV, EDA, breathing_rate) from each subject
2. Creates paired training data: (breathing_rate, motion_level) → (HRV, EDA)
3. Trains the WiFiCalibrationModel on real physiological relationships
4. Saves improved checkpoint

Integrates with:
- Neural Workflow AI Engine
- Biometric Integration Engine
- RuView WiFi Sensing Provider
- Real-time Data

Usage:
  python train_calibration_wesad.py
  python train_calibration_wesad.py --epochs 200 --output checkpoints/ruview_calibration.pt
===============================================================================
"""

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import iqr
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wifi_calibration import WiFiCalibrationModel, HRV_MIN, HRV_MAX, EDA_MIN, EDA_MAX

logger = logging.getLogger(__name__)

# WESAD data paths (check both locations)
WESAD_PATHS = [
    Path('/Users/bel/Quantara-Backend/ml-pipeline/data/wesad'),
    Path('/Users/bel/Quantara-Frontend/training/data/WESAD/WESAD'),
]

# Sampling rates
CHEST_FS = 700
WRIST_BVP_FS = 64
WRIST_EDA_FS = 4
WRIST_ACC_FS = 32

# Window parameters
WINDOW_SEC = 30       # 30-second windows
WINDOW_OVERLAP = 0.5  # 50% overlap

# Label mapping
LABELS = {1: 'baseline', 2: 'stress', 3: 'amusement', 4: 'meditation'}


def find_wesad_path() -> Path | None:
    """Find WESAD dataset directory."""
    for p in WESAD_PATHS:
        if p.exists():
            subjects = [d for d in p.iterdir() if d.is_dir() and d.name.startswith('S')]
            if subjects:
                print(f"  Found WESAD at: {p} ({len(subjects)} subjects)")
                return p
    return None


def load_subject(wesad_path: Path, subject_id: str) -> dict | None:
    """Load a subject's pickle file."""
    pkl_path = wesad_path / subject_id / f'{subject_id}.pkl'
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data
    except Exception as e:
        logger.warning(f"Failed to load {subject_id}: {e}")
        return None


def extract_hr_hrv_from_ecg(ecg: np.ndarray, fs: int = 700) -> tuple:
    """
    Extract heart rate and HRV from chest ECG signal (WESAD RespiBAN, 700 Hz).
    Uses R-peak detection on bandpass-filtered ECG for reliable RR intervals.
    Returns (heart_rate_bpm, hrv_rmssd_ms).
    """
    if len(ecg) < fs * 5:
        return None, None

    ecg = ecg.flatten()

    # Bandpass filter 5-15 Hz to isolate QRS complex
    nyq = fs / 2
    b, a = scipy_signal.butter(3, [5.0 / nyq, 15.0 / nyq], btype='band')
    filtered = scipy_signal.filtfilt(b, a, ecg)

    # Square the signal to emphasize R-peaks
    squared = filtered ** 2

    # Find R-peaks with minimum distance of 0.33s (180 BPM max)
    min_distance = int(fs * 0.33)
    # Use height threshold at 30th percentile of squared signal to reject noise
    height_thresh = np.percentile(squared, 70)
    peaks, _ = scipy_signal.find_peaks(squared, distance=min_distance, height=height_thresh)

    if len(peaks) < 3:
        return None, None

    # RR intervals in ms
    rr = np.diff(peaks) / fs * 1000

    # Filter physiologically implausible RR intervals (40-200 BPM)
    rr = rr[(rr > 300) & (rr < 1500)]
    if len(rr) < 2:
        return None, None

    # Remove outlier RR intervals (beyond 2 * IQR)
    rr_iqr = iqr(rr)
    rr_median = np.median(rr)
    rr = rr[(rr > rr_median - 2 * rr_iqr) & (rr < rr_median + 2 * rr_iqr)]
    if len(rr) < 2:
        return None, None

    hr = 60000.0 / np.mean(rr)
    hr = float(np.clip(hr, 40, 180))

    # RMSSD (root mean square of successive differences) — standard HRV metric
    successive_diffs = np.diff(rr)
    hrv_rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
    hrv_rmssd = float(np.clip(hrv_rmssd, 5, 250))

    return hr, hrv_rmssd


def extract_hr_hrv_from_bvp(bvp: np.ndarray, fs: int = 64) -> tuple:
    """
    Extract heart rate and HRV from wrist Blood Volume Pulse signal.
    NOTE: Prefer extract_hr_hrv_from_ecg() when chest ECG is available —
    wrist BVP produces artifact-inflated RMSSD during movement/stress.
    Returns (heart_rate_bpm, hrv_rmssd_ms).
    """
    if len(bvp) < fs * 5:
        return None, None

    bvp = bvp.flatten()

    # Bandpass filter 0.5-4 Hz (30-240 BPM range)
    nyq = fs / 2
    b, a = scipy_signal.butter(2, [0.5 / nyq, 4.0 / nyq], btype='band')
    filtered = scipy_signal.filtfilt(b, a, bvp)

    # Find peaks
    min_distance = int(fs * 0.4)  # Min 0.4s between beats (150 BPM max)
    peaks, _ = scipy_signal.find_peaks(filtered, distance=min_distance)

    if len(peaks) < 3:
        return None, None

    # RR intervals in ms
    rr = np.diff(peaks) / fs * 1000

    # Filter physiologically implausible RR intervals
    rr = rr[(rr > 300) & (rr < 1500)]  # 40-200 BPM
    if len(rr) < 2:
        return None, None

    hr = 60000.0 / np.mean(rr)
    hr = float(np.clip(hr, 40, 180))

    # RMSSD (root mean square of successive differences) — standard HRV metric
    successive_diffs = np.diff(rr)
    hrv_rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
    hrv_rmssd = float(np.clip(hrv_rmssd, 5, 200))

    return hr, hrv_rmssd


def extract_breathing_rate_from_resp(resp: np.ndarray, fs: int = 700) -> float | None:
    """
    Extract breathing rate from chest respiration signal.
    Returns breaths per minute.
    """
    if len(resp) < fs * 10:
        return None

    resp = resp.flatten()

    # Bandpass filter 0.1-0.5 Hz (6-30 BPM)
    nyq = fs / 2
    b, a = scipy_signal.butter(2, [0.1 / nyq, 0.5 / nyq], btype='band')
    filtered = scipy_signal.filtfilt(b, a, resp)

    # Find peaks (breath cycles)
    min_distance = int(fs * 2)  # Min 2s between breaths (30 BPM max)
    peaks, _ = scipy_signal.find_peaks(filtered, distance=min_distance)

    if len(peaks) < 2:
        return None

    # Breath intervals
    breath_intervals = np.diff(peaks) / fs  # seconds
    breath_intervals = breath_intervals[(breath_intervals > 2) & (breath_intervals < 12)]

    if len(breath_intervals) < 1:
        return None

    br = 60.0 / np.mean(breath_intervals)
    return float(np.clip(br, 4, 35))


def extract_eda_level(eda: np.ndarray) -> float | None:
    """Extract mean EDA level in µS from EDA signal."""
    if len(eda) < 4:
        return None
    eda = eda.flatten()
    # Remove obvious artifacts
    eda = eda[(eda > 0) & (eda < 40)]
    if len(eda) < 4:
        return None
    return float(np.median(eda))


def extract_motion_level(acc: np.ndarray) -> float:
    """
    Extract motion level from accelerometer signal.
    Returns 0-1 normalized activity level.
    """
    if len(acc) < 10:
        return 0.0
    # Compute magnitude of acceleration
    if acc.ndim == 2:
        mag = np.sqrt(np.sum(acc ** 2, axis=1))
    else:
        mag = np.abs(acc)
    # Activity = std of magnitude (low for still, high for moving)
    activity = float(np.std(mag))
    # Normalize: typical range 0-2 for wrist accelerometer
    return float(np.clip(activity / 2.0, 0.0, 1.0))


def extract_windows_from_subject(data: dict) -> list:
    """
    Extract windowed physiological features from one WESAD subject.
    Returns list of dicts: {hr, hrv, eda, breathing_rate, motion_level, label}
    """
    labels = np.array(data['label']).flatten()
    wrist = data['signal']['wrist']
    chest = data['signal']['chest']

    # Align lengths using chest label sampling (700 Hz)
    total_chest_samples = len(labels)
    total_seconds = total_chest_samples / CHEST_FS

    window_samples_chest = int(WINDOW_SEC * CHEST_FS)
    step_samples_chest = int(window_samples_chest * (1 - WINDOW_OVERLAP))

    results = []

    for start in range(0, total_chest_samples - window_samples_chest, step_samples_chest):
        end = start + window_samples_chest

        # Get dominant label for this window
        window_labels = labels[start:end]
        unique, counts = np.unique(window_labels, return_counts=True)
        dominant_label = unique[np.argmax(counts)]

        if dominant_label not in LABELS:
            continue

        # Time range in seconds
        t_start = start / CHEST_FS
        t_end = end / CHEST_FS

        # Extract chest respiration → breathing rate
        resp_start = start
        resp_end = end
        resp_window = chest['Resp'][resp_start:resp_end]
        breathing_rate = extract_breathing_rate_from_resp(resp_window, CHEST_FS)

        # Extract chest EDA (700 Hz, higher quality)
        chest_eda_window = chest['EDA'][resp_start:resp_end]
        eda = extract_eda_level(chest_eda_window)

        # Extract chest ECG → HR, HRV (much more reliable than wrist BVP,
        # which produces motion-artifact-inflated RMSSD during stress)
        ecg_window = chest['ECG'][resp_start:resp_end]
        hr, hrv = extract_hr_hrv_from_ecg(ecg_window, CHEST_FS)

        # Extract wrist accelerometer → motion level
        acc_start = int(t_start * WRIST_ACC_FS)
        acc_end = int(t_end * WRIST_ACC_FS)
        acc_window = wrist['ACC'][acc_start:acc_end]
        motion = extract_motion_level(acc_window)

        # Skip windows where we couldn't extract key signals
        if hr is None or hrv is None or breathing_rate is None or eda is None:
            continue

        results.append({
            'hr': hr,
            'hrv': hrv,
            'eda': eda,
            'breathing_rate': breathing_rate,
            'motion_level': motion,
            'label': int(dominant_label),
            'label_name': LABELS[dominant_label],
            'subject': None,  # filled in by caller
        })

    return results


def build_training_data(wesad_path: Path) -> tuple:
    """
    Extract training pairs from all WESAD subjects.
    Returns (inputs, targets) tensors.

    inputs: [breathing_rate, motion_level]  (what RuView would sense)
    targets: [hrv, eda]  (what a wearable measures)
    """
    subjects = sorted([d.name for d in wesad_path.iterdir()
                       if d.is_dir() and d.name.startswith('S')])

    # Collect windows per subject for per-subject normalization
    subject_windows = {}

    for sid in subjects:
        print(f"  Processing {sid}...", end=' ')
        data = load_subject(wesad_path, sid)
        if data is None:
            print("SKIP (load failed)")
            continue

        windows = extract_windows_from_subject(data)
        for w in windows:
            w['subject'] = sid
        subject_windows[sid] = windows
        print(f"{len(windows)} windows extracted")

    all_windows = [w for ws in subject_windows.values() for w in ws]
    if not all_windows:
        return None, None

    # Per-subject normalization: remove inter-subject offset while preserving
    # within-subject arousal patterns (stress → higher EDA relative to baseline)
    # Strategy: z-score within each subject, then rescale to global distribution
    global_hrv = np.array([w['hrv'] for w in all_windows])
    global_eda = np.array([w['eda'] for w in all_windows])
    global_hrv_mean, global_hrv_std = global_hrv.mean(), global_hrv.std()
    global_eda_mean, global_eda_std = global_eda.mean(), global_eda.std()

    print(f"\n  Per-subject normalization:")
    print(f"    Global HRV: mean={global_hrv_mean:.1f} ms, std={global_hrv_std:.1f} ms")
    print(f"    Global EDA: mean={global_eda_mean:.1f} µS, std={global_eda_std:.1f} µS")

    for sid, windows in subject_windows.items():
        if len(windows) < 5:
            continue
        subj_hrv = np.array([w['hrv'] for w in windows])
        subj_eda = np.array([w['eda'] for w in windows])
        subj_hrv_mean, subj_hrv_std = subj_hrv.mean(), max(subj_hrv.std(), 1.0)
        subj_eda_mean, subj_eda_std = subj_eda.mean(), max(subj_eda.std(), 0.1)

        for w in windows:
            # Z-score within subject, rescale to global distribution
            w['hrv'] = (w['hrv'] - subj_hrv_mean) / subj_hrv_std * global_hrv_std + global_hrv_mean
            w['eda'] = (w['eda'] - subj_eda_mean) / subj_eda_std * global_eda_std + global_eda_mean

    # Print statistics after normalization
    print(f"\n  Dataset statistics after per-subject normalization ({len(all_windows)} windows):")
    for label_id, label_name in LABELS.items():
        count = sum(1 for w in all_windows if w['label'] == label_id)
        if count > 0:
            subset = [w for w in all_windows if w['label'] == label_id]
            avg_hr = np.mean([w['hr'] for w in subset])
            avg_hrv = np.mean([w['hrv'] for w in subset])
            avg_eda = np.mean([w['eda'] for w in subset])
            avg_br = np.mean([w['breathing_rate'] for w in subset])
            print(f"    {label_name:12s}: {count:4d} windows | "
                  f"HR={avg_hr:.0f} bpm, HRV={avg_hrv:.0f} ms, "
                  f"EDA={avg_eda:.1f} µS, BR={avg_br:.0f} bpm")

    # Build training pairs
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

    return X, Y


def train_from_wesad(
    epochs: int = 200,
    lr: float = 0.005,
    output: str = 'checkpoints/ruview_calibration.pt',
    augment: bool = True,
):
    """
    Train calibration model from WESAD real physiological data.
    """
    print("=" * 60)
    print("  WESAD-BASED CALIBRATION MODEL TRAINING")
    print("=" * 60)

    wesad_path = find_wesad_path()
    if not wesad_path:
        print("  ERROR: WESAD dataset not found. Expected at:")
        for p in WESAD_PATHS:
            print(f"    {p}")
        sys.exit(1)

    print(f"\n  Extracting physiological features...")
    X, Y = build_training_data(wesad_path)

    if X is None or len(X) < 20:
        print("  ERROR: Not enough data extracted.")
        sys.exit(1)

    print(f"\n  Training data: {len(X)} samples")
    print(f"  Input range: BR=[{X[:,0].min():.1f}, {X[:,0].max():.1f}], "
          f"Motion=[{X[:,1].min():.2f}, {X[:,1].max():.2f}]")
    print(f"  Target range: HRV=[{Y[:,0].min():.1f}, {Y[:,0].max():.1f}], "
          f"EDA=[{Y[:,1].min():.1f}, {Y[:,1].max():.1f}]")

    # Augment with noise for robustness
    if augment:
        rng = np.random.default_rng(42)
        n = len(X)
        noise_X = torch.tensor(rng.normal(0, [1.5, 0.08], (n, 2)), dtype=torch.float32)
        X_aug = torch.cat([X, X + noise_X], dim=0)
        Y_aug = torch.cat([Y, Y], dim=0)
        # Shuffle
        perm = torch.randperm(len(X_aug))
        X_aug = X_aug[perm]
        Y_aug = Y_aug[perm]
        print(f"  Augmented to: {len(X_aug)} samples")
    else:
        X_aug, Y_aug = X, Y

    # Split train/val (90/10)
    split = int(len(X_aug) * 0.9)
    X_train, X_val = X_aug[:split], X_aug[split:]
    Y_train, Y_val = Y_aug[:split], Y_aug[split:]

    # Train from scratch — do NOT load existing checkpoint, as architecture
    # has changed (2->16->2 to 2->32->16->2) and old weights had collapsed
    # targets (HRV_MAX was 100, clipping all WESAD RMSSD values to a constant)
    model = WiFiCalibrationModel()
    print(f"  Training from scratch (new architecture with input normalization)")

    # Normalize targets to comparable scales so MSE loss weighs HRV and EDA
    # equally (HRV range ~10-250 vs EDA range ~0.5-20 would dominate loss otherwise)
    hrv_mean, hrv_std = Y_train[:, 0].mean(), Y_train[:, 0].std()
    eda_mean, eda_std = Y_train[:, 1].mean(), Y_train[:, 1].std()
    print(f"  Target stats: HRV mean={hrv_mean:.1f} std={hrv_std:.1f}, "
          f"EDA mean={eda_mean:.1f} std={eda_std:.1f}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    criterion = nn.MSELoss()

    # Scale loss by inverse variance so HRV and EDA contribute equally
    hrv_weight = 1.0 / (hrv_std ** 2 + 1e-6)
    eda_weight = 1.0 / (eda_std ** 2 + 1e-6)
    # Normalize so they sum to 2
    total = hrv_weight + eda_weight
    hrv_weight = 2.0 * hrv_weight / total
    eda_weight = 2.0 * eda_weight / total
    print(f"  Loss weights: HRV={hrv_weight:.4f}, EDA={eda_weight:.4f}")

    print(f"\n  Training for {epochs} epochs...")
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        # Training
        optimizer.zero_grad()
        pred = model(X_train)
        # Weighted MSE: balance HRV and EDA contributions
        hrv_loss = ((pred[:, 0] - Y_train[:, 0]) ** 2).mean() * hrv_weight
        eda_loss = ((pred[:, 1] - Y_train[:, 1]) ** 2).mean() * eda_weight
        loss = hrv_loss + eda_loss
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
            val_pred = model(X_val)
            val_hrv_loss = ((val_pred[:, 0] - Y_val[:, 0]) ** 2).mean() * hrv_weight
            val_eda_loss = ((val_pred[:, 1] - Y_val[:, 1]) ** 2).mean() * eda_weight
            val_loss = val_hrv_loss + val_eda_loss

        scheduler.step(val_loss)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter > 150:
            print(f"    Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % max(1, epochs // 5) == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/{epochs} | "
                  f"train_loss={loss.item():.4f} (HRV={hrv_loss.item():.2f}, EDA={eda_loss.item():.2f}) | "
                  f"val_loss={val_loss.item():.4f} | lr={current_lr:.6f}")

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    model.eval()

    # Save
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    torch.save(model.state_dict(), output)

    # Validation metrics
    with torch.no_grad():
        val_pred = model(X_val)
        hrv_mae = (val_pred[:, 0] - Y_val[:, 0]).abs().mean().item()
        eda_mae = (val_pred[:, 1] - Y_val[:, 1]).abs().mean().item()

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  HRV MAE: {hrv_mae:.1f} ms")
    print(f"  EDA MAE: {eda_mae:.1f} µS")
    print(f"  Checkpoint: {output}")

    # Quick validation on known patterns
    print(f"\n  Spot checks:")
    test_cases = [
        ([8.0, 0.05], "Slow breathing, still (expect high HRV, low EDA)"),
        ([14.0, 0.3], "Normal breathing, moderate (expect moderate HRV/EDA)"),
        ([24.0, 0.7], "Fast breathing, active (expect low HRV, high EDA)"),
        ([10.0, 0.1], "Relaxed (expect high HRV, low EDA)"),
        ([20.0, 0.6], "Stressed (expect low HRV, high EDA)"),
    ]
    with torch.no_grad():
        for inputs, desc in test_cases:
            x = torch.tensor([inputs], dtype=torch.float32)
            hrv, eda = model(x).squeeze().tolist()
            print(f"    BR={inputs[0]:5.1f}, Motion={inputs[1]:.2f} → "
                  f"HRV={hrv:5.1f} ms, EDA={eda:4.1f} µS  ({desc})")

    return model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Train WiFi calibration from WESAD dataset')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate (default: 0.005)')
    parser.add_argument('--output', default='checkpoints/ruview_calibration.pt',
                        help='Output checkpoint path')
    parser.add_argument('--no-augment', action='store_true', help='Disable data augmentation')
    args = parser.parse_args()

    train_from_wesad(
        epochs=args.epochs,
        lr=args.lr,
        output=args.output,
        augment=not args.no_augment,
    )

# train_emotion_classifier.py
"""
===============================================================================
QUANTARA - Train Emotion Classifier (32-Emotion Taxonomy)
===============================================================================
Train the fusion head on emotion-labeled text data with synthetic biometrics.
Two-stage hierarchical training: family (9) + sub-emotion (32).

Supports two embedding modes:
  --use-sentence-transformer  Use sentence-transformers (recommended, 384-dim)
  --gpt-checkpoint           Use nanoGPT character-level embeddings

Usage:
    python train_emotion_classifier.py --use-sentence-transformer
    python train_emotion_classifier.py --gpt-checkpoint out-quantara-emotion/ckpt.pt
    python train_emotion_classifier.py --use-sentence-transformer --distillation --distill-alpha 0.3 --distill-temperature 2.0
===============================================================================
"""

import os
import sys
import argparse
import pickle
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emotion_classifier import (
    BiometricEncoder, FusionHead, GoEmotionsEncoder,
    EMOTION_FAMILIES, FAMILY_NAMES, family_for_emotion,
    _EMOTION_TO_FAMILY, HAS_TRANSFORMERS,
)
from pose_encoder import PoseEncoder, POSE_FEATURE_NAMES

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from model import GPT, GPTConfig
    HAS_NANOGPT = True
except ImportError:
    HAS_NANOGPT = False


# ─── Synthetic biometric ranges for all 32 emotions (research-grounded) ──────

BIOMETRIC_RANGES = {
    # Joy family
    'joy':          {'hr': (70, 90),   'hrv': (50, 80),  'eda': (2, 4)},
    'excitement':   {'hr': (85, 110),  'hrv': (40, 60),  'eda': (4, 7)},
    'enthusiasm':   {'hr': (75, 95),   'hrv': (45, 70),  'eda': (3, 5)},
    'fun':          {'hr': (75, 95),   'hrv': (50, 75),  'eda': (2, 4)},
    'gratitude':    {'hr': (65, 80),   'hrv': (60, 85),  'eda': (1, 3)},
    'pride':        {'hr': (70, 90),   'hrv': (45, 65),  'eda': (3, 5)},
    # Sadness family
    'sadness':      {'hr': (55, 70),   'hrv': (40, 60),  'eda': (1, 2)},
    'grief':        {'hr': (50, 70),   'hrv': (30, 50),  'eda': (1, 3)},
    'boredom':      {'hr': (55, 65),   'hrv': (55, 75),  'eda': (0.5, 1.5)},
    'nostalgia':    {'hr': (60, 75),   'hrv': (45, 65),  'eda': (1.5, 3)},
    # Anger family
    'anger':        {'hr': (85, 110),  'hrv': (20, 40),  'eda': (5, 8)},
    'frustration':  {'hr': (80, 100),  'hrv': (25, 45),  'eda': (4, 7)},
    'hate':         {'hr': (85, 105),  'hrv': (20, 35),  'eda': (5, 9)},
    'contempt':     {'hr': (75, 90),   'hrv': (30, 50),  'eda': (3, 5)},
    'disgust':      {'hr': (70, 90),   'hrv': (35, 55),  'eda': (4, 7)},
    'jealousy':     {'hr': (80, 100),  'hrv': (25, 45),  'eda': (5, 8)},
    # Fear family
    'fear':         {'hr': (80, 105),  'hrv': (25, 45),  'eda': (6, 10)},
    'anxiety':      {'hr': (75, 100),  'hrv': (20, 40),  'eda': (5, 9)},
    'worry':        {'hr': (70, 90),   'hrv': (30, 50),  'eda': (3, 6)},
    'overwhelmed':  {'hr': (85, 110),  'hrv': (15, 35),  'eda': (7, 12)},
    'stressed':     {'hr': (80, 105),  'hrv': (20, 40),  'eda': (5, 9)},
    # Love family
    'love':         {'hr': (65, 85),   'hrv': (55, 75),  'eda': (2, 4)},
    'compassion':   {'hr': (60, 80),   'hrv': (60, 80),  'eda': (1.5, 3)},
    # Calm family
    'calm':         {'hr': (55, 70),   'hrv': (65, 90),  'eda': (0.5, 2)},
    'relief':       {'hr': (60, 80),   'hrv': (55, 80),  'eda': (2, 4)},
    'mindfulness':  {'hr': (55, 68),   'hrv': (70, 95),  'eda': (0.5, 1.5)},
    'resilience':   {'hr': (60, 75),   'hrv': (60, 85),  'eda': (1, 3)},
    'hope':         {'hr': (65, 80),   'hrv': (55, 75),  'eda': (1.5, 3)},
    # Self-Conscious family
    'guilt':        {'hr': (70, 90),   'hrv': (30, 50),  'eda': (4, 7)},
    'shame':        {'hr': (75, 95),   'hrv': (25, 45),  'eda': (5, 8)},
    # Atomic
    'surprise':     {'hr': (75, 100),  'hrv': (35, 55),  'eda': (4, 7)},
    'neutral':      {'hr': (60, 80),   'hrv': (50, 70),  'eda': (1, 3)},
}

EMOTION_TO_IDX = {e: i for i, e in enumerate(FusionHead.EMOTIONS)}
FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILY_NAMES)}


def generate_synthetic_biometrics(emotion: str) -> dict:
    """Generate plausible biometrics for an emotion."""
    ranges = BIOMETRIC_RANGES.get(emotion, BIOMETRIC_RANGES['neutral'])

    return {
        'heart_rate': random.uniform(*ranges['hr']),
        'hrv': random.uniform(*ranges['hrv']),
        'eda': random.uniform(*ranges['eda']),
    }


# ─── Synthetic pose ranges for root emotions (research-grounded) ────────────

POSE_RANGES = {
    'joy': {'slouch': (0.7, 1.0), 'openness': (0.6, 1.0), 'tension': (0.0, 0.2),
            'head_tilt': (-0.1, 0.2), 'gesture_speed': (0.2, 0.6), 'symmetry': (0.7, 1.0),
            'forward_lean': (-0.1, 0.2), 'stillness': (0.0, 0.1)},
    'sadness': {'slouch': (0.2, 0.5), 'openness': (0.1, 0.4), 'tension': (0.2, 0.5),
                'head_tilt': (-0.5, -0.1), 'gesture_speed': (0.0, 0.1), 'symmetry': (0.5, 0.8),
                'forward_lean': (-0.3, 0.0), 'stillness': (0.3, 0.8)},
    'fear': {'slouch': (0.5, 0.8), 'openness': (0.1, 0.3), 'tension': (0.6, 1.0),
             'head_tilt': (-0.3, 0.1), 'gesture_speed': (0.0, 0.3), 'symmetry': (0.3, 0.7),
             'forward_lean': (0.1, 0.4), 'stillness': (0.1, 0.5)},
    'anger': {'slouch': (0.6, 0.9), 'openness': (0.3, 0.6), 'tension': (0.5, 0.9),
              'head_tilt': (0.0, 0.4), 'gesture_speed': (0.3, 0.8), 'symmetry': (0.3, 0.6),
              'forward_lean': (0.2, 0.5), 'stillness': (0.0, 0.1)},
    'calm': {'slouch': (0.7, 1.0), 'openness': (0.4, 0.7), 'tension': (0.0, 0.2),
             'head_tilt': (-0.1, 0.1), 'gesture_speed': (0.0, 0.1), 'symmetry': (0.8, 1.0),
             'forward_lean': (-0.1, 0.1), 'stillness': (0.4, 0.9)},
    'neutral': {'slouch': (0.5, 0.8), 'openness': (0.3, 0.6), 'tension': (0.1, 0.4),
                'head_tilt': (-0.2, 0.2), 'gesture_speed': (0.0, 0.2), 'symmetry': (0.6, 0.9),
                'forward_lean': (-0.1, 0.1), 'stillness': (0.1, 0.4)},
}

# Map family names to root emotions for pose range lookup
_FAMILY_TO_ROOT_EMOTION = {
    'Joy': 'joy', 'Sadness': 'sadness', 'Anger': 'anger', 'Fear': 'fear',
    'Calm': 'calm', 'Neutral': 'neutral', 'Love': 'calm',
    'Self-Conscious': 'fear', 'Surprise': 'neutral',
}


def generate_synthetic_pose(emotion: str) -> dict:
    """Generate plausible pose features for an emotion.

    For sub-emotions not in POSE_RANGES, uses the family root emotion ranges
    via _EMOTION_TO_FAMILY lookup.

    Returns dict with 8 named pose features matching POSE_FEATURE_NAMES order.
    """
    if emotion in POSE_RANGES:
        ranges = POSE_RANGES[emotion]
    else:
        family = _EMOTION_TO_FAMILY.get(emotion, 'Neutral')
        root = _FAMILY_TO_ROOT_EMOTION.get(family, 'neutral')
        ranges = POSE_RANGES[root]

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


# ─── Real Biometric Data Loader ─────────────────────────────────────────────

class RealBiometricSampler:
    """Sample real biometric values from heart_rate_emotion_dataset.csv and
    stress level CSVs instead of purely synthetic ranges.

    Falls back to synthetic generation when real data is unavailable for
    a given emotion.
    """

    def __init__(self, downloads_dir: Path):
        self.real_hr_by_emotion = defaultdict(list)
        self.real_stress_scores = defaultdict(list)
        self.real_multisignal = defaultdict(list)
        self._load_heart_rate_data(downloads_dir)
        self._load_stress_data(downloads_dir)
        self._load_smartwatch_data(downloads_dir)
        self._load_empatica_data(downloads_dir)

        total = sum(len(v) for v in self.real_hr_by_emotion.values())
        stress_total = sum(len(v) for v in self.real_stress_scores.values())
        multi_total = sum(len(v) for v in self.real_multisignal.values())
        print(f"  Real biometric sampler: {total} HR readings, {stress_total} stress scores, {multi_total} multi-signal readings")

    def _load_heart_rate_data(self, downloads_dir: Path):
        """Load real heart rate values grouped by emotion."""
        hr_path = downloads_dir / "heart_rate_emotion_dataset.csv"
        if not hr_path.exists():
            print(f"  [-] heart_rate_emotion_dataset.csv not found")
            return

        df = pd.read_csv(hr_path)
        hr_col = 'HeartRate' if 'HeartRate' in df.columns else None
        label_col = 'Emotion' if 'Emotion' in df.columns else None

        if not hr_col or not label_col:
            print(f"  [-] HR dataset missing expected columns: {list(df.columns)}")
            return

        hr_map = {
            'happy': 'joy', 'sad': 'sadness', 'disgust': 'disgust',
            'anger': 'anger', 'fear': 'fear', 'surprise': 'surprise',
            'neutral': 'neutral',
        }

        for _, row in df.iterrows():
            raw = str(row[label_col]).strip().lower()
            emotion = hr_map.get(raw)
            if emotion:
                try:
                    hr = float(row[hr_col])
                    if 40 <= hr <= 180:
                        self.real_hr_by_emotion[emotion].append(hr)
                except (ValueError, TypeError):
                    pass

        # Propagate family-level HR data to sub-emotions that lack real data
        family_propagation = {
            'joy': ['excitement', 'enthusiasm', 'fun', 'gratitude', 'pride'],
            'sadness': ['grief', 'boredom', 'nostalgia'],
            'anger': ['frustration', 'hate', 'contempt', 'jealousy'],
            'fear': ['anxiety', 'worry', 'overwhelmed', 'stressed'],
        }
        for source, targets in family_propagation.items():
            if source in self.real_hr_by_emotion:
                for target in targets:
                    if not self.real_hr_by_emotion[target]:
                        self.real_hr_by_emotion[target] = list(self.real_hr_by_emotion[source])

        for emo, vals in self.real_hr_by_emotion.items():
            if vals:
                print(f"    HR [{emo}]: {len(vals)} readings, range {min(vals):.0f}-{max(vals):.0f} bpm")

    def _load_stress_data(self, downloads_dir: Path):
        """Load stress assessment scores and map to emotion-relevant stress levels."""
        for fname in ['Stress_Level_v1.csv', 'Stress_Level_v2.csv']:
            path = downloads_dir / fname
            if not path.exists():
                continue

            df = pd.read_csv(path)
            task_cols = [c for c in df.columns if c not in ['Unnamed: 0'] and c != df.columns[0]]

            for _, row in df.iterrows():
                for task in task_cols:
                    try:
                        score = float(row[task])
                    except (ValueError, TypeError):
                        continue

                    # Map score to emotion categories
                    if score >= 7.0:
                        self.real_stress_scores['overwhelmed'].append(score)
                        self.real_stress_scores['stressed'].append(score)
                    elif score >= 5.0:
                        self.real_stress_scores['stressed'].append(score)
                        self.real_stress_scores['anxiety'].append(score)
                    elif score >= 3.0:
                        self.real_stress_scores['anxiety'].append(score)
                        self.real_stress_scores['worry'].append(score)
                    elif score >= 1.5:
                        self.real_stress_scores['calm'].append(score)
                    else:
                        self.real_stress_scores['relief'].append(score)
                        self.real_stress_scores['calm'].append(score)
                        self.real_stress_scores['mindfulness'].append(score)

    def _load_smartwatch_data(self, downloads_dir: Path):
        """Load real (HR, HRV, Stress) triples from Smartwatch Health Monitoring dataset."""
        sw_path = Path('/Users/bel/Quantara-Frontend/ml-training/datasets/raw/Smartwatch_Health_Monitoring_Dataset_40000_Rows.csv')
        if not sw_path.exists():
            print(f"  [-] Smartwatch dataset not found at {sw_path}")
            return

        df = pd.read_csv(sw_path)
        required = ['Avg_Heart_Rate', 'HRV', 'Stress_Level']
        if not all(c in df.columns for c in required):
            print(f"  [-] Smartwatch dataset missing columns: {list(df.columns)}")
            return

        count = 0
        for _, row in df.iterrows():
            try:
                hr = float(row['Avg_Heart_Rate'])
                hrv = float(row['HRV'])
                stress = float(row['Stress_Level'])
            except (ValueError, TypeError):
                continue

            if not (40 <= hr <= 180 and 0 <= hrv <= 200 and 1 <= stress <= 10):
                continue

            # Map stress levels to emotions (same as prepare.py)
            emotions = []
            if stress >= 8:
                emotions = ['overwhelmed', 'stressed']
            elif stress >= 6:
                emotions = ['stressed', 'anxiety']
            elif stress >= 4:
                emotions = ['anxiety', 'worry']
            elif stress >= 2.5:
                emotions = ['calm']
            else:
                emotions = ['mindfulness', 'relief']

            for emo in emotions:
                self.real_multisignal[emo].append((hr, hrv, stress))
                count += 1

        print(f"  Smartwatch multi-signal: {count} readings loaded")

    def _load_empatica_data(self, downloads_dir: Path):
        """Load real (HR, HRV_RMSSD, EDA) triples from Empatica stress dataset."""
        emp_path = Path('/Users/bel/Quantara-Frontend/ml-training/datasets/processed/empatica_stress_processed.csv')
        if not emp_path.exists():
            print(f"  [-] Empatica dataset not found at {emp_path}")
            return

        df = pd.read_csv(emp_path)
        required = ['hr_mean', 'hrv_rmssd', 'eda_mean', 'stress_level', 'arousal_level']
        if not all(c in df.columns for c in required):
            print(f"  [-] Empatica dataset missing columns: {list(df.columns)}")
            return

        count = 0
        for _, row in df.iterrows():
            try:
                hr = float(row['hr_mean'])
                hrv = float(row['hrv_rmssd'])
                eda = float(row['eda_mean'])
                stress = str(row['stress_level']).strip().lower()
                arousal = str(row['arousal_level']).strip().lower()
            except (ValueError, TypeError):
                continue

            # Map (stress_level, arousal_level) to emotions
            emotions = []
            if stress == 'high' and arousal == 'high':
                emotions = ['overwhelmed', 'fear']
            elif stress == 'moderate' and arousal == 'high':
                emotions = ['anxiety', 'stressed']
            elif stress == 'low' and arousal == 'high':
                emotions = ['excitement', 'enthusiasm']
            elif stress == 'low' and arousal == 'medium':
                emotions = ['calm', 'relief']
            elif stress == 'low' and arousal == 'low':
                emotions = ['mindfulness', 'boredom']

            for emo in emotions:
                self.real_multisignal[emo].append((hr, hrv, eda))
                count += 1

        print(f"  Empatica multi-signal: {count} readings loaded")

    def sample(self, emotion: str) -> dict:
        """Get biometric readings — real when available, synthetic as fallback.

        Priority: multi-signal (smartwatch/empatica) > single-signal (HR/stress) > synthetic.
        """
        # Multi-signal: prefer real (HR, HRV, EDA/stress) triples when available
        multi_pool = self.real_multisignal.get(emotion)
        if multi_pool:
            hr, hrv, eda_or_stress = random.choice(multi_pool)
            heart_rate = hr + random.gauss(0, 1.0)  # ±2 bpm jitter
            heart_rate = max(40, min(180, heart_rate))
            hrv_val = hrv + random.gauss(0, 1.5)  # ±3 ms jitter
            hrv_val = max(5, min(200, hrv_val))
            # If the third value looks like a stress score (1-10), derive EDA from it
            if eda_or_stress <= 10:
                ranges = BIOMETRIC_RANGES.get(emotion, BIOMETRIC_RANGES['neutral'])
                eda = random.uniform(*ranges['eda']) + (eda_or_stress / 10.0) * 2.0
            else:
                # Real EDA value from Empatica
                eda = eda_or_stress + random.gauss(0, 0.2)
            eda = max(0.3, min(25, eda))
            return {
                'heart_rate': heart_rate,
                'hrv': hrv_val,
                'eda': eda,
            }

        # Fall through to single-signal logic
        hr_pool = self.real_hr_by_emotion.get(emotion)
        stress_pool = self.real_stress_scores.get(emotion)
        ranges = BIOMETRIC_RANGES.get(emotion, BIOMETRIC_RANGES['neutral'])

        # Heart rate: prefer real data
        if hr_pool:
            heart_rate = random.choice(hr_pool)
            # Add small jitter for diversity (±3 bpm)
            heart_rate += random.gauss(0, 1.5)
            heart_rate = max(40, min(180, heart_rate))
        else:
            heart_rate = random.uniform(*ranges['hr'])

        # HRV: derive from stress scores when available (inverse relationship)
        if stress_pool:
            stress_score = random.choice(stress_pool)
            # Higher stress → lower HRV (inverse mapping, 0-9 scale → HRV)
            # HRV typically 15-95ms, stressed people have lower HRV
            hrv = 95 - (stress_score / 9.0) * 70 + random.gauss(0, 5)
            hrv = max(10, min(100, hrv))
        else:
            hrv = random.uniform(*ranges['hrv'])

        # EDA: derive from stress + HR when available
        if stress_pool or hr_pool:
            # EDA correlates with arousal — high HR and high stress → high EDA
            base_eda = random.uniform(*ranges['eda'])
            if hr_pool and heart_rate > 90:
                base_eda *= 1.2  # Higher HR → slightly elevated EDA
            if stress_pool:
                stress_score = random.choice(stress_pool)
                base_eda += (stress_score / 9.0) * 2.0  # Stress adds EDA
            eda = max(0.5, min(20, base_eda))
        else:
            eda = random.uniform(*ranges['eda'])

        return {
            'heart_rate': heart_rate,
            'hrv': hrv,
            'eda': eda,
        }


class EmotionDataset(Dataset):
    """Dataset of text embeddings + biometrics + pose + labels (emotion + family).

    Optionally stores teacher_probs (32-dim soft labels from GoEmotions)
    for knowledge distillation.
    """

    def __init__(self, embeddings, biometrics, pose_features, emotion_labels, family_labels,
                 teacher_probs=None):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.biometrics = torch.tensor(biometrics, dtype=torch.float32)
        self.pose_features = torch.tensor(pose_features, dtype=torch.float32)
        self.emotion_labels = torch.tensor(emotion_labels, dtype=torch.long)
        self.family_labels = torch.tensor(family_labels, dtype=torch.long)
        if teacher_probs is not None:
            self.teacher_probs = torch.tensor(teacher_probs, dtype=torch.float32)
        else:
            self.teacher_probs = None

    def __len__(self):
        return len(self.emotion_labels)

    def __getitem__(self, idx):
        item = (
            self.embeddings[idx],
            self.biometrics[idx],
            self.pose_features[idx],
            self.emotion_labels[idx],
            self.family_labels[idx],
        )
        if self.teacher_probs is not None:
            item = item + (self.teacher_probs[idx],)
        return item


def load_dair_emotion_dataset(max_samples: int = 0):
    """Load dair-ai/emotion dataset directly from HuggingFace Hub.

    Returns list of (text, emotion) tuples using full 416K unsplit dataset.
    Falls back to 20K split version if unsplit unavailable.

    Connected to: ML Training & Prediction Systems
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [!] HuggingFace datasets not installed. Run: pip install datasets")
        return []

    label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    data = []

    try:
        # Try unsplit config first (416K samples)
        print("  Loading dair-ai/emotion (unsplit, ~416K samples) from HuggingFace...")
        ds = load_dataset('dair-ai/emotion', 'unsplit', split='train')
        source = "dair-ai/emotion (unsplit)"
    except Exception:
        try:
            # Fall back to split config (20K samples)
            print("  Loading dair-ai/emotion (split, ~20K samples) from HuggingFace...")
            ds = load_dataset('dair-ai/emotion', 'split', split='train')
            source = "dair-ai/emotion (split)"
        except Exception as e:
            print(f"  [!] Failed to load dair-ai/emotion: {e}")
            return []

    for row in ds:
        text = str(row['text']).strip()
        label = row['label']
        emotion = label_map.get(label, 'neutral')
        if text and len(text) > 10:
            data.append((text, emotion))

    if max_samples > 0 and len(data) > max_samples:
        random.shuffle(data)
        data = data[:max_samples]

    print(f"  Loaded {len(data)} samples from {source}")
    return data


def load_hf_go_emotions(max_samples: int = 0):
    """Load google-research-datasets/go_emotions from HuggingFace Hub.

    58K Reddit comments with 28 emotion labels mapped to Quantara 32 taxonomy.
    Uses the same mapping as GoEmotionsEncoder.GOEMOTIONS_TO_QUANTARA.

    Connected to: ML Training & Prediction Systems
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [!] HuggingFace datasets not installed. Run: pip install datasets")
        return []

    # Same mapping used by GoEmotionsEncoder
    go_to_quantara = {
        'admiration': 'gratitude',
        'amusement': 'fun',
        'anger': 'anger',
        'annoyance': 'frustration',
        'approval': 'pride',
        'caring': 'compassion',
        'confusion': 'worry',
        'curiosity': 'enthusiasm',
        'desire': 'love',
        'disappointment': 'sadness',
        'disapproval': 'contempt',
        'disgust': 'disgust',
        'embarrassment': 'shame',
        'excitement': 'excitement',
        'fear': 'fear',
        'gratitude': 'gratitude',
        'grief': 'grief',
        'joy': 'joy',
        'love': 'love',
        'nervousness': 'anxiety',
        'optimism': 'hope',
        'pride': 'pride',
        'realization': 'surprise',
        'relief': 'relief',
        'remorse': 'guilt',
        'sadness': 'sadness',
        'surprise': 'surprise',
        'neutral': 'neutral',
    }

    data = []
    try:
        print("  Loading go_emotions (~58K samples) from HuggingFace...")
        ds = load_dataset('google-research-datasets/go_emotions', 'simplified', split='train')

        # Build id-to-label map from dataset features
        label_names = ds.features['labels'].feature.names

        for row in ds:
            text = str(row['text']).strip()
            labels = row['labels']
            if not text or len(text) <= 10 or not labels:
                continue
            # Use first (primary) label
            go_label = label_names[labels[0]]
            emotion = go_to_quantara.get(go_label, 'neutral')
            data.append((text, emotion))

    except Exception as e:
        print(f"  [!] Failed to load go_emotions: {e}")
        return []

    if max_samples > 0 and len(data) > max_samples:
        random.shuffle(data)
        data = data[:max_samples]

    print(f"  Loaded {len(data)} samples from go_emotions")
    return data


def load_hf_tweet_eval_emotion(max_samples: int = 0):
    """Load tweet_eval emotion subset from HuggingFace Hub.

    4-class emotion classification from tweets: anger, joy, optimism, sadness.

    Connected to: ML Training & Prediction Systems
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [!] HuggingFace datasets not installed. Run: pip install datasets")
        return []

    label_map = {0: 'anger', 1: 'joy', 2: 'hope', 3: 'sadness'}
    data = []

    try:
        print("  Loading tweet_eval/emotion from HuggingFace...")
        ds = load_dataset('tweet_eval', 'emotion', split='train')

        for row in ds:
            text = str(row['text']).strip()
            label = row['label']
            emotion = label_map.get(label, 'neutral')
            if text and len(text) > 10:
                data.append((text, emotion))

    except Exception as e:
        print(f"  [!] Failed to load tweet_eval/emotion: {e}")
        return []

    if max_samples > 0 and len(data) > max_samples:
        random.shuffle(data)
        data = data[:max_samples]

    print(f"  Loaded {len(data)} samples from tweet_eval/emotion")
    return data


def load_hf_empathetic_dialogues(max_samples: int = 0):
    """Load empathetic_dialogues from HuggingFace Hub.

    25K dialogues with 32 emotion context labels mapped to Quantara taxonomy.

    Connected to: ML Training & Prediction Systems
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [!] HuggingFace datasets not installed. Run: pip install datasets")
        return []

    # Map empathetic_dialogues context labels to Quantara taxonomy
    ed_to_quantara = {
        'surprised': 'surprise', 'excited': 'excitement', 'angry': 'anger',
        'proud': 'pride', 'sad': 'sadness', 'annoyed': 'frustration',
        'grateful': 'gratitude', 'lonely': 'sadness', 'afraid': 'fear',
        'terrified': 'fear', 'guilty': 'guilt', 'impressed': 'surprise',
        'disgusted': 'disgust', 'hopeful': 'hope', 'confident': 'pride',
        'furious': 'anger', 'anxious': 'anxiety', 'anticipating': 'enthusiasm',
        'joyful': 'joy', 'nostalgic': 'nostalgia', 'disappointed': 'sadness',
        'prepared': 'calm', 'jealous': 'jealousy', 'content': 'calm',
        'devastated': 'grief', 'sentimental': 'nostalgia', 'caring': 'compassion',
        'trusting': 'calm', 'ashamed': 'shame', 'apprehensive': 'anxiety',
        'faithful': 'love', 'embarrassed': 'shame', 'neutral': 'neutral',
    }

    data = []
    try:
        print("  Loading empathetic_dialogues (~25K dialogues) from HuggingFace...")
        ds = load_dataset('empathetic_dialogues', split='train')

        for row in ds:
            text = str(row['utterance']).strip()
            # Remove encoding tokens that appear in this dataset
            text = text.replace('_comma_', ',').replace('_period_', '.')
            context = str(row.get('context', '')).strip().lower()
            emotion = ed_to_quantara.get(context, None)
            if text and len(text) > 10 and emotion:
                data.append((text, emotion))

    except Exception as e:
        print(f"  [!] Failed to load empathetic_dialogues: {e}")
        return []

    if max_samples > 0 and len(data) > max_samples:
        random.shuffle(data)
        data = data[:max_samples]

    print(f"  Loaded {len(data)} samples from empathetic_dialogues")
    return data


def load_hf_daily_dialog(max_samples: int = 0):
    """Load daily_dialog from HuggingFace Hub.

    13K dialogues with per-utterance emotion labels.
    Labels: 0=no_emotion, 1=anger, 2=disgust, 3=fear, 4=happiness, 5=sadness, 6=surprise.

    Connected to: ML Training & Prediction Systems
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [!] HuggingFace datasets not installed. Run: pip install datasets")
        return []

    label_map = {
        0: None,           # no emotion -- skip
        1: 'anger',
        2: 'disgust',
        3: 'fear',
        4: 'joy',
        5: 'sadness',
        6: 'surprise',
    }

    data = []
    try:
        print("  Loading daily_dialog (~13K dialogues) from HuggingFace...")
        ds = load_dataset('daily_dialog', split='train')

        for row in ds:
            utterances = row['dialog']
            emotions = row['emotion']
            for text, label in zip(utterances, emotions):
                text = str(text).strip()
                emotion = label_map.get(label, None)
                if text and len(text) > 10 and emotion:
                    data.append((text, emotion))

    except Exception as e:
        print(f"  [!] Failed to load daily_dialog: {e}")
        return []

    if max_samples > 0 and len(data) > max_samples:
        random.shuffle(data)
        data = data[:max_samples]

    print(f"  Loaded {len(data)} samples from daily_dialog")
    return data


def load_emotion_data(downloads_dir: Path, use_hf_datasets: bool = False, hf_max_samples: int = 0,
                      use_hf_all: bool = False, use_hf_go_emotions: bool = False,
                      use_hf_tweet_eval: bool = False, use_hf_empathetic: bool = False,
                      use_hf_daily_dialog: bool = False):
    """Load emotion-labeled text data for all 32 emotions."""
    all_data = []

    # Original 7-emotion label map
    label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

    # Load from HuggingFace Hub if requested (replaces text.csv with full dataset)
    if use_hf_datasets:
        hf_data = load_dair_emotion_dataset(max_samples=hf_max_samples)
        all_data.extend(hf_data)
    else:
        # Try local text.csv (legacy path)
        text_path = downloads_dir / "text.csv"
        if text_path.exists():
            df = pd.read_csv(text_path, nrows=50000)
            text_col = 'text' if 'text' in df.columns else df.columns[1]
            label_col = 'label' if 'label' in df.columns else df.columns[2]

            for _, row in df.iterrows():
                text = str(row[text_col]).strip()
                label = row[label_col]
                emotion = label_map.get(label, 'neutral')
                if text and len(text) > 10:
                    all_data.append((text, emotion))

            print(f"  Loaded {len(all_data)} samples from text.csv")

    # Load additional HuggingFace emotion datasets
    if use_hf_all or use_hf_go_emotions:
        go_data = load_hf_go_emotions(max_samples=hf_max_samples)
        all_data.extend(go_data)

    if use_hf_all or use_hf_tweet_eval:
        tweet_data = load_hf_tweet_eval_emotion(max_samples=hf_max_samples)
        all_data.extend(tweet_data)

    if use_hf_all or use_hf_empathetic:
        emp_data = load_hf_empathetic_dialogues(max_samples=hf_max_samples)
        all_data.extend(emp_data)

    if use_hf_all or use_hf_daily_dialog:
        dd_data = load_hf_daily_dialog(max_samples=hf_max_samples)
        all_data.extend(dd_data)

    # Try archive training.csv
    archive_path = downloads_dir / "archive (4) 3" / "training.csv"
    if archive_path.exists():
        df = pd.read_csv(archive_path)

        for _, row in df.iterrows():
            text = str(row['text']).strip()
            label = row['label']
            emotion = label_map.get(label, 'neutral')
            if text and len(text) > 10:
                all_data.append((text, emotion))

        print(f"  Total samples: {len(all_data)}")

    # Tweet emotions — expanded emotions (boredom, enthusiasm, worry, hate, relief, fun)
    tweet_path = downloads_dir / "text_emotion.csv"
    if tweet_path.exists():
        df = pd.read_csv(tweet_path)
        text_col = 'content' if 'content' in df.columns else df.columns[0]
        label_col = 'sentiment' if 'sentiment' in df.columns else df.columns[1]

        tweet_map = {
            'happy': 'joy', 'happiness': 'joy', 'sad': 'sadness',
            'angry': 'anger', 'boredom': 'boredom', 'enthusiasm': 'enthusiasm',
            'worry': 'worry', 'hate': 'hate', 'relief': 'relief', 'fun': 'fun',
            'love': 'love', 'surprise': 'surprise', 'fear': 'fear',
            'empty': 'neutral', 'neutral': 'neutral',
        }

        count = 0
        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            raw = str(row[label_col]).strip().lower()
            emotion = tweet_map.get(raw)
            if text and len(text) > 10 and emotion:
                all_data.append((text, emotion))
                count += 1

        print(f"  Loaded {count} tweet samples (expanded emotions)")

    # Heart rate emotion dataset — disgust
    hr_path = downloads_dir / "heart_rate_emotion_dataset.csv"
    if hr_path.exists():
        df = pd.read_csv(hr_path)
        text_col = None
        for col in ['text', 'description', 'sentence']:
            if col in df.columns:
                text_col = col
                break
        label_col = 'label' if 'label' in df.columns else 'emotion'

        hr_map = {'happy': 'joy', 'sad': 'sadness', 'disgust': 'disgust',
                   'anger': 'anger', 'fear': 'fear', 'surprise': 'surprise'}

        if text_col and label_col in df.columns:
            count = 0
            for _, row in df.iterrows():
                text = str(row[text_col]).strip()
                raw = str(row[label_col]).strip().lower()
                emotion = hr_map.get(raw)
                if text and len(text) > 10 and emotion:
                    all_data.append((text, emotion))
                    count += 1
            print(f"  Loaded {count} heart rate emotion samples")

    if not all_data:
        raise FileNotFoundError(f"No emotion data found in {downloads_dir}")

    # Reclassify derived emotions (Tier 2)
    all_data = _reclassify_for_training(all_data)

    return all_data


def _reclassify_for_training(data):
    """Apply Tier 2 reclassification to training data."""
    derivations = {
        'frustration': ('anger', ['frustrated', 'annoyed', 'irritated', 'frustrating']),
        'excitement': ('joy', ['excited', 'thrilled', 'pumped', 'can\'t wait', 'stoked']),
        'grief': ('sadness', ['lost', 'died', 'gone forever', 'passed away', 'death']),
        'overwhelmed': ('fear', ['too much', 'can\'t handle', 'overwhelmed', 'drowning']),
        'hope': ('joy', ['hope', 'looking forward', 'optimistic', 'brighter']),
        'guilt': ('sadness', ['I did', 'I shouldn\'t have', 'my fault', 'I regret']),
        'shame': ('sadness', ['ashamed', 'humiliated', 'embarrassed', 'worthless']),
    }

    extra = []
    for target, (source, keywords) in derivations.items():
        for text, emotion in data:
            if emotion == source:
                text_lower = text.lower()
                if any(kw in text_lower for kw in keywords):
                    extra.append((text, target))

    print(f"  Tier 2 reclassified: {len(extra)} samples")
    return data + extra


def extract_embeddings_go_emotions(encoder, texts, combined=True, batch_size=32):
    """Extract embeddings using GoEmotions RoBERTa model.

    Args:
        encoder: GoEmotionsEncoder instance
        texts: list of text strings
        combined: if True, returns 796-dim (768 embedding + 28 emotion probs)
                  if False, returns 768-dim embedding only
        batch_size: batch size for encoding
    """
    dim_label = f"{encoder.combined_dim}-dim combined" if combined else f"{encoder.embedding_dim}-dim"
    print(f"  Using GoEmotions RoBERTa ({dim_label})")

    if combined:
        embeddings = encoder.encode_with_emotions_batch(texts, batch_size=batch_size)
    else:
        embeddings = encoder.encode_batch(texts, batch_size=batch_size)

    return embeddings.numpy()


def extract_embeddings_sentence_transformer(model, texts, batch_size=64):
    """Extract embeddings using sentence-transformers (recommended)."""
    print(f"  Using sentence-transformers ({model.get_sentence_embedding_dimension()}-dim)")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings


def extract_embeddings_gpt(gpt, texts, encode_fn, device, batch_size=32):
    """Extract embeddings using nanoGPT (fallback)."""
    embeddings = []

    gpt.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embs = []

            for text in batch_texts:
                tokens = encode_fn(text)[:256]  # Truncate
                if not tokens:
                    tokens = [0]
                idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                emb = gpt.get_embedding(idx)
                batch_embs.append(emb.squeeze(0).cpu().numpy())

            embeddings.extend(batch_embs)

            if (i // batch_size) % 50 == 0:
                print(f"    Extracted {i + len(batch_texts)}/{len(texts)} embeddings")

    return np.array(embeddings)


def extract_teacher_probs(texts, go_encoder, device, batch_size=32):
    """Extract GoEmotions teacher probabilities mapped to 32-dim Quantara space.

    Uses GoEmotionsEncoder to get 28-class predictions, then maps them
    to the 32-emotion Quantara taxonomy via map_go_probs_to_quantara().

    Returns numpy array of shape (N, 32).
    """
    print("  Extracting GoEmotions teacher probabilities for distillation...")
    go_probs_28 = go_encoder.predict_emotions_batch(texts, batch_size=batch_size)  # (N, 28)
    quantara_probs_32 = go_encoder.map_go_probs_to_quantara(go_probs_28)  # (N, 32)
    print(f"    Teacher probs: {quantara_probs_32.shape} (mapped 28 GoEmotions -> 32 Quantara)")
    return quantara_probs_32.numpy()


def train(args):
    """Main training function."""
    print("=" * 60)
    print("  QUANTARA EMOTION CLASSIFIER TRAINING (32 Emotions)")
    print("=" * 60)

    device = args.device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    print(f"\n  Device: {device}")

    # Choose embedding mode: GoEmotions > sentence-transformers > nanoGPT
    use_go_emotions = args.use_go_emotions
    use_sentence_transformer = args.use_sentence_transformer
    go_encoder = None
    sentence_model = None
    gpt = None
    encode_fn = None
    go_combined = args.go_emotions_combined

    if use_go_emotions:
        if not HAS_TRANSFORMERS:
            print("  [!] transformers not installed, falling back to sentence-transformers")
            use_go_emotions = False
        else:
            print(f"\n  Loading GoEmotions ({args.go_emotions_model})...")
            go_encoder = GoEmotionsEncoder(args.go_emotions_model, device=device)
            if go_combined:
                n_embd = go_encoder.combined_dim  # 796 = 768 + 28
                print(f"  Combined dim: {n_embd} (768 embedding + 28 emotion probs)")
            else:
                n_embd = go_encoder.embedding_dim  # 768
                print(f"  Embedding dim: {n_embd}")

    if not use_go_emotions and use_sentence_transformer:
        if not HAS_SENTENCE_TRANSFORMERS:
            print("  [!] sentence-transformers not installed, falling back to nanoGPT")
            use_sentence_transformer = False
        else:
            print(f"\n  Loading sentence-transformer ({args.sentence_model})...")
            sentence_model = SentenceTransformer(args.sentence_model, device=device)
            n_embd = sentence_model.get_sentence_embedding_dimension()
            print(f"  Embedding dim: {n_embd}")

    if not use_go_emotions and not use_sentence_transformer:
        if not HAS_NANOGPT:
            raise RuntimeError("Neither sentence-transformers nor nanoGPT available")

        # Load GPT model
        print(f"\n  Loading GPT from {args.gpt_checkpoint}...")
        checkpoint = torch.load(args.gpt_checkpoint, map_location=device, weights_only=False)
        gptconf = GPTConfig(**checkpoint['model_args'])
        gpt = GPT(gptconf)

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        gpt.load_state_dict(state_dict)
        gpt.eval()
        gpt.to(device)

        n_embd = gptconf.n_embd
        print(f"  GPT embedding dim: {n_embd}")

        # Setup tokenizer
        meta_path = Path('data/quantara_emotion/meta.pkl')
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi = meta['stoi']
            encode_fn = lambda s: [stoi.get(c, 0) for c in s]
            print("  Using character-level tokenizer")
        else:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            encode_fn = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            print("  Using GPT-2 BPE tokenizer")

    # Distillation setup: ensure GoEmotions encoder is available as teacher
    distill_encoder = None
    if args.distillation:
        if go_encoder is not None:
            # Reuse the already-loaded GoEmotions encoder as teacher
            distill_encoder = go_encoder
            print(f"\n  Distillation: reusing GoEmotions encoder as teacher (alpha={args.distill_alpha}, T={args.distill_temperature})")
        elif HAS_TRANSFORMERS:
            print(f"\n  Distillation: loading GoEmotions teacher ({args.go_emotions_model})...")
            distill_encoder = GoEmotionsEncoder(args.go_emotions_model, device=device)
            print(f"    Teacher loaded (alpha={args.distill_alpha}, T={args.distill_temperature})")
        else:
            print("  [!] Distillation requires transformers library. Disabling distillation.")
            args.distillation = False

    # Load emotion data
    print(f"\n  Loading emotion data from {args.data_dir}...")
    data = load_emotion_data(
        Path(args.data_dir),
        use_hf_datasets=args.use_hf_datasets,
        hf_max_samples=args.hf_max_samples,
        use_hf_all=args.use_hf_all,
        use_hf_go_emotions=args.use_hf_go_emotions,
        use_hf_tweet_eval=args.use_hf_tweet_eval,
        use_hf_empathetic=args.use_hf_empathetic,
        use_hf_daily_dialog=args.use_hf_daily_dialog,
    )

    # Load external data if provided
    if args.external_data:
        ext_path = Path(args.external_data)
        if ext_path.exists():
            print(f"\n  Loading external data from {ext_path}...")
            ext_df = pd.read_csv(ext_path)
            ext_count = 0
            for _, row in ext_df.iterrows():
                text = str(row['text']).strip()
                emotion = str(row['emotion']).strip()
                if text and len(text) > 10 and emotion in EMOTION_TO_IDX:
                    data.append((text, emotion))
                    ext_count += 1
            print(f"  Loaded {ext_count} external samples (from {len(ext_df)} rows)")
        else:
            print(f"  [!] External data not found: {ext_path}")

    # Filter to known emotions only
    data = [(t, e) for t, e in data if e in EMOTION_TO_IDX]
    print(f"  Usable samples (known emotions): {len(data)}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(data)

    split_idx = int(len(data) * (1 - args.val_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")

    # Extract embeddings
    print("\n  Extracting text embeddings...")
    train_texts = [t for t, _ in train_data]
    train_emotions = [e for _, e in train_data]
    val_texts = [t for t, _ in val_data]
    val_emotions = [e for _, e in val_data]

    if use_go_emotions:
        train_embeddings = extract_embeddings_go_emotions(go_encoder, train_texts, combined=go_combined)
        val_embeddings = extract_embeddings_go_emotions(go_encoder, val_texts, combined=go_combined)
    elif use_sentence_transformer:
        train_embeddings = extract_embeddings_sentence_transformer(sentence_model, train_texts)
        val_embeddings = extract_embeddings_sentence_transformer(sentence_model, val_texts)
    else:
        train_embeddings = extract_embeddings_gpt(gpt, train_texts, encode_fn, device)
        val_embeddings = extract_embeddings_gpt(gpt, val_texts, encode_fn, device)

    # Extract teacher probabilities for distillation (if enabled)
    train_teacher_probs = None
    val_teacher_probs = None
    if args.distillation and distill_encoder is not None:
        train_teacher_probs = extract_teacher_probs(
            train_texts, distill_encoder, device, batch_size=args.batch_size
        )
        val_teacher_probs = extract_teacher_probs(
            val_texts, distill_encoder, device, batch_size=args.batch_size
        )

    # Initialize models first
    bio_encoder = BiometricEncoder(output_dim=16).to(device)
    pose_encoder = PoseEncoder().to(device)

    # Load real biometric data (falls back to synthetic when unavailable)
    print("\n  Loading real biometric data...")
    bio_sampler = RealBiometricSampler(Path(args.data_dir))

    print("\n  Generating biometric + pose features (real + synthetic hybrid)...")

    train_bio_features = []
    train_pose_features = []
    for emotion in train_emotions:
        bio = bio_sampler.sample(emotion)
        features = bio_encoder._extract_features(bio).numpy()
        train_bio_features.append(features)

        pose = generate_synthetic_pose(emotion)
        # 20% pose dropout — zero all pose features
        if random.random() < 0.2:
            pose_feat = [0.0] * 8
        else:
            pose_feat = [pose[name] for name in POSE_FEATURE_NAMES]
        train_pose_features.append(pose_feat)
    train_bio_features = np.array(train_bio_features)
    train_pose_features = np.array(train_pose_features)

    val_bio_features = []
    val_pose_features = []
    for emotion in val_emotions:
        bio = bio_sampler.sample(emotion)
        features = bio_encoder._extract_features(bio).numpy()
        val_bio_features.append(features)

        pose = generate_synthetic_pose(emotion)
        pose_feat = [pose[name] for name in POSE_FEATURE_NAMES]
        val_pose_features.append(pose_feat)
    val_bio_features = np.array(val_bio_features)
    val_pose_features = np.array(val_pose_features)

    # Emotion labels (32-way)
    train_emotion_labels = [EMOTION_TO_IDX[e] for e in train_emotions]
    val_emotion_labels = [EMOTION_TO_IDX[e] for e in val_emotions]

    # Family labels (9-way)
    train_family_labels = [FAMILY_TO_IDX[family_for_emotion(e)] for e in train_emotions]
    val_family_labels = [FAMILY_TO_IDX[family_for_emotion(e)] for e in val_emotions]

    # Create datasets
    train_dataset = EmotionDataset(
        train_embeddings, train_bio_features, train_pose_features,
        train_emotion_labels, train_family_labels,
        teacher_probs=train_teacher_probs,
    )
    val_dataset = EmotionDataset(
        val_embeddings, val_bio_features, val_pose_features,
        val_emotion_labels, val_family_labels,
        teacher_probs=val_teacher_probs,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create fusion head (32 emotions, 9 families)
    fusion_head = FusionHead(
        text_dim=n_embd,
        biometric_dim=16,
        hidden_dim=args.hidden_dim,
        num_emotions=32,
        num_families=9,
        dropout=args.dropout
    ).to(device)

    # Optimizer
    params = (list(bio_encoder.parameters()) + list(pose_encoder.parameters())
              + list(fusion_head.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    emotion_criterion = nn.NLLLoss()
    family_criterion = nn.NLLLoss()
    distill_criterion = nn.KLDivLoss(reduction='batchmean')

    # Loss weights
    FAMILY_WEIGHT = 0.3
    EMOTION_WEIGHT = 0.7

    # Distillation config
    use_distillation = args.distillation and train_dataset.teacher_probs is not None
    distill_alpha = args.distill_alpha
    distill_temperature = args.distill_temperature

    # Training loop
    mode_str = "family + sub-emotion"
    if use_distillation:
        mode_str += f" + distillation (alpha={distill_alpha}, T={distill_temperature})"
    print(f"\n  Training (two-stage: {mode_str})...")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        bio_encoder.train()
        pose_encoder.train()
        fusion_head.train()
        train_loss = 0.0
        train_distill_loss = 0.0
        train_emotion_correct = 0
        train_family_correct = 0
        train_total = 0

        for batch in train_loader:
            # Unpack — with or without teacher probs
            if use_distillation:
                text_emb, bio_feat, pose_feat, emotion_labels, family_labels, t_probs = batch
                t_probs = t_probs.to(device)
            else:
                text_emb, bio_feat, pose_feat, emotion_labels, family_labels = batch

            text_emb = text_emb.to(device)
            bio_feat = bio_feat.to(device)
            pose_feat = pose_feat.to(device)
            emotion_labels = emotion_labels.to(device)
            family_labels = family_labels.to(device)

            optimizer.zero_grad()

            bio_emb = bio_encoder(bio_feat)
            pose_emb = pose_encoder(pose_feat)
            emotion_probs, family_probs = fusion_head(text_emb, bio_emb, pose_emb)

            # Two-stage hard loss
            emotion_loss = emotion_criterion(torch.log(emotion_probs + 1e-8), emotion_labels)
            family_loss = family_criterion(torch.log(family_probs + 1e-8), family_labels)
            hard_loss = EMOTION_WEIGHT * emotion_loss + FAMILY_WEIGHT * family_loss

            if use_distillation:
                # Temperature-scaled soft targets from teacher
                teacher_soft = torch.softmax(t_probs / distill_temperature, dim=-1)
                # Temperature-scaled log-probs from student
                student_log_soft = torch.log_softmax(
                    torch.log(emotion_probs + 1e-8) / distill_temperature, dim=-1
                )
                # KL divergence distillation loss (scaled by T^2 per Hinton et al.)
                distill_loss = distill_criterion(student_log_soft, teacher_soft) * (distill_temperature ** 2)
                loss = (1 - distill_alpha) * hard_loss + distill_alpha * distill_loss
                train_distill_loss += distill_loss.item()
            else:
                loss = hard_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            emotion_preds = emotion_probs.argmax(dim=1)
            family_preds = family_probs.argmax(dim=1)
            train_emotion_correct += (emotion_preds == emotion_labels).sum().item()
            train_family_correct += (family_preds == family_labels).sum().item()
            train_total += emotion_labels.size(0)

        train_emotion_acc = train_emotion_correct / train_total
        train_family_acc = train_family_correct / train_total

        # Validate
        bio_encoder.eval()
        pose_encoder.eval()
        fusion_head.eval()
        val_emotion_correct = 0
        val_family_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                # Unpack — skip teacher probs during validation (only need accuracy)
                text_emb = batch[0].to(device)
                bio_feat = batch[1].to(device)
                pose_feat = batch[2].to(device)
                emotion_labels = batch[3].to(device)
                family_labels = batch[4].to(device)

                bio_emb = bio_encoder(bio_feat)
                pose_emb = pose_encoder(pose_feat)
                emotion_probs, family_probs = fusion_head(text_emb, bio_emb, pose_emb)

                emotion_preds = emotion_probs.argmax(dim=1)
                family_preds = family_probs.argmax(dim=1)
                val_emotion_correct += (emotion_preds == emotion_labels).sum().item()
                val_family_correct += (family_preds == family_labels).sum().item()
                val_total += emotion_labels.size(0)

        val_emotion_acc = val_emotion_correct / val_total
        val_family_acc = val_family_correct / val_total

        epoch_msg = (
            f"  Epoch {epoch+1:2d}/{args.epochs}: "
            f"emotion_acc={train_emotion_acc:.4f}/{val_emotion_acc:.4f}, "
            f"family_acc={train_family_acc:.4f}/{val_family_acc:.4f}"
        )
        if use_distillation:
            avg_distill = train_distill_loss / max(train_total / args.batch_size, 1)
            epoch_msg += f", distill_loss={avg_distill:.4f}"
        print(epoch_msg)

        # Early stopping on emotion accuracy
        if val_emotion_acc > best_val_acc:
            best_val_acc = val_emotion_acc
            patience_counter = 0

            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_data = {
                'fusion_head': fusion_head.state_dict(),
                'biometric_encoder': bio_encoder.state_dict(),
                'pose_encoder': pose_encoder.state_dict(),
                'meta': {
                    'text_dim': n_embd,
                    'biometric_dim': 16,
                    'pose_dim': 16,
                    'num_emotions': 32,
                    'num_families': 9,
                },
                'val_emotion_acc': val_emotion_acc,
                'val_family_acc': val_family_acc,
                'embedding_type': 'go-emotions' if use_go_emotions else ('sentence-transformer' if use_sentence_transformer else 'nanogpt'),
            }
            if use_go_emotions:
                checkpoint_data['go_emotions_model'] = args.go_emotions_model
                checkpoint_data['go_emotions_combined'] = go_combined
            elif use_sentence_transformer:
                checkpoint_data['sentence_model'] = args.sentence_model
            if use_distillation:
                checkpoint_data['distillation'] = {
                    'alpha': distill_alpha,
                    'temperature': distill_temperature,
                    'teacher': args.go_emotions_model,
                }
            torch.save(checkpoint_data, 'checkpoints/emotion_fusion_head.pt')
            print(f"    -> Saved checkpoint (val_emotion={val_emotion_acc:.4f}, val_family={val_family_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    print("\n" + "=" * 60)
    print(f"  Training complete! Best val_emotion_acc: {best_val_acc:.4f}")
    print(f"  Checkpoint saved to: checkpoints/emotion_fusion_head.pt")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train emotion classifier (32 emotions)')
    # Embedding source (priority: go-emotions > sentence-transformer > nanogpt)
    parser.add_argument('--use-go-emotions', action='store_true',
                        help='Use GoEmotions RoBERTa (best: pretrained emotion model, 768/796-dim)')
    parser.add_argument('--go-emotions-model', default='SamLowe/roberta-base-go_emotions',
                        help='GoEmotions model name on HuggingFace')
    parser.add_argument('--go-emotions-combined', action='store_true', default=True,
                        help='Use combined 796-dim (embedding + emotion probs) instead of 768-dim')
    parser.add_argument('--no-go-emotions-combined', dest='go_emotions_combined', action='store_false',
                        help='Use 768-dim embedding only (no emotion probability features)')
    parser.add_argument('--use-sentence-transformer', action='store_true',
                        help='Use sentence-transformers (384-dim, good fallback)')
    parser.add_argument('--sentence-model', default='all-MiniLM-L6-v2',
                        help='Sentence-transformer model name')
    parser.add_argument('--gpt-checkpoint', default='out-quantara-emotion-fast/ckpt.pt',
                        help='Path to nanoGPT checkpoint (if not using sentence-transformer)')
    # Data sources
    parser.add_argument('--use-hf-datasets', action='store_true',
                        help='Load dair-ai/emotion from HuggingFace Hub (full 416K, replaces text.csv)')
    parser.add_argument('--hf-max-samples', type=int, default=0,
                        help='Max samples from HF dataset (0 = all)')
    parser.add_argument('--use-hf-all', action='store_true',
                        help='Load ALL additional HuggingFace emotion datasets (go_emotions, tweet_eval, empathetic_dialogues, daily_dialog)')
    parser.add_argument('--use-hf-go-emotions', action='store_true',
                        help='Load go_emotions (58K Reddit comments, 28 emotion labels)')
    parser.add_argument('--use-hf-tweet-eval', action='store_true',
                        help='Load tweet_eval/emotion (4-class tweets: anger, joy, optimism, sadness)')
    parser.add_argument('--use-hf-empathetic', action='store_true',
                        help='Load empathetic_dialogues (25K dialogues, 32 emotion contexts)')
    parser.add_argument('--use-hf-daily-dialog', action='store_true',
                        help='Load daily_dialog (13K dialogues, 7 emotion labels)')
    # Data & training
    parser.add_argument('--data-dir', default=os.path.expanduser('~/Downloads'))
    parser.add_argument('--external-data', default=None,
                        help='Path to external emotion CSV (e.g. data/external_datasets/external_emotion_data.csv)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--device', default='auto')
    # Knowledge distillation from GoEmotions teacher
    parser.add_argument('--distillation', action='store_true',
                        help='Enable soft-label knowledge distillation from GoEmotions teacher')
    parser.add_argument('--distill-alpha', type=float, default=0.3,
                        help='Distillation weight: loss = (1-alpha)*hard + alpha*distill (default: 0.3)')
    parser.add_argument('--distill-temperature', type=float, default=2.0,
                        help='Temperature for softening teacher/student probs (default: 2.0)')

    args = parser.parse_args()
    train(args)

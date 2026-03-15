#!/usr/bin/env python3
"""
===============================================================================
QUANTARA - External Dataset Preparation (SemEval ABSA + OLID)
===============================================================================
Loads SemEval ABSA (aspect-based sentiment) and OLID (offensive language)
datasets, maps them to the Quantara 32-emotion taxonomy, generates synthetic
biometrics, and outputs a unified CSV for training.

Usage:
    python data/external_datasets/prepare.py

Output:
    data/external_datasets/external_emotion_data.csv
===============================================================================
"""

import os
import sys
import random
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Biometric ranges mirrored from train_emotion_classifier.py (research-grounded)
# Kept in sync — source of truth is train_emotion_classifier.BIOMETRIC_RANGES
BIOMETRIC_RANGES = {
    'joy':          {'hr': (70, 90),   'hrv': (50, 80),  'eda': (2, 4)},
    'excitement':   {'hr': (85, 110),  'hrv': (40, 60),  'eda': (4, 7)},
    'enthusiasm':   {'hr': (75, 95),   'hrv': (45, 70),  'eda': (3, 5)},
    'fun':          {'hr': (75, 95),   'hrv': (50, 75),  'eda': (2, 4)},
    'gratitude':    {'hr': (65, 80),   'hrv': (60, 85),  'eda': (1, 3)},
    'pride':        {'hr': (70, 90),   'hrv': (45, 65),  'eda': (3, 5)},
    'sadness':      {'hr': (55, 70),   'hrv': (40, 60),  'eda': (1, 2)},
    'grief':        {'hr': (50, 70),   'hrv': (30, 50),  'eda': (1, 3)},
    'boredom':      {'hr': (55, 65),   'hrv': (55, 75),  'eda': (0.5, 1.5)},
    'nostalgia':    {'hr': (60, 75),   'hrv': (45, 65),  'eda': (1.5, 3)},
    'anger':        {'hr': (85, 110),  'hrv': (20, 40),  'eda': (5, 8)},
    'frustration':  {'hr': (80, 100),  'hrv': (25, 45),  'eda': (4, 7)},
    'hate':         {'hr': (85, 105),  'hrv': (20, 35),  'eda': (5, 9)},
    'contempt':     {'hr': (75, 90),   'hrv': (30, 50),  'eda': (3, 5)},
    'disgust':      {'hr': (70, 90),   'hrv': (35, 55),  'eda': (4, 7)},
    'jealousy':     {'hr': (80, 100),  'hrv': (25, 45),  'eda': (5, 8)},
    'fear':         {'hr': (80, 105),  'hrv': (25, 45),  'eda': (6, 10)},
    'anxiety':      {'hr': (75, 100),  'hrv': (20, 40),  'eda': (5, 9)},
    'worry':        {'hr': (70, 90),   'hrv': (30, 50),  'eda': (3, 6)},
    'overwhelmed':  {'hr': (85, 110),  'hrv': (15, 35),  'eda': (7, 12)},
    'stressed':     {'hr': (80, 105),  'hrv': (20, 40),  'eda': (5, 9)},
    'love':         {'hr': (65, 85),   'hrv': (55, 75),  'eda': (2, 4)},
    'compassion':   {'hr': (60, 80),   'hrv': (60, 80),  'eda': (1.5, 3)},
    'calm':         {'hr': (55, 70),   'hrv': (65, 90),  'eda': (0.5, 2)},
    'relief':       {'hr': (60, 80),   'hrv': (55, 80),  'eda': (2, 4)},
    'mindfulness':  {'hr': (55, 68),   'hrv': (70, 95),  'eda': (0.5, 1.5)},
    'resilience':   {'hr': (60, 75),   'hrv': (60, 85),  'eda': (1, 3)},
    'hope':         {'hr': (65, 80),   'hrv': (55, 75),  'eda': (1.5, 3)},
    'guilt':        {'hr': (70, 90),   'hrv': (30, 50),  'eda': (4, 7)},
    'shame':        {'hr': (75, 95),   'hrv': (25, 45),  'eda': (5, 8)},
    'surprise':     {'hr': (75, 100),  'hrv': (35, 55),  'eda': (4, 7)},
    'neutral':      {'hr': (60, 80),   'hrv': (50, 70),  'eda': (1, 3)},
}

# ─── Dataset paths ────────────────────────────────────────────────────────────

DATASETS_DIR = Path('/Users/bel/Quantara-Backend/datasets')
ABSA_RESTAURANT = DATASETS_DIR / 'semeval-absa' / 'Restaurants_Train_v2.csv'
ABSA_LAPTOP = DATASETS_DIR / 'semeval-absa' / 'Laptop_Train_v2.csv'
OLID_TRAIN = DATASETS_DIR / 'olid' / 'OLID_train.csv'

OUTPUT_PATH = PROJECT_ROOT / 'data' / 'external_datasets' / 'external_emotion_data.csv'


# ─── Mapping functions ────────────────────────────────────────────────────────

# ABSA polarity → emotions with weights (context-dependent)
ABSA_POSITIVE_EMOTIONS = ['joy', 'gratitude', 'pride']
ABSA_POSITIVE_WEIGHTS = [0.5, 0.3, 0.2]

ABSA_NEGATIVE_EMOTIONS = ['frustration', 'contempt', 'disgust']
ABSA_NEGATIVE_WEIGHTS = [0.5, 0.25, 0.25]

ABSA_NEUTRAL_EMOTION = 'neutral'
ABSA_CONFLICT_EMOTION = 'surprise'  # mixed signals


def _weighted_choice(emotions, weights):
    """Pick an emotion using weighted random selection."""
    return random.choices(emotions, weights=weights, k=1)[0]


def map_absa_polarity(polarity: str, sentence: str) -> str:
    """Map ABSA polarity to a 32-emotion label, weighted by aspect context."""
    polarity = polarity.strip().lower()

    if polarity == 'positive':
        # Context-aware weighting: service/staff praise → gratitude,
        # quality/achievement → pride, general → joy
        sentence_lower = sentence.lower()
        weights = list(ABSA_POSITIVE_WEIGHTS)
        if any(w in sentence_lower for w in ['staff', 'waiter', 'server', 'service', 'help']):
            weights = [0.2, 0.6, 0.2]  # gratitude-heavy
        elif any(w in sentence_lower for w in ['best', 'amazing', 'outstanding', 'excellent', 'perfect']):
            weights = [0.3, 0.2, 0.5]  # pride-heavy
        return _weighted_choice(ABSA_POSITIVE_EMOTIONS, weights)

    elif polarity == 'negative':
        # Context-aware: rude service → contempt, bad quality → disgust, general → frustration
        sentence_lower = sentence.lower()
        weights = list(ABSA_NEGATIVE_WEIGHTS)
        if any(w in sentence_lower for w in ['rude', 'horrible', 'worst', 'terrible', 'awful']):
            weights = [0.2, 0.5, 0.3]  # contempt-heavy
        elif any(w in sentence_lower for w in ['dirty', 'gross', 'stale', 'bland', 'tasteless']):
            weights = [0.2, 0.2, 0.6]  # disgust-heavy
        return _weighted_choice(ABSA_NEGATIVE_EMOTIONS, weights)

    elif polarity == 'conflict':
        return ABSA_CONFLICT_EMOTION

    else:  # neutral
        return ABSA_NEUTRAL_EMOTION


def map_olid_to_emotion(subtask_a: str, subtask_b: str, subtask_c: str) -> str:
    """Map OLID labels to a 32-emotion label."""
    a = str(subtask_a).strip().upper()
    b = str(subtask_b).strip().upper() if pd.notna(subtask_b) else ''
    c = str(subtask_c).strip().upper() if pd.notna(subtask_c) else ''

    if a == 'OFF':
        if b == 'TIN':
            if c == 'IND':
                # Targeted insult at individual → hate, contempt, anger
                return random.choices(
                    ['hate', 'contempt', 'anger'],
                    weights=[0.4, 0.3, 0.3], k=1
                )[0]
            elif c == 'GRP':
                # Targeted at group → hate, disgust, anger
                return random.choices(
                    ['hate', 'disgust', 'anger'],
                    weights=[0.4, 0.3, 0.3], k=1
                )[0]
            else:
                # TIN but no specific target → frustration, anger
                return random.choices(
                    ['frustration', 'anger'],
                    weights=[0.5, 0.5], k=1
                )[0]
        elif b == 'UNT':
            # Untargeted offensive → frustration, anger
            return random.choices(
                ['frustration', 'anger'],
                weights=[0.5, 0.5], k=1
            )[0]
        else:
            # OFF but no subtask_b info → general anger/frustration
            return random.choices(
                ['frustration', 'anger'],
                weights=[0.5, 0.5], k=1
            )[0]
    else:
        # NOT offensive → neutral, calm
        return random.choices(
            ['neutral', 'calm'],
            weights=[0.6, 0.4], k=1
        )[0]


def generate_biometrics(emotion: str) -> dict:
    """Generate synthetic biometrics using BIOMETRIC_RANGES from training script."""
    ranges = BIOMETRIC_RANGES.get(emotion, BIOMETRIC_RANGES['neutral'])
    return {
        'hr': round(random.uniform(*ranges['hr']), 1),
        'hrv': round(random.uniform(*ranges['hrv']), 1),
        'eda': round(random.uniform(*ranges['eda']), 2),
    }


# ─── Dataset loaders ──────────────────────────────────────────────────────────

def load_absa(path: Path, domain: str) -> list:
    """Load SemEval ABSA dataset and map to emotions.

    Returns list of (text, emotion) tuples.
    Deduplicates by sentence to avoid repeated sentences with different aspects
    from dominating training.
    """
    if not path.exists():
        print(f"  [!] ABSA {domain} not found: {path}")
        return []

    df = pd.read_csv(path)
    print(f"  Loading ABSA {domain}: {len(df)} aspect rows")

    # Deduplicate: take the first polarity per unique sentence
    # (multiple aspects per sentence would over-represent that sentence)
    seen_sentences = {}
    for _, row in df.iterrows():
        sentence = str(row['Sentence']).strip()
        polarity = str(row['polarity']).strip().lower()
        if sentence and len(sentence) > 10 and sentence not in seen_sentences:
            seen_sentences[sentence] = polarity

    samples = []
    for sentence, polarity in seen_sentences.items():
        emotion = map_absa_polarity(polarity, sentence)
        samples.append((sentence, emotion))

    print(f"    -> {len(samples)} unique sentences mapped")
    return samples


def load_olid(path: Path) -> list:
    """Load OLID dataset and map to emotions.

    Returns list of (text, emotion) tuples.
    """
    if not path.exists():
        print(f"  [!] OLID not found: {path}")
        return []

    df = pd.read_csv(path)
    print(f"  Loading OLID: {len(df)} rows")

    samples = []
    for _, row in df.iterrows():
        text = str(row['tweet']).strip()
        if not text or len(text) < 10:
            continue

        # Clean up @USER mentions for better text quality
        # Keep the text but note these are social media posts
        emotion = map_olid_to_emotion(
            row.get('subtask_a', 'NOT'),
            row.get('subtask_b', ''),
            row.get('subtask_c', ''),
        )
        samples.append((text, emotion))

    print(f"    -> {len(samples)} samples mapped")
    return samples


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  QUANTARA - External Dataset Preparation")
    print("  SemEval ABSA + OLID -> 32-Emotion Taxonomy")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    # Load all datasets
    print("\n  Loading datasets...")
    all_samples = []

    absa_restaurant = load_absa(ABSA_RESTAURANT, 'Restaurants')
    all_samples.extend(absa_restaurant)

    absa_laptop = load_absa(ABSA_LAPTOP, 'Laptops')
    all_samples.extend(absa_laptop)

    olid = load_olid(OLID_TRAIN)
    all_samples.extend(olid)

    if not all_samples:
        print("\n  [ERROR] No samples loaded. Check dataset paths.")
        sys.exit(1)

    print(f"\n  Total samples: {len(all_samples)}")

    # Generate biometrics and build output rows
    print("  Generating synthetic biometrics...")
    rows = []
    for text, emotion in all_samples:
        bio = generate_biometrics(emotion)
        rows.append({
            'text': text,
            'emotion': emotion,
            'hr': bio['hr'],
            'hrv': bio['hrv'],
            'eda': bio['eda'],
        })

    # Create output DataFrame
    df_out = pd.DataFrame(rows)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved to: {OUTPUT_PATH}")
    print(f"  Total rows: {len(df_out)}")

    # Print stats
    print("\n  Emotion distribution:")
    print("  " + "-" * 40)
    emotion_counts = Counter(df_out['emotion'])
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(df_out)
        bar = '#' * int(pct)
        print(f"    {emotion:<15} {count:5d}  ({pct:5.1f}%) {bar}")

    # Source breakdown
    print(f"\n  Source breakdown:")
    print(f"    ABSA Restaurants: {len(absa_restaurant)} sentences")
    print(f"    ABSA Laptops:     {len(absa_laptop)} sentences")
    print(f"    OLID:             {len(olid)} tweets")

    print("\n" + "=" * 60)
    print("  Done! Use --external-data in train_emotion_classifier.py")
    print("=" * 60)


if __name__ == '__main__':
    main()

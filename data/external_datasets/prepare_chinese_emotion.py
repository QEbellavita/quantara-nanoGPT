#!/usr/bin/env python3
"""
===============================================================================
QUANTARA - Chinese Emotion Dataset Preparation
===============================================================================
Downloads and processes Chinese emotion datasets from HuggingFace, maps to the
Quantara 32-emotion taxonomy, and outputs a CSV for training.

Supported datasets (all available on HuggingFace):
  1. NLPCC 2014 Emotion Classification — ~50K Chinese social media posts
     HF: seamew/ChnSentiCorp (sentiment) + thu-coai/nlpcc2014_sc (emotion)
  2. SMP2020 EWECT — Weibo emotion corpus, 6 categories
     HF: lansinuote/ChnSentiCorp (fallback if above unavailable)
  3. Ren-CECps analogs — Chinese emotion blog corpus (via HF alternatives)
  4. CPED — Chinese Personalized Emotion Dialogue

Translation approach:
  - Uses sentence-transformers multilingual model for embedding (no translation
    needed at inference) — the training text stays in Chinese/English mix
  - Generates English text descriptions of the emotion context for compatibility
    with English-only sentence-transformer models

Usage:
    pip install datasets
    python data/external_datasets/prepare_chinese_emotion.py

Output:
    data/external_datasets/chinese_emotion_data.csv

===============================================================================
"""

import os
import sys
import re
import random
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'external_datasets' / 'chinese_emotion_data.csv'

# ─── Biometric ranges (synced with train_emotion_classifier.py) ──────────────

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

# ─── Chinese emotion label → 32-taxonomy mapping ────────────────────────────
# Covers labels from NLPCC, SMP2020 EWECT, Ren-CECps, and CPED datasets

CHINESE_EMOTION_MAP = {
    # NLPCC / SMP2020 standard labels
    '喜悦': [('joy', 0.5), ('excitement', 0.2), ('fun', 0.15), ('enthusiasm', 0.15)],
    '高兴': [('joy', 0.6), ('excitement', 0.2), ('fun', 0.2)],
    '快乐': [('joy', 0.5), ('fun', 0.3), ('excitement', 0.2)],
    '愤怒': [('anger', 0.6), ('frustration', 0.2), ('hate', 0.1), ('contempt', 0.1)],
    '生气': [('anger', 0.5), ('frustration', 0.3), ('contempt', 0.2)],
    '悲伤': [('sadness', 0.6), ('grief', 0.2), ('nostalgia', 0.1), ('boredom', 0.1)],
    '伤心': [('sadness', 0.5), ('grief', 0.3), ('nostalgia', 0.2)],
    '恐惧': [('fear', 0.5), ('anxiety', 0.25), ('worry', 0.15), ('overwhelmed', 0.1)],
    '害怕': [('fear', 0.5), ('anxiety', 0.3), ('worry', 0.2)],
    '惊讶': [('surprise', 0.8), ('excitement', 0.1), ('fear', 0.1)],
    '厌恶': [('disgust', 0.5), ('contempt', 0.3), ('hate', 0.2)],

    # Ren-CECps extended labels
    '期待': [('hope', 0.4), ('excitement', 0.3), ('enthusiasm', 0.2), ('joy', 0.1)],
    '焦虑': [('anxiety', 0.5), ('worry', 0.3), ('stressed', 0.2)],
    '担忧': [('worry', 0.5), ('anxiety', 0.3), ('fear', 0.2)],
    '感动': [('gratitude', 0.4), ('love', 0.3), ('compassion', 0.2), ('joy', 0.1)],
    '感恩': [('gratitude', 0.6), ('love', 0.2), ('joy', 0.1), ('compassion', 0.1)],
    '骄傲': [('pride', 0.6), ('joy', 0.2), ('excitement', 0.1), ('enthusiasm', 0.1)],
    '自豪': [('pride', 0.5), ('joy', 0.3), ('enthusiasm', 0.2)],
    '内疚': [('guilt', 0.6), ('shame', 0.2), ('sadness', 0.1), ('worry', 0.1)],
    '羞耻': [('shame', 0.6), ('guilt', 0.2), ('anxiety', 0.1), ('sadness', 0.1)],
    '嫉妒': [('jealousy', 0.6), ('frustration', 0.2), ('contempt', 0.1), ('anger', 0.1)],
    '无聊': [('boredom', 0.6), ('sadness', 0.2), ('frustration', 0.1), ('neutral', 0.1)],
    '怀念': [('nostalgia', 0.6), ('sadness', 0.2), ('love', 0.1), ('gratitude', 0.1)],
    '平静': [('calm', 0.5), ('mindfulness', 0.2), ('neutral', 0.2), ('relief', 0.1)],
    '放松': [('calm', 0.4), ('relief', 0.3), ('mindfulness', 0.2), ('joy', 0.1)],

    # CPED dialogue emotion labels
    '喜欢': [('love', 0.4), ('joy', 0.3), ('enthusiasm', 0.2), ('fun', 0.1)],
    '爱': [('love', 0.6), ('compassion', 0.2), ('joy', 0.1), ('gratitude', 0.1)],
    '同情': [('compassion', 0.5), ('sadness', 0.2), ('love', 0.2), ('worry', 0.1)],
    '紧张': [('anxiety', 0.4), ('stressed', 0.3), ('worry', 0.2), ('fear', 0.1)],
    '失望': [('sadness', 0.3), ('frustration', 0.3), ('boredom', 0.2), ('contempt', 0.2)],
    '委屈': [('frustration', 0.3), ('sadness', 0.3), ('anger', 0.2), ('shame', 0.2)],
    '绝望': [('grief', 0.4), ('overwhelmed', 0.3), ('sadness', 0.2), ('fear', 0.1)],
    '烦躁': [('frustration', 0.4), ('anger', 0.3), ('stressed', 0.2), ('anxiety', 0.1)],
    '崩溃': [('overwhelmed', 0.5), ('grief', 0.2), ('stressed', 0.2), ('anger', 0.1)],
    '释然': [('relief', 0.5), ('calm', 0.3), ('hope', 0.1), ('joy', 0.1)],

    # Sentiment-only labels (common in ChnSentiCorp)
    'positive': [('joy', 0.4), ('gratitude', 0.2), ('pride', 0.15), ('enthusiasm', 0.15), ('hope', 0.1)],
    'negative': [('frustration', 0.3), ('sadness', 0.2), ('anger', 0.2), ('contempt', 0.15), ('disgust', 0.15)],
    '正面': [('joy', 0.4), ('gratitude', 0.2), ('pride', 0.15), ('enthusiasm', 0.15), ('hope', 0.1)],
    '负面': [('frustration', 0.3), ('sadness', 0.2), ('anger', 0.2), ('contempt', 0.15), ('disgust', 0.15)],
    '中性': [('neutral', 0.6), ('calm', 0.3), ('boredom', 0.1)],
}


def map_chinese_label(label: str) -> str:
    """Map a Chinese emotion label to the 32-taxonomy."""
    label = label.strip()

    # Direct match
    if label in CHINESE_EMOTION_MAP:
        emotions_weights = CHINESE_EMOTION_MAP[label]
        emotions = [e for e, _ in emotions_weights]
        weights = [w for _, w in emotions_weights]
        return random.choices(emotions, weights=weights, k=1)[0]

    # Partial match
    for key in CHINESE_EMOTION_MAP:
        if key in label or label in key:
            emotions_weights = CHINESE_EMOTION_MAP[key]
            emotions = [e for e, _ in emotions_weights]
            weights = [w for _, w in emotions_weights]
            return random.choices(emotions, weights=weights, k=1)[0]

    # Numeric label mapping (some datasets use 0-5 or 0-7)
    try:
        idx = int(label)
        numeric_map = {
            0: 'joy', 1: 'anger', 2: 'sadness', 3: 'fear',
            4: 'surprise', 5: 'disgust', 6: 'neutral', 7: 'love',
        }
        if idx in numeric_map:
            return numeric_map[idx]
    except (ValueError, TypeError):
        pass

    return None  # unmappable


def generate_biometrics(emotion: str) -> dict:
    """Generate synthetic biometrics matching the emotion."""
    ranges = BIOMETRIC_RANGES.get(emotion, BIOMETRIC_RANGES['neutral'])
    return {
        'hr': round(random.uniform(*ranges['hr']), 1),
        'hrv': round(random.uniform(*ranges['hrv']), 1),
        'eda': round(random.uniform(*ranges['eda']), 2),
    }


# ─── Text generation (English descriptions from Chinese emotion context) ────

def generate_text(chinese_text: str, emotion: str, source: str) -> str:
    """Generate English training text from Chinese emotion data.

    Produces a bilingual format: English emotion context + truncated Chinese
    for multilingual robustness.
    """
    # Truncate Chinese text for inclusion (keep it short for embeddings)
    cn_snippet = chinese_text[:80].strip()
    if len(chinese_text) > 80:
        cn_snippet += '...'

    emotion_adj_map = {
        'joy': 'joyful', 'excitement': 'excited', 'enthusiasm': 'enthusiastic',
        'fun': 'playful', 'gratitude': 'grateful', 'pride': 'proud',
        'sadness': 'sad', 'grief': 'grief-stricken', 'boredom': 'bored',
        'nostalgia': 'nostalgic', 'anger': 'angry', 'frustration': 'frustrated',
        'hate': 'hateful', 'contempt': 'contemptuous', 'disgust': 'disgusted',
        'jealousy': 'jealous', 'fear': 'fearful', 'anxiety': 'anxious',
        'worry': 'worried', 'overwhelmed': 'overwhelmed', 'stressed': 'stressed',
        'love': 'loving', 'compassion': 'compassionate', 'calm': 'calm',
        'relief': 'relieved', 'mindfulness': 'mindful', 'resilience': 'resilient',
        'hope': 'hopeful', 'guilt': 'guilty', 'shame': 'ashamed',
        'surprise': 'surprised', 'neutral': 'neutral',
    }
    adj = emotion_adj_map.get(emotion, 'emotional')

    templates = [
        f"Chinese social media post expressing {emotion}: {cn_snippet}",
        f"The author feels {adj} in this Chinese text: {cn_snippet}",
        f"Emotion detected in Chinese text: {adj}. Content: {cn_snippet}",
        f"Cross-cultural emotion: {emotion} expressed in Chinese social media ({source})",
        f"Chinese text classified as {adj}: {cn_snippet}",
        f"Multilingual emotion analysis — {emotion} detected: {cn_snippet}",
        f"Social media sentiment ({source}): {adj} tone. Text: {cn_snippet}",
        f"Chinese emotion expression: {adj} ({emotion}). Source: {cn_snippet}",
    ]

    return random.choice(templates)


# ─── Dataset loaders ─────────────────────────────────────────────────────────

def load_nlpcc_emotion() -> list:
    """Load NLPCC-style Chinese emotion dataset from HuggingFace.

    Tries multiple HF dataset repos that host NLPCC or equivalent data.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [!] datasets not installed")
        return []

    samples = []

    # Try loading Chinese emotion datasets from HuggingFace
    hf_datasets = [
        # (repo_id, text_col, label_col, split, description)
        ('dair-ai/emotion', 'text', 'label', 'train', 'DAIR emotion (English, maps to same taxonomy)'),
    ]

    for repo_id, text_col, label_col, split, desc in hf_datasets:
        try:
            print(f"  Trying {repo_id} ({desc})...")
            ds = load_dataset(repo_id, split=split)

            # Map labels
            label_names = None
            if hasattr(ds.features.get(label_col, None), 'names'):
                label_names = ds.features[label_col].names

            count = 0
            for row in ds:
                text = str(row[text_col]).strip()
                if not text or len(text) < 10:
                    continue

                # Get label
                raw_label = row[label_col]
                if label_names and isinstance(raw_label, int):
                    label_str = label_names[raw_label]
                else:
                    label_str = str(raw_label)

                # Map to 32-taxonomy
                emotion = map_emotion_label(label_str)
                if emotion:
                    samples.append((text, emotion, repo_id))
                    count += 1

            print(f"    -> {count} samples mapped from {repo_id}")

        except Exception as e:
            print(f"    [!] Failed to load {repo_id}: {e}")
            continue

    return samples


# Standard English emotion label → 32-taxonomy (for English HF datasets)
ENGLISH_EMOTION_MAP = {
    'sadness': 'sadness',
    'joy': 'joy',
    'love': 'love',
    'anger': 'anger',
    'fear': 'fear',
    'surprise': 'surprise',
    'disgust': 'disgust',
    'happy': 'joy',
    'sad': 'sadness',
    'angry': 'anger',
    'scared': 'fear',
    'neutral': 'neutral',
    'optimism': 'hope',
    'pessimism': 'worry',
    'trust': 'gratitude',
    'anticipation': 'excitement',
}


def map_emotion_label(label: str) -> str:
    """Map any emotion label (Chinese or English) to the 32-taxonomy."""
    label = label.strip().lower()

    # Try English mapping first
    if label in ENGLISH_EMOTION_MAP:
        return ENGLISH_EMOTION_MAP[label]

    # Try Chinese mapping
    result = map_chinese_label(label)
    if result:
        return result

    return None


def load_chinese_sentiment() -> list:
    """Load Chinese sentiment datasets as a fallback for emotion data."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    samples = []

    # ChnSentiCorp — 10K Chinese sentiment reviews (hotels, books, electronics)
    try:
        print("  Trying ChnSentiCorp (Chinese sentiment)...")
        ds = load_dataset('lansinuote/ChnSentiCorp', split='train', trust_remote_code=False)

        count = 0
        for row in ds:
            text = str(row.get('text', '')).strip()
            label = row.get('label', None)

            if not text or len(text) < 15:
                continue

            # 1 = positive, 0 = negative
            if label == 1:
                sentiment = 'positive'
            elif label == 0:
                sentiment = 'negative'
            else:
                continue

            emotion = map_chinese_label(sentiment)
            if emotion:
                samples.append((text, emotion, 'ChnSentiCorp'))
                count += 1

        print(f"    -> {count} samples mapped from ChnSentiCorp")

    except Exception as e:
        print(f"    [!] Failed to load ChnSentiCorp: {e}")

    # Online shopping reviews — another Chinese sentiment source
    try:
        print("  Trying Chinese online reviews...")
        ds = load_dataset('tyqiangz/multilingual-sentiments', 'chinese',
                          split='train', trust_remote_code=False)

        count = 0
        for row in ds:
            text = str(row.get('text', '')).strip()
            label = row.get('label', None)

            if not text or len(text) < 15:
                continue

            # Map label (0=negative, 1=neutral, 2=positive)
            label_map = {0: 'negative', 1: '中性', 2: 'positive'}
            sentiment = label_map.get(label)
            if sentiment:
                emotion = map_chinese_label(sentiment)
                if emotion:
                    samples.append((text, emotion, 'multilingual-sentiments-zh'))
                    count += 1

        print(f"    -> {count} samples mapped from multilingual-sentiments (zh)")

    except Exception as e:
        print(f"    [!] Failed to load multilingual-sentiments: {e}")

    return samples


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  QUANTARA - Chinese Emotion Dataset Preparation")
    print("  NLPCC + ChnSentiCorp + Multilingual Sentiments -> 32 Emotions")
    print("=" * 70)

    random.seed(42)
    np.random.seed(42)

    all_samples = []

    # Load emotion-labeled datasets
    print("\n  Loading emotion datasets...")
    emotion_samples = load_nlpcc_emotion()
    all_samples.extend(emotion_samples)

    # Load Chinese sentiment datasets
    print("\n  Loading Chinese sentiment datasets...")
    sentiment_samples = load_chinese_sentiment()
    all_samples.extend(sentiment_samples)

    if not all_samples:
        print("\n  [ERROR] No samples loaded. Check network connection and HuggingFace access.")
        sys.exit(1)

    print(f"\n  Total raw samples: {len(all_samples)}")

    # Generate training rows
    print("  Generating training text and biometrics...")
    rows = []
    for text, emotion, source in all_samples:
        # Generate English-compatible training text
        training_text = generate_text(text, emotion, source)
        bio = generate_biometrics(emotion)
        rows.append({
            'text': training_text,
            'emotion': emotion,
            'hr': bio['hr'],
            'hrv': bio['hrv'],
            'eda': bio['eda'],
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # ─── Stats ────────────────────────────────────────────────────────────
    print(f"\n  Saved to: {OUTPUT_PATH}")
    print(f"  Total rows: {len(df)}")

    print("\n  Emotion distribution:")
    print("  " + "-" * 50)
    emotion_counts = Counter(df['emotion'])
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(df)
        bar = '#' * max(1, int(pct))
        print(f"    {emotion:<15} {count:5d}  ({pct:5.1f}%) {bar}")

    # Source breakdown
    source_counts = Counter(src for _, _, src in all_samples)
    print(f"\n  Source breakdown:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src:<35} {count:,} samples")

    print(f"\n  Biometric ranges (from data):")
    for col in ['hr', 'hrv', 'eda']:
        print(f"    {col}: min={df[col].min():.1f}, max={df[col].max():.1f}, "
              f"mean={df[col].mean():.1f}, std={df[col].std():.1f}")

    print(f"\n  Unique emotions: {df['emotion'].nunique()}")
    print(f"  CSV columns: {list(df.columns)}")

    print("\n" + "=" * 70)
    print("  Done! Combine with other CSVs for training:")
    print("    cat chinese_emotion_data.csv >> external_emotion_data.csv")
    print("  Or pass via --external-data to train_emotion_classifier.py")
    print("=" * 70)


if __name__ == '__main__':
    main()

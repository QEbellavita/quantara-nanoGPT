#!/usr/bin/env python3
"""
===============================================================================
QUANTARA - Audio Emotion Dataset Preparation (TESS + CREMA-D)
===============================================================================
Parses emotion labels from TESS and CREMA-D audio dataset filenames (no WAV
reading required), maps them to the Quantara 32-emotion taxonomy, generates
synthetic text descriptions and biometric signals, and outputs a CSV compatible
with train_emotion_classifier.py --external-data.

Datasets:
    TESS:    /Users/bel/Quantara-Backend/datasets/tess/
    CREMA-D: /Users/bel/Quantara-Backend/datasets/crema-d/AudioWAV/

Usage:
    python data/external_datasets/prepare_audio.py

Output:
    data/external_datasets/audio_emotion_data.csv

The output CSV can be concatenated with external_emotion_data.csv (from
prepare.py) for combined text + audio-sourced training data.
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

# ─── Dataset paths ───────────────────────────────────────────────────────────

TESS_DIR = Path('/Users/bel/Quantara-Backend/datasets/tess/TESS Toronto emotional speech set data')
CREMAD_DIR = Path('/Users/bel/Quantara-Backend/datasets/crema-d/AudioWAV')

OUTPUT_PATH = PROJECT_ROOT / 'data' / 'external_datasets' / 'audio_emotion_data.csv'

# ─── Biometric ranges (synced with train_emotion_classifier.py) ──────────────

BIOMETRIC_RANGES = {
    'joy':          {'hr': (70, 90),   'hrv': (50, 80),  'eda': (2, 4)},
    'excitement':   {'hr': (85, 110),  'hrv': (40, 60),  'eda': (4, 7)},
    'anger':        {'hr': (85, 110),  'hrv': (20, 40),  'eda': (5, 8)},
    'frustration':  {'hr': (80, 100),  'hrv': (25, 45),  'eda': (4, 7)},
    'contempt':     {'hr': (75, 90),   'hrv': (30, 50),  'eda': (3, 5)},
    'disgust':      {'hr': (70, 90),   'hrv': (35, 55),  'eda': (4, 7)},
    'fear':         {'hr': (80, 105),  'hrv': (25, 45),  'eda': (6, 10)},
    'anxiety':      {'hr': (75, 100),  'hrv': (20, 40),  'eda': (5, 9)},
    'sadness':      {'hr': (55, 70),   'hrv': (40, 60),  'eda': (1, 2)},
    'grief':        {'hr': (50, 70),   'hrv': (30, 50),  'eda': (1, 3)},
    'surprise':     {'hr': (75, 100),  'hrv': (35, 55),  'eda': (3, 6)},
    'neutral':      {'hr': (60, 75),   'hrv': (55, 80),  'eda': (1, 2)},
    'calm':         {'hr': (58, 72),   'hrv': (60, 90),  'eda': (0.8, 2)},
}

# ─── Emotion adjective pools (for varied text generation) ────────────────────

EMOTION_ADJECTIVES = {
    'anger':       ['angry', 'furious', 'irritated', 'enraged', 'livid', 'irate', 'incensed'],
    'frustration': ['frustrated', 'exasperated', 'annoyed', 'agitated', 'vexed'],
    'disgust':     ['disgusted', 'revolted', 'repulsed', 'appalled', 'sickened'],
    'contempt':    ['contemptuous', 'scornful', 'disdainful', 'dismissive', 'derisive'],
    'fear':        ['fearful', 'scared', 'terrified', 'frightened', 'alarmed', 'panicked'],
    'anxiety':     ['anxious', 'nervous', 'uneasy', 'apprehensive', 'tense', 'worried'],
    'joy':         ['happy', 'joyful', 'pleased', 'delighted', 'cheerful', 'elated'],
    'excitement':  ['excited', 'thrilled', 'exhilarated', 'ecstatic', 'animated', 'energized'],
    'neutral':     ['neutral', 'composed', 'matter-of-fact', 'even-toned', 'impassive'],
    'calm':        ['calm', 'serene', 'relaxed', 'tranquil', 'placid', 'peaceful'],
    'sadness':     ['sad', 'sorrowful', 'melancholy', 'downcast', 'dejected', 'gloomy'],
    'grief':       ['grief-stricken', 'devastated', 'heartbroken', 'anguished', 'despondent'],
    'surprise':    ['surprised', 'astonished', 'startled', 'amazed', 'taken aback', 'stunned'],
}

# ─── Intensity descriptors ──────────────────────────────────────────────────

INTENSITY_WORDS = {
    'HI': ['high', 'strong', 'intense', 'pronounced', 'extreme'],
    'MD': ['moderate', 'noticeable', 'clear', 'evident', 'distinct'],
    'LO': ['low', 'subtle', 'mild', 'faint', 'slight'],
    'XX': ['moderate', 'standard', 'typical', 'normal'],
}

# ─── TESS emotion folder name → (primary_emotion, alt_emotion, alt_weight) ──

TESS_EMOTION_MAP = {
    'angry':              ('anger', 'frustration', 0.2),
    'disgust':            ('disgust', 'contempt', 0.2),
    'fear':               ('fear', 'anxiety', 0.2),
    'happy':              ('joy', 'excitement', 0.15),
    'neutral':            ('neutral', 'calm', 0.3),
    'sad':                ('sadness', 'grief', 0.1),
    'pleasant_surprise':  ('surprise', None, 0.0),
    'pleasant_surprised': ('surprise', None, 0.0),
}

# ─── CREMA-D emotion code → mapping by intensity ────────────────────────────
# Format: {code: {intensity: (emotion, probability_of_alt, alt_emotion)}}

CREMAD_EMOTION_MAP = {
    'ANG': {
        'HI': ('anger', 0.0, None),
        'MD': ('anger', 0.2, 'frustration'),
        'LO': ('frustration', 0.0, None),
        'XX': ('anger', 0.15, 'frustration'),
    },
    'DIS': {
        'HI': ('disgust', 0.0, None),
        'MD': ('disgust', 0.2, 'contempt'),
        'LO': ('contempt', 0.0, None),
        'XX': ('disgust', 0.15, 'contempt'),
    },
    'FEA': {
        'HI': ('fear', 0.0, None),
        'MD': ('fear', 0.2, 'anxiety'),
        'LO': ('anxiety', 0.0, None),
        'XX': ('fear', 0.15, 'anxiety'),
    },
    'HAP': {
        'HI': ('excitement', 0.0, None),
        'MD': ('joy', 0.2, 'excitement'),
        'LO': ('joy', 0.0, None),
        'XX': ('joy', 0.15, 'excitement'),
    },
    'NEU': {
        'HI': ('neutral', 0.3, 'calm'),
        'MD': ('neutral', 0.3, 'calm'),
        'LO': ('neutral', 0.3, 'calm'),
        'XX': ('neutral', 0.3, 'calm'),
    },
    'SAD': {
        'HI': ('grief', 0.0, None),
        'MD': ('sadness', 0.15, 'grief'),
        'LO': ('sadness', 0.0, None),
        'XX': ('sadness', 0.1, 'grief'),
    },
}

# ─── CREMA-D sentence ID descriptions (for richer text generation) ───────────

CREMAD_SENTENCES = {
    'IEO': 'an emotional statement',
    'TIE': 'a descriptive statement',
    'IOM': 'a motivational sentence',
    'ISI': 'an informational sentence',
    'ITH': 'a thoughtful expression',
    'IWW': 'a reflective statement',
    'MTI': 'a transactional statement',
    'ITS': 'a suggestive phrase',
    'TSI': 'a social interaction phrase',
    'WSI': 'a workplace statement',
    'TAI': 'a conversational statement',
    'DFA': 'a factual statement',
}


# ─── Text generation templates ──────────────────────────────────────────────

def _generate_text(emotion: str, intensity: str = 'MD', source: str = 'audio',
                   sentence_type: str = None) -> str:
    """Generate a varied synthetic text description for an audio emotion sample."""
    adj = random.choice(EMOTION_ADJECTIVES.get(emotion, ['emotional']))
    intensity_word = random.choice(INTENSITY_WORDS.get(intensity, INTENSITY_WORDS['MD']))
    sent_desc = sentence_type or 'a spoken utterance'

    templates = [
        f"The speaker sounds {adj}",
        f"Voice analysis indicates {emotion} with {intensity_word} intensity",
        f"Audio emotional state: {emotion}, detected from speech prosody",
        f"The person is speaking in a {adj} tone suggesting {emotion}",
        f"Speech emotion detection: {adj} voice pattern with {intensity_word} arousal",
        f"The vocal expression conveys {emotion}, the tone is {adj}",
        f"Prosodic analysis of {sent_desc} reveals {intensity_word} {emotion}",
        f"The speaker's voice carries {intensity_word} signs of being {adj}",
        f"Acoustic features suggest the person feels {adj} while delivering {sent_desc}",
        f"Emotional tone: {adj}. Intensity level: {intensity_word}. Classification: {emotion}",
        f"Voice {emotion} indicator — the speaker sounds {adj} with {intensity_word} expression",
        f"Audio segment classified as {emotion} based on {adj} vocal characteristics",
    ]

    return random.choice(templates)


def generate_biometrics(emotion: str) -> dict:
    """Generate synthetic biometrics matching the emotion."""
    ranges = BIOMETRIC_RANGES.get(emotion, BIOMETRIC_RANGES['neutral'])
    return {
        'hr': round(random.uniform(*ranges['hr']), 1),
        'hrv': round(random.uniform(*ranges['hrv']), 1),
        'eda': round(random.uniform(*ranges['eda']), 2),
    }


# ─── Dataset parsers ────────────────────────────────────────────────────────

def parse_tess() -> list:
    """Walk TESS directory and extract emotion labels from folder/file names.

    TESS structure:
        .../OAF_angry/OAF_back_angry.wav
        .../YAF_happy/YAF_word_happy.wav
        .../OAF_Pleasant_surprise/OAF_word_pleasant_surprise.wav

    Returns list of dicts with keys: text, emotion, hr, hrv, eda, source
    """
    if not TESS_DIR.exists():
        print(f"  [!] TESS directory not found: {TESS_DIR}")
        return []

    samples = []
    skipped = 0

    for folder in sorted(TESS_DIR.iterdir()):
        if not folder.is_dir():
            continue

        # Parse emotion from folder name: "OAF_angry" -> "angry", "YAF_pleasant_surprised" -> "pleasant_surprised"
        parts = folder.name.split('_', 1)
        if len(parts) < 2:
            continue
        folder_emotion_raw = parts[1].lower().replace(' ', '_')

        # Look up mapping
        mapping = TESS_EMOTION_MAP.get(folder_emotion_raw)
        if mapping is None:
            # Try partial match (e.g., "pleasant_surprise" vs "pleasant_surprised")
            for key in TESS_EMOTION_MAP:
                if key.startswith(folder_emotion_raw) or folder_emotion_raw.startswith(key):
                    mapping = TESS_EMOTION_MAP[key]
                    break
        if mapping is None:
            print(f"    [!] Unknown TESS emotion folder: {folder.name} ({folder_emotion_raw})")
            skipped += 1
            continue

        primary_emotion, alt_emotion, alt_weight = mapping

        wav_files = list(folder.glob('*.wav'))
        for wav_file in wav_files:
            # Decide primary vs alt emotion
            if alt_emotion and random.random() < alt_weight:
                emotion = alt_emotion
            else:
                emotion = primary_emotion

            text = _generate_text(emotion, intensity='MD', source='tess')
            bio = generate_biometrics(emotion)
            samples.append({
                'text': text,
                'emotion': emotion,
                'hr': bio['hr'],
                'hrv': bio['hrv'],
                'eda': bio['eda'],
            })

    print(f"  TESS: {len(samples)} samples parsed ({skipped} folders skipped)")
    return samples


def parse_cremad() -> list:
    """Walk CREMA-D AudioWAV directory and extract emotion + intensity from filenames.

    CREMA-D pattern: {actor}_{sentence}_{emotion}_{intensity}.wav
    Example: 1001_IEO_ANG_HI.wav

    Returns list of dicts with keys: text, emotion, hr, hrv, eda
    """
    if not CREMAD_DIR.exists():
        print(f"  [!] CREMA-D directory not found: {CREMAD_DIR}")
        return []

    samples = []
    skipped = 0

    for wav_file in sorted(CREMAD_DIR.glob('*.wav')):
        name = wav_file.stem  # e.g., "1001_IEO_ANG_HI"
        parts = name.split('_')
        if len(parts) < 4:
            skipped += 1
            continue

        _actor = parts[0]
        sentence_id = parts[1]
        emotion_code = parts[2].upper()
        intensity = parts[3].upper()

        if emotion_code not in CREMAD_EMOTION_MAP:
            skipped += 1
            continue

        intensity_map = CREMAD_EMOTION_MAP[emotion_code]
        if intensity not in intensity_map:
            intensity = 'XX'  # fallback

        primary_emotion, alt_prob, alt_emotion = intensity_map[intensity]

        # Decide primary vs alt
        if alt_emotion and random.random() < alt_prob:
            emotion = alt_emotion
        else:
            emotion = primary_emotion

        sentence_desc = CREMAD_SENTENCES.get(sentence_id, 'a spoken phrase')
        text = _generate_text(emotion, intensity=intensity, source='cremad',
                              sentence_type=sentence_desc)
        bio = generate_biometrics(emotion)
        samples.append({
            'text': text,
            'emotion': emotion,
            'hr': bio['hr'],
            'hrv': bio['hrv'],
            'eda': bio['eda'],
        })

    print(f"  CREMA-D: {len(samples)} samples parsed ({skipped} files skipped)")
    return samples


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  QUANTARA - Audio Emotion Dataset Preparation")
    print("  TESS + CREMA-D -> 32-Emotion Taxonomy")
    print("=" * 65)

    random.seed(42)
    np.random.seed(42)

    # Parse both datasets
    print("\n  Parsing audio datasets (filename-based, no WAV reading)...\n")

    tess_samples = parse_tess()
    cremad_samples = parse_cremad()

    all_samples = tess_samples + cremad_samples

    if not all_samples:
        print("\n  [ERROR] No samples parsed. Check dataset paths.")
        sys.exit(1)

    # Build DataFrame
    df = pd.DataFrame(all_samples)

    # Shuffle to mix TESS and CREMA-D
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # ─── Stats ───────────────────────────────────────────────────────────
    print(f"\n  Saved to: {OUTPUT_PATH}")
    print(f"  Total rows: {len(df)}")

    print("\n  Emotion distribution:")
    print("  " + "-" * 50)
    emotion_counts = Counter(df['emotion'])
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(df)
        bar = '#' * int(pct)
        print(f"    {emotion:<15} {count:5d}  ({pct:5.1f}%) {bar}")

    print(f"\n  Source breakdown:")
    print(f"    TESS:    {len(tess_samples):,} samples")
    print(f"    CREMA-D: {len(cremad_samples):,} samples")
    print(f"    Total:   {len(all_samples):,} samples")

    print(f"\n  Biometric ranges (sample):")
    for col in ['hr', 'hrv', 'eda']:
        print(f"    {col}: min={df[col].min():.1f}, max={df[col].max():.1f}, "
              f"mean={df[col].mean():.1f}, std={df[col].std():.1f}")

    print(f"\n  Unique emotions: {df['emotion'].nunique()}")
    print(f"  CSV columns: {list(df.columns)}")

    print("\n" + "=" * 65)
    print("  Done! Combine with external_emotion_data.csv for training:")
    print("    cat audio_emotion_data.csv >> external_emotion_data.csv")
    print("  Or pass both via --external-data to train_emotion_classifier.py")
    print("=" * 65)


if __name__ == '__main__':
    main()

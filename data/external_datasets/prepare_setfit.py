#!/usr/bin/env python3
"""
===============================================================================
QUANTARA - SetFit/Emotion → 32-Emotion Taxonomy Data Preparation
===============================================================================
Maps the 6-class SetFit/emotion dataset (20K samples) to nanoGPT's 32-emotion
taxonomy with synthetic biometrics.

Labels: sadness(0), joy(1), love(2), anger(3), fear(4), surprise(5)

Usage:
    python data/external_datasets/prepare_setfit.py

Output: data/external_datasets/setfit_emotion_data.csv
===============================================================================
"""

import csv
import os
import random
import sys

SETFIT_DIR = os.path.expanduser('~/Quantara-Backend/datasets/setfit-emotion')
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'setfit_emotion_data.csv')

# Map SetFit 6-class → nanoGPT 32-emotion taxonomy (with weighted sub-emotions)
EMOTION_MAP = {
    'sadness': [
        ('sadness', 0.50), ('grief', 0.15), ('nostalgia', 0.15), ('boredom', 0.10), ('guilt', 0.10)
    ],
    'joy': [
        ('joy', 0.40), ('excitement', 0.20), ('enthusiasm', 0.15), ('gratitude', 0.10),
        ('pride', 0.10), ('fun', 0.05)
    ],
    'love': [
        ('love', 0.50), ('compassion', 0.25), ('gratitude', 0.15), ('joy', 0.10)
    ],
    'anger': [
        ('anger', 0.35), ('frustration', 0.25), ('contempt', 0.15), ('disgust', 0.10),
        ('hate', 0.10), ('jealousy', 0.05)
    ],
    'fear': [
        ('fear', 0.30), ('anxiety', 0.25), ('worry', 0.20), ('overwhelmed', 0.10),
        ('stressed', 0.10), ('shame', 0.05)
    ],
    'surprise': [
        ('surprise', 0.60), ('excitement', 0.20), ('joy', 0.10), ('fear', 0.10)
    ]
}

# Biometric ranges per emotion (from train_emotion_classifier.py)
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
    'overwhelmed':  {'hr': (85, 105),  'hrv': (15, 35),  'eda': (6, 10)},
    'stressed':     {'hr': (80, 100),  'hrv': (20, 40),  'eda': (5, 9)},
    'love':         {'hr': (65, 85),   'hrv': (55, 80),  'eda': (2, 4)},
    'compassion':   {'hr': (60, 80),   'hrv': (55, 80),  'eda': (1.5, 3)},
    'surprise':     {'hr': (75, 100),  'hrv': (35, 55),  'eda': (3, 6)},
    'guilt':        {'hr': (65, 85),   'hrv': (30, 50),  'eda': (3, 5)},
    'shame':        {'hr': (65, 85),   'hrv': (25, 45),  'eda': (4, 7)},
    'neutral':      {'hr': (60, 75),   'hrv': (55, 80),  'eda': (1, 2)},
}


def weighted_choice(options):
    r = random.random()
    cumulative = 0
    for emotion, weight in options:
        cumulative += weight
        if r <= cumulative:
            return emotion
    return options[0][0]


def gen_biometrics(emotion):
    rng = BIOMETRIC_RANGES.get(emotion, BIOMETRIC_RANGES['neutral'])
    return (
        round(random.uniform(*rng['hr']), 1),
        round(random.uniform(*rng['hrv']), 1),
        round(random.uniform(*rng['eda']), 2),
    )


def main():
    print('=' * 60)
    print('QUANTARA - SetFit/Emotion → 32-Emotion Taxonomy')
    print('=' * 60)

    rows = []
    for split in ['train', 'validation', 'test']:
        filepath = os.path.join(SETFIT_DIR, f'{split}.csv')
        if not os.path.exists(filepath):
            print(f'  Warning: {filepath} not found')
            continue
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_text = row.get('label_text', '')
                text = row.get('text', '')
                if label_text and text:
                    rows.append((text, label_text))
        print(f'  {split}: loaded')

    print(f'  Total samples: {len(rows)}')

    # Map to 32-emotion taxonomy
    output_rows = []
    emotion_counts = {}

    for text, label in rows:
        if label not in EMOTION_MAP:
            continue
        emotion = weighted_choice(EMOTION_MAP[label])
        hr, hrv, eda = gen_biometrics(emotion)
        output_rows.append((text, emotion, hr, hrv, eda))
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    # Write CSV
    with open(OUTPUT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['text', 'emotion', 'hr', 'hrv', 'eda'])
        for row in output_rows:
            w.writerow(row)

    print(f'\n  Saved {len(output_rows)} rows to {OUTPUT}')
    print(f'\n  Emotion distribution:')
    print(f'  {"-" * 40}')
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        pct = count / len(output_rows) * 100
        bar = '#' * int(pct)
        print(f'    {emotion:<16} {count:>5}  ({pct:5.1f}%) {bar}')

    print(f'\n  Unique emotions: {len(emotion_counts)}')
    print('=' * 60)
    print('  Done! Use --external-data to train:')
    print('  python train_emotion_classifier.py --use-sentence-transformer \\')
    print('    --external-data data/external_datasets/setfit_emotion_data.csv')
    print('=' * 60)


if __name__ == '__main__':
    main()

"""
===============================================================================
QUANTARA NANOGPT - Emotion Data Preparation (32-Emotion Taxonomy)
===============================================================================
Prepares emotion/psychology text data for nanoGPT training.
Creates a generative model that understands emotional context.

Integrates with:
- Neural Workflow AI Engine
- AI Conversational Coach
- Emotion-Aware Training Engine
- Psychology Emotion Database
- Biometric Integration Engine
- Therapist Dashboard Engine
- Real-time Dashboard Data

Data Sources:
- text_emotion.csv (40K tweets) — boredom, enthusiasm, worry, hate, relief, fun + originals
- Emotion_classify_Data.csv (6K comments)
- text.csv (416K text samples)
- archive (4) 3 - Train/test/val emotion text
- heart_rate_emotion_dataset.csv — disgust (14K) + real biometric HR data
- EEG cognitive dataset — anxiety, calm, stressed (~500 each)
- Stress_Level_v1.csv / v2.csv — real stress assessment data (biometric-enriched)

Taxonomy: 9 families, 32 emotions (see EMOTION_FAMILIES)
"""

import os
import re
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# Use tiktoken for GPT-2 BPE tokenization
try:
    import tiktoken
    USE_TIKTOKEN = True
except ImportError:
    USE_TIKTOKEN = False
    print("[!] tiktoken not found, using character-level tokenization")

DOWNLOADS = Path("/Users/bel/Downloads")
DATA_DIR = Path(os.path.dirname(__file__))

# ─── 32-Emotion Taxonomy ────────────────────────────────────────────────────

EMOTION_FAMILIES = {
    'Joy': ['joy', 'excitement', 'enthusiasm', 'fun', 'gratitude', 'pride'],
    'Sadness': ['sadness', 'grief', 'boredom', 'nostalgia'],
    'Anger': ['anger', 'frustration', 'hate', 'contempt', 'disgust', 'jealousy'],
    'Fear': ['fear', 'anxiety', 'worry', 'overwhelmed', 'stressed'],
    'Love': ['love', 'compassion'],
    'Calm': ['calm', 'relief', 'mindfulness', 'resilience', 'hope'],
    'Self-Conscious': ['guilt', 'shame'],
    'Surprise': ['surprise'],
    'Neutral': ['neutral'],
}

ALL_EMOTIONS = [e for emotions in EMOTION_FAMILIES.values() for e in emotions]

# Reverse lookup: emotion → family
_EMOTION_TO_FAMILY = {}
for family, emotions in EMOTION_FAMILIES.items():
    for e in emotions:
        _EMOTION_TO_FAMILY[e] = family

TARGET_SAMPLES_PER_EMOTION = 3000


def family_for_emotion(emotion):
    """Get family name for an emotion."""
    return _EMOTION_TO_FAMILY.get(emotion.lower(), 'Neutral')


def format_tagged(emotion, text):
    """Format text with family:emotion tags."""
    family = family_for_emotion(emotion).lower()
    return f"<{family}:{emotion}>{text}</{family}:{emotion}>"


# ─── HuggingFace Dataset Loaders ────────────────────────────────────────────

def load_hf_dair_emotion(max_samples: int = 0):
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
        print("  Loading dair-ai/emotion (unsplit, ~416K samples) from HuggingFace...")
        ds = load_dataset('dair-ai/emotion', 'unsplit', split='train')
        source = "dair-ai/emotion (unsplit)"
    except Exception:
        try:
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

    Connected to: ML Training & Prediction Systems
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [!] HuggingFace datasets not installed. Run: pip install datasets")
        return []

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
        label_names = ds.features['labels'].feature.names

        for row in ds:
            text = str(row['text']).strip()
            labels = row['labels']
            if not text or len(text) <= 10 or not labels:
                continue
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


def _merge_hf_into_emotion_data(emotion_data, hf_pairs):
    """Merge list of (text, emotion) tuples into the emotion_data dict."""
    count = 0
    for text, emotion in hf_pairs:
        if emotion in emotion_data:
            emotion_data[emotion].append(text)
            count += 1
    return count


# ─── Dataset Loaders ─────────────────────────────────────────────────────────

def load_emotion_datasets(use_hf=False):
    """Load all emotion datasets and combine into training corpus.

    Returns dict: {emotion: [text, text, ...]}
    """
    emotion_data = {e: [] for e in ALL_EMOTIONS}

    print("=" * 60)
    print("  QUANTARA EMOTION DATA PREPARATION (32 Emotions)")
    print("=" * 60)

    # Dataset 1: Tweet emotions (40K) — includes boredom, enthusiasm, worry, hate, relief, fun
    tweet_path = DOWNLOADS / "text_emotion.csv"
    if tweet_path.exists():
        df = pd.read_csv(tweet_path)
        text_col = 'content' if 'content' in df.columns else df.columns[0]
        label_col = 'sentiment' if 'sentiment' in df.columns else df.columns[1]

        tweet_label_map = {
            'happy': 'joy', 'happiness': 'joy',
            'sad': 'sadness', 'sadness': 'sadness',
            'angry': 'anger', 'anger': 'anger',
            'fear': 'fear', 'surprise': 'surprise',
            'love': 'love', 'neutral': 'neutral',
            # Direct mappings for expanded emotions
            'boredom': 'boredom', 'enthusiasm': 'enthusiasm',
            'worry': 'worry', 'hate': 'hate',
            'relief': 'relief', 'fun': 'fun',
            'empty': 'neutral',
        }

        count = 0
        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            raw_emotion = str(row[label_col]).strip().lower()
            emotion = tweet_label_map.get(raw_emotion, raw_emotion)
            if text and len(text) > 10 and emotion in emotion_data:
                emotion_data[emotion].append(text)
                count += 1

        print(f"  [+] Loaded tweets: {count} samples")
    else:
        print(f"  [-] Tweet data not found: {tweet_path}")

    # Dataset 1b: GoEmotions (24K Reddit comments — pride, guilt, contempt, compassion, etc.)
    goemotions_path = DOWNLOADS / "goemotions_extracted.csv"
    if goemotions_path.exists():
        df = pd.read_csv(goemotions_path)
        count = 0
        for _, row in df.iterrows():
            text = str(row['text']).strip()
            emotion = str(row['emotion']).strip().lower()
            if text and len(text) > 10 and emotion in emotion_data:
                emotion_data[emotion].append(text)
                count += 1
        print(f"  [+] Loaded GoEmotions: {count} samples")
    else:
        print(f"  [-] GoEmotions not found: {goemotions_path}")

    # Dataset 2: Emotion classify (6K)
    classify_path = DOWNLOADS / "Emotion_classify_Data.csv"
    if classify_path.exists():
        df = pd.read_csv(classify_path)
        text_col = 'Comment' if 'Comment' in df.columns else df.columns[0]
        label_col = 'Emotion' if 'Emotion' in df.columns else df.columns[1]

        count = 0
        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            emotion = str(row[label_col]).strip().lower()
            if text and emotion and len(text) > 10 and emotion in emotion_data:
                emotion_data[emotion].append(text)
                count += 1

        print(f"  [+] Loaded emotion classify: {count} samples")
    else:
        print(f"  [-] Emotion classify not found: {classify_path}")

    # Dataset 3: Large text dataset (sample 100K)
    text_path = DOWNLOADS / "text.csv"
    if text_path.exists():
        df = pd.read_csv(text_path, nrows=100000)
        text_col = 'text' if 'text' in df.columns else df.columns[1]
        label_col = 'label' if 'label' in df.columns else df.columns[2]

        label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

        count = 0
        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            label = row[label_col]
            emotion = label_map.get(label)
            if text and len(text) > 10 and emotion:
                emotion_data[emotion].append(text)
                count += 1

        print(f"  [+] Loaded large text: {count} samples")
    else:
        print(f"  [-] Large text not found: {text_path}")

    # Dataset 4: Archive emotion (train/test/val)
    label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    for csv_name in ['training.csv', 'validation.csv']:
        csv_path = DOWNLOADS / "archive (4) 3" / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            count = 0
            for _, row in df.iterrows():
                text = str(row['text']).strip()
                label = row['label']
                emotion = label_map.get(label)
                if text and len(text) > 10 and emotion:
                    emotion_data[emotion].append(text)
                    count += 1
            print(f"  [+] Loaded {csv_name}: {count} samples")

    # Dataset 5: Heart rate emotion dataset — real biometric HR data + emotion labels
    hr_path = DOWNLOADS / "heart_rate_emotion_dataset.csv"
    if hr_path.exists():
        df = pd.read_csv(hr_path)
        label_col = 'Emotion' if 'Emotion' in df.columns else ('label' if 'label' in df.columns else 'emotion')
        hr_col = 'HeartRate' if 'HeartRate' in df.columns else None

        hr_label_map = {
            'happy': 'joy', 'sad': 'sadness', 'disgust': 'disgust',
            'anger': 'anger', 'fear': 'fear', 'surprise': 'surprise',
            'neutral': 'neutral',
        }

        # Check for text column (may not exist — dataset is HR + Emotion only)
        text_col = None
        for col in ['text', 'description', 'sentence']:
            if col in df.columns:
                text_col = col
                break

        count = 0
        bio_count = 0

        if text_col and label_col in df.columns:
            for _, row in df.iterrows():
                text = str(row[text_col]).strip()
                raw_label = str(row[label_col]).strip().lower()
                emotion = hr_label_map.get(raw_label)
                if text and len(text) > 10 and emotion and emotion in emotion_data:
                    emotion_data[emotion].append(text)
                    count += 1

        # Generate biometric-enriched training samples from real HR + emotion pairs
        if hr_col and label_col in df.columns:
            hr_templates = {
                'joy': [
                    "Heart rate at {hr} bpm — feeling happy and energized right now.",
                    "My heart is beating at {hr} bpm. I feel genuinely joyful.",
                    "Biometric reading: {hr} bpm heart rate. Experiencing happiness and warmth.",
                ],
                'sadness': [
                    "Heart rate dropped to {hr} bpm. Feeling low and withdrawn.",
                    "My heart rate is {hr} bpm. A heavy sadness sits in my chest.",
                    "Biometric reading: {hr} bpm. The weight of sadness is slowing everything down.",
                ],
                'anger': [
                    "Heart rate spiking at {hr} bpm. I can feel the anger rising.",
                    "My heart is pounding at {hr} bpm. This fury is consuming me.",
                    "Biometric reading: {hr} bpm. Anger has my whole body on edge.",
                ],
                'fear': [
                    "Heart rate racing at {hr} bpm. The fear is gripping my chest.",
                    "My heart is at {hr} bpm and climbing. I feel deeply afraid.",
                    "Biometric reading: {hr} bpm. Fight-or-flight activated by this fear.",
                ],
                'disgust': [
                    "Heart rate at {hr} bpm. This revulsion is visceral.",
                    "My heart rate is {hr} bpm. Feeling deeply disgusted by what I saw.",
                    "Biometric reading: {hr} bpm. A wave of disgust washed over me.",
                ],
                'surprise': [
                    "Heart rate jumped to {hr} bpm. I did not see that coming at all.",
                    "My heart is at {hr} bpm after that shock. Completely surprised.",
                    "Biometric reading: {hr} bpm. The surprise caught me totally off guard.",
                ],
                'neutral': [
                    "Heart rate steady at {hr} bpm. Feeling calm and balanced right now.",
                    "My heart rate is {hr} bpm. Nothing particularly strong emotionally.",
                    "Biometric reading: {hr} bpm. A neutral, relaxed state.",
                ],
            }

            for _, row in df.iterrows():
                raw_label = str(row[label_col]).strip().lower()
                emotion = hr_label_map.get(raw_label)
                if emotion and emotion in hr_templates:
                    hr_val = int(round(float(row[hr_col])))
                    template = random.choice(hr_templates[emotion])
                    text = template.format(hr=hr_val)
                    emotion_data[emotion].append(text)
                    bio_count += 1

            print(f"  [+] Loaded heart rate emotion: {count} text + {bio_count} biometric-enriched samples")
        else:
            print(f"  [+] Loaded heart rate emotion: {count} text samples")
            if not hr_col:
                print(f"  [-] No HeartRate column found (cols: {list(df.columns)})")
    else:
        print(f"  [-] Heart rate emotion not found: {hr_path}")

    # Dataset 5b: Stress Level datasets — real stress assessment data
    stress_count = 0
    for stress_file in ['Stress_Level_v1.csv', 'Stress_Level_v2.csv']:
        stress_path = DOWNLOADS / stress_file
        if stress_path.exists():
            df = pd.read_csv(stress_path)
            # Columns are cognitive/stress tasks with numeric stress scores (0-9 scale)
            task_cols = [c for c in df.columns if c not in ['Unnamed: 0'] and c != df.columns[0]]

            stress_templates = {
                'stressed': [
                    "Stress assessment score {score:.1f} during {task}. Feeling the pressure building.",
                    "Scored {score:.1f} on {task} stress test. My body is tense and my mind is racing.",
                    "Stress level at {score:.1f} for {task}. The cognitive load feels overwhelming.",
                ],
                'calm': [
                    "Stress assessment score {score:.1f} during {task}. Feeling centered and relaxed.",
                    "Scored {score:.1f} on {task}. Baseline calm — breathing is steady.",
                    "Stress level at {score:.1f} for {task}. A peaceful, grounded state.",
                ],
                'anxiety': [
                    "Stress assessment score {score:.1f} during {task}. Anxiety is creeping in.",
                    "Scored {score:.1f} on {task}. The anticipation is making me anxious.",
                    "Stress level at {score:.1f} for {task}. Worried about how I'm performing.",
                ],
                'overwhelmed': [
                    "Stress assessment score {score:.1f} during {task}. Everything feels like too much.",
                    "Scored {score:.1f} on {task}. I can't keep up — feeling completely overwhelmed.",
                ],
                'relief': [
                    "Stress assessment score {score:.1f} during {task}. The tension is finally easing.",
                    "Scored {score:.1f} on {task} rest period. Relief washing over me.",
                ],
            }

            for _, row in df.iterrows():
                for task in task_cols:
                    try:
                        score = float(row[task])
                    except (ValueError, TypeError):
                        continue

                    # Map stress score to emotion based on thresholds
                    if score >= 7.0:
                        emotion = 'overwhelmed'
                    elif score >= 5.0:
                        emotion = 'stressed'
                    elif score >= 3.0:
                        emotion = 'anxiety'
                    elif score >= 1.5:
                        emotion = 'calm'
                    else:
                        emotion = 'relief'

                    # Rest periods are more likely calm/relief
                    if 'rest' in task.lower():
                        if score < 4.0:
                            emotion = 'relief'
                        elif score < 6.0:
                            emotion = 'calm'

                    templates = stress_templates.get(emotion, [])
                    if templates:
                        template = random.choice(templates)
                        task_name = task.replace('_', ' ').strip()
                        text = template.format(score=score, task=task_name)
                        emotion_data[emotion].append(text)
                        stress_count += 1

            print(f"  [+] Loaded {stress_file}: stress assessment data")

    if stress_count > 0:
        print(f"  [+] Total stress-derived samples: {stress_count}")

    # Dataset 5c: Smartwatch Health Monitoring (40K rows with real HRV, HR, Stress, Sleep)
    smartwatch_path = Path("/Users/bel/Quantara-Frontend/ml-training/datasets/raw/Smartwatch_Health_Monitoring_Dataset_40000_Rows.csv")
    if smartwatch_path.exists():
        df = pd.read_csv(smartwatch_path)
        sw_count = 0

        # Sample max 5000 rows to keep balanced
        if len(df) > 5000:
            df = df.sample(n=5000, random_state=42)

        sw_templates = {
            'overwhelmed': [
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Everything is crashing down at once and I can't breathe.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Too many demands, too little energy — I'm breaking.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. My body is screaming stop but the world won't pause.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. I feel buried under the weight of everything right now.",
            ],
            'stressed': [
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. The pressure is building and I can feel it in my shoulders.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Tension is mounting with every passing hour today.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. My jaw is clenched and my mind won't stop racing.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. The stress is like a constant hum I can't turn off.",
            ],
            'anxiety': [
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Worry is creeping in and settling in my chest.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. I keep anticipating something going wrong.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. An uneasy feeling follows me through the day.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. My thoughts spiral into what-if scenarios I can't control.",
            ],
            'calm': [
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Feeling centered and grounded right now.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. A steady, peaceful rhythm in my body today.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Everything feels balanced and manageable.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. My breathing is slow and my mind is quiet.",
            ],
            'mindfulness': [
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Present in this moment, noticing peace and stillness.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Fully aware of each breath, each sensation, just being.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. No judgment, just observing — the world is quiet inside.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Grounded in the here and now, letting thoughts drift past.",
            ],
            'relief': [
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. The tension has finally released and I can breathe again.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. A wave of relief — the storm has passed.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. My body is unwinding and peace is settling in.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Everything eased up and I feel light again.",
            ],
            'resilience': [
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. My body is strong and adaptable — I can handle this.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Recovery is happening — I bounce back from hard days.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Steady signals remind me I've survived worse.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Strength flows through me even after tough stretches.",
            ],
            'gratitude': [
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Grateful for this well-rested, restored feeling.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Thankful my body is healthy and at peace today.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Appreciating the calm after so many restful hours.",
                "Heart rate {hr} bpm, HRV {hrv}ms, stress level {stress:.1f}/10. Deep gratitude for the gift of recovery and rest.",
            ],
        }

        for _, row in df.iterrows():
            try:
                hr = float(row.get('Avg_Heart_Rate', 0))
                hrv = float(row.get('HRV', 0))
                stress = float(row.get('Stress_Level', 0))
                sleep = float(row.get('Sleep_Hours', 0))
            except (ValueError, TypeError):
                continue

            # Determine emotion(s) from multiple signals
            emotions_to_add = []

            # Primary mapping: stress level (1-10 scale)
            if stress >= 8:
                emotions_to_add.append('overwhelmed')
            elif stress >= 6:
                emotions_to_add.append('stressed')
            elif stress >= 4:
                emotions_to_add.append('anxiety')
            elif stress >= 2.5:
                emotions_to_add.append('calm')
            else:
                emotions_to_add.append('mindfulness')
                emotions_to_add.append('relief')

            # Secondary mapping: HRV signal
            if hrv > 80:
                for e in ['calm', 'mindfulness', 'resilience']:
                    if e not in emotions_to_add:
                        emotions_to_add.append(e)
            elif hrv < 35:
                for e in ['stressed', 'anxiety', 'overwhelmed']:
                    if e not in emotions_to_add:
                        emotions_to_add.append(e)

            # Tertiary mapping: sleep quality
            if sleep < 5:
                for e in ['stressed', 'overwhelmed']:
                    if e not in emotions_to_add:
                        emotions_to_add.append(e)
            elif sleep > 7.5:
                for e in ['calm', 'relief', 'gratitude']:
                    if e not in emotions_to_add:
                        emotions_to_add.append(e)

            # Generate one sample per emotion for this row
            for emotion in emotions_to_add:
                templates = sw_templates.get(emotion)
                if templates:
                    template = random.choice(templates)
                    text = template.format(hr=int(hr), hrv=int(hrv), stress=stress)
                    if emotion in emotion_data:
                        emotion_data[emotion].append(text)
                        sw_count += 1

        print(f"  [+] Loaded Smartwatch Health Monitoring: {sw_count} biometric-enriched samples")
    else:
        print(f"  [-] Smartwatch Health Monitoring not found: {smartwatch_path}")

    # Dataset 5d: Empatica E4 Wearable Stress (2112 rows with HR, HRV, EDA, temp)
    empatica_path = Path("/Users/bel/Quantara-Frontend/ml-training/datasets/processed/empatica_stress_processed.csv")
    if empatica_path.exists():
        df = pd.read_csv(empatica_path)
        emp_count = 0

        emp_templates = {
            'overwhelmed': [
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Everything is too much — my body is in overdrive.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. I can't process anything more — system overload.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Maxed out on every level — need to shut down.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Drowning in demands, my nervous system is screaming.",
            ],
            'fear': [
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Fear grips me — fight or flight fully activated.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Something feels deeply wrong — my body knows before my mind.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. A primal dread floods through me and I can't shake it.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Every nerve is on edge — the fear is overwhelming.",
            ],
            'anxiety': [
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Anxious energy buzzing through my body won't stop.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Can't sit still — the worry is physical now.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. My skin is tingling and my thoughts race ahead.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Elevated arousal with persistent unease — anxiety in full swing.",
            ],
            'stressed': [
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. The stress is registered in every signal my body sends.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Moderate tension — I feel it accumulating through the day.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Cortisol running higher than baseline — the strain is real.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. My body is wound tight with unresolved pressure.",
            ],
            'excitement': [
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Pumped up and ready to go — this energy is electric!",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. High arousal, zero stress — pure excitement flowing through me.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. My heart is racing with anticipation and joy.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Everything feels alive and thrilling right now.",
            ],
            'enthusiasm': [
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Fired up and eager — can't wait to dive in!",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Positive activation — I'm fully engaged and motivated.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. This buzzing energy is driving me forward with purpose.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Enthusiastic arousal — my body says let's go!",
            ],
            'calm': [
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Relaxed and at ease — parasympathetic dominance.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Low stress, low arousal — a perfect restful state.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. My nervous system is settled and peaceful.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Everything is quiet inside — truly calm.",
            ],
            'relief': [
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. The tension dropped and relief washed through me.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Stress signals fading — finally letting go.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Recovery mode activated — the worst is over.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. A deep exhale — relief flowing through every cell.",
            ],
            'mindfulness': [
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Present and aware — simply observing without reacting.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Each signal is just information — no judgment, just awareness.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Tuned into my body's rhythms with gentle attention.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Mindfully resting in baseline — the body knows peace.",
            ],
            'boredom': [
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Flat arousal, flat engagement — nothing is stimulating.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Low everything — just existing without interest.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. Understimulated and restless — need something to change.",
                "Wearable readings: HR {hr:.0f}bpm, HRV-RMSSD {hrv:.1f}ms, EDA {eda:.2f}\u00b5S, temp {temp:.1f}\u00b0C. The monotony is reflected in every flat reading.",
            ],
        }

        for _, row in df.iterrows():
            try:
                hr = float(row.get('hr_mean', 0))
                hrv = float(row.get('hrv_rmssd', 0))
                eda = float(row.get('eda_mean', 0))
                temp = float(row.get('temp_mean', 0))
                stress_lvl = str(row.get('stress_level', '')).strip().lower()
                arousal_lvl = str(row.get('arousal_level', '')).strip().lower()
            except (ValueError, TypeError):
                continue

            # Map (stress_level, arousal_level) to emotions via string matching
            emotions_to_add = []
            if stress_lvl == 'high' and arousal_lvl == 'high':
                emotions_to_add = ['overwhelmed', 'fear']
            elif stress_lvl == 'moderate' and arousal_lvl == 'high':
                emotions_to_add = ['anxiety', 'stressed']
            elif stress_lvl == 'low' and arousal_lvl == 'high':
                emotions_to_add = ['excitement', 'enthusiasm']
            elif stress_lvl == 'low' and arousal_lvl == 'medium':
                emotions_to_add = ['calm', 'relief']
            elif stress_lvl == 'low' and arousal_lvl == 'low':
                emotions_to_add = ['mindfulness', 'boredom']

            for emotion in emotions_to_add:
                templates = emp_templates.get(emotion)
                if templates:
                    template = random.choice(templates)
                    text = template.format(hr=hr, hrv=hrv, eda=eda, temp=temp)
                    if emotion in emotion_data:
                        emotion_data[emotion].append(text)
                        emp_count += 1

        print(f"  [+] Loaded Empatica E4 Wearable Stress: {emp_count} biometric-enriched samples")
    else:
        print(f"  [-] Empatica E4 Wearable Stress not found: {empatica_path}")

    # Dataset 5e: WESAD Processed (1802 rows with ECG, EDA, EMG)
    wesad_path = Path("/Users/bel/Quantara-Frontend/ml-training/datasets/processed/wesad_full_processed.csv")
    if wesad_path.exists():
        df = pd.read_csv(wesad_path)
        wesad_count = 0

        wesad_templates = {
            'neutral': [
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Baseline state — nothing notable, just steady.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Neutral equilibrium — neither stressed nor excited.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. A flat, unremarkable emotional state.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Even-keeled and emotionally neutral right now.",
            ],
            'calm': [
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Relaxed baseline — my body is at rest.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Calm and composed — all signals steady.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Peaceful regulation — parasympathetic in control.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. A settled, grounded sense of calm.",
            ],
            'joy': [
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Laughing and delighted — pure joy in this moment.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Amusement lights up my whole system — genuine happiness.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Grinning ear to ear — this feeling is wonderful.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Joy bubbles up — my body reflects the lightness inside.",
            ],
            'fun': [
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Having a great time — this is genuinely fun.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Amused and entertained — loving every second of this.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Light-hearted amusement — my body relaxes into the fun.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Playful energy — everything feels lighter and brighter.",
            ],
            'stressed': [
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Stress markers elevated — my body is under pressure.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Tense and on edge — the stress is physiologically obvious.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Sympathetic activation — cortisol flooding my system.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Every physiological marker confirms: I'm stressed.",
            ],
            'anxiety': [
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Anxious activation — my body anticipates threat.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Worry manifests in every biosignal — anxiety is real.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Heightened vigilance — my nervous system won't calm down.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. The anxiety shows in my skin conductance and heart rhythm.",
            ],
            'relief': [
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Returning to baseline — relief is settling in.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. The stress response is fading — finally some relief.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. My body is deactivating — the pressure has lifted.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Calm after the storm — relief washes through me.",
            ],
            'overwhelmed': [
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Maximum stress response — completely overwhelmed.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Every signal is maxed — I can't take any more.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Physiological overload — my system is breaking down.",
                "Physiological state: ECG {ecg:.3f}, EDA {eda:.3f}\u00b5S, EMG {emg:.3f}. Beyond stressed — fully overwhelmed and shutting down.",
            ],
        }

        for _, row in df.iterrows():
            try:
                ecg = float(row.get('ecg_mean', 0))
                eda = float(row.get('eda_mean', 0))
                emg = float(row.get('emg_mean', 0))
                stress_lvl = int(row.get('stress_level', 0))
                emotion_state = int(row.get('emotion_state', 0))
            except (ValueError, TypeError):
                continue

            emotions_to_add = []

            # Map emotion_state
            if emotion_state == 0:
                emotions_to_add.extend(['neutral', 'calm'])
            elif emotion_state == 1:
                emotions_to_add.extend(['joy', 'fun'])
            elif emotion_state == 4:
                emotions_to_add.extend(['stressed', 'anxiety'])

            # Map stress_level
            if stress_lvl == 0:
                for e in ['calm', 'relief']:
                    if e not in emotions_to_add:
                        emotions_to_add.append(e)
            elif stress_lvl == 2:
                for e in ['stressed', 'overwhelmed']:
                    if e not in emotions_to_add:
                        emotions_to_add.append(e)

            for emotion in emotions_to_add:
                templates = wesad_templates.get(emotion)
                if templates:
                    template = random.choice(templates)
                    text = template.format(ecg=ecg, eda=eda, emg=emg)
                    if emotion in emotion_data:
                        emotion_data[emotion].append(text)
                        wesad_count += 1

        print(f"  [+] Loaded WESAD Processed: {wesad_count} biometric-enriched samples")
    else:
        print(f"  [-] WESAD Processed not found: {wesad_path}")

    # Dataset 6: EEG cognitive dataset — anxiety, calm, stressed
    eeg_path = DOWNLOADS / "eeg_cognitive_dataset.csv"
    if not eeg_path.exists():
        eeg_path = DOWNLOADS / "EEG_cognitive_dataset.csv"
    if eeg_path.exists():
        df = pd.read_csv(eeg_path)
        text_col = None
        label_col = None
        for col in df.columns:
            if col.lower() in ('text', 'sentence', 'description', 'content'):
                text_col = col
            if col.lower() in ('label', 'emotion', 'state', 'condition'):
                label_col = col

        eeg_label_map = {
            'anxious': 'anxiety', 'anxiety': 'anxiety',
            'calm': 'calm', 'relaxed': 'calm',
            'stressed': 'stressed', 'stress': 'stressed',
        }

        if text_col and label_col:
            count = 0
            for _, row in df.iterrows():
                text = str(row[text_col]).strip()
                raw_label = str(row[label_col]).strip().lower()
                emotion = eeg_label_map.get(raw_label)
                if text and len(text) > 10 and emotion and emotion in emotion_data:
                    emotion_data[emotion].append(text)
                    count += 1
            print(f"  [+] Loaded EEG cognitive: {count} samples")
        else:
            print(f"  [-] EEG dataset: columns not matched (cols: {list(df.columns)})")
    else:
        print(f"  [-] EEG cognitive not found: {eeg_path}")

    # ─── HuggingFace datasets (when --use-hf is set) ──────────────────────
    if use_hf:
        print("\n  Loading HuggingFace datasets...")
        hf_total = 0

        hf_data = load_hf_dair_emotion()
        n = _merge_hf_into_emotion_data(emotion_data, hf_data)
        hf_total += n
        print(f"  [+] Merged dair-ai/emotion: {n} samples")

        hf_data = load_hf_go_emotions()
        n = _merge_hf_into_emotion_data(emotion_data, hf_data)
        hf_total += n
        print(f"  [+] Merged go_emotions: {n} samples")

        hf_data = load_hf_empathetic_dialogues()
        n = _merge_hf_into_emotion_data(emotion_data, hf_data)
        hf_total += n
        print(f"  [+] Merged empathetic_dialogues: {n} samples")

        hf_data = load_hf_daily_dialog()
        n = _merge_hf_into_emotion_data(emotion_data, hf_data)
        hf_total += n
        print(f"  [+] Merged daily_dialog: {n} samples")

        print(f"  [+] Total HuggingFace samples merged: {hf_total}")

    # Print per-emotion counts
    print("\n  Per-emotion counts (Tier 1 — direct):")
    for family, emotions in EMOTION_FAMILIES.items():
        counts = [f"{e}:{len(emotion_data[e])}" for e in emotions]
        print(f"    {family}: {', '.join(counts)}")

    return emotion_data


# ─── Tier 2: Derived via Reclassification ────────────────────────────────────

def reclassify_derived_emotions(emotion_data):
    """Derive new emotion categories from existing data via keyword filtering."""

    print("\n  Tier 2: Reclassifying derived emotions...")

    derivations = {
        'frustration': {
            'source': 'anger',
            'keywords': ['frustrated', 'annoyed', 'irritated', 'irritating', 'frustrating', 'ugh'],
        },
        'excitement': {
            'source': 'joy',
            'keywords': ['excited', 'thrilled', 'pumped', 'can\'t wait', 'stoked', 'hyped', 'amazing'],
        },
        'grief': {
            'source': 'sadness',
            'keywords': ['lost', 'died', 'gone forever', 'passed away', 'death', 'mourning', 'funeral'],
        },
        'overwhelmed': {
            'source': 'stressed' if emotion_data.get('stressed') else 'fear',
            'keywords': ['too much', 'can\'t handle', 'overwhelmed', 'drowning', 'breaking point', 'overloaded'],
        },
        'hope': {
            'source': 'joy',
            'keywords': ['hope', 'looking forward', 'someday', 'optimistic', 'brighter', 'better days'],
        },
        'guilt': {
            'source': 'sadness',
            'keywords': ['I did', 'I shouldn\'t have', 'my fault', 'I regret', 'I\'m sorry', 'apologize'],
        },
        'shame': {
            'source': 'sadness',
            'keywords': ['I am', 'I\'m worthless', 'ashamed', 'humiliated', 'embarrassed', 'disgrace'],
        },
    }

    for target_emotion, config in derivations.items():
        source = config['source']
        keywords = config['keywords']
        source_texts = emotion_data.get(source, [])

        derived = []
        for text in source_texts:
            text_lower = text.lower()
            if any(kw in text_lower for kw in keywords):
                derived.append(text)

        # Don't remove from source — just copy to new category
        emotion_data[target_emotion].extend(derived)
        print(f"    {target_emotion}: {len(derived)} samples (from {source})")

    return emotion_data


# ─── Tier 3: Synthetic Generation ─────────────────────────────────────────────

SYNTHETIC_TEMPLATES = {
    'nostalgia': [
        "I remember when {subject}. Those were simpler times.",
        "Looking back at {subject} fills me with a bittersweet warmth.",
        "I miss the days when {subject}. Things were different then.",
        "Sometimes I think about {subject} and wish I could go back.",
        "There's something about {subject} that takes me right back to those moments.",
        "The memory of {subject} always makes me smile, even if it hurts a little.",
        "I can still picture {subject} so clearly, like it was yesterday.",
        "Nothing quite compares to when {subject}. That feeling is irreplaceable.",
    ],
    'jealousy': [
        "I can't help but feel envious when I see {subject}.",
        "Why does {subject} come so easily to everyone else but me?",
        "It's hard watching {subject} while I'm stuck here struggling.",
        "I know I shouldn't compare, but seeing {subject} really stings.",
        "Everyone seems to have {subject} except me. It's not fair.",
        "I feel a pang of jealousy every time {subject} comes up.",
        "They don't even appreciate {subject} the way I would.",
        "It eats at me knowing that {subject} was handed to them so easily.",
    ],
    'contempt': [
        "I have no respect for people who {subject}.",
        "The way they {subject} is beneath any decent person.",
        "How can anyone {subject} and still look at themselves in the mirror?",
        "People who {subject} deserve every bit of scorn they get.",
        "It's pathetic how they {subject} without any self-awareness.",
        "I look down on anyone who would {subject}.",
        "The arrogance of someone who {subject} is truly disgusting.",
        "They {subject} as if it's something to be proud of. Pathetic.",
    ],
    'pride': [
        "I worked so hard for {subject} and it finally paid off.",
        "I'm genuinely proud of myself for {subject}.",
        "Achieving {subject} is one of my greatest accomplishments.",
        "Nobody believed I could {subject}, but here I am.",
        "Looking at {subject} fills me with a deep sense of accomplishment.",
        "All that effort toward {subject} was absolutely worth it.",
        "I proved myself when I {subject}. That means everything.",
        "The satisfaction of {subject} is something no one can take from me.",
    ],
    'resilience': [
        "I went through {subject} and came out stronger on the other side.",
        "No matter how hard {subject} was, I refused to give up.",
        "Getting through {subject} showed me what I'm really made of.",
        "I've survived {subject} before. I can survive anything.",
        "The struggle with {subject} only made me more determined.",
        "I bounced back from {subject} and I'll do it again if I have to.",
        "They tried to break me with {subject}, but I'm still standing.",
        "Every time {subject} knocked me down, I got back up stronger.",
    ],
    'mindfulness': [
        "Right now, in this moment, I notice {subject} without any judgment.",
        "I'm sitting with {subject} and just observing how it feels.",
        "Breathing in, I'm aware of {subject}. Breathing out, I let it be.",
        "I don't need to change {subject}. I just need to be present with it.",
        "There's peace in simply noticing {subject} without reacting.",
        "I'm grounding myself in {subject}, feeling each sensation fully.",
        "This moment with {subject} is enough. I don't need anything else.",
        "I observe {subject} with curiosity, not judgment.",
    ],
    'gratitude': [
        "I'm so thankful for {subject}. It means the world to me.",
        "I don't say it enough, but I'm truly grateful for {subject}.",
        "Having {subject} in my life is such a blessing.",
        "I appreciate {subject} more than words can express.",
        "Thank you for {subject}. I never want to take it for granted.",
        "Every day I'm reminded of how lucky I am to have {subject}.",
        "The gift of {subject} fills my heart with warmth.",
        "I count {subject} among my greatest blessings.",
    ],
    'compassion': [
        "My heart goes out to those who are dealing with {subject}.",
        "I wish I could take away the pain of {subject} for them.",
        "Seeing someone struggle with {subject} moves me deeply.",
        "I want to help anyone going through {subject}. They shouldn't face it alone.",
        "The suffering caused by {subject} touches something deep in me.",
        "I care deeply about people affected by {subject}.",
        "No one should have to endure {subject} without support.",
        "Knowing they face {subject} makes me want to do whatever I can.",
    ],
}

SYNTHETIC_SUBJECTS = {
    'nostalgia': [
        "we used to play outside all summer", "my grandmother's house smelled like fresh bread",
        "we'd stay up late talking about nothing", "the neighborhood kids would gather at the park",
        "music sounded different back then", "Saturday mornings meant cartoons and cereal",
        "we wrote letters instead of texting", "the whole family sat together for dinner",
        "summer break felt like it lasted forever", "we didn't need phones to have fun",
        "the ice cream truck would come every afternoon", "my best friend lived next door",
    ],
    'jealousy': [
        "their effortless success", "their perfect relationship", "their natural talent",
        "how easily they make friends", "their financial freedom", "their happy family",
        "their career advancement", "their carefree lifestyle", "their confidence",
        "getting praised by everyone", "traveling the world", "their opportunities",
    ],
    'contempt': [
        "lie to people's faces", "take credit for others' work", "bully the vulnerable",
        "cheat without remorse", "manipulate people for gain", "betray trust repeatedly",
        "pretend to care when they don't", "exploit kindness", "act superior to everyone",
        "mock those who are struggling", "abuse their power", "show zero accountability",
    ],
    'pride': [
        "finishing my degree", "landing my dream job", "running my first marathon",
        "overcoming my biggest fear", "building something from nothing", "standing up for myself",
        "mastering a difficult skill", "helping someone in need", "publishing my work",
        "recovering from illness", "breaking a personal record", "earning their respect",
    ],
    'resilience': [
        "losing everything and starting over", "a devastating diagnosis",
        "being told I'd never succeed", "the hardest year of my life",
        "betrayal by someone I trusted", "financial ruin", "a painful breakup",
        "chronic illness", "workplace discrimination", "losing my home",
        "failing publicly", "years of self-doubt",
    ],
    'mindfulness': [
        "the weight of my body in this chair", "the sound of rain on the window",
        "the rhythm of my breathing", "the warmth of sunlight on my skin",
        "the taste of this cup of tea", "tension in my shoulders",
        "thoughts passing like clouds", "the feeling of my feet on the ground",
        "the colors in the sky right now", "the space between each breath",
        "ambient sounds around me", "the texture of this surface",
    ],
    'gratitude': [
        "my health", "the people who stood by me", "another beautiful morning",
        "a warm place to sleep", "friends who truly care", "the food on my table",
        "being alive right now", "the lessons I've learned", "small acts of kindness",
        "my family's love", "the opportunity to grow", "peaceful quiet moments",
    ],
    'compassion': [
        "homelessness", "childhood trauma", "chronic pain",
        "losing a loved one", "mental health struggles", "discrimination",
        "loneliness in old age", "poverty", "domestic violence",
        "refugee displacement", "addiction", "invisible disabilities",
    ],
}


def generate_synthetic_samples(emotion_data):
    """Generate Tier 3 synthetic samples using templates and augmentation."""

    print("\n  Tier 3: Generating synthetic samples...")

    for emotion, templates in SYNTHETIC_TEMPLATES.items():
        subjects = SYNTHETIC_SUBJECTS.get(emotion, [])
        if not subjects:
            continue

        base_samples = []
        for template in templates:
            for subject in subjects:
                base_samples.append(template.format(subject=subject))

        # Augment: synonym substitution + sentence restructuring
        augmented = list(base_samples)
        synonym_map = {
            'happy': 'joyful', 'sad': 'sorrowful', 'angry': 'furious',
            'great': 'wonderful', 'bad': 'terrible', 'hard': 'difficult',
            'good': 'excellent', 'feel': 'sense', 'think': 'believe',
            'really': 'truly', 'very': 'deeply', 'always': 'constantly',
            'never': 'not once', 'sometimes': 'occasionally', 'love': 'cherish',
        }

        for sample in base_samples:
            # Synonym substitution
            augmented_text = sample
            for original, replacement in synonym_map.items():
                if original in augmented_text.lower():
                    pattern = re.compile(re.escape(original), re.IGNORECASE)
                    augmented_text = pattern.sub(replacement, augmented_text, count=1)
                    break
            if augmented_text != sample:
                augmented.append(augmented_text)

            # Sentence restructuring: move first clause to end
            if '. ' in sample:
                parts = sample.split('. ', 1)
                augmented.append(f"{parts[1]} {parts[0]}.")
            elif ', ' in sample:
                parts = sample.split(', ', 1)
                augmented.append(f"{parts[1].capitalize()}, {parts[0].lower()}.")

        emotion_data[emotion].extend(augmented)
        print(f"    {emotion}: {len(augmented)} synthetic samples")

    return emotion_data


# ─── Data Balancing ───────────────────────────────────────────────────────────

def balance_emotion_data(emotion_data):
    """Balance all emotions to TARGET_SAMPLES_PER_EMOTION."""

    print(f"\n  Balancing to ~{TARGET_SAMPLES_PER_EMOTION} samples per emotion...")

    balanced = {}
    for emotion in ALL_EMOTIONS:
        samples = emotion_data.get(emotion, [])

        if not samples:
            print(f"    [!] {emotion}: 0 samples — skipping")
            balanced[emotion] = []
            continue

        n = len(samples)
        if n >= TARGET_SAMPLES_PER_EMOTION:
            # Downsample
            random.seed(42)
            balanced[emotion] = random.sample(samples, TARGET_SAMPLES_PER_EMOTION)
            print(f"    {emotion}: {n} → {TARGET_SAMPLES_PER_EMOTION} (downsampled)")
        elif n >= 500:
            # Oversample to target
            random.seed(42)
            oversampled = samples * (TARGET_SAMPLES_PER_EMOTION // n + 1)
            balanced[emotion] = oversampled[:TARGET_SAMPLES_PER_EMOTION]
            print(f"    {emotion}: {n} → {TARGET_SAMPLES_PER_EMOTION} (oversampled)")
        else:
            # Very few samples — oversample but warn
            random.seed(42)
            if n > 0:
                oversampled = samples * (TARGET_SAMPLES_PER_EMOTION // n + 1)
                balanced[emotion] = oversampled[:TARGET_SAMPLES_PER_EMOTION]
                print(f"    {emotion}: {n} → {TARGET_SAMPLES_PER_EMOTION} (oversampled, LOW SOURCE)")
            else:
                balanced[emotion] = []
                print(f"    {emotion}: 0 → 0 (NO DATA)")

    return balanced


# ─── Psychology Prompts (Expanded) ────────────────────────────────────────────

def create_psychology_prompts():
    """Create psychology-focused training prompts for conversational AI.

    Expanded to cover all 32 emotions with therapy techniques,
    transitions, and coaching prompts from the design spec.
    """
    prompts = []

    # Empathetic responses — all families
    empathy_patterns = [
        ("sadness", "I understand this is difficult. What you're feeling is valid, and I'm here to support you through this."),
        ("anger", "I can see why you'd feel frustrated. Let's take a moment to explore what's driving these feelings."),
        ("fear", "It's natural to feel anxious about this. Let's break it down together and find ways to feel more grounded."),
        ("joy", "That's wonderful! I'm glad to hear things are going well. Let's explore what's contributing to this positive feeling."),
        ("grief", "There's no timeline for this. What you're feeling honors what you lost. I'm here."),
        ("frustration", "I hear the frustration. Let's separate what you can control from what you can't — and start there."),
        ("anxiety", "Your mind is racing ahead. Let's bring you back to right now — what do you feel in your body?"),
        ("overwhelmed", "Everything at once is too much. What's the one smallest thing you could do in the next 5 minutes?"),
        ("guilt", "Guilt means your values are intact. What would making it right look like?"),
        ("shame", "What happened doesn't define who you are. Can you tell me what you're telling yourself about this?"),
        ("excitement", "That energy is powerful. Let's channel it — what's the one thing you most want to direct this toward?"),
        ("gratitude", "That appreciation you feel — have you told them? Sometimes expressing it multiplies it."),
        ("hope", "Hold onto that vision. Now let's build a bridge to it — what's one step you can take this week?"),
        ("resilience", "You've weathered storms before. What got you through last time? That strength is still in you."),
        ("compassion", "Your empathy is a gift. How can you channel this caring into something meaningful?"),
        ("boredom", "Boredom is often a signal. What would feel meaningful to you right now, even something small?"),
        ("nostalgia", "Those memories shaped who you are. What from that time do you still carry with you today?"),
        ("contempt", "What if there's a reason you haven't considered? Let's explore what might be driving their behavior."),
        ("jealousy", "That comparison is stealing your peace. What have you accomplished that you're not giving yourself credit for?"),
        ("stressed", "Your system is running hot. Let's cool it down — breathe with me: in 4, hold 4, out 4, hold 4."),
    ]

    for emotion, response in empathy_patterns:
        prompts.append(f"<empathy>User feels: {emotion} | Response: {response}</empathy>")

    # Therapeutic techniques — expanded
    therapy_patterns = [
        ("Cognitive Reframing", "When you notice a negative thought, ask yourself: What evidence supports this? What evidence contradicts it? Is there another way to look at this situation?"),
        ("Grounding 5-4-3-2-1", "Notice 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste. This brings you back to the present moment."),
        ("Breathing Exercise", "Breathe in for 4 counts, hold for 4 counts, breathe out for 6 counts. This activates your parasympathetic nervous system."),
        ("Progressive Muscle Relaxation", "Tense each muscle group for 5 seconds, then release. Start with your feet and work up to your head."),
        ("Behavioral Activation", "Schedule small, pleasurable activities. Even brief moments of engagement can shift mood."),
        ("Box Breathing", "Breathe in for 4 counts, hold for 4, breathe out for 4, hold for 4. This resets your autonomic nervous system."),
        ("Self-Compassion Practice", "Treat yourself with the same kindness you'd show a good friend. You deserve that same care."),
        ("Shame Resilience", "Separate what happened from who you are. Actions can be wrong without you being wrong."),
        ("Task Chunking", "Break the overwhelming whole into the smallest possible next step. Just one thing at a time."),
        ("Values Exploration", "Connect with what matters most to you. Let your values guide your next action."),
        ("Grief Journaling", "Write about your loss. Honor it. There's no right way to grieve, but expression helps."),
        ("Gratitude Letter", "Write a letter of thanks to someone who made a difference. You don't have to send it."),
        ("Future Self Visualization", "Picture your future self having achieved your goals. What steps did they take?"),
        ("Strength Spotting", "Identify the strengths you used to get through past challenges. They're still in you."),
        ("Self-Worth Inventory", "List your accomplishments, qualities, and the people who value you. Comparison steals joy."),
    ]

    for technique, description in therapy_patterns:
        prompts.append(f"<therapy>Technique: {technique} | {description}</therapy>")

    # Emotion transitions — expanded
    transition_patterns = [
        ("anxiety", "calm", "Progressive muscle relaxation combined with slow breathing helps shift from fight-or-flight to rest-and-digest."),
        ("sadness", "acceptance", "Allow yourself to feel the emotion without judgment. Emotions are temporary visitors, not permanent residents."),
        ("anger", "understanding", "Consider what need is not being met. Anger often masks fear, hurt, or frustration about unmet needs."),
        ("frustration", "agency", "Separate what you can control from what you can't. Focus energy on actionable steps."),
        ("grief", "acceptance", "There's no timeline. Meaning-making happens when you honor what was lost while staying present."),
        ("overwhelmed", "manageable", "Find the smallest next step. Everything big started with one tiny action."),
        ("excitement", "focused calm", "Ground that energy. Channel the spark into the one thing that matters most right now."),
        ("guilt", "repair", "Values intact means repair is possible. What would amends look like?"),
        ("shame", "self-worth", "Separate the action from your identity. You are not your worst moment."),
        ("boredom", "engagement", "Boredom signals misalignment. What micro-goal would feel meaningful?"),
        ("jealousy", "self-acceptance", "Compare to your past self, not others. Your growth is the only fair measure."),
        ("hope", "motivation", "Anchor that hope to a concrete next step. Vision without action stays a dream."),
        ("stressed", "regulated", "Autonomic reset: box breathing, cold water on wrists, or 2-minute walk."),
        ("contempt", "curiosity", "What if there's context you're missing? Perspective shift starts with a question."),
    ]

    for from_e, to_e, method in transition_patterns:
        prompts.append(f"<transition>From: {from_e} | To: {to_e} | Method: {method}</transition>")

    return prompts


# ─── Main Preparation ─────────────────────────────────────────────────────────

def prepare_data(use_hf=False):
    """Main data preparation function.

    Args:
        use_hf: If True, load additional data from HuggingFace Hub
                (dair-ai/emotion, go_emotions, empathetic_dialogues, daily_dialog)
                and merge with any local CSV data that exists.
    """

    # Load all emotion data (Tier 1 — direct from datasets)
    emotion_data = load_emotion_datasets(use_hf=use_hf)

    # Tier 2 — reclassify derived emotions
    emotion_data = reclassify_derived_emotions(emotion_data)

    # Tier 3 — generate synthetic samples
    emotion_data = generate_synthetic_samples(emotion_data)

    # Balance all emotions
    balanced_data = balance_emotion_data(emotion_data)

    # Format as tagged text
    all_data = []
    for emotion, texts in balanced_data.items():
        for text in texts:
            all_data.append(format_tagged(emotion, text))

    # Add psychology prompts (expanded for 32 emotions)
    psych_prompts = create_psychology_prompts()
    psych_prompts = psych_prompts * 100  # Amplify these patterns
    all_data.extend(psych_prompts)

    # Shuffle
    np.random.seed(42)
    np.random.shuffle(all_data)

    print(f"\n  Total training samples: {len(all_data)}")

    # Print final distribution
    emotion_counts = Counter()
    for sample in all_data:
        for emotion in ALL_EMOTIONS:
            family = family_for_emotion(emotion).lower()
            tag = f"<{family}:{emotion}>"
            if tag in sample:
                emotion_counts[emotion] += 1
                break

    print("\n  Final distribution:")
    for family, emotions in EMOTION_FAMILIES.items():
        for e in emotions:
            count = emotion_counts.get(e, 0)
            marker = " [LOW]" if 0 < count < 1000 else ""
            print(f"    {e}: {count}{marker}")

    # Join into single text corpus
    data = "\n\n".join(all_data)
    print(f"\n  Total characters: {len(data):,}")

    # Split into train/val (90/10)
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    if USE_TIKTOKEN:
        # GPT-2 BPE tokenization
        print("\n  Using GPT-2 BPE tokenization...")
        enc = tiktoken.get_encoding("gpt2")

        train_ids = enc.encode_ordinary(train_data)
        val_ids = enc.encode_ordinary(val_data)

        print(f"  Train tokens: {len(train_ids):,}")
        print(f"  Val tokens: {len(val_ids):,}")

        # Export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)

    else:
        # Character-level tokenization (fallback)
        print("\n  Using character-level tokenization...")
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        print(f"  Vocab size: {vocab_size}")

        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        def encode(s):
            return [stoi[c] for c in s]

        train_ids = np.array(encode(train_data), dtype=np.uint16)
        val_ids = np.array(encode(val_data), dtype=np.uint16)

        # Save meta for char-level
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        with open(DATA_DIR / 'meta.pkl', 'wb') as f:
            pickle.dump(meta, f)
        print(f"  Saved meta.pkl")

    # Save binary files
    train_ids.tofile(DATA_DIR / 'train.bin')
    val_ids.tofile(DATA_DIR / 'val.bin')

    print(f"\n  [+] Saved train.bin ({len(train_ids):,} tokens)")
    print(f"  [+] Saved val.bin ({len(val_ids):,} tokens)")

    # Save config info
    all_tags = list(ALL_EMOTIONS) + ['empathy', 'therapy', 'transition']
    family_tags = [f.lower() for f in EMOTION_FAMILIES.keys()]
    all_tags.extend(family_tags)

    config = {
        'total_samples': len(all_data),
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
        'tokenizer': 'gpt2' if USE_TIKTOKEN else 'char',
        'emotion_tags': all_tags,
        'emotion_families': {k: v for k, v in EMOTION_FAMILIES.items()},
        'num_emotions': len(ALL_EMOTIONS),
    }
    with open(DATA_DIR / 'config.pkl', 'wb') as f:
        pickle.dump(config, f)

    print("\n" + "=" * 60)
    print("  DATA PREPARATION COMPLETE (32-Emotion Taxonomy)")
    print("=" * 60)
    print(f"\n  Ready for training with:")
    print(f"  python train.py config/train_quantara_emotion.py")
    print("=" * 60 + "\n")

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Quantara emotion data for nanoGPT training (32-emotion taxonomy)"
    )
    parser.add_argument(
        '--use-hf', action='store_true',
        help='Load additional data from HuggingFace Hub (dair-ai/emotion, go_emotions, '
             'empathetic_dialogues, daily_dialog) and merge with local CSV data'
    )
    args = parser.parse_args()
    prepare_data(use_hf=args.use_hf)

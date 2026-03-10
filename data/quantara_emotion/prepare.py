"""
===============================================================================
QUANTARA NANOGPT - Emotion Data Preparation
===============================================================================
Prepares emotion/psychology text data for nanoGPT training.
Creates a generative model that understands emotional context.

Integrates with:
- Neural Workflow AI Engine
- AI Conversational Coach
- Emotion-Aware Training Engine
- Psychology Emotion Database

Data Sources:
- text_emotion.csv (40K tweets)
- Emotion_classify_Data.csv (6K comments)
- text.csv (416K text samples)
- archive (4) 3 - Train/test/val emotion text
"""

import os
import pickle
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


def load_emotion_datasets():
    """Load all emotion datasets and combine into training corpus"""
    all_data = []

    print("=" * 60)
    print("  QUANTARA EMOTION DATA PREPARATION")
    print("=" * 60)

    # Dataset 1: Tweet emotions (40K)
    tweet_path = DOWNLOADS / "text_emotion.csv"
    if tweet_path.exists():
        df = pd.read_csv(tweet_path)
        text_col = 'content' if 'content' in df.columns else df.columns[0]
        label_col = 'sentiment' if 'sentiment' in df.columns else df.columns[1]

        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            emotion = str(row[label_col]).strip()
            if text and emotion and len(text) > 10:
                # Format: <emotion>text</emotion>
                all_data.append(f"<{emotion}>{text}</{emotion}>")

        print(f"  [+] Loaded tweets: {len(df)} samples")
    else:
        print(f"  [-] Tweet data not found: {tweet_path}")

    # Dataset 2: Emotion classify (6K)
    classify_path = DOWNLOADS / "Emotion_classify_Data.csv"
    if classify_path.exists():
        df = pd.read_csv(classify_path)
        text_col = 'Comment' if 'Comment' in df.columns else df.columns[0]
        label_col = 'Emotion' if 'Emotion' in df.columns else df.columns[1]

        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            emotion = str(row[label_col]).strip().lower()
            if text and emotion and len(text) > 10:
                all_data.append(f"<{emotion}>{text}</{emotion}>")

        print(f"  [+] Loaded emotion classify: {len(df)} samples")
    else:
        print(f"  [-] Emotion classify not found: {classify_path}")

    # Dataset 3: Large text dataset (sample 100K)
    text_path = DOWNLOADS / "text.csv"
    if text_path.exists():
        df = pd.read_csv(text_path, nrows=100000)
        text_col = 'text' if 'text' in df.columns else df.columns[1]
        label_col = 'label' if 'label' in df.columns else df.columns[2]

        # Map numeric labels to emotions
        label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            label = row[label_col]
            emotion = label_map.get(label, str(label))
            if text and len(text) > 10:
                all_data.append(f"<{emotion}>{text}</{emotion}>")

        print(f"  [+] Loaded large text: {len(df)} samples")
    else:
        print(f"  [-] Large text not found: {text_path}")

    # Dataset 4: Archive emotion (train/test/val)
    archive_path = DOWNLOADS / "archive (4) 3" / "training.csv"
    if archive_path.exists():
        df = pd.read_csv(archive_path)
        label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

        for _, row in df.iterrows():
            text = str(row['text']).strip()
            label = row['label']
            emotion = label_map.get(label, str(label))
            if text and len(text) > 10:
                all_data.append(f"<{emotion}>{text}</{emotion}>")

        print(f"  [+] Loaded archive emotion: {len(df)} samples")
    else:
        print(f"  [-] Archive emotion not found: {archive_path}")

    # Also try to load validation data
    val_path = DOWNLOADS / "archive (4) 3" / "validation.csv"
    if val_path.exists():
        df = pd.read_csv(val_path)
        label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

        for _, row in df.iterrows():
            text = str(row['text']).strip()
            label = row['label']
            emotion = label_map.get(label, str(label))
            if text and len(text) > 10:
                all_data.append(f"<{emotion}>{text}</{emotion}>")

        print(f"  [+] Loaded validation data: {len(df)} samples")

    return all_data


def create_psychology_prompts():
    """Create psychology-focused training prompts for conversational AI"""
    prompts = []

    # Emotion-aware response patterns
    patterns = [
        # Empathetic responses
        ("<empathy>User feels: sadness | Response: I understand this is difficult. What you're feeling is valid, and I'm here to support you through this.</empathy>",),
        ("<empathy>User feels: anger | Response: I can see why you'd feel frustrated. Let's take a moment to explore what's driving these feelings.</empathy>",),
        ("<empathy>User feels: fear | Response: It's natural to feel anxious about this. Let's break it down together and find ways to feel more grounded.</empathy>",),
        ("<empathy>User feels: joy | Response: That's wonderful! I'm glad to hear things are going well. Let's explore what's contributing to this positive feeling.</empathy>",),

        # Therapeutic techniques
        ("<therapy>Technique: Cognitive Reframing | When you notice a negative thought, ask yourself: What evidence supports this? What evidence contradicts it? Is there another way to look at this situation?</therapy>",),
        ("<therapy>Technique: Grounding 5-4-3-2-1 | Notice 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste. This brings you back to the present moment.</therapy>",),
        ("<therapy>Technique: Breathing Exercise | Breathe in for 4 counts, hold for 4 counts, breathe out for 6 counts. This activates your parasympathetic nervous system.</therapy>",),

        # Emotion transitions
        ("<transition>From: anxiety | To: calm | Method: Progressive muscle relaxation combined with slow breathing helps shift from fight-or-flight to rest-and-digest.</transition>",),
        ("<transition>From: sadness | To: acceptance | Method: Allow yourself to feel the emotion without judgment. Emotions are temporary visitors, not permanent residents.</transition>",),
        ("<transition>From: anger | To: understanding | Method: Consider what need is not being met. Anger often masks fear, hurt, or frustration about unmet needs.</transition>",),
    ]

    for pattern in patterns:
        prompts.append(pattern[0])

    return prompts


def prepare_data():
    """Main data preparation function"""

    # Load all emotion data
    emotion_data = load_emotion_datasets()

    # Add psychology prompts
    psych_prompts = create_psychology_prompts()
    # Repeat psychology prompts to balance with emotion data
    psych_prompts = psych_prompts * 100  # Amplify these patterns

    all_data = emotion_data + psych_prompts

    # Shuffle
    np.random.seed(42)
    np.random.shuffle(all_data)

    print(f"\n  Total training samples: {len(all_data)}")

    # Join into single text corpus
    data = "\n\n".join(all_data)
    print(f"  Total characters: {len(data):,}")

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
    config = {
        'total_samples': len(all_data),
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
        'tokenizer': 'gpt2' if USE_TIKTOKEN else 'char',
        'emotion_tags': ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise', 'neutral', 'empathy', 'therapy', 'transition'],
    }
    with open(DATA_DIR / 'config.pkl', 'wb') as f:
        pickle.dump(config, f)

    print("\n" + "=" * 60)
    print("  DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\n  Ready for training with:")
    print(f"  python train.py config/train_quantara_emotion.py")
    print("=" * 60 + "\n")

    return config


if __name__ == "__main__":
    prepare_data()

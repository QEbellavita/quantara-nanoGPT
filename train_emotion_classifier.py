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
===============================================================================
"""

import os
import sys
import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emotion_classifier import (
    BiometricEncoder, FusionHead, EMOTION_FAMILIES, FAMILY_NAMES, family_for_emotion
)

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


class EmotionDataset(Dataset):
    """Dataset of text embeddings + biometrics + labels (emotion + family)."""

    def __init__(self, embeddings, biometrics, emotion_labels, family_labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.biometrics = torch.tensor(biometrics, dtype=torch.float32)
        self.emotion_labels = torch.tensor(emotion_labels, dtype=torch.long)
        self.family_labels = torch.tensor(family_labels, dtype=torch.long)

    def __len__(self):
        return len(self.emotion_labels)

    def __getitem__(self, idx):
        return (
            self.embeddings[idx],
            self.biometrics[idx],
            self.emotion_labels[idx],
            self.family_labels[idx],
        )


def load_emotion_data(downloads_dir: Path):
    """Load emotion-labeled text data for all 32 emotions."""
    all_data = []

    # Original 7-emotion label map
    label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

    # Try text.csv
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

    # Choose embedding mode
    use_sentence_transformer = args.use_sentence_transformer
    sentence_model = None
    gpt = None
    encode_fn = None

    if use_sentence_transformer:
        if not HAS_SENTENCE_TRANSFORMERS:
            print("  [!] sentence-transformers not installed, falling back to nanoGPT")
            use_sentence_transformer = False
        else:
            print(f"\n  Loading sentence-transformer ({args.sentence_model})...")
            sentence_model = SentenceTransformer(args.sentence_model, device=device)
            n_embd = sentence_model.get_sentence_embedding_dimension()
            print(f"  Embedding dim: {n_embd}")

    if not use_sentence_transformer:
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

    # Load emotion data
    print(f"\n  Loading emotion data from {args.data_dir}...")
    data = load_emotion_data(Path(args.data_dir))

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

    if use_sentence_transformer:
        train_embeddings = extract_embeddings_sentence_transformer(sentence_model, train_texts)
        val_embeddings = extract_embeddings_sentence_transformer(sentence_model, val_texts)
    else:
        train_embeddings = extract_embeddings_gpt(gpt, train_texts, encode_fn, device)
        val_embeddings = extract_embeddings_gpt(gpt, val_texts, encode_fn, device)

    # Initialize models first
    bio_encoder = BiometricEncoder(output_dim=16).to(device)

    # Generate synthetic biometrics
    print("\n  Generating synthetic biometrics...")

    train_bio_features = []
    for emotion in train_emotions:
        bio = generate_synthetic_biometrics(emotion)
        features = bio_encoder._extract_features(bio).numpy()
        train_bio_features.append(features)
    train_bio_features = np.array(train_bio_features)

    val_bio_features = []
    for emotion in val_emotions:
        bio = generate_synthetic_biometrics(emotion)
        features = bio_encoder._extract_features(bio).numpy()
        val_bio_features.append(features)
    val_bio_features = np.array(val_bio_features)

    # Emotion labels (32-way)
    train_emotion_labels = [EMOTION_TO_IDX[e] for e in train_emotions]
    val_emotion_labels = [EMOTION_TO_IDX[e] for e in val_emotions]

    # Family labels (9-way)
    train_family_labels = [FAMILY_TO_IDX[family_for_emotion(e)] for e in train_emotions]
    val_family_labels = [FAMILY_TO_IDX[family_for_emotion(e)] for e in val_emotions]

    # Create datasets
    train_dataset = EmotionDataset(
        train_embeddings, train_bio_features, train_emotion_labels, train_family_labels
    )
    val_dataset = EmotionDataset(
        val_embeddings, val_bio_features, val_emotion_labels, val_family_labels
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
    params = list(bio_encoder.parameters()) + list(fusion_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    emotion_criterion = nn.NLLLoss()
    family_criterion = nn.NLLLoss()

    # Loss weights
    FAMILY_WEIGHT = 0.3
    EMOTION_WEIGHT = 0.7

    # Training loop
    print("\n  Training (two-stage: family + sub-emotion)...")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        bio_encoder.train()
        fusion_head.train()
        train_loss = 0.0
        train_emotion_correct = 0
        train_family_correct = 0
        train_total = 0

        for text_emb, bio_feat, emotion_labels, family_labels in train_loader:
            text_emb = text_emb.to(device)
            bio_feat = bio_feat.to(device)
            emotion_labels = emotion_labels.to(device)
            family_labels = family_labels.to(device)

            optimizer.zero_grad()

            bio_emb = bio_encoder(bio_feat)
            emotion_probs, family_probs = fusion_head(text_emb, bio_emb)

            # Two-stage loss
            emotion_loss = emotion_criterion(torch.log(emotion_probs + 1e-8), emotion_labels)
            family_loss = family_criterion(torch.log(family_probs + 1e-8), family_labels)
            loss = EMOTION_WEIGHT * emotion_loss + FAMILY_WEIGHT * family_loss

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
        fusion_head.eval()
        val_emotion_correct = 0
        val_family_correct = 0
        val_total = 0

        with torch.no_grad():
            for text_emb, bio_feat, emotion_labels, family_labels in val_loader:
                text_emb = text_emb.to(device)
                bio_feat = bio_feat.to(device)
                emotion_labels = emotion_labels.to(device)
                family_labels = family_labels.to(device)

                bio_emb = bio_encoder(bio_feat)
                emotion_probs, family_probs = fusion_head(text_emb, bio_emb)

                emotion_preds = emotion_probs.argmax(dim=1)
                family_preds = family_probs.argmax(dim=1)
                val_emotion_correct += (emotion_preds == emotion_labels).sum().item()
                val_family_correct += (family_preds == family_labels).sum().item()
                val_total += emotion_labels.size(0)

        val_emotion_acc = val_emotion_correct / val_total
        val_family_acc = val_family_correct / val_total

        print(
            f"  Epoch {epoch+1:2d}/{args.epochs}: "
            f"emotion_acc={train_emotion_acc:.4f}/{val_emotion_acc:.4f}, "
            f"family_acc={train_family_acc:.4f}/{val_family_acc:.4f}"
        )

        # Early stopping on emotion accuracy
        if val_emotion_acc > best_val_acc:
            best_val_acc = val_emotion_acc
            patience_counter = 0

            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_data = {
                'fusion_head': fusion_head.state_dict(),
                'biometric_encoder': bio_encoder.state_dict(),
                'n_embd': n_embd,
                'num_emotions': 32,
                'num_families': 9,
                'val_emotion_acc': val_emotion_acc,
                'val_family_acc': val_family_acc,
                'embedding_type': 'sentence-transformer' if use_sentence_transformer else 'nanogpt',
            }
            if use_sentence_transformer:
                checkpoint_data['sentence_model'] = args.sentence_model
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
    # Embedding source
    parser.add_argument('--use-sentence-transformer', action='store_true',
                        help='Use sentence-transformers (recommended for text-only accuracy)')
    parser.add_argument('--sentence-model', default='all-MiniLM-L6-v2',
                        help='Sentence-transformer model name')
    parser.add_argument('--gpt-checkpoint', default='out-quantara-emotion-fast/ckpt.pt',
                        help='Path to nanoGPT checkpoint (if not using sentence-transformer)')
    # Data & training
    parser.add_argument('--data-dir', default='.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--device', default='auto')

    args = parser.parse_args()
    train(args)

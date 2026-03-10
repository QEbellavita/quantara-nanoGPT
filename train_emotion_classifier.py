# train_emotion_classifier.py
"""
===============================================================================
QUANTARA - Train Emotion Classifier
===============================================================================
Train the fusion head on emotion-labeled text data with synthetic biometrics.

Usage:
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

from model import GPT, GPTConfig
from emotion_classifier import BiometricEncoder, FusionHead


# Synthetic biometric ranges per emotion
BIOMETRIC_RANGES = {
    'joy':      {'hr': (70, 90),   'hrv': (50, 80),  'eda': (2, 4)},
    'sadness':  {'hr': (55, 70),   'hrv': (40, 60),  'eda': (1, 2)},
    'anger':    {'hr': (85, 110),  'hrv': (20, 40),  'eda': (5, 8)},
    'fear':     {'hr': (80, 105),  'hrv': (25, 45),  'eda': (6, 10)},
    'surprise': {'hr': (75, 100),  'hrv': (35, 55),  'eda': (4, 7)},
    'love':     {'hr': (65, 85),   'hrv': (55, 75),  'eda': (2, 4)},
    'neutral':  {'hr': (60, 80),   'hrv': (50, 70),  'eda': (1, 3)},
}

EMOTION_TO_IDX = {e: i for i, e in enumerate(FusionHead.EMOTIONS)}


def generate_synthetic_biometrics(emotion: str) -> dict:
    """Generate plausible biometrics for an emotion."""
    ranges = BIOMETRIC_RANGES.get(emotion, BIOMETRIC_RANGES['neutral'])

    return {
        'heart_rate': random.uniform(*ranges['hr']),
        'hrv': random.uniform(*ranges['hrv']),
        'eda': random.uniform(*ranges['eda']),
    }


class EmotionDataset(Dataset):
    """Dataset of text embeddings + biometrics + labels."""

    def __init__(self, embeddings, biometrics, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.biometrics = torch.tensor(biometrics, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.biometrics[idx], self.labels[idx]


def load_emotion_data(downloads_dir: Path):
    """Load emotion-labeled text data."""
    all_data = []
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

    if not all_data:
        raise FileNotFoundError(f"No emotion data found in {downloads_dir}")

    return all_data


def extract_embeddings(gpt, texts, encode_fn, device, batch_size=32):
    """Extract embeddings for all texts."""
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
    print("  QUANTARA EMOTION CLASSIFIER TRAINING")
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
    train_embeddings = extract_embeddings(gpt, train_texts, encode_fn, device)

    val_texts = [t for t, _ in val_data]
    val_emotions = [e for _, e in val_data]
    val_embeddings = extract_embeddings(gpt, val_texts, encode_fn, device)

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

    # Labels
    train_labels = [EMOTION_TO_IDX[e] for e in train_emotions]
    val_labels = [EMOTION_TO_IDX[e] for e in val_emotions]

    # Create datasets
    train_dataset = EmotionDataset(train_embeddings, train_bio_features, train_labels)
    val_dataset = EmotionDataset(val_embeddings, val_bio_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    fusion_head = FusionHead(
        text_dim=n_embd,
        biometric_dim=16,
        hidden_dim=args.hidden_dim,
        num_emotions=7,
        dropout=args.dropout
    ).to(device)

    # Optimizer
    params = list(bio_encoder.parameters()) + list(fusion_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    criterion = nn.NLLLoss()

    # Training loop
    print("\n  Training...")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        bio_encoder.train()
        fusion_head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for text_emb, bio_feat, labels in train_loader:
            text_emb = text_emb.to(device)
            bio_feat = bio_feat.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            bio_emb = bio_encoder(bio_feat)
            probs = fusion_head(text_emb, bio_emb)

            loss = criterion(torch.log(probs + 1e-8), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = probs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # Validate
        bio_encoder.eval()
        fusion_head.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for text_emb, bio_feat, labels in val_loader:
                text_emb = text_emb.to(device)
                bio_feat = bio_feat.to(device)
                labels = labels.to(device)

                bio_emb = bio_encoder(bio_feat)
                probs = fusion_head(text_emb, bio_emb)

                preds = probs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"  Epoch {epoch+1:2d}/{args.epochs}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'fusion_head': fusion_head.state_dict(),
                'biometric_encoder': bio_encoder.state_dict(),
                'n_embd': n_embd,
                'val_acc': val_acc,
            }, 'checkpoints/emotion_fusion_head.pt')
            print(f"    -> Saved checkpoint (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    print("\n" + "=" * 60)
    print(f"  Training complete! Best val_acc: {best_val_acc:.4f}")
    print(f"  Checkpoint saved to: checkpoints/emotion_fusion_head.pt")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train emotion classifier')
    parser.add_argument('--gpt-checkpoint', default='out-quantara-emotion/ckpt.pt')
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

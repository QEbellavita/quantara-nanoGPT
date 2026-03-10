# Multimodal Emotion Analyzer Design

**Date:** 2026-03-10
**Status:** Approved
**Goal:** Replace keyword-based emotion analysis with ML-powered multimodal classification

## Overview

Upgrade the emotion analysis system from simple keyword matching to a hybrid approach combining nanoGPT embeddings with a lightweight classifier head, supporting text-only and multimodal (text + biometrics) inference.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      MultimodalEmotionAnalyzer                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐     ┌──────────────────┐     ┌────────────────┐  │
│  │  TextEncoder     │     │  BiometricEncoder │     │  FusionHead    │  │
│  │  (nanoGPT)       │     │  (normalize+MLP)  │     │  (MLP+softmax) │  │
│  │                  │     │                   │     │                │  │
│  │  text → 512-dim  │     │  biometrics→16-dim│     │  528-dim → 7   │  │
│  └────────┬─────────┘     └─────────┬─────────┘     └───────┬────────┘  │
│           │                         │                       │           │
│           └────────────┬────────────┘                       │           │
│                        ▼                                    │           │
│                   [concatenate]──────────────────────────────┘           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| `TextEncoder` | text string | 512-dim vector | Extract semantic features using trained nanoGPT |
| `BiometricEncoder` | HR, HRV, EDA | 16-dim vector | Normalize and encode physiological signals |
| `FusionHead` | 528-dim concat | 7 emotion probs | Classify fused representation |

## API Design

### Enhanced `/api/emotion/analyze`

**Request:**
```json
{
    "text": "I've been feeling overwhelmed lately",
    "biometrics": {
        "heart_rate": 92,
        "hrv": 35,
        "eda": 4.2
    },
    "return_embedding": false
}
```

**Response:**
```json
{
    "status": "success",
    "dominant_emotion": "fear",
    "confidence": 0.68,
    "scores": {
        "sadness": 0.12,
        "joy": 0.02,
        "love": 0.01,
        "anger": 0.08,
        "fear": 0.68,
        "surprise": 0.04,
        "neutral": 0.05
    },
    "biometric_contribution": 0.23,
    "latency_ms": 8.2
}
```

### Biometric Feature Engineering

| Raw Signal | Derived Features | Emotion Correlation |
|------------|------------------|---------------------|
| Heart Rate | HR, HR_delta | High HR → anger, fear, surprise |
| HRV | HRV, HRV_normalized | Low HRV → stress/anxiety |
| EDA | EDA, EDA_peaks | High EDA → arousal (strong emotion) |

### Fallback Behavior

- No biometrics → text-only classification (512-dim)
- Partial biometrics → zero-fill missing, reduce biometric weight
- No text → biometric-only mode (experimental)

## nanoGPT Modification

Add `get_embedding()` method to `model.py`:

```python
def get_embedding(self, idx):
    """Extract pooled embedding from input tokens."""
    device = idx.device
    b, t = idx.shape

    pos = torch.arange(0, t, dtype=torch.long, device=device)
    tok_emb = self.transformer.wte(idx)
    pos_emb = self.transformer.wpe(pos)
    x = self.transformer.drop(tok_emb + pos_emb)

    for block in self.transformer.h:
        x = block(x)

    x = self.transformer.ln_f(x)

    # Mean pooling over sequence
    embedding = x.mean(dim=1)
    return embedding
```

## Training Strategy

### Data Source

Use existing ~150K labeled emotion samples from `data/quantara_emotion/prepare.py`.

### Synthetic Biometrics

Generate plausible biometrics based on emotion labels for training:

| Emotion | HR Range | HRV Range | EDA Range |
|---------|----------|-----------|-----------|
| joy | 70-90 | 50-80 | 2-4 |
| sadness | 55-70 | 40-60 | 1-2 |
| anger | 85-110 | 20-40 | 5-8 |
| fear | 80-105 | 25-45 | 6-10 |
| surprise | 75-100 | 35-55 | 4-7 |
| love | 65-85 | 55-75 | 2-4 |
| neutral | 60-80 | 50-70 | 1-3 |

### Training Config

```python
{
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "hidden_dim": 128,
    "dropout": 0.3,
    "val_split": 0.1,
    "early_stopping_patience": 3
}
```

### Expected Results

- Text-only accuracy: ~85-90% (vs current keyword ~40%)
- With biometrics: ~88-93% (fusion boost)
- Inference latency: <10ms on CPU

## File Structure

```
quantara-nanoGPT/
├── model.py                      # ADD: get_embedding() method
├── emotion_api_server.py         # MODIFY: use MultimodalEmotionAnalyzer
├── emotion_classifier.py         # NEW: FusionHead + MultimodalEmotionAnalyzer
├── train_emotion_classifier.py   # NEW: Training script for fusion head
└── checkpoints/
    └── emotion_fusion_head.pt    # NEW: Trained classifier weights
```

## Class Structure

```python
# emotion_classifier.py

class BiometricEncoder(nn.Module):
    """Normalize and encode biometric signals → 16-dim"""

class FusionHead(nn.Module):
    """MLP: 528-dim → 128 → 7 emotions"""

class MultimodalEmotionAnalyzer:
    """Main interface - replaces keyword analysis"""

    def __init__(self, gpt_checkpoint, classifier_checkpoint):
        self.gpt = GPT(...)
        self.classifier = FusionHead(...)

    def analyze(self, text, biometrics=None) -> dict:
        # 1. Extract text embedding
        # 2. Encode biometrics (or zeros)
        # 3. Concatenate + classify
        # 4. Return scores

    def analyze_stream(self, text_buffer, biometric_state) -> dict:
        # Real-time variant with state management
```

## Integration

Replace keyword analysis in `emotion_api_server.py`:

```python
# OLD (line 161-196)
def analyze(self, text: str) -> dict:
    # keyword matching...

# NEW
def analyze(self, text: str, biometrics: dict = None) -> dict:
    return self.multimodal_analyzer.analyze(text, biometrics)
```

### Backward Compatibility

- API signature unchanged (biometrics optional)
- Existing `/api/emotion/analyze` calls continue to work
- `/api/emotion/coach` automatically benefits from better analysis

## Neural Ecosystem Integration

Connected to:
- Neural Workflow AI Engine
- AI Conversational Coach
- Emotion-Aware Training Engine
- Biometric Integration Engine
- Real-time Dashboard Data

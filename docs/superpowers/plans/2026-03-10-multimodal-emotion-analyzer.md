# Multimodal Emotion Analyzer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace keyword-based emotion analysis with ML-powered multimodal classification using nanoGPT embeddings + fusion head.

**Architecture:** Extract 512-dim embeddings from trained nanoGPT model, concatenate with 16-dim encoded biometric features, pass through 2-layer MLP classifier to predict 7 emotion categories.

**Tech Stack:** PyTorch, Flask, NumPy, existing nanoGPT model

**Spec:** `docs/superpowers/specs/2026-03-10-multimodal-emotion-analyzer-design.md`

---

## File Structure

| File | Responsibility | Status |
|------|----------------|--------|
| `model.py` | Add `get_embedding()` method | Modify |
| `emotion_classifier.py` | BiometricEncoder, FusionHead, MultimodalEmotionAnalyzer | Create |
| `train_emotion_classifier.py` | Training script for fusion head | Create |
| `tests/test_emotion_classifier.py` | Unit tests for all components | Create |
| `emotion_api_server.py` | Integrate MultimodalEmotionAnalyzer | Modify |
| `checkpoints/` | Store trained classifier weights | Create dir |

---

## Chunk 1: nanoGPT Embedding Extraction

### Task 1: Add get_embedding() to GPT class

**Files:**
- Modify: `model.py:~line 140` (after `generate` method)
- Test: `tests/test_emotion_classifier.py`

- [ ] **Step 1: Create test file with embedding extraction test**

```python
# tests/test_emotion_classifier.py
"""Tests for multimodal emotion analyzer components."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig


class TestGPTEmbedding:
    """Test GPT embedding extraction."""

    @pytest.fixture
    def small_gpt(self):
        """Create a small GPT for testing."""
        config = GPTConfig(
            block_size=64,
            vocab_size=256,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
            bias=True
        )
        model = GPT(config)
        model.eval()
        return model

    def test_get_embedding_returns_correct_shape(self, small_gpt):
        """Embedding should be (batch, n_embd)."""
        batch_size = 2
        seq_len = 10
        idx = torch.randint(0, 256, (batch_size, seq_len))

        embedding = small_gpt.get_embedding(idx)

        assert embedding.shape == (batch_size, 64)

    def test_get_embedding_is_deterministic(self, small_gpt):
        """Same input should produce same embedding."""
        idx = torch.randint(0, 256, (1, 10))

        emb1 = small_gpt.get_embedding(idx)
        emb2 = small_gpt.get_embedding(idx)

        assert torch.allclose(emb1, emb2)

    def test_get_embedding_different_inputs_differ(self, small_gpt):
        """Different inputs should produce different embeddings."""
        idx1 = torch.zeros(1, 10, dtype=torch.long)
        idx2 = torch.ones(1, 10, dtype=torch.long)

        emb1 = small_gpt.get_embedding(idx1)
        emb2 = small_gpt.get_embedding(idx2)

        assert not torch.allclose(emb1, emb2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_emotion_classifier.py::TestGPTEmbedding -v`
Expected: FAIL with "GPT object has no attribute 'get_embedding'"

- [ ] **Step 3: Implement get_embedding() in model.py**

Add after the `generate` method (around line 140):

```python
    def get_embedding(self, idx):
        """
        Extract pooled embedding from input tokens.

        Args:
            idx: (batch, seq_len) tensor of token indices

        Returns:
            (batch, n_embd) tensor of mean-pooled embeddings
        """
        device = idx.device
        b, t = idx.shape
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # Mean pooling over sequence dimension
        embedding = x.mean(dim=1)
        return embedding
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_emotion_classifier.py::TestGPTEmbedding -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_emotion_classifier.py
git commit -m "feat(model): add get_embedding() for emotion classification"
```

---

## Chunk 2: BiometricEncoder Component

### Task 2: Implement BiometricEncoder

**Files:**
- Create: `emotion_classifier.py`
- Test: `tests/test_emotion_classifier.py`

- [ ] **Step 1: Add BiometricEncoder tests**

Append to `tests/test_emotion_classifier.py`:

```python
class TestBiometricEncoder:
    """Test biometric signal encoding."""

    @pytest.fixture
    def encoder(self):
        from emotion_classifier import BiometricEncoder
        return BiometricEncoder()

    def test_encoder_output_shape(self, encoder):
        """Output should be (batch, 16)."""
        biometrics = {
            'heart_rate': 80.0,
            'hrv': 50.0,
            'eda': 3.0
        }

        output = encoder.encode(biometrics)

        assert output.shape == (1, 16)

    def test_encoder_handles_missing_values(self, encoder):
        """Missing biometrics should default to neutral."""
        biometrics = {'heart_rate': 80.0}  # missing hrv, eda

        output = encoder.encode(biometrics)

        assert output.shape == (1, 16)
        assert not torch.isnan(output).any()

    def test_encoder_handles_empty_dict(self, encoder):
        """Empty biometrics should return neutral encoding."""
        output = encoder.encode({})

        assert output.shape == (1, 16)
        assert not torch.isnan(output).any()

    def test_encoder_handles_none(self, encoder):
        """None should return neutral encoding."""
        output = encoder.encode(None)

        assert output.shape == (1, 16)

    def test_encoder_batch_processing(self, encoder):
        """Should handle batch of biometrics."""
        batch = [
            {'heart_rate': 80.0, 'hrv': 50.0, 'eda': 3.0},
            {'heart_rate': 100.0, 'hrv': 30.0, 'eda': 7.0},
        ]

        output = encoder.encode_batch(batch)

        assert output.shape == (2, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_emotion_classifier.py::TestBiometricEncoder -v`
Expected: FAIL with "No module named 'emotion_classifier'"

- [ ] **Step 3: Create emotion_classifier.py with BiometricEncoder**

```python
# emotion_classifier.py
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Multimodal Emotion Classifier
===============================================================================
ML-powered emotion analysis combining text embeddings with biometric signals.

Integrates with:
- Neural Workflow AI Engine
- AI Conversational Coach
- Biometric Integration Engine
- Real-time Dashboard Data
===============================================================================
"""

import torch
import torch.nn as nn
import numpy as np


class BiometricEncoder(nn.Module):
    """
    Encode biometric signals into a dense representation.

    Input features:
    - heart_rate: beats per minute (40-180 typical)
    - hrv: heart rate variability in ms (10-100 typical)
    - eda: electrodermal activity in microsiemens (0.5-20 typical)

    Output: 16-dimensional encoded representation
    """

    # Normalization constants (based on physiological ranges)
    DEFAULTS = {
        'heart_rate': 70.0,
        'hrv': 50.0,
        'eda': 2.0
    }

    RANGES = {
        'heart_rate': (40.0, 180.0),
        'hrv': (10.0, 100.0),
        'eda': (0.5, 20.0)
    }

    def __init__(self, output_dim: int = 16):
        super().__init__()
        self.output_dim = output_dim

        # Input: 6 features (3 raw + 3 derived)
        # - heart_rate_norm, hrv_norm, eda_norm
        # - hr_arousal (high HR indicator)
        # - hrv_stress (low HRV indicator)
        # - eda_activation (high EDA indicator)
        self.encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Tanh()
        )

    def _normalize(self, value: float, key: str) -> float:
        """Normalize value to [-1, 1] range."""
        min_val, max_val = self.RANGES[key]
        normalized = (value - min_val) / (max_val - min_val)
        return max(-1.0, min(1.0, 2.0 * normalized - 1.0))

    def _extract_features(self, biometrics: dict) -> torch.Tensor:
        """Extract and normalize features from biometrics dict."""
        if biometrics is None:
            biometrics = {}

        hr = biometrics.get('heart_rate', self.DEFAULTS['heart_rate'])
        hrv = biometrics.get('hrv', self.DEFAULTS['hrv'])
        eda = biometrics.get('eda', self.DEFAULTS['eda'])

        features = [
            self._normalize(hr, 'heart_rate'),
            self._normalize(hrv, 'hrv'),
            self._normalize(eda, 'eda'),
            1.0 if hr > 100 else 0.0,   # hr_arousal
            1.0 if hrv < 30 else 0.0,   # hrv_stress
            1.0 if eda > 5 else 0.0,    # eda_activation
        ]

        return torch.tensor(features, dtype=torch.float32)

    def encode(self, biometrics: dict) -> torch.Tensor:
        """
        Encode single biometrics dict to tensor.

        Args:
            biometrics: dict with heart_rate, hrv, eda (all optional)

        Returns:
            (1, output_dim) tensor
        """
        features = self._extract_features(biometrics).unsqueeze(0)
        with torch.no_grad():
            return self.encoder(features)

    def encode_batch(self, batch: list) -> torch.Tensor:
        """
        Encode batch of biometrics dicts.

        Args:
            batch: list of biometrics dicts

        Returns:
            (batch_size, output_dim) tensor
        """
        features = torch.stack([self._extract_features(b) for b in batch])
        with torch.no_grad():
            return self.encoder(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for training."""
        return self.encoder(features)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_emotion_classifier.py::TestBiometricEncoder -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add emotion_classifier.py tests/test_emotion_classifier.py
git commit -m "feat(classifier): add BiometricEncoder for physiological signals"
```

---

## Chunk 3: FusionHead Component

### Task 3: Implement FusionHead MLP

**Files:**
- Modify: `emotion_classifier.py`
- Test: `tests/test_emotion_classifier.py`

- [ ] **Step 1: Add FusionHead tests**

Append to `tests/test_emotion_classifier.py`:

```python
class TestFusionHead:
    """Test fusion classification head."""

    @pytest.fixture
    def fusion_head(self):
        from emotion_classifier import FusionHead
        return FusionHead(text_dim=512, biometric_dim=16, num_emotions=7)

    def test_output_shape(self, fusion_head):
        """Output should be (batch, num_emotions)."""
        text_emb = torch.randn(2, 512)
        bio_emb = torch.randn(2, 16)

        output = fusion_head(text_emb, bio_emb)

        assert output.shape == (2, 7)

    def test_output_is_probability_distribution(self, fusion_head):
        """Output should sum to 1 (softmax)."""
        text_emb = torch.randn(1, 512)
        bio_emb = torch.randn(1, 16)

        output = fusion_head(text_emb, bio_emb)

        assert torch.isclose(output.sum(), torch.tensor(1.0), atol=1e-5)

    def test_text_only_mode(self, fusion_head):
        """Should work with None biometrics."""
        text_emb = torch.randn(1, 512)

        output = fusion_head(text_emb, None)

        assert output.shape == (1, 7)
        assert torch.isclose(output.sum(), torch.tensor(1.0), atol=1e-5)

    def test_different_inputs_different_outputs(self, fusion_head):
        """Different embeddings should produce different predictions."""
        text1 = torch.randn(1, 512)
        text2 = torch.randn(1, 512)
        bio = torch.randn(1, 16)

        out1 = fusion_head(text1, bio)
        out2 = fusion_head(text2, bio)

        assert not torch.allclose(out1, out2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_emotion_classifier.py::TestFusionHead -v`
Expected: FAIL with "cannot import name 'FusionHead'"

- [ ] **Step 3: Add FusionHead to emotion_classifier.py**

Append to `emotion_classifier.py`:

```python
class FusionHead(nn.Module):
    """
    Multimodal fusion classification head.

    Concatenates text and biometric embeddings, then classifies
    into emotion categories via 2-layer MLP.
    """

    EMOTIONS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise', 'neutral']

    def __init__(
        self,
        text_dim: int = 512,
        biometric_dim: int = 16,
        hidden_dim: int = 128,
        num_emotions: int = 7,
        dropout: float = 0.3
    ):
        super().__init__()
        self.text_dim = text_dim
        self.biometric_dim = biometric_dim
        self.num_emotions = num_emotions

        # Fusion MLP
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + biometric_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_emotions),
        )

        # Zero biometric embedding for text-only mode
        self.register_buffer(
            'zero_biometric',
            torch.zeros(1, biometric_dim)
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        biometric_embedding: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            text_embedding: (batch, text_dim) from GPT
            biometric_embedding: (batch, biometric_dim) or None

        Returns:
            (batch, num_emotions) probability distribution
        """
        batch_size = text_embedding.shape[0]

        if biometric_embedding is None:
            biometric_embedding = self.zero_biometric.expand(batch_size, -1)

        # Concatenate embeddings
        fused = torch.cat([text_embedding, biometric_embedding], dim=-1)

        # Classify
        logits = self.classifier(fused)
        probs = torch.softmax(logits, dim=-1)

        return probs

    def get_emotion_name(self, index: int) -> str:
        """Get emotion name from index."""
        return self.EMOTIONS[index]

    def get_emotion_index(self, name: str) -> int:
        """Get index from emotion name."""
        return self.EMOTIONS.index(name.lower())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_emotion_classifier.py::TestFusionHead -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add emotion_classifier.py tests/test_emotion_classifier.py
git commit -m "feat(classifier): add FusionHead MLP for multimodal classification"
```

---

## Chunk 4: MultimodalEmotionAnalyzer

### Task 4: Implement main analyzer class

**Files:**
- Modify: `emotion_classifier.py`
- Test: `tests/test_emotion_classifier.py`

- [ ] **Step 1: Add MultimodalEmotionAnalyzer tests**

Append to `tests/test_emotion_classifier.py`:

```python
class TestMultimodalEmotionAnalyzer:
    """Test the main analyzer interface."""

    @pytest.fixture
    def mock_analyzer(self):
        """Create analyzer with randomly initialized weights (no checkpoint)."""
        from emotion_classifier import MultimodalEmotionAnalyzer
        return MultimodalEmotionAnalyzer(
            gpt_checkpoint=None,  # Will create random model
            classifier_checkpoint=None,
            device='cpu'
        )

    def test_analyze_returns_required_fields(self, mock_analyzer):
        """Analysis result should have all required fields."""
        result = mock_analyzer.analyze("I feel happy today")

        assert 'dominant_emotion' in result
        assert 'confidence' in result
        assert 'scores' in result
        assert len(result['scores']) == 7

    def test_analyze_with_biometrics(self, mock_analyzer):
        """Should accept biometric data."""
        biometrics = {'heart_rate': 90, 'hrv': 40, 'eda': 5.0}
        result = mock_analyzer.analyze("I feel anxious", biometrics=biometrics)

        assert 'dominant_emotion' in result
        assert 'biometric_contribution' in result

    def test_scores_sum_to_one(self, mock_analyzer):
        """Emotion scores should sum to 1."""
        result = mock_analyzer.analyze("Test text")

        total = sum(result['scores'].values())
        assert abs(total - 1.0) < 1e-5

    def test_confidence_matches_dominant_score(self, mock_analyzer):
        """Confidence should equal the dominant emotion's score."""
        result = mock_analyzer.analyze("I am so excited!")

        dominant = result['dominant_emotion']
        assert result['confidence'] == result['scores'][dominant]

    def test_empty_text_handled(self, mock_analyzer):
        """Empty text should not crash."""
        result = mock_analyzer.analyze("")

        assert 'dominant_emotion' in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_emotion_classifier.py::TestMultimodalEmotionAnalyzer -v`
Expected: FAIL with "cannot import name 'MultimodalEmotionAnalyzer'"

- [ ] **Step 3: Add MultimodalEmotionAnalyzer to emotion_classifier.py**

Append to `emotion_classifier.py`:

```python
import pickle
import time
from pathlib import Path


class MultimodalEmotionAnalyzer:
    """
    Main interface for multimodal emotion analysis.

    Combines nanoGPT text embeddings with biometric signals
    for accurate emotion classification.

    Connected to:
    - Neural Workflow AI Engine
    - AI Conversational Coach
    - Biometric Integration Engine
    """

    def __init__(
        self,
        gpt_checkpoint: str = None,
        classifier_checkpoint: str = None,
        device: str = 'auto'
    ):
        self.device = self._detect_device(device)

        # Load or create GPT model
        if gpt_checkpoint and Path(gpt_checkpoint).exists():
            self._load_gpt(gpt_checkpoint)
        else:
            self._create_dummy_gpt()

        # Initialize biometric encoder
        self.biometric_encoder = BiometricEncoder(output_dim=16)
        self.biometric_encoder.to(self.device)

        # Load or create fusion head
        self.fusion_head = FusionHead(
            text_dim=self.n_embd,
            biometric_dim=16,
            num_emotions=7
        )

        if classifier_checkpoint and Path(classifier_checkpoint).exists():
            state = torch.load(classifier_checkpoint, map_location=self.device)
            self.fusion_head.load_state_dict(state['fusion_head'])
            self.biometric_encoder.load_state_dict(state['biometric_encoder'])

        self.fusion_head.to(self.device)
        self.fusion_head.eval()

        # Setup tokenizer
        self._setup_tokenizer()

    def _detect_device(self, device: str) -> str:
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            return 'cpu'
        return device

    def _load_gpt(self, checkpoint_path: str):
        """Load trained GPT model."""
        from model import GPT, GPTConfig

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        gptconf = GPTConfig(**checkpoint['model_args'])
        self.gpt = GPT(gptconf)
        self.n_embd = gptconf.n_embd

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        self.gpt.load_state_dict(state_dict)
        self.gpt.eval()
        self.gpt.to(self.device)

    def _create_dummy_gpt(self):
        """Create a small GPT for testing without checkpoint."""
        from model import GPT, GPTConfig

        config = GPTConfig(
            block_size=256,
            vocab_size=256,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
            bias=True
        )
        self.gpt = GPT(config)
        self.gpt.eval()
        self.gpt.to(self.device)
        self.n_embd = 64

        # Adjust fusion head for smaller embedding
        self.fusion_head = FusionHead(
            text_dim=64,
            biometric_dim=16,
            num_emotions=7
        )

    def _setup_tokenizer(self):
        """Setup tokenizer (char-level or BPE)."""
        meta_path = Path('data/quantara_emotion/meta.pkl')

        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            self.encode = lambda s: [stoi.get(c, 0) for c in s]
        else:
            try:
                import tiktoken
                enc = tiktoken.get_encoding("gpt2")
                self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            except ImportError:
                # Fallback: simple ASCII encoding
                self.encode = lambda s: [ord(c) % 256 for c in s]

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Extract embedding from text."""
        if not text:
            text = " "  # Avoid empty tensor

        tokens = self.encode(text)
        # Truncate to block size
        max_len = min(len(tokens), self.gpt.config.block_size)
        tokens = tokens[:max_len]

        idx = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            embedding = self.gpt.get_embedding(idx)

        return embedding

    def analyze(
        self,
        text: str,
        biometrics: dict = None,
        return_embedding: bool = False
    ) -> dict:
        """
        Analyze emotional content of text with optional biometrics.

        Args:
            text: Input text to analyze
            biometrics: Optional dict with heart_rate, hrv, eda
            return_embedding: If True, include raw embedding in result

        Returns:
            Dict with emotion scores, dominant emotion, confidence
        """
        start_time = time.time()

        # Get text embedding
        text_embedding = self._get_text_embedding(text)

        # Get biometric embedding
        if biometrics:
            bio_embedding = self.biometric_encoder.encode(biometrics).to(self.device)
            biometric_contribution = 0.3  # Placeholder - could be learned
        else:
            bio_embedding = None
            biometric_contribution = 0.0

        # Classify
        with torch.no_grad():
            probs = self.fusion_head(text_embedding, bio_embedding)

        probs = probs.squeeze(0).cpu().numpy()

        # Build result
        scores = {
            emotion: float(probs[i])
            for i, emotion in enumerate(FusionHead.EMOTIONS)
        }

        dominant_idx = int(np.argmax(probs))
        dominant_emotion = FusionHead.EMOTIONS[dominant_idx]
        confidence = float(probs[dominant_idx])

        latency_ms = (time.time() - start_time) * 1000

        result = {
            'dominant_emotion': dominant_emotion,
            'confidence': confidence,
            'scores': scores,
            'biometric_contribution': biometric_contribution,
            'latency_ms': round(latency_ms, 2),
            'status': 'success'
        }

        if return_embedding:
            result['embedding'] = text_embedding.squeeze(0).cpu().numpy().tolist()

        return result

    def analyze_batch(
        self,
        texts: list,
        biometrics_list: list = None
    ) -> list:
        """Analyze batch of texts."""
        if biometrics_list is None:
            biometrics_list = [None] * len(texts)

        return [
            self.analyze(text, bio)
            for text, bio in zip(texts, biometrics_list)
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_emotion_classifier.py::TestMultimodalEmotionAnalyzer -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add emotion_classifier.py tests/test_emotion_classifier.py
git commit -m "feat(classifier): add MultimodalEmotionAnalyzer main interface"
```

---

## Chunk 5: Training Script

### Task 5: Create training script for fusion head

**Files:**
- Create: `train_emotion_classifier.py`

- [ ] **Step 1: Create training script**

```python
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

    # Generate synthetic biometrics
    print("\n  Generating synthetic biometrics...")
    bio_encoder = BiometricEncoder(output_dim=16)

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

    # Initialize models
    bio_encoder = BiometricEncoder(output_dim=16).to(device)
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
    criterion = nn.CrossEntropyLoss()

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
    parser.add_argument('--data-dir', default='/Users/bel/Downloads')
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
```

- [ ] **Step 2: Verify script syntax**

Run: `python -m py_compile train_emotion_classifier.py`
Expected: No output (syntax OK)

- [ ] **Step 3: Commit**

```bash
git add train_emotion_classifier.py
git commit -m "feat(training): add fusion head training script with synthetic biometrics"
```

---

## Chunk 6: API Integration

### Task 6: Integrate MultimodalEmotionAnalyzer into API server

**Files:**
- Modify: `emotion_api_server.py`

- [ ] **Step 1: Add import and initialization**

At top of `emotion_api_server.py`, after existing imports (around line 50), add:

```python
try:
    from emotion_classifier import MultimodalEmotionAnalyzer
    HAS_MULTIMODAL = True
except ImportError:
    HAS_MULTIMODAL = False
```

- [ ] **Step 2: Add multimodal analyzer to EmotionGPTModel.__init__**

In `EmotionGPTModel.__init__` (around line 70), after `self._load_model()`, add:

```python
        # Initialize multimodal analyzer if available
        self.multimodal_analyzer = None
        classifier_path = Path(__file__).parent / 'checkpoints' / 'emotion_fusion_head.pt'
        if HAS_MULTIMODAL and classifier_path.exists():
            try:
                self.multimodal_analyzer = MultimodalEmotionAnalyzer(
                    gpt_checkpoint=checkpoint_path,
                    classifier_checkpoint=str(classifier_path),
                    device=self.device
                )
                print("[EmotionGPT] Multimodal analyzer loaded")
            except Exception as e:
                print(f"[EmotionGPT] Multimodal analyzer failed: {e}")
```

- [ ] **Step 3: Replace analyze method**

Replace the `analyze` method in `EmotionGPTModel` (lines 161-196) with:

```python
    def analyze(self, text: str, biometrics: dict = None) -> dict:
        """Analyze emotional content of text with optional biometrics."""

        # Use multimodal analyzer if available
        if self.multimodal_analyzer is not None:
            return self.multimodal_analyzer.analyze(text, biometrics)

        # Fallback to keyword-based analysis
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'love', 'blessed'],
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'crying', 'tears', 'grief', 'lonely'],
            'anger': ['angry', 'furious', 'mad', 'annoyed', 'frustrated', 'hate', 'rage'],
            'fear': ['scared', 'afraid', 'anxious', 'worried', 'nervous', 'terrified', 'panic'],
            'love': ['love', 'adore', 'cherish', 'caring', 'affection', 'romantic'],
            'surprise': ['surprised', 'shocked', 'amazed', 'unexpected', 'wow', 'astonished'],
        }

        text_lower = text.lower()
        scores = {}

        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            scores[emotion] = count

        total = sum(scores.values()) or 1
        scores = {k: v / total for k, v in scores.items()}

        if max(scores.values()) < 0.3:
            scores['neutral'] = 0.5

        dominant = max(scores, key=scores.get)

        return {
            'text': text[:200] + '...' if len(text) > 200 else text,
            'scores': scores,
            'dominant_emotion': dominant,
            'confidence': scores[dominant],
            'status': 'success'
        }
```

- [ ] **Step 4: Update /api/emotion/analyze endpoint**

Replace the analyze endpoint (around line 343-365) with:

```python
    @app.route('/api/emotion/analyze', methods=['POST'])
    def analyze():
        """Analyze emotional content

        Request body:
        {
            "text": "I'm feeling great today!",
            "biometrics": {  // optional
                "heart_rate": 80,
                "hrv": 50,
                "eda": 3.0
            }
        }
        """
        try:
            data = request.json or {}
            text = data.get('text', '')

            if not text:
                return jsonify({'error': 'text is required'}), 400

            biometrics = data.get('biometrics')
            result = model.analyze(text, biometrics)

            return jsonify({**result, 'status': 'success'})

        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500
```

- [ ] **Step 5: Run existing tests**

Run: `python test_emotion_api.py` (with server running)
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add emotion_api_server.py
git commit -m "feat(api): integrate MultimodalEmotionAnalyzer with fallback to keywords"
```

---

## Chunk 7: Final Integration Test

### Task 7: End-to-end test

- [ ] **Step 1: Run all unit tests**

Run: `pytest tests/test_emotion_classifier.py -v`
Expected: All 17 tests pass

- [ ] **Step 2: Train the classifier (if data available)**

Run: `python train_emotion_classifier.py --gpt-checkpoint out-quantara-emotion/ckpt.pt --epochs 5`
Expected: Training completes, checkpoint saved

- [ ] **Step 3: Test API with multimodal analyzer**

Start server: `python emotion_api_server.py --port 5050`

Test with curl:
```bash
curl -X POST http://localhost:5050/api/emotion/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so anxious about tomorrow", "biometrics": {"heart_rate": 95, "hrv": 28, "eda": 7.5}}'
```

Expected: Response with `dominant_emotion`, `scores`, `biometric_contribution`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete multimodal emotion analyzer implementation"
```

---

## Summary

| Task | Component | Files |
|------|-----------|-------|
| 1 | GPT embedding extraction | `model.py` |
| 2 | BiometricEncoder | `emotion_classifier.py` |
| 3 | FusionHead MLP | `emotion_classifier.py` |
| 4 | MultimodalEmotionAnalyzer | `emotion_classifier.py` |
| 5 | Training script | `train_emotion_classifier.py` |
| 6 | API integration | `emotion_api_server.py` |
| 7 | End-to-end test | All files |

**Expected accuracy improvement:** 40% (keywords) → 85-90% (ML-powered)

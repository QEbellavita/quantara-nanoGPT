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
import pickle
import time
from pathlib import Path


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
            # Fallback: simple ASCII encoding compatible with vocab_size=256
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

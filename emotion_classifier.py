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

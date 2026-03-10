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

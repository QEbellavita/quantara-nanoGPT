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

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
        return FusionHead(text_dim=512, biometric_dim=16, pose_dim=16)

    def test_output_shape(self, fusion_head):
        """Output should be (batch, num_emotions) and (batch, num_families)."""
        text_emb = torch.randn(2, 512)
        bio_emb = torch.randn(2, 16)

        emotion_probs, family_probs = fusion_head(text_emb, bio_emb)

        assert emotion_probs.shape == (2, 32)
        assert family_probs.shape == (2, 9)

    def test_output_is_probability_distribution(self, fusion_head):
        """Emotion probs should sum to 1 (softmax)."""
        text_emb = torch.randn(1, 512)
        bio_emb = torch.randn(1, 16)

        emotion_probs, family_probs = fusion_head(text_emb, bio_emb)

        assert torch.isclose(emotion_probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(family_probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_text_only_mode(self, fusion_head):
        """Should work with None biometrics."""
        text_emb = torch.randn(1, 512)

        emotion_probs, family_probs = fusion_head(text_emb, None)

        assert emotion_probs.shape == (1, 32)
        assert torch.isclose(emotion_probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_different_inputs_different_outputs(self, fusion_head):
        """Different embeddings should produce different predictions."""
        text1 = torch.randn(1, 512)
        text2 = torch.randn(1, 512)
        bio = torch.randn(1, 16)

        out1, _ = fusion_head(text1, bio)
        out2, _ = fusion_head(text2, bio)

        assert not torch.allclose(out1, out2)


class TestAttentionFusionHead:
    """Test cross-modal attention fusion head."""

    @pytest.fixture
    def fusion_head(self):
        from emotion_classifier import AttentionFusionHead
        return AttentionFusionHead(
            text_dim=384, biometric_dim=16, pose_dim=16,
            hidden_dim=128, num_emotions=32, num_families=9
        )

    def test_forward_output_shapes(self, fusion_head):
        text = torch.randn(1, 384)
        bio = torch.randn(1, 16)
        pose = torch.randn(1, 16)
        emotion_probs, family_probs = fusion_head(text, bio, pose)
        assert emotion_probs.shape == (1, 32)
        assert family_probs.shape == (1, 9)

    def test_forward_without_biometrics(self, fusion_head):
        text = torch.randn(1, 384)
        emotion_probs, family_probs = fusion_head(text, None, None)
        assert emotion_probs.shape == (1, 32)
        assert family_probs.shape == (1, 9)

    def test_probs_sum_to_one(self, fusion_head):
        text = torch.randn(1, 384)
        bio = torch.randn(1, 16)
        emotion_probs, family_probs = fusion_head(text, bio, None)
        assert abs(emotion_probs.sum().item() - 1.0) < 1e-5
        assert abs(family_probs.sum().item() - 1.0) < 1e-5

    def test_classify_with_fallback_returns_modality_weights(self, fusion_head):
        text = torch.randn(1, 384)
        bio = torch.randn(1, 16)
        result = fusion_head.classify_with_fallback(text, bio, None)
        assert 'modality_weights' in result
        assert 'text' in result['modality_weights']
        assert 'bio' in result['modality_weights']
        assert 'pose' in result['modality_weights']

    def test_modality_weights_sum_to_one(self, fusion_head):
        text = torch.randn(1, 384)
        bio = torch.randn(1, 16)
        pose = torch.randn(1, 16)
        result = fusion_head.classify_with_fallback(text, bio, pose)
        weights = result['modality_weights']
        total = weights['text'] + weights['bio'] + weights['pose']
        assert abs(total - 1.0) < 1e-4

    def test_attention_mask_excludes_absent_modalities(self, fusion_head):
        text = torch.randn(1, 384)
        result = fusion_head.classify_with_fallback(text, None, None)
        weights = result['modality_weights']
        assert weights['text'] == 1.0
        assert weights['bio'] == 0.0
        assert weights['pose'] == 0.0

    def test_batch_forward(self, fusion_head):
        text = torch.randn(4, 384)
        bio = torch.randn(4, 16)
        emotion_probs, family_probs = fusion_head(text, bio, None)
        assert emotion_probs.shape == (4, 32)
        assert family_probs.shape == (4, 9)

    def test_backward_compat_with_fusionhead_interface(self, fusion_head):
        from emotion_classifier import FusionHead
        old = FusionHead(text_dim=384, biometric_dim=16, pose_dim=16)
        for method in ['forward', 'classify_with_fallback', 'get_emotion_name', 'get_emotion_index']:
            assert hasattr(fusion_head, method), f"Missing method: {method}"


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
        assert len(result['scores']) == 32

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

    def test_confidence_is_valid(self, mock_analyzer):
        """Confidence should be between 0 and 1."""
        result = mock_analyzer.analyze("I am so excited!")

        assert 0.0 <= result['confidence'] <= 1.0
        assert result['dominant_emotion'] in result['scores']

    def test_empty_text_handled(self, mock_analyzer):
        """Empty text should not crash."""
        result = mock_analyzer.analyze("")

        assert 'dominant_emotion' in result

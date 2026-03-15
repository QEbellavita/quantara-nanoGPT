# tests/test_go_emotions.py
"""Tests for GoEmotionsEncoder and distillation mapping."""

import pytest
import torch
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGoEmotionsMapping:
    """Test GOEMOTIONS_TO_QUANTARA mapping coverage."""

    def test_mapping_covers_all_28_labels(self):
        """All 28 GoEmotions labels should be in the mapping."""
        from emotion_classifier import GoEmotionsEncoder

        go_emotions_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
            'pride', 'realization', 'relief', 'remorse', 'sadness',
            'surprise', 'neutral',
        ]

        mapping = GoEmotionsEncoder.GOEMOTIONS_TO_QUANTARA

        assert len(go_emotions_labels) == 28
        for label in go_emotions_labels:
            assert label in mapping, f"GoEmotions label '{label}' missing from mapping"

    def test_mapping_has_exactly_28_entries(self):
        """Mapping should have exactly 28 entries (one per GoEmotions label)."""
        from emotion_classifier import GoEmotionsEncoder

        mapping = GoEmotionsEncoder.GOEMOTIONS_TO_QUANTARA
        assert len(mapping) == 28

    def test_all_mapped_emotions_exist_in_fusionhead(self):
        """Every mapped Quantara emotion should be in FusionHead.EMOTIONS."""
        from emotion_classifier import GoEmotionsEncoder, FusionHead

        mapping = GoEmotionsEncoder.GOEMOTIONS_TO_QUANTARA

        for go_label, quantara_emotion in mapping.items():
            assert quantara_emotion in FusionHead.EMOTIONS, (
                f"Mapped emotion '{quantara_emotion}' (from '{go_label}') "
                f"not in FusionHead.EMOTIONS"
            )

    def test_mapping_values_are_lowercase(self):
        """All mapped Quantara emotions should be lowercase."""
        from emotion_classifier import GoEmotionsEncoder

        for go_label, q_emotion in GoEmotionsEncoder.GOEMOTIONS_TO_QUANTARA.items():
            assert q_emotion == q_emotion.lower(), (
                f"Mapping for '{go_label}' is not lowercase: '{q_emotion}'"
            )


class TestGoEmotionsEncoderInit:
    """Test GoEmotionsEncoder initialization with mocked model loading."""

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Mock transformers model and tokenizer to avoid downloads."""
        mock_config = MagicMock()
        mock_config.hidden_size = 768
        mock_config.num_labels = 28
        mock_config.id2label = {
            i: label for i, label in enumerate([
                'admiration', 'amusement', 'anger', 'annoyance', 'approval',
                'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
                'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
                'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                'pride', 'realization', 'relief', 'remorse', 'sadness',
                'surprise', 'neutral',
            ])
        }

        mock_model = MagicMock()
        mock_model.config = mock_config
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.to = MagicMock(return_value=mock_model)

        mock_tokenizer = MagicMock()

        return mock_model, mock_tokenizer

    @patch('emotion_classifier.AutoModelForSequenceClassification.from_pretrained')
    @patch('emotion_classifier.AutoTokenizer.from_pretrained')
    def test_initialization_sets_dimensions(self, mock_tok_cls, mock_model_cls,
                                            mock_model_and_tokenizer):
        """Encoder should set correct embedding_dim, num_go_labels, combined_dim."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_model_cls.return_value = mock_model
        mock_tok_cls.return_value = mock_tokenizer

        from emotion_classifier import GoEmotionsEncoder
        encoder = GoEmotionsEncoder(device='cpu')

        assert encoder.embedding_dim == 768
        assert encoder.num_go_labels == 28
        assert encoder.combined_dim == 796  # 768 + 28

    @patch('emotion_classifier.AutoModelForSequenceClassification.from_pretrained')
    @patch('emotion_classifier.AutoTokenizer.from_pretrained')
    def test_initialization_builds_label_index(self, mock_tok_cls, mock_model_cls,
                                               mock_model_and_tokenizer):
        """Encoder should build go_label_names list with 28 entries."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_model_cls.return_value = mock_model
        mock_tok_cls.return_value = mock_tokenizer

        from emotion_classifier import GoEmotionsEncoder
        encoder = GoEmotionsEncoder(device='cpu')

        assert len(encoder.go_label_names) == 28
        assert 'joy' in encoder.go_label_names
        assert 'neutral' in encoder.go_label_names

    @patch('emotion_classifier.AutoModelForSequenceClassification.from_pretrained')
    @patch('emotion_classifier.AutoTokenizer.from_pretrained')
    def test_initialization_builds_quantara_idx_mapping(self, mock_tok_cls,
                                                        mock_model_cls,
                                                        mock_model_and_tokenizer):
        """Encoder should build _go_to_quantara_idx list with 28 valid indices."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        mock_model_cls.return_value = mock_model
        mock_tok_cls.return_value = mock_tokenizer

        from emotion_classifier import GoEmotionsEncoder, FusionHead
        encoder = GoEmotionsEncoder(device='cpu')

        assert len(encoder._go_to_quantara_idx) == 28
        for idx in encoder._go_to_quantara_idx:
            assert 0 <= idx < len(FusionHead.EMOTIONS), (
                f"Index {idx} out of range for FusionHead.EMOTIONS"
            )

    @patch('emotion_classifier.HAS_TRANSFORMERS', False)
    def test_initialization_raises_without_transformers(self):
        """Should raise ImportError when transformers is not available."""
        from emotion_classifier import GoEmotionsEncoder

        with pytest.raises(ImportError, match="transformers not installed"):
            GoEmotionsEncoder(device='cpu')


class TestMapGoProbsToQuantara:
    """Test map_go_probs_to_quantara probability remapping."""

    @pytest.fixture
    def encoder(self):
        """Create GoEmotionsEncoder with mocked model."""
        mock_config = MagicMock()
        mock_config.hidden_size = 768
        mock_config.num_labels = 28
        mock_config.id2label = {
            i: label for i, label in enumerate([
                'admiration', 'amusement', 'anger', 'annoyance', 'approval',
                'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
                'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
                'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                'pride', 'realization', 'relief', 'remorse', 'sadness',
                'surprise', 'neutral',
            ])
        }

        mock_model = MagicMock()
        mock_model.config = mock_config
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.to = MagicMock(return_value=mock_model)

        mock_tokenizer = MagicMock()

        with patch('emotion_classifier.AutoModelForSequenceClassification.from_pretrained',
                   return_value=mock_model), \
             patch('emotion_classifier.AutoTokenizer.from_pretrained',
                   return_value=mock_tokenizer):
            from emotion_classifier import GoEmotionsEncoder
            return GoEmotionsEncoder(device='cpu')

    def test_output_shape_single(self, encoder):
        """Single 28-dim input should produce (1, 32) output."""
        go_probs = torch.rand(28)
        result = encoder.map_go_probs_to_quantara(go_probs)
        assert result.shape == (1, 32)

    def test_output_shape_batch(self, encoder):
        """Batch of 28-dim inputs should produce (batch, 32) output."""
        go_probs = torch.rand(4, 28)
        result = encoder.map_go_probs_to_quantara(go_probs)
        assert result.shape == (4, 32)

    def test_output_values_non_negative(self, encoder):
        """All mapped values should be >= 0."""
        go_probs = torch.rand(2, 28)
        result = encoder.map_go_probs_to_quantara(go_probs)
        assert (result >= 0).all()

    def test_zero_input_gives_zero_output(self, encoder):
        """Zero GoEmotions probs should produce zero Quantara probs."""
        go_probs = torch.zeros(28)
        result = encoder.map_go_probs_to_quantara(go_probs)
        assert result.sum().item() == 0.0

    def test_high_prob_maps_correctly(self, encoder):
        """A high probability for a specific GoEmotions label should appear
        at the corresponding Quantara index."""
        from emotion_classifier import FusionHead

        go_probs = torch.zeros(28)
        # 'joy' is at index 17 in the standard GoEmotions order
        joy_idx = encoder.go_label_names.index('joy')
        go_probs[joy_idx] = 0.95

        result = encoder.map_go_probs_to_quantara(go_probs)

        quantara_joy_idx = FusionHead.EMOTIONS.index('joy')
        assert result[0, quantara_joy_idx].item() == pytest.approx(0.95, abs=1e-5)

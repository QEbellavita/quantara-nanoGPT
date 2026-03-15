import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTaxonomyMapping:
    def test_goemotions_mapping_covers_all_27(self):
        from evaluate import GOEMOTIONS_MAPPING
        assert len(GOEMOTIONS_MAPPING) == 27

    def test_goemotions_mapping_targets_are_valid_emotions(self):
        from evaluate import GOEMOTIONS_MAPPING
        from emotion_classifier import FusionHead
        valid = set(FusionHead.EMOTIONS)
        for source, target in GOEMOTIONS_MAPPING.items():
            if target is not None:
                assert target in valid, f"{source} maps to invalid emotion: {target}"

    def test_compute_metrics_returns_expected_keys(self):
        from evaluate import compute_metrics
        y_true = ['joy', 'sadness', 'anger', 'joy']
        y_pred = ['joy', 'sadness', 'fear', 'excitement']
        metrics = compute_metrics(y_true, y_pred)
        assert 'weighted_f1' in metrics
        assert 'accuracy' in metrics
        assert 'per_emotion' in metrics

    def test_semeval_mapping_covers_all_11(self):
        from evaluate import SEMEVAL_MAPPING
        assert len(SEMEVAL_MAPPING) == 11

    def test_semeval_mapping_targets_are_valid(self):
        from evaluate import SEMEVAL_MAPPING
        from emotion_classifier import FusionHead
        valid = set(FusionHead.EMOTIONS)
        for source, target in SEMEVAL_MAPPING.items():
            assert target in valid, f"{source} maps to invalid emotion: {target}"

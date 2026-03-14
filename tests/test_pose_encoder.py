"""Tests for PoseFeatureExtractor and PoseEncoder.

Covers Task 6 (feature extraction from COCO keypoints) and
Task 7 (neural pose encoder network).
"""

import pytest
import torch

from pose_encoder import PoseFeatureExtractor, PoseEncoder, POSE_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Helper poses
# ---------------------------------------------------------------------------

def _make_standing_pose():
    return [
        [0.5, 0.1, 0.9], [0.48, 0.08, 0.9], [0.52, 0.08, 0.9],
        [0.45, 0.1, 0.9], [0.55, 0.1, 0.9],
        [0.4, 0.3, 0.9], [0.6, 0.3, 0.9],
        [0.38, 0.5, 0.9], [0.62, 0.5, 0.9],
        [0.38, 0.7, 0.9], [0.62, 0.7, 0.9],
        [0.45, 0.6, 0.9], [0.55, 0.6, 0.9],
        [0.45, 0.8, 0.9], [0.55, 0.8, 0.9],
        [0.45, 1.0, 0.9], [0.55, 1.0, 0.9],
    ]


def _make_slouched_pose():
    pose = _make_standing_pose()
    pose[5][1] = 0.45  # left shoulder lower
    pose[6][1] = 0.45  # right shoulder lower
    pose[0][1] = 0.35  # nose drooped
    return pose


# ===================================================================
# Task 6 – PoseFeatureExtractor
# ===================================================================

class TestPoseFeatureExtractor:

    def setup_method(self):
        self.extractor = PoseFeatureExtractor()

    # -- basic extraction ---------------------------------------------------

    def test_produces_8_features_from_valid_keypoints(self):
        features = self.extractor.extract(_make_standing_pose())
        assert features is not None
        assert len(features) == 8
        for name in POSE_FEATURE_NAMES:
            assert name in features

    # -- slouch score -------------------------------------------------------

    def test_slouch_score_lower_when_slouched(self):
        standing = self.extractor.extract(_make_standing_pose())
        slouched = self.extractor.extract(_make_slouched_pose())
        assert slouched['slouch_score'] < standing['slouch_score']

    # -- symmetry -----------------------------------------------------------

    def test_symmetry_score_high_for_symmetric_pose(self):
        features = self.extractor.extract(_make_standing_pose())
        assert features['symmetry_score'] > 0.8

    # -- temporal defaults --------------------------------------------------

    def test_temporal_features_default_zero_empty_buffer(self):
        extractor = PoseFeatureExtractor()  # fresh, empty buffer
        features = extractor.extract(_make_standing_pose())
        assert features['gesture_speed'] == 0.0
        assert features['stillness_duration'] == 0.0

    # -- gesture speed after movement ---------------------------------------

    def test_gesture_speed_positive_after_movement(self):
        pose1 = _make_standing_pose()
        self.extractor.extract(pose1)  # first frame populates buffer

        pose2 = _make_standing_pose()
        # move wrists significantly
        pose2[9][0] += 0.3
        pose2[10][0] -= 0.3
        features = self.extractor.extract(pose2)
        assert features['gesture_speed'] > 0.0

    # -- invalid input ------------------------------------------------------

    @pytest.mark.parametrize("bad_input", [
        None,
        [],
        [[0.0, 0.0, 0.0]] * 5,   # wrong count
    ])
    def test_returns_none_for_invalid_input(self, bad_input):
        assert self.extractor.extract(bad_input) is None

    # -- value ranges -------------------------------------------------------

    def test_all_features_within_documented_ranges(self):
        features = self.extractor.extract(_make_standing_pose())
        assert 0.0 <= features['slouch_score'] <= 1.0
        assert 0.0 <= features['openness_score'] <= 1.0
        assert 0.0 <= features['tension_score'] <= 1.0
        assert -1.0 <= features['head_tilt'] <= 1.0
        assert 0.0 <= features['gesture_speed'] <= 1.0
        assert 0.0 <= features['symmetry_score'] <= 1.0
        assert -1.0 <= features['forward_lean'] <= 1.0
        assert 0.0 <= features['stillness_duration'] <= 1.0


# ===================================================================
# Task 7 – PoseEncoder neural network
# ===================================================================

class TestPoseEncoder:

    def setup_method(self):
        self.encoder = PoseEncoder()

    def test_output_shape(self):
        features = PoseFeatureExtractor().extract(_make_standing_pose())
        out = self.encoder.encode(features)
        assert out.shape == (1, 16)

    def test_zero_embedding_for_none(self):
        out = self.encoder.encode(None)
        assert out.shape == (1, 16)
        assert torch.all(out == 0.0)

    def test_batch_encode_shape(self):
        extractor = PoseFeatureExtractor()
        f1 = extractor.extract(_make_standing_pose())
        f2 = extractor.extract(_make_standing_pose())
        out = self.encoder.encode_batch([f1, f2, None])
        assert out.shape == (3, 16)

    def test_forward_training(self):
        t = torch.randn(4, 8)
        out = self.encoder(t)
        assert out.shape == (4, 16)

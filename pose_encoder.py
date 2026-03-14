"""Pose feature extraction and neural encoding for emotion-aware training.

Integrates with Neural Workflow AI Engine, Biometric Integration Engine,
RuView WiFi Sensing Provider, Emotion-Aware Training Engine, and Real-time
Data pipelines within the Quantara intelligent ecosystem (Phases 1-5).

PoseFeatureExtractor converts 17 COCO keypoints into 8 emotion-relevant
features. PoseEncoder maps those features into a 16-dimensional embedding
suitable for cross-modal fusion with other biometric signals.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Canonical feature order used by PoseEncoder
POSE_FEATURE_NAMES = [
    'slouch_score',
    'openness_score',
    'tension_score',
    'head_tilt',
    'gesture_speed',
    'symmetry_score',
    'forward_lean',
    'stillness_duration',
]

# COCO keypoint indices
NOSE = 0
L_EYE = 1; R_EYE = 2
L_EAR = 3; R_EAR = 4
L_SHOULDER = 5; R_SHOULDER = 6
L_ELBOW = 7; R_ELBOW = 8
L_WRIST = 9; R_WRIST = 10
L_HIP = 11; R_HIP = 12
L_KNEE = 13; R_KNEE = 14
L_ANKLE = 15; R_ANKLE = 16

EXPECTED_UPRIGHT_RATIO = 0.3
TEMPORAL_BUFFER_SIZE = 10  # ~0.5s at 20 Hz


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


class PoseFeatureExtractor:
    """Extracts 8 emotion-relevant features from 17 COCO keypoints.

    Y-axis increases downward (screen coordinates). Maintains a rolling
    temporal buffer of the last 10 frames for gesture_speed and
    stillness_duration computation.
    """

    def __init__(self) -> None:
        self._buffer: deque = deque(maxlen=TEMPORAL_BUFFER_SIZE)
        self._still_frames: int = 0

    def extract(self, keypoints: Optional[List]) -> Optional[Dict[str, float]]:
        """Return dict of 8 named features, or None for invalid input."""
        if keypoints is None or not isinstance(keypoints, list) or len(keypoints) != 17:
            return None

        kp = keypoints  # alias

        # Derived landmarks
        shoulder_mid_x = (kp[L_SHOULDER][0] + kp[R_SHOULDER][0]) / 2.0
        shoulder_mid_y = (kp[L_SHOULDER][1] + kp[R_SHOULDER][1]) / 2.0
        hip_mid_x = (kp[L_HIP][0] + kp[R_HIP][0]) / 2.0
        hip_mid_y = (kp[L_HIP][1] + kp[R_HIP][1]) / 2.0
        shoulder_width = abs(kp[R_SHOULDER][0] - kp[L_SHOULDER][0])
        ear_mid_y = (kp[L_EAR][1] + kp[R_EAR][1]) / 2.0

        # 1. slouch_score – bigger hip_y-shoulder_y gap = more upright
        torso_len = hip_mid_y - shoulder_mid_y
        slouch_score = _clamp(torso_len / EXPECTED_UPRIGHT_RATIO, 0.0, 1.0)

        # 2. openness_score – wrist spread normalised by shoulder width
        wrist_spread = abs(kp[R_WRIST][0] - kp[L_WRIST][0])
        if shoulder_width > 0:
            openness_score = _clamp(wrist_spread / (3.0 * shoulder_width), 0.0, 1.0)
        else:
            openness_score = 0.0

        # 3. tension_score – smaller shoulder-ear gap = raised shoulders
        shoulder_ear_gap = shoulder_mid_y - ear_mid_y
        tension_score = _clamp(1.0 - shoulder_ear_gap / 0.25, 0.0, 1.0)

        # 4. head_tilt
        head_tilt = _clamp((kp[NOSE][0] - shoulder_mid_x) / 0.2, -1.0, 1.0)

        # 5. gesture_speed (temporal)
        gesture_speed = 0.0
        if len(self._buffer) > 0:
            prev = self._buffer[-1]
            displacements = []
            for i in range(17):
                dx = kp[i][0] - prev[i][0]
                dy = kp[i][1] - prev[i][1]
                displacements.append(math.sqrt(dx * dx + dy * dy))
            mean_disp = sum(displacements) / len(displacements)
            gesture_speed = _clamp(math.tanh(mean_disp / 50.0), 0.0, 1.0)

        # 6. symmetry_score
        mirror_pairs = [
            (L_SHOULDER, R_SHOULDER),
            (L_ELBOW, R_ELBOW),
            (L_WRIST, R_WRIST),
            (L_HIP, R_HIP),
        ]
        midline_x = shoulder_mid_x
        diffs = []
        for li, ri in mirror_pairs:
            left_off = abs(kp[li][0] - midline_x)
            right_off = abs(kp[ri][0] - midline_x)
            diffs.append(abs(left_off - right_off))
        mean_diff = sum(diffs) / len(diffs)
        symmetry_score = _clamp(1.0 - mean_diff / 0.2, 0.0, 1.0)

        # 7. forward_lean – shoulder midpoint vs hip midpoint along X
        forward_lean = _clamp((shoulder_mid_x - hip_mid_x) / 0.15, -1.0, 1.0)

        # 8. stillness_duration (temporal)
        if gesture_speed < 0.02:
            self._still_frames += 1
        else:
            self._still_frames = 0

        stillness_duration = 0.0
        if len(self._buffer) > 0:
            stillness_duration = _clamp(math.tanh(self._still_frames / 10.0), 0.0, 1.0)

        # Update temporal buffer
        self._buffer.append([list(pt) for pt in kp])

        return {
            'slouch_score': slouch_score,
            'openness_score': openness_score,
            'tension_score': tension_score,
            'head_tilt': head_tilt,
            'gesture_speed': gesture_speed,
            'symmetry_score': symmetry_score,
            'forward_lean': forward_lean,
            'stillness_duration': stillness_duration,
        }


class PoseEncoder(nn.Module):
    """Neural encoder: 8 pose features -> 16-dim embedding.

    Integrates with Neural Workflow AI Engine, Biometric Integration Engine,
    RuView WiFi Sensing Provider, Emotion-Aware Training Engine, and
    Real-time Data streams for cross-modal fusion within the Quantara
    intelligent ecosystem (Phases 1-5).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.tanh = nn.Tanh()
        self.register_buffer('zero_pose', torch.zeros(1, 16))

    def forward(self, features_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass for training. Input shape: (batch, 8)."""
        x = self.relu(self.fc1(features_tensor))
        return self.tanh(self.fc2(x))

    def _extract_features(self, feature_dict: Dict[str, float]) -> torch.Tensor:
        """Convert a feature dict to a (1, 8) tensor in canonical order."""
        vals = [feature_dict[name] for name in POSE_FEATURE_NAMES]
        return torch.tensor([vals], dtype=torch.float32)

    def encode(self, pose_features_dict: Optional[Dict[str, float]]) -> torch.Tensor:
        """Encode a single feature dict -> (1, 16) tensor. Returns zero_pose if None."""
        if pose_features_dict is None:
            return self.zero_pose.clone()
        with torch.no_grad():
            t = self._extract_features(pose_features_dict)
            return self.forward(t)

    def encode_batch(self, list_of_dicts: List[Optional[Dict[str, float]]]) -> torch.Tensor:
        """Encode a batch -> (batch, 16) tensor. None entries get zero embedding."""
        embeddings = [self.encode(d) for d in list_of_dicts]
        return torch.cat(embeddings, dim=0)

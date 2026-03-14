# RuView Calibration Model & Pose-Based Emotion Features

**Date:** 2026-03-15
**Status:** Approved
**Scope:** Enhance RuView WiFi sensing integration with learned calibration and body pose emotion features

## Problem

The current RuView integration uses linear approximations to map WiFi-sensed signals (breathing rate, motion level) to Quantara's biometric format (HRV, EDA). These approximations are coarse and don't account for per-user or per-environment variation. Additionally, RuView's 17-keypoint body pose data is collected but not used for emotion classification.

## Solution

Two enhancements:
1. **Calibration Model** — Replace linear mappings with a learned neural network, trained on existing real biometric datasets, with online per-user adaptation.
2. **Pose Encoder** — Extract emotion-relevant posture features from body keypoints and feed them as a parallel input to the emotion classifier.

---

## Enhancement 1: WiFi Calibration Model

### Architecture

`WiFiCalibrationModel` — small 2-layer network:
- Input: 2 features (breathing_rate, motion_level) — heart_rate is passed through from RuView unmodified, as RuView already provides direct HR measurement
- Hidden: 16 units, ReLU
- Output: 2 derived values (hrv, eda), clamped via sigmoid + rescale to physiological ranges (HRV: 10-100ms, EDA: 0.5-20uS)
- Output activation: `sigmoid(x) * (max - min) + min` per output, guaranteeing values stay within `BiometricEncoder.RANGES`

### Paired Data Construction

Real biometric datasets (Empatica E4, smartwatch, HR emotion dataset) contain wearable-only readings. To create training pairs:
1. For each real wearable reading (HR, HRV, EDA), derive corresponding WiFi signals using the existing linear mappings inverted: `breathing_rate = inverse of _breathing_to_hrv(hrv)`, `motion_level = inverse of _motion_to_eda(eda)`
2. Add Gaussian noise (sigma=10% of value range) to simulate WiFi sensor imprecision
3. Augment with random perturbations: jitter breathing_rate +-3 BPM, motion_level +-0.15
4. Train the calibration model to reconstruct the original (HRV, EDA) from the noisy (breathing_rate, motion_level)

The model learns a denoised, nonlinear mapping that outperforms the linear approximation, particularly at physiological extremes. The key value is that online adaptation (#Online Adaptation) will later refine this base with real paired WiFi+wearable data.

### Online Adaptation

`PersonalCalibrationBuffer`:
- Per-environment rolling buffer (last 100 paired readings), keyed by `profile_id` (defaults to `"default"`, can be set to a room name, user ID, or any string passed via API or env var `RUVIEW_PROFILE_ID`)
- When a wearable is simultaneously present, stores (wifi_reading, wearable_reading) pairs
- Fine-tuning triggers after 20 accumulated pairs, then every 50 new pairs
- Fine-tunes on a cloned model copy, then atomically swaps weights behind a `threading.Lock` — inference reads are never blocked during fine-tuning computation, only during the microsecond weight swap
- Calibration state saved to `calibration_profiles/{profile_id}.pt` (directory auto-created, path configurable via `RUVIEW_CALIBRATION_DIR` env var)
- Falls back to base model when no personal calibration exists or on any file I/O error

### Integration

`ruview_provider.get_biometrics()` uses the calibration model instead of `_breathing_to_hrv()` / `_motion_to_eda()`. Output format unchanged — still `{heart_rate, hrv, eda}`. Falls back to linear approximation if calibration model checkpoint is missing.

Data paths: `heart_rate` and `breathing_rate` come from `_averaged_vitals()` (vitals buffer). `motion_level` comes from `_presence_data` (presence stream). Both are available in WebSocket mode. In REST mode, `_poll_vitals()` is updated to also fetch motion_level from `/api/v1/sensing/latest` alongside vital signs.

### Training Invocation

Calibration training runs as a separate entry point: `python wifi_calibration.py --train`. It does NOT run as part of `train_emotion_classifier.py`. The emotion classifier training script imports the already-trained calibration model.

### File: `wifi_calibration.py`

Contains:
- `WiFiCalibrationModel(nn.Module)` — the calibration network (2→16→2)
- `PersonalCalibrationBuffer` — online adaptation buffer, fine-tuning with lock
- `train_calibration_model()` — offline training with bootstrapped pairs
- `load_calibration_model()` — loads base model + optional user profile

---

## Enhancement 2: Pose Encoder

### Keypoint Format

RuView outputs 17 keypoints in COCO format with Y-axis increasing downward (screen coordinates). `get_pose_features()` expects a list of 17 `[x, y, confidence]` arrays:

```
Index: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
       5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
       9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
       13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
```

### Pose Feature Extraction

Extracts 8 emotion-relevant features from 17-keypoint body pose:

| Feature | Keypoints Used | Computation | Emotion Signal |
|---------|---------------|-------------|----------------|
| `slouch_score` | 5,6 (shoulders), 11,12 (hips) | Vertical distance ratio: `(hip_y - shoulder_y) / expected_upright` (Y-down coords: smaller ratio = more slouch) | Depression, low energy |
| `openness_score` | 9,10 (wrists), 5,6 (shoulders) | `wrist_spread / shoulder_width`, clamped 0-1 | Joy, confidence |
| `tension_score` | 5,6 (shoulders), 3,4 (ears) | `(ear_y - shoulder_y)` deviation from relaxed baseline | Anxiety, stress |
| `head_tilt` | 0 (nose), 5,6 (shoulders) | Nose position relative to shoulder midpoint, normalized | Engagement vs withdrawal |
| `gesture_speed` | All keypoints | Mean L2 displacement vs previous frame, requires pose buffer | Excitement vs stillness |
| `symmetry_score` | Left/right pairs (5/6, 7/8, 9/10, 11/12) | Mean absolute difference of mirrored pairs | Calm (symmetric) vs distress |
| `forward_lean` | 5,6 (shoulders), 11,12 (hips) | Horizontal offset of shoulder midpoint vs hip midpoint | Interest/engagement |
| `stillness_duration` | All keypoints | Consecutive frames where gesture_speed < threshold, requires pose buffer | Deep focus, freeze, low energy |

### Temporal Pose Buffer

`gesture_speed` and `stillness_duration` require temporal context:
- `PoseFeatureExtractor` maintains a rolling buffer of the last 10 frames (~0.5s at 20Hz)
- At startup (buffer empty), these two features default to 0.0
- In REST polling mode (single snapshot), these features default to 0.0
- In WebSocket streaming mode, buffer fills naturally from the stream

### PoseEncoder Architecture

`PoseEncoder(nn.Module)`:
- Input: 8 pose features
- Hidden: 32 units, ReLU
- Output: 16-dimensional embedding (matches BiometricEncoder output dim)
- Includes `zero_pose` registered buffer for missing pose data

### FusionHead Changes

`FusionHead.forward()` signature:
```python
def forward(self, text_embedding, biometric_embedding=None, pose_embedding=None) -> tuple:
```

`FusionHead.classify_with_fallback()` signature updated similarly:
```python
def classify_with_fallback(self, text_embedding, biometric_embedding=None, pose_embedding=None, threshold=0.6) -> dict:
```

- Current input: `text(384) + biometric(16) = 400`
- New input: `text(384) + biometric(16) + pose(16) = 416`
- When pose unavailable, `zero_pose` buffer used (same pattern as `zero_biometric`)
- Requires retraining the fusion head shared layers

### FusionHead Constructor Change

```python
class FusionHead(nn.Module):
    def __init__(self, text_dim=512, biometric_dim=16, pose_dim=16, ...):
        # pose_dim is a separate parameter, defaults to 16
        # shared layer input: text_dim + biometric_dim + pose_dim
```

### Checkpoint Format

Checkpoint `emotion_fusion_head.pt` state dict adds a new key:
```python
{
    'fusion_head': fusion_head.state_dict(),
    'biometric_encoder': biometric_encoder.state_dict(),
    'pose_encoder': pose_encoder.state_dict(),  # NEW
    'meta': {
        'text_dim': 384,
        'biometric_dim': 16,
        'pose_dim': 16,  # NEW — enables unambiguous dimension detection
        'num_emotions': 32,
        'num_families': 9,
    }
}
```

### Checkpoint Dimension Detection

`MultimodalEmotionAnalyzer.__init__()` uses `meta.pose_dim` if present, else infers:
```python
saved_pose_dim = state.get('meta', {}).get('pose_dim', 0)
saved_total = state['fusion_head']['shared.0.weight'].shape[1]
# If no meta: pose_dim = saved_total - saved_text_dim - 16 (biometric)
```
This catches old→new, new→old, and ambiguous cases.

### `get_pose_features()` Return Type and Ownership

`RuViewProvider` instantiates a `PoseFeatureExtractor` internally (owns the temporal pose buffer lifecycle). `get_pose_features()` delegates to it:

```python
# In RuViewProvider.__init__():
self._pose_extractor = PoseFeatureExtractor()

# get_pose_features() calls:
self._pose_extractor.extract(self._pose_data)
```

Returns a dict with named features:
```python
{
    'slouch_score': float,      # 0.0-1.0
    'openness_score': float,    # 0.0-1.0
    'tension_score': float,     # 0.0-1.0
    'head_tilt': float,         # -1.0 to 1.0
    'gesture_speed': float,     # 0.0+ normalized via tanh(speed / 50.0) to 0.0-1.0
    'symmetry_score': float,    # 0.0-1.0
    'forward_lean': float,      # -1.0 to 1.0
    'stillness_duration': float, # 0.0+ normalized via tanh(duration / 10.0) to 0.0-1.0
}
```

`PoseEncoder._extract_features()` converts this dict to a tensor. All values are pre-normalized by `PoseFeatureExtractor` — no additional normalization needed.

### File: `pose_encoder.py`

Contains:
- `PoseFeatureExtractor` — converts 17 COCO keypoints to 8 features, maintains pose buffer
- `PoseEncoder(nn.Module)` — neural encoder with `zero_pose` buffer

---

## Training Pipeline Changes

### New Training Data

- `generate_synthetic_pose(emotion)` — generates pose features per emotion using research-grounded posture-emotion correlations
- `WiFiCalibrationDataset` — bootstrapped pairs from existing biometric datasets (see Paired Data Construction)

### Training Robustness: Pose Dropout

During training, 20% of samples have pose features zeroed out. This teaches the model to produce valid classifications when pose data is missing at inference time, preventing the zero embedding from being out-of-distribution.

### Updated Training Flow

1. Train WiFi calibration model separately: `python wifi_calibration.py --train` → `checkpoints/ruview_calibration.pt`
2. Train emotion classifier with `python train_emotion_classifier.py` which now:
   - Generates training samples with text + biometrics + pose features per emotion
   - Applies 20% pose dropout during training
   - Saves updated FusionHead (416-dim) + PoseEncoder weights → `checkpoints/emotion_fusion_head.pt`
3. Online: `calibration_profiles/{user_id}.pt` created at runtime by PersonalCalibrationBuffer

### Checkpoint Compatibility

The updated `emotion_fusion_head.pt` has different input dimensions (416 vs 400). The dimension check in `MultimodalEmotionAnalyzer.__init__()` is updated to detect biometric+pose dim mismatches in both directions (old→new and new→old). On mismatch, logs a retraining message and recreates the fusion head with correct dimensions.

---

## API Changes

### `POST /api/ruview/analyze` (updated)

Request (backward-compatible — no new required fields):
```json
{
    "text": "I feel anxious",
    "biometrics": null
}
```

Internal flow:
1. Endpoint calls `context_provider.get_ruview_biometrics()` → calibrated `{hr, hrv, eda}`
2. Endpoint calls `context_provider.ruview.get_pose_features()` → `{slouch_score, ...}` or None
3. Calls `multimodal_analyzer.analyze(text, biometrics=bio, pose=pose_features)`
4. `MultimodalEmotionAnalyzer.analyze()` signature updated: `def analyze(self, text, biometrics=None, pose=None, ...)`
5. Inside `analyze()`: if `pose` is not None, `self.pose_encoder.encode(pose)` → pose embedding; else `zero_pose`

Response (new fields added, existing fields unchanged):
```json
{
    "dominant_emotion": "anxiety",
    "family": "Fear",
    "confidence": 0.82,
    "ruview_source": true,
    "ruview_confidence": 0.87,
    "ruview_insight": "...",
    "presence": {"detected": true, "occupancy": 1, "motion_level": 0.3},
    "pose_features": {"slouch_score": 0.6, "tension_score": 0.8, "...": "..."},
    "calibration_profile": "default",
    "status": "success"
}
```

Clients that don't read `pose_features` or `calibration_profile` are unaffected.

### JS Client (`RuViewClient`)

New method:
```javascript
async getPoseFeatures() → {slouch_score, openness_score, ...}
```

`analyzeWithWiFiBiometrics()` response now includes `pose_features` and `calibration_profile` fields.

---

## Files

### New Files
- `wifi_calibration.py` — calibration model, personal buffer, training entry point
- `pose_encoder.py` — pose feature extraction (with temporal buffer) and encoder
- `tests/test_wifi_calibration.py` — calibration model tests
- `tests/test_pose_encoder.py` — pose encoder tests

### Modified Files
- `emotion_classifier.py` — FusionHead accepts pose embedding (400→416), PoseEncoder integrated, dimension check updated, `classify_with_fallback()` signature updated
- `ruview_provider.py` — `get_biometrics()` uses calibration model, add `get_pose_features()` returning named dict
- `train_emotion_classifier.py` — synthetic pose generation, 20% pose dropout, updated training loop
- `emotion_api_server.py` — `/api/ruview/analyze` passes pose data, returns `pose_features` and `calibration_profile`
- `neural_ecosystem_connector.js` — `RuViewClient.getPoseFeatures()`, updated response handling

---

## Test Criteria

### Calibration Tests (`tests/test_wifi_calibration.py`)
- Calibration model output shapes correct (2 inputs → 2 outputs)
- Calibrated HRV within BiometricEncoder range (10-100 ms) for all valid breathing rates
- Calibrated EDA within BiometricEncoder range (0.5-20 µS) for all valid motion levels
- PersonalCalibrationBuffer triggers fine-tune after 20 pairs
- Fine-tuning does not block concurrent inference (lock test)
- Fallback to linear approximation when checkpoint missing
- User profile save/load round-trips correctly

### Pose Tests (`tests/test_pose_encoder.py`)
- PoseFeatureExtractor produces 8 features from 17 COCO keypoints
- All features within documented ranges
- Temporal features (gesture_speed, stillness_duration) default to 0.0 with empty buffer
- Temporal features update correctly after buffer fills
- PoseEncoder output shape is (1, 16)
- Zero pose embedding when keypoints are None
- FusionHead accepts 416-dim input and produces valid classification
- 20% pose dropout during training produces model robust to missing pose

---

## Error Handling

All new paths fail gracefully:
- Missing calibration checkpoint → linear approximation fallback (existing `_breathing_to_hrv` / `_motion_to_eda`)
- Missing pose data → zero embedding (no impact on classification, trained with 20% dropout)
- Missing personal calibration profile → base calibration model
- RuView offline → no biometric or pose contribution, text-only classification
- Malformed keypoint data → `get_pose_features()` returns None → zero embedding
- Thread safety: calibration fine-tuning uses lock, non-blocking (<100ms)

No regression if RuView is unavailable. Existing tests must continue to pass.

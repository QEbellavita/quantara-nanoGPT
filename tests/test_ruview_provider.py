"""
Tests for RuView WiFi Sensing Provider integration.
Tests data mapping, mood signals, caching, and graceful failure.
"""

import sys
import os
import time
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ruview_provider import RuViewProvider


@pytest.fixture
def provider():
    """Create a RuViewProvider without connecting."""
    return RuViewProvider(use_websocket=False)


class TestBreathingToHRV:
    """Test breathing rate → HRV approximation."""

    def test_slow_breathing_high_hrv(self, provider):
        # Slow breathing (meditation) → high HRV (relaxed)
        hrv = provider._breathing_to_hrv(8.0)
        assert hrv > 75.0

    def test_fast_breathing_low_hrv(self, provider):
        # Fast breathing (stress) → low HRV
        hrv = provider._breathing_to_hrv(28.0)
        assert hrv < 30.0

    def test_normal_breathing_moderate_hrv(self, provider):
        # Normal range
        hrv = provider._breathing_to_hrv(15.0)
        assert 40.0 < hrv < 70.0

    def test_zero_breathing_default(self, provider):
        hrv = provider._breathing_to_hrv(0)
        assert hrv == 50.0

    def test_clamped_above_max(self, provider):
        # Above max should clamp
        hrv = provider._breathing_to_hrv(50.0)
        assert hrv == provider.HRV_FROM_BREATHING['min_hrv']

    def test_clamped_below_min(self, provider):
        # Below min should clamp
        hrv = provider._breathing_to_hrv(3.0)
        assert hrv == provider.HRV_FROM_BREATHING['max_hrv']


class TestMotionToEDA:
    """Test motion level → EDA approximation."""

    def test_no_motion_low_eda(self, provider):
        eda = provider._motion_to_eda(0.0)
        assert eda == provider.EDA_FROM_MOTION['min_eda']

    def test_max_motion_high_eda(self, provider):
        eda = provider._motion_to_eda(1.0)
        assert eda == provider.EDA_FROM_MOTION['max_eda']

    def test_moderate_motion(self, provider):
        eda = provider._motion_to_eda(0.5)
        assert 3.0 < eda < 5.0


class TestGetBiometrics:
    """Test biometric output format for emotion classifier compatibility."""

    def test_returns_none_when_no_data(self, provider):
        assert provider.get_biometrics() is None

    def test_maps_vitals_to_quantara_format(self, provider):
        # Simulate buffered data
        provider._vitals_buffer.append({
            'heart_rate': 82.0,
            'breathing_rate': 16.0,
            'confidence': 0.85,
            'timestamp': time.time(),
        })
        provider._presence_data = {
            'detected': True,
            'occupancy': 1,
            'motion_level': 0.3,
        }

        bio = provider.get_biometrics()
        assert bio is not None
        assert 'heart_rate' in bio
        assert 'hrv' in bio
        assert 'eda' in bio
        assert bio['source'] == 'ruview_wifi'
        assert bio['heart_rate'] == 82.0

    def test_skips_low_confidence(self, provider):
        provider._vitals_buffer.append({
            'heart_rate': 82.0,
            'breathing_rate': 16.0,
            'confidence': 0.1,  # Too low
            'timestamp': time.time(),
        })
        provider._presence_data = {'detected': True, 'occupancy': 1, 'motion_level': 0}
        assert provider.get_biometrics() is None

    def test_hr_range_valid(self, provider):
        """Ensure mapped HR stays within BiometricEncoder's expected range."""
        provider._vitals_buffer.append({
            'heart_rate': 95.0,
            'breathing_rate': 20.0,
            'confidence': 0.9,
            'timestamp': time.time(),
        })
        provider._presence_data = {'detected': True, 'occupancy': 1, 'motion_level': 0.5}

        bio = provider.get_biometrics()
        assert 40.0 <= bio['heart_rate'] <= 180.0
        assert 10.0 <= bio['hrv'] <= 100.0
        assert 0.5 <= bio['eda'] <= 20.0


class TestMoodSignals:
    """Test mood signal derivation from RuView data."""

    def test_elevated_hr_signal(self, provider):
        provider._vitals_buffer.append({
            'heart_rate': 110.0,
            'breathing_rate': 14.0,
            'confidence': 0.9,
            'timestamp': time.time(),
        })
        provider._presence_data = {'detected': True, 'occupancy': 1, 'motion_level': 0.2}
        provider._update_mood_signals()
        assert 'elevated_heart_rate' in provider._mood_signals

    def test_rapid_breathing_signal(self, provider):
        provider._vitals_buffer.append({
            'heart_rate': 80.0,
            'breathing_rate': 25.0,
            'confidence': 0.9,
            'timestamp': time.time(),
        })
        provider._presence_data = {'detected': True, 'occupancy': 1, 'motion_level': 0.1}
        provider._update_mood_signals()
        assert 'rapid_breathing' in provider._mood_signals

    def test_alone_signal(self, provider):
        provider._presence_data = {'detected': False, 'occupancy': 0, 'motion_level': 0}
        provider._update_mood_signals()
        assert 'alone' in provider._mood_signals

    def test_crowded_signal(self, provider):
        provider._presence_data = {'detected': True, 'occupancy': 5, 'motion_level': 0.4}
        provider._update_mood_signals()
        assert 'crowded_environment' in provider._mood_signals


class TestWebSocketMessage:
    """Test WebSocket message processing."""

    def test_processes_vital_signs(self, provider):
        msg = '{"vital_signs": {"heart_rate": 75, "breathing_rate": 14, "confidence": 0.9}}'
        provider._on_ws_message(None, msg)
        assert len(provider._vitals_buffer) == 1
        assert provider._vitals_buffer[0]['heart_rate'] == 75

    def test_processes_presence(self, provider):
        msg = '{"presence": true, "occupancy": 2, "motion_level": 0.5}'
        provider._on_ws_message(None, msg)
        assert provider._presence_data['occupancy'] == 2

    def test_processes_pose(self, provider):
        msg = '{"pose": {"keypoints": [[0,0], [1,1]]}}'
        provider._on_ws_message(None, msg)
        assert provider._pose_data is not None

    def test_handles_malformed_json(self, provider):
        # Should not raise
        provider._on_ws_message(None, 'not json')
        assert provider._latest_data is None


class TestRESTFallback:
    """Test REST polling fallback."""

    @patch('ruview_provider.requests.get')
    def test_health_check_success(self, mock_get, provider):
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.raise_for_status = MagicMock()
        assert provider._check_rest_health() is True
        assert provider._connected is True

    @patch('ruview_provider.requests.get')
    def test_health_check_failure(self, mock_get, provider):
        mock_get.side_effect = Exception("Connection refused")
        assert provider._check_rest_health() is False
        assert provider._connected is False

    @patch('ruview_provider.requests.get')
    def test_poll_vitals_caches(self, mock_get, provider):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            'heart_rate': 72, 'breathing_rate': 15, 'confidence': 0.8, 'motion_level': 0.2
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        # First call
        result = provider._poll_vitals()
        assert result['heart_rate'] == 72

        # Second call should use cache
        result2 = provider._poll_vitals()
        assert result2['heart_rate'] == 72
        assert mock_get.call_count == 1  # Only one actual request


class TestInsight:
    """Test human-readable insight generation."""

    def test_insight_with_elevated_hr(self, provider):
        provider._vitals_buffer.append({
            'heart_rate': 105, 'breathing_rate': 14, 'confidence': 0.9,
            'timestamp': time.time(),
        })
        provider._presence_data = {'detected': True, 'occupancy': 1, 'motion_level': 0.2}
        provider._update_mood_signals()

        insight = provider.get_insight()
        assert insight is not None
        assert 'heart rate' in insight.lower()

    def test_insight_returns_none_without_data(self, provider):
        assert provider.get_insight() is None


class TestPoseFeatures:
    """Test pose feature extraction integration."""

    STANDING_KEYPOINTS = [
        [0.5, 0.1, 0.9], [0.48, 0.08, 0.9], [0.52, 0.08, 0.9],
        [0.45, 0.1, 0.9], [0.55, 0.1, 0.9],
        [0.4, 0.3, 0.9], [0.6, 0.3, 0.9],
        [0.38, 0.5, 0.9], [0.62, 0.5, 0.9],
        [0.38, 0.7, 0.9], [0.62, 0.7, 0.9],
        [0.45, 0.6, 0.9], [0.55, 0.6, 0.9],
        [0.45, 0.8, 0.9], [0.55, 0.8, 0.9],
        [0.45, 1.0, 0.9], [0.55, 1.0, 0.9],
    ]

    def test_get_pose_features_with_data(self, provider):
        provider._pose_data = {'keypoints': self.STANDING_KEYPOINTS}
        features = provider.get_pose_features()
        assert features is not None
        assert len(features) == 8

    def test_get_pose_features_returns_none_when_no_data(self, provider):
        provider._pose_data = None
        assert provider.get_pose_features() is None

    def test_get_pose_features_handles_raw_list(self, provider):
        provider._pose_data = self.STANDING_KEYPOINTS
        features = provider.get_pose_features()
        assert features is not None
        assert len(features) == 8


class TestCalibrationIntegration:
    """Test calibration model integration with RuViewProvider."""

    def test_uses_calibration_when_available(self, provider):
        """When calibration model is set, hrv/eda come from it, not linear."""
        mock_model = MagicMock()
        # Model returns tensor-like output
        import torch
        mock_model.return_value = torch.tensor([[65.0, 4.5]])
        provider._calibration_model = mock_model
        provider._calibration_buffer = None

        # Set up vitals data
        provider._vitals_buffer.append({
            'heart_rate': 80.0,
            'breathing_rate': 16.0,
            'confidence': 0.9,
            'timestamp': time.time(),
        })
        provider._presence_data = {
            'detected': True, 'occupancy': 1, 'motion_level': 0.3,
        }

        bio = provider.get_biometrics()
        assert bio is not None
        assert bio['hrv'] == 65.0
        assert bio['eda'] == 4.5
        mock_model.assert_called_once()

    def test_uses_calibration_buffer_when_available(self, provider):
        """When both model and buffer are set, buffer.calibrate() is used."""
        import torch
        mock_model = MagicMock()
        mock_buffer = MagicMock()
        mock_buffer.calibrate.return_value = torch.tensor([[70.0, 3.2]])
        provider._calibration_model = mock_model
        provider._calibration_buffer = mock_buffer

        provider._vitals_buffer.append({
            'heart_rate': 80.0,
            'breathing_rate': 16.0,
            'confidence': 0.9,
            'timestamp': time.time(),
        })
        provider._presence_data = {
            'detected': True, 'occupancy': 1, 'motion_level': 0.3,
        }

        bio = provider.get_biometrics()
        assert bio is not None
        assert abs(bio['hrv'] - 70.0) < 0.01
        assert abs(bio['eda'] - 3.2) < 0.01
        mock_buffer.calibrate.assert_called_once()
        mock_model.assert_not_called()

    def test_falls_back_to_linear_when_no_calibration(self, provider):
        """When calibration_model is None, linear fallback is used."""
        provider._calibration_model = None
        provider._calibration_buffer = None

        provider._vitals_buffer.append({
            'heart_rate': 80.0,
            'breathing_rate': 16.0,
            'confidence': 0.9,
            'timestamp': time.time(),
        })
        provider._presence_data = {
            'detected': True, 'occupancy': 1, 'motion_level': 0.3,
        }

        bio = provider.get_biometrics()
        assert bio is not None
        # Verify these match the linear methods
        expected_hrv = provider._breathing_to_hrv(16.0)
        expected_eda = provider._motion_to_eda(0.3)
        assert bio['hrv'] == expected_hrv
        assert bio['eda'] == expected_eda

    def test_load_calibration_returns_none_when_no_checkpoint(self):
        """load_calibration_model returns (None, None) for nonexistent path."""
        from wifi_calibration import load_calibration_model
        model, buf = load_calibration_model('/nonexistent/path.pt')
        assert model is None
        assert buf is None

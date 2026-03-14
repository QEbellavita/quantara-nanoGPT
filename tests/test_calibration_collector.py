"""Tests for CalibrationCollector."""

import sys
import os
import json
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration_collector import CalibrationCollector, retrain_from_collected_data


class TestCalibrationCollector:
    def test_init_defaults(self):
        c = CalibrationCollector()
        assert c.ruview_url == 'http://localhost:8080'
        assert c.poll_interval == 5
        assert c.pairs == []

    @patch('calibration_collector.requests.get')
    def test_get_ruview_reading_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            'status': 'success',
            'heart_rate': 75.0,
            'breathing_rate': 14.0,
            'motion_level': 0.3,
            'confidence': 0.85,
        }
        mock_get.return_value = mock_resp

        c = CalibrationCollector()
        reading = c._get_ruview_reading()
        assert reading is not None
        assert reading['heart_rate'] == 75.0

    @patch('calibration_collector.requests.get')
    def test_get_ruview_reading_failure(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        c = CalibrationCollector()
        reading = c._get_ruview_reading()
        assert reading is None

    @patch('calibration_collector.requests.get')
    def test_get_healthkit_reading_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            'heart_rate': 72.0,
            'hrv': 55.0,
            'respiratory_rate': 15.0,
            'source': 'apple_watch',
        }
        mock_get.return_value = mock_resp

        c = CalibrationCollector()
        reading = c._get_healthkit_reading()
        assert reading is not None
        assert reading['hrv'] == 55.0

    def test_collect_one_pair_both_available(self):
        c = CalibrationCollector()
        c._get_ruview_reading = MagicMock(return_value={
            'heart_rate': 78, 'breathing_rate': 16, 'motion_level': 0.2, 'confidence': 0.9
        })
        c._get_healthkit_reading = MagicMock(return_value={
            'heart_rate': 76, 'hrv': 52, 'eda': 2.5, 'source': 'watch'
        })

        pair = c._collect_one_pair()
        assert pair is not None
        assert pair['wifi']['heart_rate'] == 78
        assert pair['wearable']['hrv'] == 52

    def test_collect_one_pair_wifi_missing(self):
        c = CalibrationCollector()
        c._get_ruview_reading = MagicMock(return_value=None)
        c._get_healthkit_reading = MagicMock(return_value={
            'heart_rate': 76, 'hrv': 52
        })
        assert c._collect_one_pair() is None

    def test_save_and_load_session(self, tmp_path):
        c = CalibrationCollector()
        c._session_start = __import__('datetime').datetime.now()
        c.pairs = [
            {
                'timestamp': '2026-03-15T10:00:00',
                'wifi': {'heart_rate': 78, 'breathing_rate': 16, 'motion_level': 0.2},
                'wearable': {'heart_rate': 76, 'hrv': 52, 'eda': 2.5},
            }
        ]

        output = str(tmp_path / 'test_session.json')
        c._save_session(output)

        with open(output) as f:
            data = json.load(f)
        assert data['pair_count'] == 1
        assert len(data['pairs']) == 1
        assert data['pairs'][0]['wifi']['heart_rate'] == 78


class TestRetrainFromData:
    def test_retrain_from_session_files(self, tmp_path):
        # Create a fake session file with enough pairs
        pairs = []
        for i in range(30):
            pairs.append({
                'timestamp': f'2026-03-15T10:{i:02d}:00',
                'wifi': {'heart_rate': 70 + i, 'breathing_rate': 12 + i * 0.5, 'motion_level': 0.1 + i * 0.02},
                'wearable': {'heart_rate': 70 + i, 'hrv': 60 - i, 'eda': 1.5 + i * 0.1},
            })

        session_file = tmp_path / 'session.json'
        with open(session_file, 'w') as f:
            json.dump({'pairs': pairs}, f)

        output = str(tmp_path / 'calibration.pt')
        retrain_from_collected_data(str(tmp_path), output=output)
        assert os.path.exists(output)

    def test_retrain_empty_dir(self, tmp_path):
        # Should not crash
        retrain_from_collected_data(str(tmp_path))

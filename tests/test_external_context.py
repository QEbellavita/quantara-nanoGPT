"""Tests for external context providers."""
import os
import time
import pytest
import requests
from unittest.mock import patch, MagicMock
from external_context import WeatherProvider


class TestWeatherProvider:
    def setup_method(self):
        self.provider = WeatherProvider()

    @patch('external_context.requests.get')
    def test_get_weather_success(self, mock_get):
        """Successful weather fetch returns structured data with air quality."""
        weather_resp = MagicMock()
        weather_resp.status_code = 200
        weather_resp.json.return_value = {
            'current': {
                'temperature_2m': 15.2,
                'relative_humidity_2m': 72,
                'weather_code': 3,
                'uv_index': 2.1,
            }
        }
        aq_resp = MagicMock()
        aq_resp.status_code = 200
        aq_resp.json.return_value = {
            'current': {'pm2_5': 12.3, 'pm10': 18.7}
        }
        mock_get.side_effect = [weather_resp, aq_resp]

        result = self.provider.get_weather(37.77, -122.42)

        assert result is not None
        assert result['temperature_c'] == 15.2
        assert result['humidity'] == 72
        assert result['weather_code'] == 3
        assert result['air_quality'] == {'pm2_5': 12.3, 'pm10': 18.7}
        assert 'mood_signals' in result

    @patch('external_context.requests.get')
    def test_get_weather_timeout(self, mock_get):
        """Timeout returns None, not an exception."""
        mock_get.side_effect = requests.exceptions.Timeout()

        result = self.provider.get_weather(37.77, -122.42)
        assert result is None

    @patch('external_context.requests.get')
    def test_get_weather_http_error(self, mock_get):
        """HTTP error returns None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = Exception("Server error")
        mock_get.return_value = mock_resp

        result = self.provider.get_weather(37.77, -122.42)
        assert result is None

    def test_mood_signals_low_sunlight(self):
        """Weather code >= 50 triggers low_sunlight signal."""
        signals = self.provider._derive_mood_signals(
            weather_code=61, humidity=50, temp_c=20, uv_index=3, pm2_5=10
        )
        assert 'low_sunlight' in signals

    def test_mood_signals_high_humidity(self):
        """Humidity > 80% triggers high_humidity signal."""
        signals = self.provider._derive_mood_signals(
            weather_code=0, humidity=85, temp_c=20, uv_index=3, pm2_5=10
        )
        assert 'high_humidity' in signals

    def test_mood_signals_extreme_temp_cold(self):
        """Temperature < 0 triggers extreme_temp signal."""
        signals = self.provider._derive_mood_signals(
            weather_code=0, humidity=50, temp_c=-5, uv_index=1, pm2_5=10
        )
        assert 'extreme_temp' in signals

    def test_mood_signals_extreme_temp_hot(self):
        """Temperature > 35 triggers extreme_temp signal."""
        signals = self.provider._derive_mood_signals(
            weather_code=0, humidity=50, temp_c=38, uv_index=8, pm2_5=10
        )
        assert 'extreme_temp' in signals

    def test_mood_signals_poor_air_quality(self):
        """PM2.5 > 35 triggers poor_air_quality signal."""
        signals = self.provider._derive_mood_signals(
            weather_code=0, humidity=50, temp_c=20, uv_index=3, pm2_5=40
        )
        assert 'poor_air_quality' in signals

    def test_mood_signals_high_uv(self):
        """UV > 7 triggers high_uv signal."""
        signals = self.provider._derive_mood_signals(
            weather_code=0, humidity=50, temp_c=20, uv_index=9, pm2_5=10
        )
        assert 'high_uv' in signals

    def test_mood_signals_none_when_normal(self):
        """Normal conditions produce no mood signals."""
        signals = self.provider._derive_mood_signals(
            weather_code=0, humidity=50, temp_c=20, uv_index=3, pm2_5=10
        )
        assert signals == []

    @patch('external_context.requests.get')
    def test_cache_hit(self, mock_get):
        """Second call with same coords uses cache, no additional HTTP requests."""
        weather_resp = MagicMock()
        weather_resp.status_code = 200
        weather_resp.json.return_value = {
            'current': {
                'temperature_2m': 15.2,
                'relative_humidity_2m': 72,
                'weather_code': 3,
                'uv_index': 2.1,
            }
        }
        aq_resp = MagicMock()
        aq_resp.status_code = 200
        aq_resp.json.return_value = {'current': {'pm2_5': 10, 'pm10': 15}}
        mock_get.side_effect = [weather_resp, aq_resp]

        self.provider.get_weather(37.77, -122.42)
        self.provider.get_weather(37.77, -122.42)

        # First call makes 2 HTTP requests (weather + air quality), second uses cache
        assert mock_get.call_count == 2

    @patch('external_context.requests.get')
    def test_cache_eviction_at_max(self, mock_get):
        """Cache evicts oldest entry when max size reached."""
        weather_resp = MagicMock()
        weather_resp.status_code = 200
        weather_resp.json.return_value = {
            'current': {
                'temperature_2m': 15.0, 'relative_humidity_2m': 50,
                'weather_code': 0, 'uv_index': 3,
            }
        }
        aq_resp = MagicMock()
        aq_resp.status_code = 200
        aq_resp.json.return_value = {'current': {'pm2_5': 10, 'pm10': 15}}
        # Each get_weather call makes 2 HTTP requests, need enough for 4 calls
        mock_get.side_effect = [weather_resp, aq_resp] * 4

        self.provider._max_cache_size = 3  # Override for test
        for i in range(4):
            self.provider.get_weather(float(i), 0.0)

        assert len(self.provider._cache) <= 3

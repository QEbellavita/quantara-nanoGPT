"""Tests for external context providers."""
import os
import time
import pytest
import requests
from unittest.mock import patch, MagicMock
from external_context import WeatherProvider
from external_context import NutritionProvider


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


class TestNutritionProvider:
    def setup_method(self):
        self.provider = NutritionProvider()

    @patch.dict(os.environ, {
        'NUTRITIONIX_APP_ID': 'test_id',
        'NUTRITIONIX_API_KEY': 'test_key',
    })
    @patch('external_context.requests.post')
    def test_get_nutrition_success(self, mock_post):
        """Successful nutrition fetch returns structured data."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            'foods': [
                {
                    'food_name': 'coffee',
                    'serving_qty': 3,
                    'nf_calories': 15,
                    'nf_protein': 0.9,
                    'nf_total_carbohydrate': 0,
                    'nf_total_fat': 0,
                    'nf_sugars': 0,
                    'nf_caffeine': 285,
                },
            ]
        }
        mock_post.return_value = mock_resp

        provider = NutritionProvider()
        result = provider.get_nutrition(['3 cups of coffee'])

        assert result is not None
        assert result['total_calories'] == 15
        assert result['caffeine_mg'] == 285
        assert 'items' in result
        assert 'mood_signals' in result

    def test_get_nutrition_no_keys(self):
        """Missing env keys returns None with no crash."""
        with patch.dict(os.environ, {}, clear=True):
            provider = NutritionProvider()
            result = provider.get_nutrition(['a donut'])
            assert result is None

    @patch.dict(os.environ, {
        'NUTRITIONIX_APP_ID': 'test_id',
        'NUTRITIONIX_API_KEY': 'test_key',
    })
    @patch('external_context.requests.post')
    def test_get_nutrition_timeout(self, mock_post):
        """Timeout returns None."""
        mock_post.side_effect = requests.exceptions.Timeout()
        provider = NutritionProvider()
        result = provider.get_nutrition(['a donut'])
        assert result is None

    @patch.dict(os.environ, {
        'NUTRITIONIX_APP_ID': 'test_id',
        'NUTRITIONIX_API_KEY': 'test_key',
    })
    @patch('external_context.requests.post')
    def test_get_nutrition_rate_limited(self, mock_post):
        """HTTP 429 returns None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_resp)
        mock_post.return_value = mock_resp
        provider = NutritionProvider()
        result = provider.get_nutrition(['a donut'])
        assert result is None

    def test_mood_signals_high_caffeine(self):
        """Caffeine > 300mg triggers high_caffeine signal."""
        signals = self.provider._derive_mood_signals(
            calories=500, protein_g=20, sugar_g=10, caffeine_mg=350
        )
        assert 'high_caffeine' in signals

    def test_mood_signals_low_protein(self):
        """Protein < 15g triggers low_protein signal."""
        signals = self.provider._derive_mood_signals(
            calories=500, protein_g=10, sugar_g=10, caffeine_mg=0
        )
        assert 'low_protein' in signals

    def test_mood_signals_sugar_crash(self):
        """Sugar > 50g triggers sugar_crash_risk signal."""
        signals = self.provider._derive_mood_signals(
            calories=500, protein_g=20, sugar_g=60, caffeine_mg=0
        )
        assert 'sugar_crash_risk' in signals

    def test_mood_signals_undereating(self):
        """Calories < 500 triggers undereating signal."""
        signals = self.provider._derive_mood_signals(
            calories=300, protein_g=20, sugar_g=10, caffeine_mg=0
        )
        assert 'undereating' in signals

    def test_mood_signals_post_meal_fatigue(self):
        """Calories > 1000 triggers post_meal_fatigue signal."""
        signals = self.provider._derive_mood_signals(
            calories=1200, protein_g=40, sugar_g=30, caffeine_mg=0
        )
        assert 'post_meal_fatigue' in signals

    def test_mood_signals_none_when_normal(self):
        """Normal nutrition produces no mood signals."""
        signals = self.provider._derive_mood_signals(
            calories=600, protein_g=25, sugar_g=20, caffeine_mg=100
        )
        assert signals == []

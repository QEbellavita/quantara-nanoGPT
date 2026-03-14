# External Context Enrichment Layer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Open-Meteo (weather), Nutritionix (nutrition), and NLP Cloud (sentiment cross-validation) APIs into the Quantara emotion coaching pipeline with standalone endpoints and optional coaching enrichment.

**Architecture:** New `external_context.py` module with three providers (WeatherProvider, NutritionProvider, SentimentValidator) orchestrated by ExternalContextProvider. Providers are wired into `emotion_api_server.py` as 3 standalone endpoints and optional enrichment in the coaching endpoint. All external calls are non-fatal with 5-second timeouts.

**Tech Stack:** Python, Flask, `requests` library, Open-Meteo API (free), Nutritionix API (env key), NLP Cloud API (env key)

**Spec:** `docs/superpowers/specs/2026-03-14-external-context-enrichment-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `external_context.py` | Create | WeatherProvider, NutritionProvider, SentimentValidator, ExternalContextProvider orchestrator |
| `tests/test_external_context.py` | Create | Unit tests for all providers, cache, mood signals, enrichment, graceful degradation |
| `emotion_api_server.py` | Modify | Add 3 `/api/context/*` endpoints, wire `context` param into coach endpoint |
| `quantara_integration.py` | Modify | Add `context` param to `get_coaching_response()` for parity |

---

## Chunk 1: WeatherProvider

### Task 1: WeatherProvider — tests and implementation

**Files:**
- Create: `tests/test_external_context.py`
- Create: `external_context.py`

- [ ] **Step 1: Write failing tests for WeatherProvider**

Create `tests/test_external_context.py`:

```python
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
        import requests
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_external_context.py::TestWeatherProvider -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'external_context'`

- [ ] **Step 3: Implement WeatherProvider**

Create `external_context.py`:

```python
"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - External Context Enrichment Layer
===============================================================================
Integrates external APIs (weather, nutrition, sentiment) as context signals
for emotion-aware coaching and standalone data access.

Integrates with:
- Neural Workflow AI Engine
- AI Conversational Coach
- Biometric Integration Engine
- Dashboard Data Integration
- Real-time Data

Providers:
- WeatherProvider: Open-Meteo (free, no auth)
- NutritionProvider: Nutritionix (env key auth)
- SentimentValidator: NLP Cloud (env key auth)
===============================================================================
"""

import os
import time
import logging

import requests

logger = logging.getLogger(__name__)

# WMO Weather Code descriptions (subset)
WMO_CODES = {
    0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
    45: 'Fog', 48: 'Depositing rime fog',
    51: 'Light drizzle', 53: 'Moderate drizzle', 55: 'Dense drizzle',
    61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
    71: 'Slight snow', 73: 'Moderate snow', 75: 'Heavy snow',
    80: 'Slight rain showers', 81: 'Moderate rain showers', 82: 'Violent rain showers',
    95: 'Thunderstorm', 96: 'Thunderstorm with slight hail', 99: 'Thunderstorm with heavy hail',
}

API_TIMEOUT = 5  # seconds


class WeatherProvider:
    """
    Fetches weather and air quality data from Open-Meteo.
    Free, no auth required. Results cached for 30 minutes.

    Connected to:
    - Neural Workflow AI Engine
    - Dashboard Data Integration
    - Real-time Data
    """

    FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'
    AIR_QUALITY_URL = 'https://air-quality-api.open-meteo.com/v1/air-quality'
    CACHE_TTL = 1800  # 30 minutes

    def __init__(self):
        self._cache = {}
        self._max_cache_size = 100

    def _cache_key(self, lat: float, lon: float) -> str:
        return f"{lat:.2f},{lon:.2f}"

    def _evict_if_needed(self):
        """Remove oldest-accessed entry if cache exceeds max size."""
        if len(self._cache) >= self._max_cache_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k]['accessed_at'])
            del self._cache[oldest_key]

    def _derive_mood_signals(
        self, weather_code: int, humidity: float, temp_c: float,
        uv_index: float, pm2_5: float
    ) -> list:
        """Derive mood-relevant signals from weather data."""
        signals = []
        if weather_code >= 50:
            signals.append('low_sunlight')
        if humidity > 80:
            signals.append('high_humidity')
        if temp_c < 0 or temp_c > 35:
            signals.append('extreme_temp')
        if pm2_5 > 35:
            signals.append('poor_air_quality')
        if uv_index > 7:
            signals.append('high_uv')
        return signals

    def get_weather(self, lat: float, lon: float) -> dict | None:
        """
        Fetch current weather and air quality for a location.
        Returns None on any failure.
        """
        key = self._cache_key(lat, lon)

        # Check cache
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['fetched_at'] < self.CACHE_TTL:
                entry['accessed_at'] = time.time()
                return entry['data']
            else:
                del self._cache[key]

        try:
            # Fetch weather
            weather_resp = requests.get(self.FORECAST_URL, params={
                'latitude': lat, 'longitude': lon,
                'current': 'temperature_2m,relative_humidity_2m,weather_code,uv_index',
            }, timeout=API_TIMEOUT)
            weather_resp.raise_for_status()
            weather = weather_resp.json()['current']

            # Fetch air quality
            pm2_5, pm10 = 0.0, 0.0
            try:
                aq_resp = requests.get(self.AIR_QUALITY_URL, params={
                    'latitude': lat, 'longitude': lon,
                    'current': 'pm2_5,pm10',
                }, timeout=API_TIMEOUT)
                aq_resp.raise_for_status()
                aq = aq_resp.json().get('current', {})
                pm2_5 = aq.get('pm2_5', 0.0)
                pm10 = aq.get('pm10', 0.0)
            except Exception as e:
                logger.warning(f"[WeatherProvider] Air quality fetch failed: {e}")

            temp_c = weather.get('temperature_2m', 0)
            humidity = weather.get('relative_humidity_2m', 0)
            weather_code = weather.get('weather_code', 0)
            uv_index = weather.get('uv_index', 0)

            mood_signals = self._derive_mood_signals(
                weather_code, humidity, temp_c, uv_index, pm2_5
            )

            result = {
                'temperature_c': temp_c,
                'humidity': humidity,
                'weather_code': weather_code,
                'weather_description': WMO_CODES.get(weather_code, 'Unknown'),
                'uv_index': uv_index,
                'air_quality': {'pm2_5': pm2_5, 'pm10': pm10},
                'mood_signals': mood_signals,
            }

            # Cache result
            self._evict_if_needed()
            now = time.time()
            self._cache[key] = {'data': result, 'fetched_at': now, 'accessed_at': now}

            return result

        except Exception as e:
            logger.warning(f"[WeatherProvider] Weather fetch failed: {e}")
            return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_external_context.py::TestWeatherProvider -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add external_context.py tests/test_external_context.py
git commit -m "feat: add WeatherProvider with Open-Meteo integration, caching, and mood signals"
```

---

## Chunk 2: NutritionProvider

### Task 2: NutritionProvider — tests and implementation

**Files:**
- Modify: `tests/test_external_context.py`
- Modify: `external_context.py`

- [ ] **Step 1: Write failing tests for NutritionProvider**

Append to `tests/test_external_context.py`:

```python
from external_context import NutritionProvider


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_external_context.py::TestNutritionProvider -v`
Expected: FAIL with `ImportError: cannot import name 'NutritionProvider'`

- [ ] **Step 3: Implement NutritionProvider**

Append to `external_context.py`:

```python
class NutritionProvider:
    """
    Analyzes food logs via Nutritionix Natural Language API.
    Auth: NUTRITIONIX_APP_ID and NUTRITIONIX_API_KEY env vars.

    Connected to:
    - AI Conversational Coach
    - Biometric Integration Engine
    - Dashboard Data Integration
    """

    API_URL = 'https://trackapi.nutritionix.com/v2/natural/nutrients'

    def _derive_mood_signals(
        self, calories: float, protein_g: float, sugar_g: float, caffeine_mg: float
    ) -> list:
        """Derive mood-relevant signals from nutrition data."""
        signals = []
        if caffeine_mg > 300:
            signals.append('high_caffeine')
        if protein_g < 15:
            signals.append('low_protein')
        if sugar_g > 50:
            signals.append('sugar_crash_risk')
        if calories < 500:
            signals.append('undereating')
        if calories > 1000:
            signals.append('post_meal_fatigue')
        return signals

    def get_nutrition(self, food_log: list[str]) -> dict | None:
        """
        Analyze a food log using Nutritionix natural language API.
        Returns None if keys are missing or API fails.
        """
        app_id = os.environ.get('NUTRITIONIX_APP_ID')
        api_key = os.environ.get('NUTRITIONIX_API_KEY')

        if not app_id or not api_key:
            logger.warning("[NutritionProvider] Missing NUTRITIONIX_APP_ID or NUTRITIONIX_API_KEY")
            return None

        query = '. '.join(food_log)

        try:
            resp = requests.post(self.API_URL, json={
                'query': query,
            }, headers={
                'x-app-id': app_id,
                'x-app-key': api_key,
                'Content-Type': 'application/json',
            }, timeout=API_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            foods = data.get('foods', [])
            total_calories = sum(f.get('nf_calories', 0) for f in foods)
            protein_g = sum(f.get('nf_protein', 0) for f in foods)
            carbs_g = sum(f.get('nf_total_carbohydrate', 0) for f in foods)
            fat_g = sum(f.get('nf_total_fat', 0) for f in foods)
            sugar_g = sum(f.get('nf_sugars', 0) for f in foods)
            caffeine_mg = sum(f.get('nf_caffeine', 0) or 0 for f in foods)

            items = [
                {
                    'name': f.get('food_name', 'Unknown'),
                    'qty': f.get('serving_qty', 1),
                    'calories': f.get('nf_calories', 0),
                    'caffeine_mg': f.get('nf_caffeine', 0) or 0,
                }
                for f in foods
            ]

            mood_signals = self._derive_mood_signals(
                total_calories, protein_g, sugar_g, caffeine_mg
            )

            return {
                'total_calories': total_calories,
                'protein_g': protein_g,
                'carbs_g': carbs_g,
                'fat_g': fat_g,
                'caffeine_mg': caffeine_mg,
                'sugar_g': sugar_g,
                'items': items,
                'mood_signals': mood_signals,
            }

        except Exception as e:
            logger.warning(f"[NutritionProvider] API call failed: {e}")
            return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_external_context.py::TestNutritionProvider -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add external_context.py tests/test_external_context.py
git commit -m "feat: add NutritionProvider with Nutritionix API integration and mood signals"
```

---

## Chunk 3: SentimentValidator

### Task 3: SentimentValidator — tests and implementation

**Files:**
- Modify: `tests/test_external_context.py`
- Modify: `external_context.py`

- [ ] **Step 1: Write failing tests for SentimentValidator**

Append to `tests/test_external_context.py`:

```python
from external_context import SentimentValidator


class TestSentimentValidator:
    def setup_method(self):
        self.validator = SentimentValidator()

    @patch.dict(os.environ, {'NLPCLOUD_API_KEY': 'test_key'})
    @patch('external_context.requests.post')
    def test_validate_sentiment_success(self, mock_post):
        """Successful sentiment validation returns scores."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            'scored_labels': [
                {'label': 'POSITIVE', 'score': 0.12},
                {'label': 'NEGATIVE', 'score': 0.88},
            ]
        }
        mock_post.return_value = mock_resp

        validator = SentimentValidator()
        result = validator.validate_sentiment("I feel awful")

        assert result is not None
        assert result['nlpcloud_sentiment'] == 'NEGATIVE'
        assert result['negative_score'] == 0.88
        assert result['positive_score'] == 0.12

    def test_validate_sentiment_no_key(self):
        """Missing API key returns None."""
        with patch.dict(os.environ, {}, clear=True):
            validator = SentimentValidator()
            result = validator.validate_sentiment("test")
            assert result is None

    @patch.dict(os.environ, {'NLPCLOUD_API_KEY': 'test_key'})
    @patch('external_context.requests.post')
    def test_validate_sentiment_timeout(self, mock_post):
        """Timeout returns None."""
        mock_post.side_effect = requests.exceptions.Timeout()
        validator = SentimentValidator()
        result = validator.validate_sentiment("test")
        assert result is None

    def test_agrees_with_local_positive(self):
        """POSITIVE sentiment agrees with Joy family."""
        agrees = self.validator._check_agreement('POSITIVE', 'Joy')
        assert agrees is True

    def test_agrees_with_local_negative(self):
        """NEGATIVE sentiment agrees with Sadness family."""
        agrees = self.validator._check_agreement('NEGATIVE', 'Sadness')
        assert agrees is True

    def test_disagrees_positive_vs_sadness(self):
        """POSITIVE sentiment disagrees with Sadness family."""
        agrees = self.validator._check_agreement('POSITIVE', 'Sadness')
        assert agrees is False

    def test_neutral_always_agrees(self):
        """Neutral family always agrees."""
        agrees = self.validator._check_agreement('POSITIVE', 'Neutral')
        assert agrees is True
        agrees = self.validator._check_agreement('NEGATIVE', 'Neutral')
        assert agrees is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_external_context.py::TestSentimentValidator -v`
Expected: FAIL with `ImportError: cannot import name 'SentimentValidator'`

- [ ] **Step 3: Implement SentimentValidator**

Append to `external_context.py`:

```python
class SentimentValidator:
    """
    Cross-validates emotion classification using NLP Cloud sentiment analysis.
    Informational only — does not override the local 32-emotion classifier.
    Auth: NLPCLOUD_API_KEY env var.

    Connected to:
    - AI Conversational Coach (cross-validation field)
    - Dashboard Data Integration
    """

    API_URL = 'https://api.nlpcloud.io/v1/distilbert-base-uncased-finetuned-sst-2-english/sentiment'

    POSITIVE_FAMILIES = {'Joy', 'Love', 'Calm', 'Surprise'}
    NEGATIVE_FAMILIES = {'Sadness', 'Anger', 'Fear', 'Self-Conscious'}

    def _check_agreement(self, nlp_sentiment: str, local_family: str) -> bool:
        """Check if NLP Cloud sentiment direction agrees with local family valence."""
        if local_family == 'Neutral':
            return True
        if nlp_sentiment == 'POSITIVE':
            return local_family in self.POSITIVE_FAMILIES
        if nlp_sentiment == 'NEGATIVE':
            return local_family in self.NEGATIVE_FAMILIES
        return True

    def validate_sentiment(self, text: str, local_family: str = None) -> dict | None:
        """
        Get sentiment scores from NLP Cloud.
        Returns None if key is missing or API fails.
        """
        api_key = os.environ.get('NLPCLOUD_API_KEY')

        if not api_key:
            logger.warning("[SentimentValidator] Missing NLPCLOUD_API_KEY")
            return None

        try:
            resp = requests.post(self.API_URL, json={
                'text': text,
            }, headers={
                'Authorization': f'Token {api_key}',
                'Content-Type': 'application/json',
            }, timeout=API_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            scored_labels = data.get('scored_labels', [])
            scores = {item['label']: item['score'] for item in scored_labels}

            positive_score = scores.get('POSITIVE', 0.0)
            negative_score = scores.get('NEGATIVE', 0.0)
            sentiment = 'POSITIVE' if positive_score >= negative_score else 'NEGATIVE'

            result = {
                'nlpcloud_sentiment': sentiment,
                'positive_score': positive_score,
                'negative_score': negative_score,
            }

            if local_family:
                result['agrees_with_local'] = self._check_agreement(sentiment, local_family)

            return result

        except Exception as e:
            logger.warning(f"[SentimentValidator] API call failed: {e}")
            return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_external_context.py::TestSentimentValidator -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add external_context.py tests/test_external_context.py
git commit -m "feat: add SentimentValidator with NLP Cloud cross-validation"
```

---

## Chunk 4: ExternalContextProvider orchestrator and coaching enrichment

### Task 4: ExternalContextProvider — tests and implementation

**Files:**
- Modify: `tests/test_external_context.py`
- Modify: `external_context.py`

- [ ] **Step 1: Write failing tests for ExternalContextProvider**

Append to `tests/test_external_context.py`:

```python
from external_context import ExternalContextProvider


class TestExternalContextProvider:
    def setup_method(self):
        self.provider = ExternalContextProvider()

    @patch.object(WeatherProvider, 'get_weather')
    def test_enrich_coaching_weather_only(self, mock_weather):
        """Context with only location returns weather insight."""
        mock_weather.return_value = {
            'temperature_c': 5.0, 'humidity': 60, 'weather_code': 61,
            'weather_description': 'Slight rain', 'uv_index': 1,
            'air_quality': {'pm2_5': 10, 'pm10': 15},
            'mood_signals': ['low_sunlight'],
        }

        context = {'location': [37.77, -122.42]}
        result = self.provider.enrich_coaching(context, "I feel sad")

        assert result['weather_insight'] is not None
        assert result['weather_data'] is not None
        assert result['nutrition_insight'] is None
        assert result['cross_validation'] is None

    @patch.object(NutritionProvider, 'get_nutrition')
    def test_enrich_coaching_nutrition_only(self, mock_nutrition):
        """Context with only food_log returns nutrition insight."""
        mock_nutrition.return_value = {
            'total_calories': 15, 'protein_g': 0.9, 'carbs_g': 0,
            'fat_g': 0, 'caffeine_mg': 350, 'sugar_g': 0,
            'items': [{'name': 'coffee', 'qty': 3, 'calories': 15, 'caffeine_mg': 350}],
            'mood_signals': ['high_caffeine'],
        }

        context = {'food_log': ['3 cups of coffee']}
        result = self.provider.enrich_coaching(context, "I feel anxious")

        assert result['nutrition_insight'] is not None
        assert result['nutrition_data'] is not None
        assert result['weather_insight'] is None

    @patch.object(SentimentValidator, 'validate_sentiment')
    def test_enrich_coaching_sentiment_only(self, mock_sentiment):
        """Context with validate_sentiment returns cross_validation."""
        mock_sentiment.return_value = {
            'nlpcloud_sentiment': 'NEGATIVE',
            'positive_score': 0.1, 'negative_score': 0.9,
            'agrees_with_local': True,
        }

        context = {'validate_sentiment': True}
        result = self.provider.enrich_coaching(context, "I feel terrible", local_family='Sadness')

        assert result['cross_validation'] is not None
        assert result['cross_validation']['agrees_with_local'] is True
        assert result['weather_insight'] is None
        assert result['nutrition_insight'] is None

    def test_enrich_coaching_empty_context(self):
        """Empty context dict returns all None."""
        result = self.provider.enrich_coaching({}, "hello")

        assert result['weather_insight'] is None
        assert result['nutrition_insight'] is None
        assert result['cross_validation'] is None

    @patch.object(WeatherProvider, 'get_weather')
    def test_enrich_coaching_weather_failure_graceful(self, mock_weather):
        """Weather failure still returns structured result with Nones."""
        mock_weather.return_value = None

        context = {'location': [37.77, -122.42]}
        result = self.provider.enrich_coaching(context, "I feel sad")

        assert result['weather_insight'] is None
        assert result['weather_data'] is None

    def test_weather_insight_text_low_sunlight(self):
        """Low sunlight produces appropriate insight text."""
        insight = self.provider._weather_insight({
            'weather_description': 'Slight rain',
            'temperature_c': 10,
            'mood_signals': ['low_sunlight'],
        })
        assert 'sunlight' in insight.lower() or 'overcast' in insight.lower() or 'rain' in insight.lower()

    def test_nutrition_insight_text_high_caffeine(self):
        """High caffeine produces appropriate insight text."""
        insight = self.provider._nutrition_insight({
            'caffeine_mg': 350,
            'mood_signals': ['high_caffeine'],
        })
        assert 'caffeine' in insight.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_external_context.py::TestExternalContextProvider -v`
Expected: FAIL with `ImportError: cannot import name 'ExternalContextProvider'`

- [ ] **Step 3: Implement ExternalContextProvider**

Append to `external_context.py`:

```python
class ExternalContextProvider:
    """
    Orchestrator for external context enrichment.
    Combines weather, nutrition, and sentiment providers.

    Connected to:
    - Neural Workflow AI Engine
    - AI Conversational Coach
    - Biometric Integration Engine
    - Dashboard Data Integration
    """

    def __init__(self):
        self.weather = WeatherProvider()
        self.nutrition = NutritionProvider()
        self.sentiment = SentimentValidator()

    def get_weather(self, lat: float, lon: float) -> dict | None:
        return self.weather.get_weather(lat, lon)

    def get_nutrition(self, food_log: list[str]) -> dict | None:
        return self.nutrition.get_nutrition(food_log)

    def validate_sentiment(self, text: str, local_family: str = None) -> dict | None:
        return self.sentiment.validate_sentiment(text, local_family)

    def _weather_insight(self, weather_data: dict) -> str:
        """Generate human-readable weather insight from mood signals."""
        signals = weather_data.get('mood_signals', [])
        desc = weather_data.get('weather_description', 'Unknown')
        temp = weather_data.get('temperature_c', 0)

        parts = []
        if 'low_sunlight' in signals:
            parts.append(f"{desc} conditions with low sunlight may be affecting your mood")
        if 'extreme_temp' in signals:
            if temp < 0:
                parts.append(f"Cold temperatures ({temp}°C) can contribute to low energy")
            else:
                parts.append(f"High heat ({temp}°C) can increase irritability and fatigue")
        if 'high_humidity' in signals:
            parts.append("High humidity can contribute to feelings of discomfort")
        if 'poor_air_quality' in signals:
            parts.append("Poor air quality may be affecting how you feel physically")
        if 'high_uv' in signals:
            parts.append("High UV levels — consider staying hydrated and seeking shade")

        if not parts:
            return f"Current conditions: {desc}, {temp}°C — no significant weather-mood factors detected."

        return '. '.join(parts) + '.'

    def _nutrition_insight(self, nutrition_data: dict) -> str:
        """Generate human-readable nutrition insight from mood signals."""
        signals = nutrition_data.get('mood_signals', [])
        caffeine = nutrition_data.get('caffeine_mg', 0)
        calories = nutrition_data.get('total_calories', 0)

        parts = []
        if 'high_caffeine' in signals:
            parts.append(f"High caffeine intake (~{caffeine:.0f}mg) may be amplifying anxiety or jitters")
        if 'low_protein' in signals:
            parts.append("Low protein in your food log may contribute to fatigue")
        if 'sugar_crash_risk' in signals:
            parts.append("High sugar intake could lead to an energy crash")
        if 'undereating' in signals:
            parts.append(f"Low calorie intake ({calories:.0f} cal) may be causing irritability or low energy")
        if 'post_meal_fatigue' in signals:
            parts.append("Large meal may cause post-meal drowsiness")

        if not parts:
            return "No significant nutrition-mood factors detected in your food log."

        return '. '.join(parts) + '.'

    def enrich_coaching(self, context: dict, text: str, local_family: str = None) -> dict:
        """
        Enrich coaching response with external context.
        Each context key is independent — client can pass any combination.

        Args:
            context: Raw context object from API request.
                     Keys: 'location' (lat/lon), 'food_log' (list), 'validate_sentiment' (bool)
            text: User's original message (used for sentiment validation).
            local_family: Emotion family from local classifier (for agreement check).

        Returns:
            Dict with weather_insight, weather_data, nutrition_insight,
            nutrition_data, and cross_validation fields (any may be None).
        """
        result = {
            'weather_insight': None,
            'weather_data': None,
            'nutrition_insight': None,
            'nutrition_data': None,
            'cross_validation': None,
        }

        # Weather enrichment
        location = context.get('location')
        if location and isinstance(location, (list, tuple)) and len(location) == 2:
            weather_data = self.get_weather(location[0], location[1])
            if weather_data:
                result['weather_data'] = weather_data
                result['weather_insight'] = self._weather_insight(weather_data)

        # Nutrition enrichment
        food_log = context.get('food_log')
        if food_log and isinstance(food_log, list):
            nutrition_data = self.get_nutrition(food_log)
            if nutrition_data:
                result['nutrition_data'] = nutrition_data
                result['nutrition_insight'] = self._nutrition_insight(nutrition_data)

        # Sentiment cross-validation
        if context.get('validate_sentiment'):
            sentiment_data = self.validate_sentiment(text, local_family)
            if sentiment_data:
                result['cross_validation'] = sentiment_data

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_external_context.py::TestExternalContextProvider -v`
Expected: All tests PASS

- [ ] **Step 5: Run all external context tests**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_external_context.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add external_context.py tests/test_external_context.py
git commit -m "feat: add ExternalContextProvider orchestrator with coaching enrichment"
```

---

## Chunk 5: Wire into emotion_api_server.py and quantara_integration.py

### Task 5: Add standalone API endpoints

**Files:**
- Modify: `emotion_api_server.py:762-958` (inside `create_app()` function)

- [ ] **Step 1: Add ExternalContextProvider import to emotion_api_server.py**

Add after the existing imports at top of `emotion_api_server.py` (around line 67):

```python
try:
    from external_context import ExternalContextProvider
    HAS_EXTERNAL_CONTEXT = True
except ImportError:
    HAS_EXTERNAL_CONTEXT = False
```

- [ ] **Step 2: Initialize ExternalContextProvider in create_app()**

Add inside `create_app()` function, after `CORS(app)` line (~line 766):

```python
    # External context provider (weather, nutrition, sentiment)
    context_provider = ExternalContextProvider() if HAS_EXTERNAL_CONTEXT else None
```

- [ ] **Step 3: Add 3 standalone endpoints**

Add before the `return app` line (~line 958) inside `create_app()`:

```python
    # ─── External Context Endpoints ─────────────────────────────────────────

    @app.route('/api/context/weather', methods=['POST'])
    def context_weather():
        """Get weather and air quality for a location"""
        if not context_provider:
            return jsonify({'error': 'External context not available', 'status': 'error'}), 503
        try:
            data = request.json or {}
            lat = data.get('latitude')
            lon = data.get('longitude')
            if lat is None or lon is None:
                return jsonify({'error': 'latitude and longitude are required', 'status': 'error'}), 400

            result = context_provider.get_weather(float(lat), float(lon))
            if result is None:
                return jsonify({'error': 'Weather service unavailable', 'status': 'error'}), 502
            return jsonify({**result, 'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    @app.route('/api/context/nutrition', methods=['POST'])
    def context_nutrition():
        """Analyze food log for mood-relevant nutrition data"""
        if not context_provider:
            return jsonify({'error': 'External context not available', 'status': 'error'}), 503
        try:
            data = request.json or {}
            food_log = data.get('food_log')
            if not food_log or not isinstance(food_log, list):
                return jsonify({'error': 'food_log (list of strings) is required', 'status': 'error'}), 400

            result = context_provider.get_nutrition(food_log)
            if result is None:
                return jsonify({'error': 'Nutrition service unavailable', 'status': 'error'}), 502
            return jsonify({**result, 'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500

    @app.route('/api/context/sentiment', methods=['POST'])
    def context_sentiment():
        """Cross-validate text sentiment via NLP Cloud"""
        if not context_provider:
            return jsonify({'error': 'External context not available', 'status': 'error'}), 503
        try:
            data = request.json or {}
            text = data.get('text')
            if not text:
                return jsonify({'error': 'text is required', 'status': 'error'}), 400

            result = context_provider.validate_sentiment(text)
            if result is None:
                return jsonify({'error': 'Sentiment service unavailable', 'status': 'error'}), 502
            return jsonify({**result, 'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'}), 500
```

- [ ] **Step 4: Commit**

```bash
git add emotion_api_server.py
git commit -m "feat: add /api/context/weather, /api/context/nutrition, /api/context/sentiment endpoints"
```

### Task 6: Wire context into coaching endpoints

**Files:**
- Modify: `emotion_api_server.py:686-741` (EmotionGPTModel.coach method)
- Modify: `emotion_api_server.py:850-870` (coach endpoint in create_app)
- Modify: `quantara_integration.py:219-268` (get_coaching_response)

- [ ] **Step 1: Add context parameter to EmotionGPTModel.coach()**

Modify the `coach()` method signature at line 686 of `emotion_api_server.py`:

```python
    def coach(
        self,
        message: str,
        emotion: str = None,
        biometric_data: dict = None,
        use_model: bool = False,
        context: dict = None,
        context_provider=None
    ) -> dict:
```

Then, before the `return` statement (~line 733), add context enrichment. Note: `context_provider` is passed in from the `create_app()` scope (see Task 5 Step 2):

```python
        # External context enrichment (only when context is provided)
        weather_insight = None
        nutrition_insight = None
        cross_validation = None
        if context and context_provider:
            enrichment = context_provider.enrich_coaching(context, message, local_family=family)
            weather_insight = enrichment.get('weather_insight')
            nutrition_insight = enrichment.get('nutrition_insight')
            cross_validation = enrichment.get('cross_validation')
```

Then modify the return dict to include the new fields (only when not None):

```python
        result = {
            'response': response,
            'detected_emotion': emotion,
            'family': family,
            'biometric_insight': biometric_insight,
            'coaching_prompt': COACHING_PROMPTS.get(emotion, ''),
            'user_message': message,
            'model': 'quantara-emotion-gpt'
        }

        if weather_insight is not None:
            result['weather_insight'] = weather_insight
        if nutrition_insight is not None:
            result['nutrition_insight'] = nutrition_insight
        if cross_validation is not None:
            result['cross_validation'] = cross_validation

        return result
```

- [ ] **Step 2: Wire context in the coach endpoint**

Modify the coach endpoint in `create_app()` (~line 860) to pass context and provider:

```python
            result = model.coach(
                message=message,
                emotion=data.get('emotion'),
                biometric_data=data.get('biometric'),
                use_model=data.get('use_model', False),
                context=data.get('context'),
                context_provider=context_provider
            )
```

- [ ] **Step 3: Add context parameter to quantara_integration.py for parity**

Modify `get_coaching_response()` at line 219 of `quantara_integration.py`:

```python
    def get_coaching_response(
        self,
        user_message: str,
        detected_emotion: str = None,
        biometric_data: dict = None,
        context: dict = None
    ) -> dict:
```

Then before the return statement (~line 262), add:

```python
        # External context enrichment
        if context:
            try:
                from external_context import ExternalContextProvider
                ext = ExternalContextProvider()
                family = self._emotion_to_family.get(detected_emotion, 'Neutral')
                enrichment = ext.enrich_coaching(context, user_message, local_family=family)
                if enrichment.get('weather_insight'):
                    result['weather_insight'] = enrichment['weather_insight']
                if enrichment.get('nutrition_insight'):
                    result['nutrition_insight'] = enrichment['nutrition_insight']
                if enrichment.get('cross_validation'):
                    result['cross_validation'] = enrichment['cross_validation']
            except ImportError:
                pass
```

And change the return to use a `result` variable instead of inline dict:

```python
        result = {
            'response': response,
            'detected_emotion': detected_emotion,
            'biometric_insight': biometric_insight,
            'user_message': user_message,
            'model': 'quantara-emotion-gpt'
        }

        # External context enrichment
        if context:
            ...

        return result
```

- [ ] **Step 4: Commit**

```bash
git add emotion_api_server.py quantara_integration.py
git commit -m "feat: wire external context enrichment into coaching endpoints"
```

### Task 7: Backwards compatibility test

**Files:**
- Modify: `tests/test_external_context.py`

- [ ] **Step 1: Write backwards compatibility test**

Append to `tests/test_external_context.py`:

```python
class TestBackwardsCompatibility:
    """Ensure coaching works identically without context parameter."""

    def test_coach_no_context_no_new_fields(self):
        """Coach response without context has no weather/nutrition/cross_validation fields."""
        # Simulate what coach() returns with context=None
        provider = ExternalContextProvider()
        result = provider.enrich_coaching({}, "hello")

        assert result['weather_insight'] is None
        assert result['nutrition_insight'] is None
        assert result['cross_validation'] is None
```

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/test_external_context.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_external_context.py
git commit -m "test: add backwards compatibility test for coaching without context"
```

---

## Chunk 6: Update API server status endpoint and final verification

### Task 8: Update status endpoint and verify

**Files:**
- Modify: `emotion_api_server.py:768-778` (status endpoint)

- [ ] **Step 1: Update status endpoint to show external context availability**

Modify the status endpoint return in `create_app()` (~line 771):

```python
        return jsonify({
            'status': 'online',
            'model': 'quantara-emotion-gpt',
            'device': model.device,
            'timestamp': datetime.now().isoformat(),
            'version': '2.1.0',
            'taxonomy': '32-emotion / 9-family',
            'external_context': {
                'available': HAS_EXTERNAL_CONTEXT,
                'weather': True,  # Open-Meteo, no key needed
                'nutrition': bool(os.environ.get('NUTRITIONIX_APP_ID')),
                'sentiment': bool(os.environ.get('NLPCLOUD_API_KEY')),
            }
        })
```

Add `import os` at the top of the status endpoint if not already available in scope (it's already imported at file top).

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/bel/quantara-nanoGPT && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add emotion_api_server.py
git commit -m "feat: update status endpoint to report external context availability"
```

- [ ] **Step 4: Verify server starts without errors**

Run: `cd /Users/bel/quantara-nanoGPT && python -c "from external_context import ExternalContextProvider; p = ExternalContextProvider(); print('OK:', p)"`
Expected: `OK: <external_context.ExternalContextProvider object at ...>`

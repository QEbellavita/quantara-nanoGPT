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

# External Context Enrichment Layer — Design Spec

**Date:** 2026-03-14
**Status:** Approved
**Scope:** Integrate Open-Meteo, NLP Cloud, and Nutritionix APIs into the Quantara Neural Ecosystem as external context signals for emotion-aware coaching and standalone data access.

## Problem

The coaching pipeline currently uses text emotion analysis and biometric signals (HR, HRV, EDA) to generate responses. Environmental and dietary factors — which significantly affect emotional state — are not captured. There is also no external cross-validation of the local emotion classifier's output.

## Decision Summary

- **Integration pattern:** Standalone API endpoints + optional coaching enrichment via `context` parameter (Option C)
- **Authentication:** Environment variables for API keys (Option A)
- **Coaching enrichment:** Lazy — client opts in by passing a `context` object; no external calls otherwise (Option C)
- **Sentiment cross-validation:** Trust local classifier as authority; NLP Cloud sentiment is informational only (Option A)

## Architecture

### New Module: `external_context.py`

Single module containing `ExternalContextProvider` and three sub-providers.

#### WeatherProvider

- **API:** Open-Meteo (https://api.open-meteo.com/v1/forecast) — free, no auth, CORS-friendly
- **Input:** latitude, longitude
- **Output:**
  - `temperature_c`: Current temperature in Celsius
  - `humidity`: Relative humidity %
  - `weather_code`: WMO weather code (maps to description)
  - `uv_index`: Current UV index
  - `air_quality`: PM2.5 and PM10 from Open-Meteo Air Quality API
  - `mood_signals`: Derived mood-relevant factors
- **Mood signal mapping:**
  - Low sunlight (weather_code ≥ 50) → `low_sunlight` flag (seasonal affective risk)
  - High humidity (> 80%) → `high_humidity` flag (discomfort)
  - Extreme temperature (< 0°C or > 35°C) → `extreme_temp` flag
  - Poor air quality (PM2.5 > 35) → `poor_air_quality` flag
  - High UV (> 7) → `high_uv` flag
- **Caching:** 30-minute TTL keyed on rounded lat/lon (2 decimal places)
- **Timeout:** 5 seconds
- **Failure mode:** Returns `None`; coaching proceeds without weather insight

#### NutritionProvider

- **API:** Nutritionix Natural Language API (https://trackapi.nutritionix.com/v2/natural/nutrients)
- **Auth:** `NUTRITIONIX_APP_ID` and `NUTRITIONIX_API_KEY` environment variables
- **Input:** List of food descriptions (natural language, e.g., `["3 cups of coffee", "a donut"]`)
- **Output:**
  - `total_calories`: Sum across all items
  - `protein_g`, `carbs_g`, `fat_g`: Macronutrient totals
  - `caffeine_mg`: Total caffeine (when available)
  - `sugar_g`: Total sugar
  - `mood_signals`: Derived mood-relevant factors
- **Mood signal mapping:**
  - High caffeine (> 300mg) → `high_caffeine` (anxiety/jitter risk)
  - Low protein (< 15g in log) → `low_protein` (fatigue risk)
  - High sugar (> 50g) → `sugar_crash_risk`
  - Very low calories (< 500 in log) → `undereating` (irritability/fatigue risk)
  - High calorie meal (> 1000) → `post_meal_fatigue` risk
- **Timeout:** 5 seconds
- **Failure mode:** Returns `None`; coaching proceeds without nutrition insight

#### SentimentValidator

- **API:** NLP Cloud sentiment analysis endpoint
- **Auth:** `NLPCLOUD_API_KEY` environment variable
- **Model:** `distilbert-base-uncased-finetuned-sst-2-english` (fast, free tier eligible)
- **Input:** Text string
- **Output:**
  - `positive_score`: 0.0–1.0
  - `negative_score`: 0.0–1.0
  - `nlpcloud_sentiment`: "POSITIVE" or "NEGATIVE"
  - `agrees_with_local`: Boolean — whether NLP Cloud sentiment direction aligns with the local classifier's emotion family valence
- **Valence mapping for agreement check:**
  - Positive families: Joy, Love, Calm, Surprise
  - Negative families: Sadness, Anger, Fear, Self-Conscious
  - Neutral: always "agrees"
- **Timeout:** 5 seconds
- **Failure mode:** Returns `None`; response includes no cross-validation field

### ExternalContextProvider

Orchestrator class that holds all three providers and exposes:

```python
class ExternalContextProvider:
    def __init__(self):
        self.weather = WeatherProvider()
        self.nutrition = NutritionProvider()
        self.sentiment = SentimentValidator()

    def get_weather(self, lat, lon) -> dict | None
    def get_nutrition(self, food_log: list[str]) -> dict | None
    def validate_sentiment(self, text: str) -> dict | None
    def enrich_coaching(self, context: dict, text: str) -> dict
```

`enrich_coaching()` takes the raw `context` object from the API request and returns:

```python
{
    "weather_insight": str | None,    # Human-readable weather insight
    "weather_data": dict | None,      # Raw weather data
    "nutrition_insight": str | None,  # Human-readable nutrition insight
    "nutrition_data": dict | None,    # Raw nutrition data
    "cross_validation": dict | None,  # NLP Cloud sentiment comparison
}
```

### Coaching Integration

**Modified method signature:**

```python
def get_coaching_response(
    self,
    user_message: str,
    detected_emotion: str = None,
    biometric_data: dict = None,
    context: dict = None            # NEW — optional external context
) -> dict
```

**Behavior when `context` is provided:**

```python
context = {
    "location": [37.7749, -122.4194],      # lat, lon — triggers weather lookup
    "food_log": ["3 cups of coffee"],        # triggers nutrition analysis
    "validate_sentiment": true               # triggers NLP Cloud cross-validation
}
```

Each key is independent. Client can pass any combination.

**Response additions (only when context is provided):**

```python
{
    # Existing fields unchanged
    "response": "...",
    "detected_emotion": "anxiety",
    "biometric_insight": "...",

    # New fields (present only when requested via context)
    "weather_insight": "Cold, overcast conditions with low sunlight may be contributing to your mood.",
    "nutrition_insight": "High caffeine intake (estimated 300mg+) may be amplifying feelings of anxiety.",
    "cross_validation": {
        "nlpcloud_sentiment": "NEGATIVE",
        "positive_score": 0.12,
        "negative_score": 0.88,
        "agrees_with_local": true
    }
}
```

When `context` is `None` (default), none of these fields are added. Fully backwards-compatible.

### New API Endpoints

Added to `emotion_api_server.py`:

#### `POST /api/context/weather`

```json
// Request
{"latitude": 37.7749, "longitude": -122.4194}

// Response
{
    "temperature_c": 15.2,
    "humidity": 72,
    "weather_code": 3,
    "weather_description": "Overcast",
    "uv_index": 2.1,
    "air_quality": {"pm2_5": 12.3, "pm10": 18.7},
    "mood_signals": ["low_sunlight"],
    "status": "success"
}
```

#### `POST /api/context/nutrition`

```json
// Request
{"food_log": ["3 cups of coffee", "a blueberry muffin", "grilled chicken salad"]}

// Response
{
    "total_calories": 685,
    "protein_g": 32.5,
    "carbs_g": 78.2,
    "fat_g": 24.1,
    "caffeine_mg": 285,
    "sugar_g": 38.0,
    "items": [
        {"name": "Coffee", "qty": 3, "calories": 15, "caffeine_mg": 285},
        {"name": "Blueberry Muffin", "qty": 1, "calories": 350},
        {"name": "Grilled Chicken Salad", "qty": 1, "calories": 320}
    ],
    "mood_signals": ["high_caffeine"],
    "status": "success"
}
```

#### `POST /api/context/sentiment`

```json
// Request
{"text": "I've been feeling really overwhelmed and stressed out lately"}

// Response
{
    "nlpcloud_sentiment": "NEGATIVE",
    "positive_score": 0.08,
    "negative_score": 0.92,
    "status": "success"
}
```

### Error Handling

All external API failures are non-fatal:

- Each provider wraps its HTTP call in try/except with a 5-second timeout
- On failure, the provider returns `None`
- The coaching response omits the corresponding insight field
- Errors are logged (not raised) with the provider name and error type
- The `/api/context/*` standalone endpoints return `{"status": "error", "message": "..."}` with HTTP 502 when the upstream API fails

### Caching

- **Weather:** 30-minute TTL, keyed on `f"{lat:.2f},{lon:.2f}"`. Simple dict cache with timestamp (no external dependency).
- **Nutrition:** No caching (food logs vary per request).
- **Sentiment:** No caching (text varies per request).

### Dependencies

- `requests` — HTTP client (likely already installed; needed for external API calls)
- No new heavy dependencies. All three APIs are standard REST/JSON.

### Environment Variables

| Variable | Required | Used By |
|----------|----------|---------|
| `NUTRITIONIX_APP_ID` | For nutrition features | NutritionProvider |
| `NUTRITIONIX_API_KEY` | For nutrition features | NutritionProvider |
| `NLPCLOUD_API_KEY` | For sentiment cross-validation | SentimentValidator |

Open-Meteo requires no authentication.

Missing keys cause the respective provider to return `None` with a logged warning — never a crash.

### Integration Points

| Quantara Component | Connection |
|--------------------|------------|
| Neural Workflow AI Engine | Weather/nutrition context in workflow triggers |
| AI Conversational Coach | Enriched coaching responses with environmental + dietary signals |
| Dashboard Data Integration | External context signals for real-time analytics |
| Biometric Integration Engine | Nutrition + biometrics correlation (caffeine × heart rate) |
| Real-time Data | Weather and air quality as ambient data streams |

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `external_context.py` | Create | ExternalContextProvider with three sub-providers |
| `emotion_api_server.py` | Modify | Add 3 standalone endpoints, wire context into coaching |
| `quantara_integration.py` | Modify | Add `context` parameter to `get_coaching_response()` |
| `tests/test_external_context.py` | Create | Unit tests for all providers and enrichment logic |

### Testing Strategy

- Unit tests mock HTTP responses for all three APIs
- Test graceful degradation: mock timeouts and errors, verify `None` returns
- Test coaching backwards compatibility: ensure no `context` = identical behavior
- Test mood signal derivation for each threshold
- Integration test with real APIs (optional, gated behind env var presence)

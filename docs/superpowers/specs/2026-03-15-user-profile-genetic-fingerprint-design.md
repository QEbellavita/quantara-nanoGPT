# User Profile & Genetic Fingerprint Engine — Design Spec

**Date:** 2026-03-15
**Status:** Draft
**Project:** quantara-nanoGPT

## Overview

A centralized User Profile Engine in quantara-nanoGPT that serves as the connective tissue of the entire Quantara ecosystem. It ingests data from all services (local nanoGPT modules + Quantara Backend/Frontend/Master), synthesizes it into an 8-domain genetic fingerprint, tracks user evolution through 5 maturity stages, and pushes intelligence back out to personalize every service.

### Architecture

Centralized engine (Approach A) with event-sourced ingestion (elements of Approach C). Every data point is logged as a timestamped event before being processed into the fingerprint. The engine is the single owner of user profile truth.

## 1. Data Model

### Event Log (SQLite — append-only)

| Column | Type | Description |
|--------|------|-------------|
| event_id | INTEGER PRIMARY KEY | Auto-increment |
| user_id | TEXT NOT NULL | User identifier |
| timestamp | REAL NOT NULL | Unix timestamp |
| domain | TEXT NOT NULL | DNA domain (biometric, cognitive, emotional, temporal, behavioral, linguistic, social, aspirational) |
| event_type | TEXT NOT NULL | Specific event (e.g., emotion_classified, hr_reading, session_completed) |
| payload | TEXT NOT NULL | JSON blob with event-specific data |
| source | TEXT NOT NULL | Origin service (nanogpt, backend, frontend, master, external) |
| confidence | REAL | 0.0–1.0 confidence in this data point |

Indexes: `(user_id, domain, timestamp)`, `(user_id, timestamp)`, `(user_id, event_type)`

#### Event Rate Limiting & Retention

High-frequency biometric events (HR, HRV, EDA) are downsampled before logging:
- **Raw biometric readings:** Aggregated into 1-minute windows (min, max, mean) before logging as a single event. At most 1 event per domain per minute per user.
- **All other events:** Logged individually (emotion classifications, session events, etc. are naturally low-frequency).
- **Retention tiers:**
  - Raw events < 30 days: kept as-is
  - Events 30–90 days: aggregated into hourly summaries, raw events purged
  - Events 90–180 days: aggregated into daily summaries, hourly summaries purged
  - Events > 180 days: aggregated into weekly summaries, daily summaries purged
- **Storage ceiling:** If event table exceeds 500K rows per user, early aggregation is triggered regardless of age.

### User Profile (SQLite)

| Column | Type | Description |
|--------|------|-------------|
| user_id | TEXT PRIMARY KEY | User identifier |
| created_at | REAL | Unix timestamp |
| fingerprint_json | TEXT | Full 8-domain fingerprint as JSON |
| schema_version | INTEGER | Fingerprint JSON schema version (starts at 1) |
| confidence | REAL | Overall fingerprint confidence 0.0–1.0 |
| evolution_stage | INTEGER | Current stage 1–5 |
| evolution_count | INTEGER | Total evolution cycles completed |
| last_evolved | REAL | Timestamp of last evolution |
| last_synced | REAL | Timestamp of last ecosystem sync |

### Evolution Snapshots (SQLite)

| Column | Type | Description |
|--------|------|-------------|
| snapshot_id | INTEGER PRIMARY KEY | Auto-increment |
| user_id | TEXT NOT NULL | User identifier |
| timestamp | REAL NOT NULL | When snapshot was taken |
| snapshot_type | TEXT | weekly, monthly, stage_change |
| fingerprint_json | TEXT | Full fingerprint at this point |
| stage | INTEGER | Evolution stage at snapshot time |
| confidence | REAL | Confidence at snapshot time |
| trends_json | TEXT | Per-domain trend data |

### Synergies (SQLite)

| Column | Type | Description |
|--------|------|-------------|
| synergy_id | INTEGER PRIMARY KEY | Auto-increment |
| user_id | TEXT NOT NULL | User identifier |
| domain_a | TEXT NOT NULL | First domain |
| domain_b | TEXT NOT NULL | Second domain |
| correlation | REAL | Correlation strength -1.0 to 1.0 |
| insight | TEXT | Human-readable insight |
| detected_at | REAL | When synergy was first detected |
| last_confirmed | REAL | Most recent confirmation |

## 2. 8 DNA Domains

### Biometric DNA
- **Sources:** WiFi calibration readings (RuView), wearable sensors, HealthKit sync, cross-modal fusion (Backend)
- **Computed metrics:** Resting HR baseline, HRV baseline, EDA baseline, stress response curve (onset speed, peak intensity, recovery time), breathing rate patterns
- **Key events:** `hr_reading`, `hrv_reading`, `eda_reading`, `stress_response`, `recovery_event`

### Cognitive DNA
- **Sources:** Transition session outcomes, therapy technique effectiveness, learning patterns from session data
- **Computed metrics:** Learning style (visual/auditory/kinesthetic preference), processing speed (session step completion times), decision-making patterns, problem-solving approach, technique retention rate
- **Key events:** `session_step_completed`, `technique_applied`, `learning_outcome`

### Emotional DNA
- **Sources:** Emotion classifications (32 emotions), pattern detection, emotion transition tracker history
- **Computed metrics:** Baseline mood distribution, emotional range (breadth of emotions expressed), trigger patterns, recovery mechanisms, emotional intelligence score, dominant emotion family
- **Key events:** `emotion_classified`, `pattern_detected`, `transition_recorded`

### Temporal DNA
- **Sources:** Session timing from API requests, HealthKit sleep data, external context (time/calendar), emotion-time correlations
- **Computed metrics:** Chronotype estimate, peak emotional hours, session frequency patterns, circadian mood curve, day-of-week patterns
- **Key events:** `session_started`, `sleep_data`, `time_correlation`

### Behavioral DNA
- **Sources:** Transition session engagement, therapy adherence, intervention response rates, workflow interactions
- **Computed metrics:** Habit strength score, routine consistency, novelty-seeking tendency, intervention response rate by technique type, engagement streak
- **Key events:** `session_started`, `session_completed`, `session_abandoned`, `technique_response`, `workflow_action`

### Linguistic DNA
- **Sources:** Text input from `/api/emotion/analyze` requests, AI Coach conversations (Backend)
- **Computed metrics:** Average sentence length, vocabulary complexity (unique word ratio), dominant tone, communication style (direct/indirect, formal/casual), feedback style preference
- **Key events:** `text_analyzed`, `conversation_turn`, `language_pattern`

### Social DNA
- **Sources:** App interaction patterns (Frontend), session sharing behavior, group vs solo preferences, community engagement
- **Computed metrics:** Interaction frequency, session sharing rate, group affinity score, feedback receptivity, introversion/extraversion estimate
- **Key events:** `app_interaction`, `content_shared`, `community_action`, `feedback_received`

### Aspirational DNA
- **Sources:** User-set goals (Frontend), growth area selections, motivation prompts, AI Coach goal discussions
- **Computed metrics:** Core values (ranked), primary motivators, long-term goals, active growth areas, goal completion rate
- **Key events:** `goal_set`, `goal_updated`, `goal_completed`, `growth_area_selected`, `value_expressed`

## 3. Ecosystem Hub — Bidirectional Intelligence

### Data flows IN (ingestion)

| Source | Transport | Domains Fed |
|--------|-----------|-------------|
| nanoGPT Emotion API | In-process (direct call) | Emotional, Linguistic, Temporal |
| nanoGPT Transition Engine | In-process (direct call) | Behavioral, Cognitive |
| nanoGPT WiFi/RuView | In-process (direct call) | Biometric |
| nanoGPT Pattern Detection | In-process (direct call) | Emotional (pattern events also trigger evolution stage evaluation) |
| nanoGPT External Context | In-process (direct call) | Temporal |
| Quantara Backend Engines | Pull (REST, 5min interval) | Biometric, Cognitive, Emotional |
| Quantara Frontend | Push (webhook to /api/profile/ingest) | Temporal, Social, Aspirational |
| Quantara Master | Pull (REST, 5min interval) | Behavioral |

### Intelligence flows OUT (consumption)

| Consumer | What It Reads | How It Uses It |
|----------|---------------|----------------|
| Emotion Transition Engine | Behavioral + Cognitive DNA | Weights therapy paths by user's technique response rates |
| AI Coach (Backend) | Linguistic + Social + Aspirational DNA | Adapts tone, depth, pacing; aligns with user goals |
| Proactive Alerts | All domains + Evolution stage | Matches current patterns against historical distress signatures |
| Frontend Dashboards | Full fingerprint + Evolution | 3D visualization, DNA strands, evolution timeline |
| Predictive Intelligence | Temporal + Emotional + Biometric DNA | Next-state forecasts and pre-emptive recommendations |
| Neural Workflows (Master) | Profile context summary | Auto-prioritizes cases by emotional state + stage |

### Ingestion Patterns

- **Push (in-process):** Local nanoGPT modules call `profile_engine.log_event()` directly. Synchronous, zero latency.
- **Pull (polling):** `ProfileSyncWorker` thread polls Backend and Master REST APIs every 5 minutes for new data. Exponential backoff on failure (5min → 10min → 20min → max 60min). Resets to 5min after any successful poll.
- **Webhook (push from external):** `/api/profile/ingest` endpoint accepts event payloads from authenticated ecosystem services (see Authentication section).

### Concurrency Strategy

SQLite is configured with WAL (Write-Ahead Logging) mode for concurrent read/write access. All database writes (event logging, profile updates, snapshot creation) are serialized through a single `ProfileDBWriter` thread with a queue. The `ProfileSyncWorker` and Flask request threads enqueue writes; only the writer thread executes them. Domain processors and API reads use separate read-only connections.

## 4. API Surface

### Frontend-Compatible Endpoints (matching existing GeneticFingerprintService.ts contract)

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/api/v1/users/:id/genetic-fingerprint/sync` | POST | `{ user_id }` | Full fingerprint: 8 domains, confidence, evolution stage, synergies, evolution count |
| `/api/v1/users/:id/insights` | POST | `{ user_id, domains? }` | AI-generated insights array with confidence and actionable recommendations |
| `/api/v1/cognitive/theory-of-mind` | POST | `{ user_id }` | Personality evolution: Big Five traits inferred from Emotional + Social + Behavioral DNA (Openness from novelty-seeking + emotional range; Conscientiousness from habit strength + routine consistency; Extraversion from social interaction frequency + group affinity; Agreeableness from feedback receptivity + communication style; Neuroticism from stress response curve + emotional volatility) |

### Profile Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/profile/ingest` | POST | Webhook — accepts event payloads from authenticated ecosystem services (requires service API key) |
| `/api/profile/:id/snapshot` | GET | Current fingerprint snapshot |
| `/api/profile/:id/evolution` | GET | Full evolution timeline: all snapshots, stage history, trends |
| `/api/profile/:id/evolution/stage` | GET | Current stage + progress toward next stage (criteria checklist) |
| `/api/profile/:id/domain/:domain` | GET | Deep-dive into a single DNA domain (full metrics + event history) |
| `/api/profile/:id/synergies` | GET | Cross-domain correlations and insights |
| `/api/profile/:id/predictions` | GET | Next-state forecasts: predicted dominant emotion family, expected stress level, optimal intervention timing, and risk factors — computed from Temporal + Emotional + Biometric DNA trend extrapolation over a 24-48 hour window |
| `/api/profile/:id/events` | GET | Paginated event log (query params: domain, start_time, end_time, limit, offset) |
| `/api/profile/:id/export` | GET | Full profile export as JSON (data portability) |
| `/api/profile/:id` | DELETE | Purge all user data (events, snapshots, profile, synergies) |
| `/api/profile/health` | GET | Profile engine health check |

### Ecosystem Service Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/profile/:id/context` | GET | Lightweight profile summary for service personalization |
| `/api/profile/:id/domain/:domain/score` | GET | Single domain score for quick lookups |

### WebSocket Channels (socket.io)

| Channel | Payload | Trigger |
|---------|---------|---------|
| `profile:updated` | `{ user_id, domains_changed, confidence }` | Any domain score change |
| `profile:stage-change` | `{ user_id, old_stage, new_stage, criteria_met }` | Evolution stage transition |
| `profile:alert` | `{ user_id, alert_type, message, severity }` | Proactive alerts (predicted distress, anomalies) |

## 5. Evolution Stage System

### Stage Definitions

| Stage | Name | Confidence Range | Entry Criteria | User Experience |
|-------|------|-----------------|----------------|-----------------|
| 1 | **Nascent** | <0.3 | User created | "We're getting to know you" — progress bar toward first snapshot |
| 2 | **Awareness** | 0.3–0.55 | 50+ events across 3+ domains, confidence ≥0.3 | First complete fingerprint revealed. Basic insights unlocked. |
| 3 | **Regulation** | 0.55–0.75 | 4+ weeks of data, positive pattern detected, confidence >0.55 | Evolution timeline visible. Personalized therapy. Predictions start. |
| 4 | **Integration** | 0.75–0.9 | 3+ domain synergies, 8+ weeks of data, confidence >0.75 | Synergy map unlocked. Proactive alerts. Fully personalized AI Coach. |
| 5 | **Mastery** | >0.9 | Sustained multi-domain growth 12+ weeks, stability >0.85, confidence >0.9 | Full predictive intelligence. Mentor candidate. |

### Stage Transition Rules

- **Advance:** Criteria must be met for 3 consecutive evaluation cycles (prevents noise-driven flickering).
- **Regress:** If sustained negative patterns detected for 2+ weeks, user drops back one stage with a supportive "recalibration" message (never punitive framing).
- **Evaluation frequency:** Every 24 hours or after 20+ new events, whichever comes first.

### Confidence Computation

Overall fingerprint confidence is a weighted mean of per-domain confidences:

- **Per-domain confidence** = `min(1.0, event_count / 100) * data_recency_factor * source_diversity_factor`
  - `data_recency_factor`: 1.0 if events within last 7 days, decays by 0.1 per week of inactivity (floor 0.3)
  - `source_diversity_factor`: 1.0 if 2+ sources feed the domain, 0.8 if single source
- **Overall confidence** = weighted mean of all 8 domain confidences, weighted by event count per domain. Domains with zero events contribute 0.0.
- Confidence naturally rises as more diverse, recent data flows in and naturally decays during inactivity.

### Synergy Detection

A synergy is a statistically significant correlation between two DNA domains over a rolling 4-week window:

- **Computation:** For each pair of domains, compute Pearson correlation between their daily aggregate scores over the past 28 days. Requires at least 14 days of data for both domains.
- **Detection threshold:** |correlation| > 0.6 with p-value < 0.05.
- **Insight generation:** Positive correlations produce insights like "Your sleep quality improvements correlate with better emotional regulation." Negative correlations: "Higher work stress periods coincide with reduced social engagement."
- **Re-evaluation:** Synergies are recomputed weekly. A synergy is removed if it drops below |0.4| for 2 consecutive weeks.

### Snapshot Schedule

- **Weekly snapshots:** Taken every Sunday at midnight (user's local time if known, else UTC). Contains full fingerprint + per-domain trends.
- **Monthly snapshots:** First of each month. Contains fingerprint + aggregated trends + stage history.
- **Stage-change snapshots:** Taken immediately on any stage transition.
- **Catch-up on startup:** When the server starts, it checks for any missed scheduled snapshots (by comparing last snapshot timestamp against expected schedule) and generates them retroactively from available event data.

## 6. Module Structure

```
quantara-nanoGPT/
├── user_profile_engine.py              # Core orchestrator
│   ├── UserProfileEngine               # Main class — init, process, evolve
│   ├── EventLogger                     # Append-only event logging to SQLite
│   └── ProfileSyncWorker               # Background thread polling ecosystem services
│
├── domain_processors/                  # One processor per DNA domain
│   ├── __init__.py                     # Base class + processor registry
│   ├── biometric_processor.py          # HR/HRV/EDA baselines, stress curves
│   ├── cognitive_processor.py          # Learning patterns, decision style
│   ├── emotional_processor.py          # Mood distribution, triggers, recovery
│   ├── temporal_processor.py           # Circadian patterns, session timing
│   ├── behavioral_processor.py         # Habits, intervention response rates
│   ├── linguistic_processor.py         # Text analysis, tone, vocabulary
│   ├── social_processor.py            # Interaction patterns, preferences
│   └── aspirational_processor.py       # Goals, values, growth areas
│
├── evolution_engine.py                 # Stage system + snapshot management
│   ├── EvolutionEngine                 # Stage transitions, criteria checking
│   ├── SnapshotManager                 # Weekly/monthly/stage-change snapshots
│   └── SynergyDetector                 # Cross-domain correlation finder
│
├── profile_api.py                      # Flask Blueprint — all profile endpoints
│   ├── Frontend-compatible routes      # /api/v1/users/:id/...
│   ├── Profile routes                  # /api/profile/:id/...
│   └── Ingest webhook                  # /api/profile/ingest
│
├── profile_db.py                       # SQLite schema, queries, migrations
│   └── Tables: events, profiles, snapshots, synergies
│
└── tests/
    ├── test_profile_engine.py          # Core engine tests
    ├── test_domain_processors.py       # Per-domain processor tests
    ├── test_evolution_engine.py         # Stage system tests
    └── test_profile_api.py             # API endpoint tests
```

### Key Design Decisions

- **Domain processors are isolated:** Each processor only knows how to compute its own DNA domain from events. Easy to test, easy to add new domains.
- **UserProfileEngine is the orchestrator:** Wires event logging → domain processing → fingerprint synthesis → evolution checks → API serving.
- **Flask Blueprint:** Profile endpoints register on the existing Flask app in `emotion_api_server.py`, so everything runs in one process.
- **SQLite:** Consistent with the rest of the codebase. Events table indexed on `(user_id, domain, timestamp)`.

## 7. Integration with Existing Modules

### emotion_api_server.py
- Register profile Blueprint on the existing Flask app.
- Add `profile_engine.log_event()` calls to existing endpoints:
  - `/api/emotion/analyze` → logs emotion result + text metadata (Emotional + Linguistic + Temporal DNA)
  - `/api/emotion/transition/record` → logs transition (Behavioral DNA)
  - `/api/ruview/*` → logs biometric readings (Biometric DNA)
- Add `/api/profile/health` alongside existing `/api/emotion/health`.

### emotion_transition_tracker.py
- `record()` also pushes an event to the profile engine.
- `detect_patterns()` results feed into evolution stage evaluation.
- Existing per-user JSON files continue to work — profile engine is additive, not a replacement.

### emotion_transition_engine.py
- Session outcomes feed Behavioral + Cognitive DNA.
- Engine reads from profile to personalize transition path weights (e.g., "this user's behavioral DNA shows 40% better response to cognitive techniques than somatic ones").

### wifi_calibration.py
- Each `PersonalCalibrationBuffer` reading becomes a Biometric DNA event.
- Existing `profile_id` maps directly to the profile engine's `user_id`.

### external_context.py
- Weather, calendar, time context enriches Temporal DNA events (attached as metadata, not separate events).

### auto_retrain.py
- Drift detection events feed into the profile as system-level events (visibility into when user patterns shifted enough to trigger model retraining).

### Integration Pattern
All integrations use a lightweight pattern:
```python
if profile_engine:
    profile_engine.log_event(user_id, domain, event_type, payload, source, confidence)
```
If the profile engine isn't initialized (e.g., standalone training), the calls are no-ops. No existing module behavior is changed.

## 8. Error Handling & Privacy

### Error Handling

- **Event logging failures are non-blocking:** If event log write fails, the original operation (emotion analysis, therapy session, etc.) still succeeds. Events are buffered in-memory and retried on next cycle.
- **Domain processor failures are isolated:** If one processor errors, the other 7 domains still update. Failed domain keeps its last known state; confidence drops slightly.
- **Ecosystem sync failures are graceful:** If Backend/Master is unreachable, pull worker backs off exponentially (5min → 10min → 20min). Profile continues updating from local sources.
- **Evolution stage transitions require confirmation:** Stage advances only trigger after criteria are met for 3 consecutive evaluation cycles.

### Authentication & Authorization

- **User-facing endpoints** (`/api/profile/:id/*`, `/api/v1/users/:id/*`): Require a user token (JWT or API key) in the `Authorization` header. The token's `user_id` claim must match the `:id` path parameter — users can only access their own profile. Returns 403 if mismatched.
- **Ingest webhook** (`/api/profile/ingest`): Requires a service API key in the `X-Service-Key` header. Each ecosystem service (Backend, Frontend, Master) gets a unique key configured in environment variables (`PROFILE_SERVICE_KEY_BACKEND`, `PROFILE_SERVICE_KEY_FRONTEND`, `PROFILE_SERVICE_KEY_MASTER`). Returns 401 if missing/invalid.
- **Ecosystem service endpoints** (`/api/profile/:id/context`, `/api/profile/:id/domain/:domain/score`): Require service API key (same as ingest). These are service-to-service calls, not user-facing.
- **Rate limiting:** Export and event log endpoints limited to 10 requests/minute per user. Ingest webhook limited to 100 events/second per service key.

### Privacy

- **Per-user isolation:** All SQLite queries scoped by `user_id`. API authorization enforces ownership — users can only access their own profile.
- **Event log retention:** Tiered aggregation (see Section 1 — Event Rate Limiting & Retention).
- **Data export:** `/api/profile/:id/export` returns full profile as JSON for data portability.
- **Data deletion:** `DELETE /api/profile/:id` purges all events, snapshots, profile, and synergies data for a user.

## 9. Connection to Neural Workflow AI Engine

This profile engine connects to all 5 phases of the Quantara Neural Ecosystem:

- **Phase 1 (Foundation):** Profile DB extends the existing SQLite infrastructure. Event log provides the data layer.
- **Phase 2 (ML Training & Prediction):** Domain processors use the trained emotion classifier. Evolution predictions leverage the multimodal analyzer.
- **Phase 3 (Backend APIs):** Profile API Blueprint integrates with the existing Flask server. Ecosystem sync pulls from Backend engines.
- **Phase 4 (Dashboard Integration):** WebSocket channels push real-time profile updates. Frontend-compatible endpoints serve the 3D fingerprint visualization.
- **Phase 5 (Real-time Data):** Live biometric feeds (RuView, HealthKit) stream directly into the event log. Proactive alerts push via WebSocket.

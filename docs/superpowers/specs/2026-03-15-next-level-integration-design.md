# Next-Level Integration — Event Bus, Intelligence Loops, Alerts, WebSocket

**Date:** 2026-03-15
**Status:** Draft
**Project:** quantara-nanoGPT
**Depends on:** User Profile & Genetic Fingerprint Engine (implemented)

## Overview

Elevate the profile engine from a data collector into a fully bidirectional intelligence hub. An in-process event bus unifies ingestion, intelligence delivery, and real-time streaming. All ecosystem services push events in and receive personalization back. Proactive alerts detect and predict concerning patterns. WebSocket smart streaming delivers updates based on what the user is viewing.

### Architecture

Central ProfileEventBus with topic-based pub/sub. Services publish events → profile engine processes → publishes intelligence + alerts → WebSocket router streams to frontend. Tiered autonomy: advisory at stages 1-2, active override at stages 3-5.

## 1. ProfileEventBus — Core Pub/Sub

### Components

**ProfileEventBus** — In-process topic-based publish/subscribe dispatcher. No external dependencies (not Kafka/Redis). Topics are dot-separated strings with wildcard support.

- `publish(topic, payload)` — Broadcast to all matching subscribers
- `subscribe(topic_pattern, callback)` — Register callback. Patterns support `*` wildcards (e.g., `event.*` matches `event.emotional`, `event.biometric`)
- `unsubscribe(subscription_id)` — Remove a subscription

**TopicMatcher** — Resolves wildcard patterns against published topics using glob-style matching.

### Topic Hierarchy

| Topic | Publisher | Purpose |
|-------|-----------|---------|
| `event.emotional` | Ecosystem ingest | Raw emotional event received |
| `event.biometric` | Ecosystem ingest | Raw biometric event received |
| `event.linguistic` | Ecosystem ingest | Raw linguistic event received |
| `event.temporal` | Ecosystem ingest | Raw temporal event received |
| `event.behavioral` | Ecosystem ingest | Raw behavioral event received |
| `event.social` | Ecosystem ingest | Raw social event received |
| `event.aspirational` | Ecosystem ingest | Raw aspirational event received |
| `event.cognitive` | Ecosystem ingest | Raw cognitive event received |
| `profile.updated` | Profile engine | Fingerprint updated after process() |
| `profile.domain.<domain>` | Profile engine | Specific domain score changed |
| `profile.stage.changed` | Profile engine | Evolution stage transition |
| `profile.snapshot.created` | Profile engine | New snapshot taken |
| `intelligence.therapy` | IntelligencePublisher | Therapy personalization data |
| `intelligence.coaching` | IntelligencePublisher | Coaching profile update |
| `intelligence.workflow` | IntelligencePublisher | Workflow prioritization data |
| `intelligence.calibration` | IntelligencePublisher | Emotion classifier baseline |
| `alert.reactive` | AlertEngine | Pattern-based alert fired |
| `alert.predictive` | AlertEngine | Prediction-based alert fired |

### Threading Model

The bus runs on the main thread. `publish()` calls subscriber callbacks synchronously in the publishing thread. For I/O-bound subscribers (outbound webhooks, WebSocket emit), callbacks enqueue work to their own worker threads rather than blocking the bus.

## 2. Ecosystem Ingestion — All Data Sources Connected

All services push events to the nanoGPT `/api/profile/ingest` webhook. The EcosystemConnector receives them and publishes to the bus.

### Backend Engine Event Forwarders

| Backend Engine | Event Type | Domain | Trigger |
|---|---|---|---|
| Stress Detection | `stress_detected` | biometric | Stress level change |
| Cross-Modal Fusion | `fusion_result` | biometric, emotional | New fusion reading |
| AI Coach | `conversation_turn` | linguistic, social, aspirational | Each user message (summarized: word count, tone, topics, goals) |
| Anomaly Detection | `anomaly_detected` | emotional, behavioral | Anomaly flag raised |
| Sentiment Analysis | `sentiment_result` | emotional | Sentiment computed |
| Social Intelligence | `social_interaction` | social | User interaction event |

### Frontend Event Sources

| Source | Event Type | Domain | Trigger |
|---|---|---|---|
| HealthKit | `health_reading` | biometric, temporal | Sleep, steps, HR from Apple Watch |
| App Usage | `app_interaction` | social, temporal | Screen views, session duration |
| User Goals | `goal_action` | aspirational | Goal set/updated/completed |
| Growth Selections | `growth_selected` | aspirational | User selects growth area |
| Sharing | `content_shared` | social | User shares content |

### Master Event Sources

| Source | Event Type | Domain | Trigger |
|---|---|---|---|
| Workflow Engine | `workflow_transition` | behavioral | Case state change |
| Case Management | `case_action` | behavioral, cognitive | User action on case |

### Backend Event Forwarder Pattern

Each engine adds a lightweight fire-and-forget POST:

```python
requests.post(PROFILE_INGEST_URL, json={
    'user_id': user_id,
    'domain': 'emotional',
    'event_type': 'stress_detected',
    'payload': {summarized_data},
    'source': 'backend'
}, headers={'X-Service-Key': SERVICE_KEY}, timeout=2)
```

Timeout of 2 seconds. Failure does not block the engine's main work. The existing `/api/profile/ingest` endpoint and service key authentication are reused unchanged.

### AI Coach Conversation Summarization

Each conversation turn is summarized before sending (not full transcript):

```json
{
    "word_count": 42,
    "tone": "reflective",
    "topics": ["work_stress", "sleep"],
    "goals_mentioned": ["reduce_anxiety"],
    "sentiment_score": 0.3,
    "communication_style": "indirect"
}
```

This feeds Linguistic DNA (word count, tone, style), Social DNA (interaction pattern), and Aspirational DNA (goals mentioned).

## 3. Intelligence Feedback Loops — Personalization Flowing Out

After each `process()` cycle, IntelligencePublisher computes and publishes personalization data.

### Tiered Autonomy

- **Stages 1-2 (Nascent, Awareness):** `mode: 'advisory'` — Services receive recommendations as metadata but make their own decisions
- **Stages 3-5 (Regulation, Integration, Mastery):** `mode: 'active'` — Services should apply personalization directly

### Therapy Personalization (→ Transition Engine)

Computed from Behavioral + Cognitive DNA:
- Technique scores: per-technique effectiveness rate from session outcomes
- Preferred step types: calming vs activation vs cognitive based on response rates
- Advisory mode: Scores attached as metadata on transition pathway responses
- Active mode: Directly adjust edge weights in AdaptiveWeightTracker SQLite

### Coaching Personalization (→ AI Coach on Backend)

Derived from Linguistic + Social + Aspirational DNA:
- `preferred_tone`: direct or supportive (from communication_style metric)
- `vocabulary_level`: simple or complex (from vocabulary_complexity metric)
- `response_depth`: brief or detailed (from avg_sentence_length metric)
- `active_goals`: from Aspirational DNA goal tracking
- `communication_style`: direct/indirect, formal/casual
- Advisory mode: Included as context hints in coach prompts
- Active mode: Override coach system prompt parameters

### Workflow Prioritization (→ Master)

Derived from Emotional + Behavioral DNA:
- `emotional_readiness_score`: 0-1 composite of stability, current dominant emotion valence, stress level
- `engagement_momentum`: from engagement streak and completion rate
- `evolution_stage`: user's current stage
- Advisory mode: Metadata attached to case listings
- Active mode: Auto-reorder case queue by readiness score

### Emotion Analysis Enhancement (→ nanoGPT Classifier)

Derived from Biometric DNA:
- User's resting HR/HRV/EDA baselines for per-user stress threshold calibration
- Applied at all stages (improves accuracy without changing behavior)

### Intelligence Delivery API

Each subscribing service registers an endpoint with the EcosystemConnector:

```
POST <service_url>/api/intelligence/notify
{
    "user_id": "...",
    "intelligence_type": "therapy_personalization",
    "mode": "advisory" | "active",
    "stage": 3,
    "confidence": 0.78,
    "payload": { ... service-specific data ... }
}
```

Delivery: 3 retry attempts with exponential backoff (1s, 2s, 4s). After 3 failures, event dropped, service marked degraded. After 5 consecutive failures, service deregistered (auto-re-registers when reachable).

## 4. Proactive Alerts — Reactive + Predictive

AlertEngine subscribes to all `event.*` topics and runs two detection systems.

### Reactive Alerts (Pattern-Based)

| Alert Type | Trigger | Severity |
|---|---|---|
| Emotional Spiral | 5+ negative emotions in 2 hours | high |
| Rapid Cycling | 8+ emotion changes in 1 hour | medium |
| Sustained Stress | Biometric stress_ratio > 0.5 for 30+ minutes | high |
| Emotional Flatline | Same emotion for 24+ hours | medium |
| Engagement Drop | No events for 3+ days after daily streak | low |
| Recovery Detected | Sustained negative → positive transition | positive |

### Predictive Alerts (Fingerprint-Based)

**Signature matching:** Store the fingerprint snapshot from the last 3 times the user entered a concerning pattern. When current readings match a stored signature (cosine similarity > 0.8 across relevant domains), fire a predictive alert with the matched pattern type.

**Trend extrapolation:** If emotional volatility has been increasing linearly over the past 7 days, compute when it will cross the spiral threshold. Alert 24 hours before predicted crossing.

**Temporal patterns:** If the user historically has stress spikes at specific times (e.g., Monday mornings), alert beforehand with a pre-emptive recommendation.

### Alert Payload

```json
{
    "user_id": "...",
    "alert_type": "emotional_spiral" | "sustained_stress" | "predicted_anxiety" | ...,
    "detection_method": "reactive" | "predictive",
    "severity": "high" | "medium" | "low" | "positive",
    "message": "Your emotional patterns suggest increasing stress...",
    "recommended_action": "Try a 5-minute breathing exercise",
    "confidence": 0.82,
    "timestamp": 1773569000.0
}
```

### Alert Fatigue Prevention

- Same alert type fires at most once per 4 hours per user
- Low severity alerts suppressed while a high severity alert is active
- Stage 1 (Nascent) users only receive positive alerts — not enough data for accurate warnings
- Predictive alerts require confidence > 0.7 to fire

## 5. WebSocket Smart Streaming

WebSocketRouter subscribes to the bus and maintains per-connection subscription state.

### Connection Lifecycle

1. Frontend connects via socket.io with `{user_id, active_screen}`
2. Router creates subscription profile for that connection
3. Frontend emits `screen:changed` with new screen name on navigation
4. Router adjusts which bus topics forward to that connection
5. On disconnect, subscriptions cleaned up, buffer preserved 5 minutes

### Screen-to-Topic Mapping

| Frontend Screen | Bus Topics Forwarded | Update Frequency |
|---|---|---|
| DNA Strands | `profile.domain.*` | On domain score change |
| Evolution Timeline | `profile.stage.*`, `profile.snapshot.*` | On stage/snapshot change |
| 3D Fingerprint | `profile.updated` | On any fingerprint update |
| Coaching Session | `intelligence.coaching`, `alert.*` | Real-time during session |
| Therapist Dashboard | `profile.updated`, `alert.*`, `intelligence.therapy` | Real-time |
| App Backgrounded | Nothing — events batched | Delivered on resume |

### Backgrounded App Batching

- When no active screen, events accumulate in a per-user buffer (max 100 events, FIFO)
- On reconnect/resume within 5 minutes, deliver batch as `profile:batch-update` event
- After 5 minutes, buffer discarded
- Frontend diffs against its last known state

### Payload Optimization

- Domain updates send only the changed domain data, not full fingerprint
- Alerts send full payload (infrequent)
- Stage changes send old_stage, new_stage, criteria_met

### Socket.io Namespaces

- `/profile` — Domain updates, stage changes, snapshots
- `/alerts` — Reactive + predictive alerts
- `/intelligence` — Intelligence events relevant to current screen

## 6. Module Structure

```
quantara-nanoGPT/
├── profile_event_bus.py              # Core pub/sub bus
│   ├── ProfileEventBus               # Topic-based publish/subscribe
│   └── TopicMatcher                  # Wildcard topic matching
│
├── ecosystem_connector.py            # Inbound/outbound webhook management
│   ├── EcosystemConnector            # Service registration, event routing
│   └── OutboundWebhook               # Retry logic for intelligence delivery
│
├── alert_engine.py                   # Reactive + predictive alert system
│   ├── AlertEngine                   # Bus subscriber, runs detectors
│   ├── ReactiveDetector              # Pattern-based (spirals, stress, flatline)
│   ├── PredictiveDetector            # Signature matching, trend extrapolation
│   └── AlertThrottler                # 4hr cooldown, severity suppression
│
├── intelligence_publisher.py         # Pushes personalization to services
│   ├── IntelligencePublisher         # Publishes after process() cycle
│   ├── TherapyPersonalizer           # Therapy reweighting from DNA
│   ├── CoachingPersonalizer          # Coaching profile from DNA
│   └── WorkflowPrioritizer           # Emotional readiness scores
│
├── websocket_router.py               # Smart streaming to frontend
│   ├── WebSocketRouter               # Per-connection subscription management
│   ├── ScreenSubscriptionMap         # Screen → topic mapping
│   └── BatchBuffer                   # Buffers for backgrounded apps
│
└── tests/
    ├── test_event_bus.py
    ├── test_ecosystem_connector.py
    ├── test_alert_engine.py
    ├── test_intelligence_publisher.py
    └── test_websocket_router.py
```

### Integration with Existing Modules

**user_profile_engine.py:**
- After `process()`, publish to bus: `profile.updated`, `profile.domain.<domain>` for changed domains, `profile.stage.changed` on transitions
- Initialize AlertEngine and IntelligencePublisher as bus subscribers
- Add `set_event_bus(bus)` method

**emotion_api_server.py:**
- Initialize ProfileEventBus on startup
- Wire bus to profile engine, alert engine, intelligence publisher, WebSocket router
- Register socket.io namespace handlers for `/profile`, `/alerts`, `/intelligence`

**profile_api.py:**
- EcosystemConnector receives `/api/profile/ingest` events and publishes to bus (backward compatible — same endpoint, same auth)
- New endpoint: `POST /api/profile/services/register` for services to register their intelligence webhook URL

## 7. Error Handling & Resilience

### Bus Failures
- In-process, no external dependencies — cannot go down independently
- Subscriber callback exceptions caught and logged; other subscribers still receive event
- Failed subscriber retried once after 1 second

### Outbound Intelligence Delivery
- 3 retries with exponential backoff (1s, 2s, 4s)
- After 3 failures: event dropped, service marked degraded
- After 5 consecutive failures: service deregistered, warning logged
- Service auto-re-registers when it becomes reachable on next inbound event

### WebSocket Failures
- Disconnected clients cleaned up immediately (subscriptions removed)
- Buffer preserved 5 minutes for reconnection
- After 5 minutes, buffer discarded

### Alert Engine
- ReactiveDetector and PredictiveDetector run independently — one failing doesn't block the other
- Prediction errors are swallowed, never converted to false alerts
- Alert delivery failures don't block event processing

### Ecosystem Connector
- Inbound webhooks with bad auth/missing fields return 4xx immediately
- Valid events that fail processing queued in dead letter buffer (max 1000 events, FIFO)
- Dead letter events retried on next processing cycle

## 8. Connection to Neural Workflow AI Engine

This integration connects to all 5 phases:

- **Phase 1 (Foundation):** Event bus extends the SQLite + profile infrastructure with real-time pub/sub
- **Phase 2 (ML Training & Prediction):** Intelligence publisher feeds user baselines back to the emotion classifier for per-user calibration. Predictive alerts use ML-style signature matching.
- **Phase 3 (Backend APIs):** EcosystemConnector receives events from all Backend engines. Intelligence delivery notifies Backend services of personalization updates.
- **Phase 4 (Dashboard Integration):** WebSocket smart streaming pushes real-time updates to 3D fingerprint, DNA strands, evolution timeline. Alert events stream to therapist dashboard.
- **Phase 5 (Real-time Data):** The entire system is real-time — events flow in from biometrics/HealthKit, through the bus, processed by the profile engine, intelligence published, alerts evaluated, and results streamed to the frontend, all in a continuous loop.

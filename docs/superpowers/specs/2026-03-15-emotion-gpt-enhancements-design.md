# Emotion GPT Enhancement Suite — Design Spec

**Date:** 2026-03-15
**Project:** quantara-nanoGPT
**Approach:** Modular Enhancement (Approach A) — each feature is a self-contained module with clean interfaces

## Overview

Six features that elevate the Emotion GPT from a working classifier + API into a production-grade, self-improving emotion AI system with real-time streaming and formal benchmarks.

**Build order:** 1 → 2 → 5 → 6 → 3 → 4 (attention fusion first since transitions and streaming depend on it; training and eval last since they validate everything)

---

## Feature 1: Cross-Modal Attention Fusion

**Problem:** Current `FusionHead` concatenates text + bio + pose embeddings and runs them through dense layers. This treats all modalities equally regardless of context — EDA matters more for anxiety than joy, but the model can't learn that.

**Solution:** `AttentionFusionHead` in `emotion_classifier.py` using multi-head cross-attention.

### Architecture

```
Text (384-dim) → Linear → 128-dim ──┐
Bio  (16-dim)  → Linear → 128-dim ──┼→ MultiHeadCrossAttention(4 heads, text=Q, bio+pose=KV)
Pose (16-dim)  → Linear → 128-dim ──┘         │
                                        Attended (128-dim)
                                               │
                                     ┌─────────┴──────────┐
                              Dense (128 → 64) + ReLU
                                     ┌─────────┴──────────┐
                               Family Head (9)    Emotion Head (32)
```

- Text embedding is the **query**; bio and pose are **key/value**
- 4 attention heads, 128-dim projected space
- When bio/pose are absent, an **attention mask** excludes zero-input positions from the softmax — prevents uniform attention over meaningless vectors. The mask is derived from whether biometric/pose inputs were provided (not from the embedding values themselves).
- Dropout (0.3) on attention weights for regularization

### Interface

Same as existing `FusionHead`:
- `forward(text_embedding, biometric_embedding, pose_embedding)` → `(emotion_probs, family_probs)`
- `classify_with_fallback(...)` → `{emotion, family, confidence, is_fallback, ...}`
- **New field in output:** `modality_weights: {text: float, bio: float, pose: float}` — normalized attention contributions

### Checkpoint compatibility

- Save format includes `version: 2` flag
- `MultimodalEmotionAnalyzer` auto-detects version on load: v1 → `FusionHead`, v2 → `AttentionFusionHead`
- Old checkpoints continue to work without changes

### File changes

| File | Change |
|------|--------|
| `emotion_classifier.py` | Add `AttentionFusionHead` class (~80 lines). Update `MultimodalEmotionAnalyzer.__init__` to use it. Keep `FusionHead` for backward compat. |

---

## Feature 2: Emotion Transition Engine

**Problem:** `TRANSITION_PATHWAYS` is a flat dict of strings — no multi-step pathways, no target emotion routing, no adaptive learning from outcomes.

**Solution:** New `emotion_transition_engine.py` with a directed weighted graph, session tracking, and adaptive weight refinement.

### Architecture

**TransitionGraph:** Directed graph over 32 emotion nodes.
- ~80 curated edges with initial weights, therapy techniques, and estimated durations
- Pathfinding: Dijkstra shortest path from any emotion to any target emotion
- Intermediate nodes represent natural emotional stepping stones (e.g., anxiety → grounded → relief → calm)

**TransitionSession:** Tracks a user through a multi-step pathway.
- Created when a transition is requested
- Each step has: technique, exercise, success criteria, estimated duration (2-10 minutes per step)
- **Success criteria per step type:**
  - Calming steps: HR drops ≥5 bpm from step start OR HRV rises ≥10ms
  - Activation steps: HR rises ≥5 bpm OR EDA rises ≥0.5 µS
  - Cognitive steps: time-based only (complete exercise duration)
  - Default: time-based (if no biometric data available — text-only users advance after exercise duration elapses)
- Advances via biometric criteria, manual advance (`POST /api/neural/transition-session` with `action: advance`), or time-based auto-advance
- Emits events for WebSocket streaming (Feature 5)

**AdaptiveWeightTracker:** SQLite-backed learning from outcomes.
- Logs every transition attempt: user_id, from_emotion, to_emotion, path_taken, outcome (success/partial/abandoned), duration
- Periodically recomputes edge weights: edges with higher success rates get lower weights (preferred by Dijkstra)
- Stored in `data/transitions.db` (WAL mode enabled, 5s busy timeout for concurrent access safety)

### API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/neural/emotion-transition` | POST | Enhanced: accepts `to_emotion`, returns multi-step pathway |
| `/api/neural/transition-session` | POST | Start/advance/query active session |
| `/api/neural/transition-feedback` | POST | Log outcome to feed adaptive weights |

### File changes

| File | Change |
|------|--------|
| `emotion_transition_engine.py` | New file (~400 lines) |
| `emotion_api_server.py` | Import engine, update transition endpoint, add 2 new endpoints (~40 lines) |

---

## Feature 3: Production Model Training (Colab)

**Problem:** Current deployed model is medium (6-layer, 384-dim). The full production config (8-layer, 512-dim, 10K iters) requires GPU.

**Solution:** Colab notebook with Google Drive checkpointing for resilient GPU training.

### Notebook: `notebooks/train_production.ipynb`

| Cell | Purpose |
|------|---------|
| 1. Setup | Clone repo, install deps, detect GPU type (T4/A100), mount Google Drive |
| 2. Data prep | Run `data/quantara_emotion/prepare.py`, verify tokenized `.bin` files |
| 3. Config | Load `config/train_quantara_emotion.py`. Override batch size for VRAM (T4=32, A100=64) |
| 4. Train | Run `train.py` with live loss plotting. Checkpoint every 1000 iters to Drive |
| 5. Evaluate | Validation loss + sample generations per emotion family |
| 6. Export | Download `ckpt.pt`, save to Drive |
| 7. Retrain FusionHead | Train `AttentionFusionHead` on new 512-dim embeddings against emotion dataset |

### Key details

- Drive checkpointing survives Colab disconnects — resume from last checkpoint
- Cell 7 is critical: production GPT changes embedding dim (384 → 512), so the fusion head must be retrained
- Estimated: ~2-3h on T4 free tier, ~45min on A100 Pro
- Deploying the production checkpoint (512-dim) requires updating the `MultimodalEmotionAnalyzer` dimension auto-detection logic — already handled by the v1/v2 checkpoint compat in Feature 1, since the `AttentionFusionHead` reads `text_dim` from checkpoint metadata

### File changes

| File | Change |
|------|--------|
| `notebooks/train_production.ipynb` | New file |

---

## Feature 4: Formal Evaluation

**Problem:** No standardized benchmarks — can't measure classifier quality or track regressions.

**Solution:** `evaluate.py` script benchmarking against GoEmotions, SemEval, and held-out data.

### Datasets

| Dataset | Source | Labels | Mapping |
|---------|--------|--------|---------|
| GoEmotions | HuggingFace `google-research-datasets/goemotions` | 27 emotions | 22 direct, 5 merged (annoyance→frustration, desire→love, nervousness→anxiety, curiosity→enthusiasm, realization→mindfulness) |
| SemEval 2018 Task 1 | HuggingFace `sem_eval_2018_task_1` | 11 emotions | All direct mappings |
| Held-out | 20% of `data/quantara_emotion/` training data | 32 emotions | Native taxonomy |

### Metrics

- **Per-emotion:** precision, recall, F1
- **Per-family:** accuracy, macro-F1
- **Overall:** weighted F1, accuracy
- **Fusion lift:** F1 delta between text-only and text+biometrics (held-out only)
- **Confusion matrix:** heatmap PNG

### Output

- `results/eval_YYYY-MM-DD.json` — full metrics report
- `results/confusion_matrix.png` — visualization
- Console summary table

### Usage

```bash
python evaluate.py --datasets all --output results/
python evaluate.py --datasets goemotions --text-only
python evaluate.py --datasets held-out --with-biometrics
```

### File changes

| File | Change |
|------|--------|
| `evaluate.py` | New file (~300 lines) |
| `requirements-dev.txt` | Add `datasets` (HuggingFace) — dev-only, not in production Docker image |

---

## Feature 5: Real-time WebSocket Streaming

**Problem:** Consumers must poll REST endpoints for emotion state. No live streaming for dashboards or real-time workflow triggers.

**Solution:** Socket.io server mounted on the existing Flask app via `flask-socketio`.

### Namespaces and events

| Namespace | Event | Payload | Trigger |
|-----------|-------|---------|---------|
| `/emotion` | `emotion_update` | `{emotion, family, confidence, scores, modality_weights, timestamp}` | Every `analyze()` call |
| `/emotion` | `transition_step` | `{session_id, step, technique, from_emotion, to_emotion}` | Transition session advance |
| `/biometrics` | `biometric_stream` | `{heart_rate, hrv, eda, breathing_rate, source}` | RuView data at ~1Hz |
| `/system` | `model_retrained` | `{trigger, samples_used, mae_before, mae_after, timestamp}` | Auto-retrain completes (stub namespace — populated by Feature 6) |

### Authentication

- WebSocket clients pass a `token` query parameter on connect (e.g., `io.connect('/emotion?token=xxx')`)
- `on_connect` handler validates the token (same JWT validation as REST middleware)
- Unauthenticated connections are rejected with `ConnectionRefusedError`

### Room-based routing

- Clients join rooms by `user_id` or `session_id`
- Dashboard clients can subscribe to specific emotion families (e.g., only Fear family alerts)

### Server integration

- `flask-socketio` mounts on the Flask app — same port, no separate process
- Gunicorn switches to `eventlet` worker class with **`--workers 1`** (required — flask-socketio with eventlet cannot use multiple workers since WebSocket state is not shared across processes)
- `emotion_websocket.py` exports `init_websocket(app)` called from `create_app()`
- Hook into `analyze()`: after classification, call `emit_emotion_update(result)`

### Consumers

- **Frontend dashboards:** connect to `/emotion`, render live emotion state + modality weight visualization
- **Quantara-Backend:** connect to `/emotion`, listen for `transition_step` to trigger Neural Workflow AI Engine state changes

### File changes

| File | Change |
|------|--------|
| `emotion_websocket.py` | New file (~200 lines) |
| `emotion_api_server.py` | Import and mount socketio in `create_app()` (~15 lines) |
| `requirements.txt` | Add `flask-socketio`, `eventlet` |
| `Dockerfile` | Change gunicorn to: `--worker-class eventlet --workers 1` |

---

## Feature 6: Auto-Retraining Pipeline

**Problem:** WiFi calibration model is static after deployment. Per-user calibration buffer collects data but never triggers retraining. Model accuracy degrades as conditions change.

**Solution:** Background auto-retraining with dual triggers (threshold + drift) and safety guardrails.

### Components

**DriftDetector:**
- Rolling window of last 200 prediction errors (predicted vs actual HRV/EDA when ground truth is available)
- Kolmogorov-Smirnov test comparing recent errors vs baseline distribution
- Triggers retrain when p-value < 0.05 (significant drift detected)
- Cooldown: 1 hour minimum between retrains (extended to 4h after failed retrain)

**ThresholdMonitor:**
- Watches `PersonalCalibrationBuffer` sample count per user
- Initial retrain at 20 paired samples (first personalization — matches existing buffer threshold)
- Subsequent retrains every 50 new samples
- `PersonalCalibrationBuffer.MAX_BUFFER_SIZE` increased from 100 → 500 to support drift detection and retrain history
- ThresholdMonitor tracks a cumulative `total_samples_seen` counter (separate from buffer size) to correctly count "every 50 new samples"

**RetrainWorker (background thread):**
1. Receives retrain request (drift or threshold trigger)
2. Copies current model weights
3. Fine-tunes on accumulated paired data (learning rate = 1e-4, max 50 epochs, early stopping)
4. **Validation gate:** new model must have lower MAE than current on 20% held-out split
5. If worse → discard, log warning, extend cooldown
6. If better → hot-swap weights (thread-safe with `threading.Lock`)
7. Save checkpoint to `checkpoints/ruview_calibration.pt`
8. Emit `model_retrained` event on WebSocket `/system` namespace

**RetrainLog (SQLite in `data/retrain_log.db`, WAL mode, 5s busy timeout):**
- Tracks: timestamp, trigger type, samples_used, mae_before, mae_after, outcome (applied/discarded)

### API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/calibration/retrain-status` | GET | Current drift score, last retrain time, buffer sizes |
| `/api/calibration/retrain` | POST | Manual retrain trigger |
| `/api/calibration/retrain-log` | GET | History of all retrains with metrics |

### File changes

| File | Change |
|------|--------|
| `auto_retrain.py` | New file (~300 lines) |
| `wifi_calibration.py` | Add `get_drift_score()` to `PersonalCalibrationBuffer` (~20 lines) |
| `emotion_api_server.py` | Import auto_retrain, start monitor, add 3 endpoints (~30 lines) |

---

## Build Order

```
1. AttentionFusionHead  (foundation — other features emit its modality_weights)
    │
2. EmotionTransitionEngine  (depends on classifier output)
    │
3. WebSocket Streaming  (wires up events from 1 + 2)
    │
4. Auto-Retraining  (emits events via WebSocket from 3)
    │
5. Production Training (Colab)  (trains everything built in 1-4)
    │
6. Formal Evaluation  (validates the production model from 5)
```

## New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `flask-socketio` | >=5.3 | WebSocket server |
| `eventlet` | >=0.35 | Async worker for gunicorn |
| `datasets` | >=2.14 | HuggingFace dataset loading for eval |

## New Files

| File | Lines (est.) | Feature |
|------|-------------|---------|
| `emotion_transition_engine.py` | ~400 | Feature 2 |
| `emotion_websocket.py` | ~200 | Feature 5 |
| `auto_retrain.py` | ~300 | Feature 6 |
| `evaluate.py` | ~300 | Feature 4 |
| `notebooks/train_production.ipynb` | — | Feature 3 |

## Modified Files

| File | Features | Summary |
|------|----------|---------|
| `emotion_classifier.py` | 1 | Add `AttentionFusionHead`, update analyzer init |
| `emotion_api_server.py` | 2, 5, 6 | Mount WebSocket, add transition + retrain endpoints |
| `wifi_calibration.py` | 6 | Add drift scoring method, increase buffer size to 500 |
| `requirements.txt` | 5 | Add flask-socketio, eventlet |
| `requirements-dev.txt` | 4 | Add datasets (HuggingFace) — dev only |
| `Dockerfile` | 5 | Switch to eventlet worker with 1 worker |

## Rollback Strategy

If WebSocket/eventlet integration causes production issues:
- Set env var `DISABLE_WEBSOCKET=1` — `create_app()` skips `init_websocket()` and the Dockerfile falls back to sync gunicorn (`--workers 2` without `--worker-class eventlet`)
- This env var is checked at startup, not per-request

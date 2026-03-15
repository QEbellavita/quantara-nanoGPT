# Emotion Pipeline Completion Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the emotion training pipeline by benchmarking the newly trained model, integrating wearable biometric data, testing the transition engine, connecting to Quantara Backend, and committing all work.

**Architecture:** The pipeline flows: external datasets (prepare scripts) -> training (train_emotion_classifier.py) -> checkpoint (emotion_fusion_head.pt) -> benchmark validation -> API server (emotion_api_server.py) -> Quantara Backend integration. The transition engine provides graph-based emotion pathfinding for therapy workflows.

**Tech Stack:** Python, PyTorch, Flask, sentence-transformers, HuggingFace datasets, SQLite, pytest

---

## Chunk 1: Benchmark & Wearable Data

### Task 1: Run Benchmark on Newly Trained Model

**Files:**
- Existing: `benchmark_emotion.py`
- Existing: `checkpoints/emotion_fusion_head.pt`

- [ ] **Step 1: Run the benchmark suite against the new checkpoint**

```bash
python benchmark_emotion.py
```

Expected: Per-emotion accuracy table, family-level accuracy, edge-case results. Note any emotions with < 70% accuracy — these are candidates for targeted data augmentation.

- [ ] **Step 2: Run benchmark with dair-ai/emotion validation set**

```bash
python benchmark_emotion.py --validate --val-size 2000
```

Expected: Validation accuracy on external dataset. Compare against the 92.9% training val_acc.

- [ ] **Step 3: Record results**

Save benchmark output to a results file for tracking:

```bash
python benchmark_emotion.py --validate --val-size 2000 2>&1 | tee benchmark_results_2026-03-15.txt
```

---

### Task 2: Run Wearable HuggingFace Dataset Preparation

**Files:**
- Existing: `data/external_datasets/prepare_wearable_hf.py`
- Output: `data/external_datasets/wearable_emotion_data.csv`

- [ ] **Step 1: Install datasets library if needed**

```bash
pip install datasets
```

- [ ] **Step 2: Run the wearable preparation script**

```bash
python data/external_datasets/prepare_wearable_hf.py
```

Expected: Downloads oscarzhang/Wearable_TimeSeries_HealthRecommendation_Dataset (87 samples) and infinite-dataset-hub/BiometricHealthTrends (100 samples), processes them, outputs `wearable_emotion_data.csv` with ~274 rows of real biometric data.

- [ ] **Step 3: Verify output quality**

```bash
head -5 data/external_datasets/wearable_emotion_data.csv
wc -l data/external_datasets/wearable_emotion_data.csv
```

Expected: CSV with columns `text,emotion,hr,hrv,eda`. HR values should include real wearable readings (not just synthetic ranges).

- [ ] **Step 4: Combine with existing training data**

```bash
tail -n +2 data/external_datasets/wearable_emotion_data.csv >> data/external_datasets/external_emotion_data.csv
```

Note: `tail -n +2` skips the header row to avoid duplicate headers in the combined CSV.

---

## Chunk 2: Transition Engine Testing

### Task 3: Run Transition Engine Tests

**Files:**
- Existing: `tests/test_transition_engine.py`
- Existing: `emotion_transition_engine.py`

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest tests/test_transition_engine.py -v
```

Expected: 16 tests pass. Tests cover:
- TransitionGraph: 32 emotions reachable, 160+ edges, Dijkstra correctness
- TransitionSession: step tracking, biometric validation, completion
- AdaptiveWeightTracker: SQLite persistence, weight calculation
- EmotionTransitionEngine: session lifecycle, graph rebuilding

- [ ] **Step 2: Fix any failing tests**

If failures occur, analyze the error and fix. Common issues:
- Import errors: ensure `emotion_transition_engine.py` is importable from project root
- SQLite temp file issues: tests use temp directories, verify cleanup

- [ ] **Step 3: Run with coverage**

```bash
python -m pytest tests/test_transition_engine.py -v --cov=emotion_transition_engine --cov-report=term-missing
```

Expected: >80% coverage on emotion_transition_engine.py. Note any uncovered paths.

---

## Chunk 3: API Server Verification

### Task 4: Verify API Server Loads New Model

**Files:**
- Existing: `emotion_api_server.py`
- Existing: `checkpoints/emotion_fusion_head.pt`

- [ ] **Step 1: Start the API server**

```bash
python emotion_api_server.py &
```

Expected: Server starts on port 5000 (or configured port). Should log:
- Model checkpoint loaded
- MultimodalEmotionAnalyzer initialized
- Encoder type detected (sentence-transformer or go-emotions)

- [ ] **Step 2: Test the health endpoint**

```bash
curl http://localhost:5000/api/emotion/status
```

Expected: `{"status": "ok", ...}` with model info.

- [ ] **Step 3: Test emotion analysis with the new model**

```bash
curl -X POST http://localhost:5000/api/emotion/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so grateful for everything you have done"}'
```

Expected: `{"emotion": "gratitude", "family": "Joy", "confidence": >0.6, ...}`

- [ ] **Step 4: Test emotion transition endpoint**

```bash
curl -X POST http://localhost:5000/api/neural/emotion-transition \
  -H "Content-Type: application/json" \
  -d '{"from_emotion": "anxiety", "to_emotion": "calm"}'
```

Expected: Multi-step pathway with exercises and duration estimates.

- [ ] **Step 5: Test therapy endpoint**

```bash
curl -X POST http://localhost:5000/api/emotion/therapy \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel overwhelmed with everything going on"}'
```

Expected: Therapy technique recommendation with exercise.

- [ ] **Step 6: Stop the API server**

```bash
kill %1
```

---

## Chunk 4: Backend Connection

### Task 5: Connect Updated Checkpoint to Quantara Backend

**Files:**
- Existing: `checkpoints/emotion_fusion_head.pt`
- Target: `/Users/bel/Quantara-Backend/` (Backend server)

- [ ] **Step 1: Check if Backend references this checkpoint**

```bash
grep -r "emotion_fusion_head" /Users/bel/Quantara-Backend/ --include="*.js" --include="*.py" -l
grep -r "nanoGPT\|quantara-nanoGPT" /Users/bel/Quantara-Backend/ --include="*.js" --include="*.py" -l
```

Expected: Find configuration files or engine modules that reference the nanoGPT checkpoint path. This determines if the Backend pulls the model directly or if it's served via the API server.

- [ ] **Step 2: Determine integration path**

Two possible integration patterns:
- **Direct**: Backend loads checkpoint directly → copy checkpoint to Backend's model directory
- **API**: Backend calls nanoGPT API server → verify API server URL in Backend config

Check Backend emotion engine configuration:

```bash
grep -r "emotion" /Users/bel/Quantara-Backend/engines/ --include="*.js" -l | head -10
```

- [ ] **Step 3: If Direct — copy updated checkpoint**

```bash
# Only if Backend loads checkpoint directly
cp checkpoints/emotion_fusion_head.pt /Users/bel/Quantara-Backend/models/emotion_fusion_head.pt
```

- [ ] **Step 4: If API — verify Backend API config points to nanoGPT server**

```bash
# Only if Backend calls the API
grep -r "5000\|emotion-api\|nanoGPT" /Users/bel/Quantara-Backend/ --include="*.js" --include="*.env" -l
```

Ensure the Backend's emotion engine is configured to call the correct API URL.

- [ ] **Step 5: Verify end-to-end**

Start both servers and test the full pipeline:

```bash
# Terminal 1: nanoGPT API
python emotion_api_server.py

# Terminal 2: Quantara Backend
cd /Users/bel/Quantara-Backend && npm run dev

# Terminal 3: Test
curl http://localhost:3000/api/health  # Backend health
curl -X POST http://localhost:5000/api/emotion/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel anxious about the upcoming deadline"}'
```

---

## Chunk 5: Commit

### Task 6: Commit All Work

**Files:**
- New: `data/external_datasets/prepare_wearable_hf.py`
- New: `data/external_datasets/prepare_chinese_emotion.py`
- New: `emotion_transition_engine.py`
- New: `tests/test_transition_engine.py`
- New: `config/train_quantara_emotion_medium.py`
- New: `benchmark_emotion.py`
- Modified: `data/external_datasets/prepare.py`
- Modified: `emotion_classifier.py`
- Modified: `train_emotion_classifier.py`
- Modified: `emotion_api_server.py`
- Modified: `checkpoints/emotion_fusion_head.pt`

- [ ] **Step 1: Review all changes**

```bash
git status
git diff --stat
```

- [ ] **Step 2: Stage new files**

```bash
git add data/external_datasets/prepare_wearable_hf.py \
        data/external_datasets/prepare_chinese_emotion.py \
        emotion_transition_engine.py \
        tests/test_transition_engine.py \
        config/train_quantara_emotion_medium.py \
        benchmark_emotion.py
```

- [ ] **Step 3: Stage modified files**

```bash
git add data/external_datasets/prepare.py \
        emotion_classifier.py \
        train_emotion_classifier.py \
        emotion_api_server.py \
        checkpoints/emotion_fusion_head.pt
```

- [ ] **Step 4: Verify nothing sensitive is staged**

```bash
git diff --cached --name-only
```

Ensure no `.env`, credentials, or large data files (CSVs, logs) are included.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add Chinese/wearable datasets, transition engine, and retrained emotion model

- Add prepare_wearable_hf.py (HuggingFace wearable biometric datasets with real HR/HRV)
- Add prepare_chinese_emotion.py (NLPCC, ChnSentiCorp, multilingual sentiment datasets)
- Add emotion_transition_engine.py (Dijkstra pathfinding, 160+ curated edges, adaptive weights)
- Add benchmark_emotion.py (edge-case + dair-ai/emotion validation)
- Retrain emotion_fusion_head.pt on combined dataset (92.9% val accuracy)
- Add transition engine test suite (16 tests)
- Add medium training config

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Verify commit**

```bash
git log --oneline -3
git status
```

Expected: Clean working tree (except data CSVs, logs, .coverage which are gitignored or intentionally untracked).

# Training TODO

> Last updated: **2026-03-16 ~7:36am** — 4 active training jobs, 1 completed

---

## Currently Running Jobs

| # | PID | Command | Started | Elapsed | MEM | Purpose | Status |
|---|-----|---------|---------|---------|-----|---------|--------|
| 1 | 23114 | `python train_emotion_classifier.py --use-sentence-transformer --external-data data/external_datasets/augmented_emotion_data.csv --epochs 20 --lr 0.0005` | Mar 16 ~7:05am | 31 min | 1.9% | Sentence-transformer + augmented data (69K rows) | **Running** |
| 2 | 54628 | `python train_emotion_classifier.py --use-go-emotions --use-hf-datasets --epochs 20 --batch-size 64 --device cpu` | Mar 15 ~6pm | **13.5 hrs** | 3.3% | GoEmotions + HuggingFace datasets (211K+ rows) | **Running** |
| 3 | 72566 | `python3 train.py config/train_quantara_emotion_medium.py` | Mar 15 ~2pm | **17.7 hrs** | 24.2% | nanoGPT medium emotion model (6L/6H/384d, 5K iters) | **Running** |
| 5 | 24303 | `python train_emotion_classifier.py --use-sentence-transformer --epochs 20` | Mar 16 ~7:18am | 17 min | 1.5% | Base classifier with pose features (FusionHead 416-dim) | **Running** |

### How to Check Status

```bash
# Check all training processes
ps aux | grep -E "train|python" | grep -v grep

# Check specific PID
ps -p 23114 -o pid,etime,%cpu,%mem,command
ps -p 54628 -o pid,etime,%cpu,%mem,command
ps -p 72566 -o pid,etime,%cpu,%mem,command

# Check nanoGPT checkpoint progress (updates every 500 iters)
ls -lh out-quantara-emotion-medium/ckpt.pt
# Current: 344M, last updated Mar 15 5:20am

# Check emotion classifier checkpoint
ls -lh checkpoints/emotion_fusion_head.pt
# Current: 271K, last updated Mar 16 2:48am
```

### Completed

| PID | Command | Result | Checkpoint |
|-----|---------|--------|------------|
| 24300 | `python train_calibration_dreamer.py --epochs 300` | **HRV MAE: 31.2 ms, EDA MAE: 1.6 µS** — 8125 samples (38 subjects), 300 epochs | `checkpoints/ruview_calibration.pt` (5.4K) |

### Dead

| PID | Command | Status | Checkpoint? |
|-----|---------|--------|-------------|
| 93208 | `python train_emotion_classifier.py --use-sentence-transformer --external-data data/external_datasets/augmented_emotion_data.csv --epochs 20 --lr 0.0005` | **Dead** — replaced by PID 23114 | Yes — `checkpoints/emotion_fusion_head.pt` (271K) |

---

## Restart Commands

If any job dies or you need to re-run, copy-paste these:

```bash
cd /Users/bel/quantara-nanoGPT

# ── Training 1: Sentence-transformer + augmented emotion data ──
python train_emotion_classifier.py \
  --use-sentence-transformer \
  --external-data data/external_datasets/augmented_emotion_data.csv \
  --epochs 20 --lr 0.0005

# ── Training 2: GoEmotions + HuggingFace datasets (long-running) ──
python train_emotion_classifier.py \
  --use-go-emotions --use-hf-datasets \
  --epochs 20 --batch-size 64 --device cpu

# ── Training 3: nanoGPT medium emotion model ──
python3 train.py config/train_quantara_emotion_medium.py

# ── Base classifier with pose features only (no external data) ──
# Trains FusionHead (416-dim) with PoseEncoder
python train_emotion_classifier.py --use-sentence-transformer --epochs 20

# ── Classifier with raw external emotion data ──
python train_emotion_classifier.py \
  --use-sentence-transformer \
  --external-data data/external_datasets/external_emotion_data.csv \
  --epochs 20
```

---

## Full Dataset Combining Steps

### Step 1: Combine all external CSV sources into one master file

All CSVs in `data/external_datasets/` share the same schema: `text,emotion,hr,hrv,eda`

| File | Rows | Source |
|------|------|--------|
| `all_combined_emotion_data.csv` | 198,711 | **Master combined** (already built) |
| `wearable_stress_emotion_data.csv` | 100,000 | HF wearable/stress datasets |
| `augmented_emotion_data.csv` | 69,242 | Augmented from hard cases |
| `external_emotion_data.csv` | 42,878 | External emotion texts |
| `chinese_emotion_data.csv` | 25,591 | Chinese emotion data |
| `setfit_emotion_data.csv` | 20,000 | SetFit emotion embeddings |
| `audio_emotion_data.csv` | 10,242 | Audio emotion features |
| `wearable_emotion_data.csv` | 274 | Wearable biometric emotions |

**Prep scripts** (already in `data/external_datasets/`):
```bash
# These generated the CSVs above:
python data/external_datasets/prepare.py                  # → external_emotion_data.csv
python data/external_datasets/prepare_audio.py            # → audio_emotion_data.csv
python data/external_datasets/prepare_chinese_emotion.py  # → chinese_emotion_data.csv
python data/external_datasets/prepare_setfit.py           # → setfit_emotion_data.csv
python data/external_datasets/prepare_wearable_hf.py      # → wearable_stress_emotion_data.csv

# Augmentation / balancing:
python data/external_datasets/augment_balanced.py         # → all_combined_emotion_data.csv
python data/external_datasets/augment_hard_cases.py       # → augmented_emotion_data.csv
```

### Step 2: Rebuild nanoGPT tokenized dataset (if CSVs changed)

```bash
cd /Users/bel/quantara-nanoGPT
python data/quantara_emotion/prepare.py
# Outputs: data/quantara_emotion/train.bin (6.5M) + val.bin (740K)
```

### Step 3: Retrain calibration model (biometrics)

```bash
# WESAD + DREAMER combined: 8125 windows from 38 subjects
# Checkpoint: checkpoints/ruview_calibration.pt (5.4K)
python train_calibration_dreamer.py --epochs 300
```

### Step 4: Full retrain — emotion classifier with ALL sources

```bash
# This is the big one — combines everything:
#   - Sentence-transformer embeddings (384-dim)
#   - Augmented emotion data (69K rows)
#   - GoEmotions dataset (211K rows from HF)
#   - Pose features (from pose_encoder.py, 32-dim)
#   - WESAD-calibrated biometrics
python train_emotion_classifier.py \
  --use-sentence-transformer \
  --use-go-emotions --use-hf-datasets \
  --external-data data/external_datasets/augmented_emotion_data.csv \
  --epochs 30 --lr 0.0003

# Alternative: use the master combined CSV instead of augmented-only:
python train_emotion_classifier.py \
  --use-sentence-transformer \
  --use-go-emotions --use-hf-datasets \
  --external-data data/external_datasets/all_combined_emotion_data.csv \
  --epochs 30 --lr 0.0003
```

### Step 5: Real-device calibration (future — requires WiFi-paired data)

```bash
python calibration_collector.py retrain --data-dir calibration_data/
```

---

## Commit & Deploy After Training

```bash
cd /Users/bel/quantara-nanoGPT

# Commit updated checkpoints
git add -f checkpoints/emotion_fusion_head.pt checkpoints/ruview_calibration.pt
git commit -m "feat: retrained emotion classifier and calibration with full datasets"
git push origin master

# If nanoGPT checkpoint also updated:
git add -f out-quantara-emotion-medium/ckpt.pt
git commit -m "feat: nanoGPT medium emotion model checkpoint"
git push origin master
```

---

## Autoresearch Results (quantara-autoresearch/emotion)

### Experiment History — text_emotion (13 classes, 40K rows)

| Exp | Accuracy | F1 | Time | Method |
|-----|----------|-----|------|--------|
| 1 | 20.6% | 0.220 | 515s | Baseline RF + TF-IDF |
| 4 | 28.1% | 0.274 | 368s | ExtraTrees n=200 |
| 5 | 32.3% | 0.294 | 2661s | ExtraTrees n=300 |
| 9 | 33.0% | 0.306 | 4237s | Voting RF300+ET300 |
| 13 | 33.1% | 0.307 | 6238s | Voting RF400+ET400 |
| 15 | 34.5% | 0.325 | 29188s | Voting RF500+ET500+LGBM300+XGB300 |
| 16 | 35.0% | 0.320 | 12704s | Voting RF500+ET500+LGBM500+XGB500 |
| 17 | 35.5% | 0.307 | 7594s | Voting + MiniLM-L6 384d embeddings |
| **18** | **35.7%** | **0.307** | **12149s** | **Voting + MPNet-base 768d embeddings** |

**Conclusion:** Plateaued at ~35.7%. 13-class Twitter emotion with extreme class imbalance (22-8638 samples per class) limits accuracy. Embeddings helped marginally over TF-IDF.

### Multi-Dataset Results (Best Models)

| Dataset | Classes | Samples | Accuracy | F1 weighted | Model Path |
|---------|---------|---------|----------|-------------|------------|
| **speech_emotion** | 8 | 85K | **88.6%** | 0.886 | `autoresearch_speech_emotion_voting_20260315_093758.pkl` |
| **emotion_labels** | 4 | 7K | **73.0%** | 0.728 | `autoresearch_emotion_labels_voting_20260315_102224.pkl` |
| **text_emotion_6** | 6 | 100K | **68.6%** | 0.672 | `autoresearch_text_emotion_6_voting_20260316_053820.pkl` |
| **goemotions** | 13 | 24K | **51.9%** | 0.504 | `autoresearch_goemotions_voting_20260315_124249.pkl` |
| **text_emotion** | 13 | 40K | 35.7% | 0.307 | `autoresearch_text_emotion_voting_embeddings_20260315_020705.pkl` |

All models saved to: `/Users/bel/Quantara-Frontend/models/`
Uploading to HuggingFace: `belindaswitzer/quantara-emotion-models` (PID 23490, in progress)

### Autoresearch Restart Commands

```bash
cd /Users/bel/quantara-autoresearch/emotion

# Re-run any dataset:
python3 -u train.py speech_emotion 2>&1 | tee run_speech.log
python3 -u train.py emotion_labels 2>&1 | tee run_emotion_labels.log
python3 -u train.py text_emotion_6 2>&1 | tee run_text6.log
python3 -u train.py goemotions 2>&1 | tee run_goemotions.log
python3 -u train.py text_emotion 2>&1 | tee run.log

# Config in train.py: MODEL_TYPE, USE_EMBEDDINGS, EMBEDDING_MODEL, estimator counts
# Results tracked in: results.tsv
```

---

## Complete Dataset Inventory

### Biometric / Physiological Datasets

| Dataset | Location | Records | Signals | Schema | Used By |
|---------|----------|---------|---------|--------|---------|
| WESAD | `Quantara-Backend/ml-pipeline/data/wesad/` | 2,996 windows (15 subjects) | ECG→HR/HRV, Resp→BR, EDA, ACC | MAT/pkl | `train_calibration_wesad.py` |
| DREAMER | `data/dreamer/DREAMER.mat` (432M) | 5,129 windows (23 subjects) | ECG→HR/HRV, valence/arousal labels | MAT | `train_calibration_dreamer.py` |
| HR Emotion | `~/Downloads/heart_rate_emotion_dataset.csv` (1M) | 100K readings | HR + emotion label | CSV | `train_emotion_classifier.py` |
| Stress Level v1 | `~/Downloads/Stress_Level_v1.csv` | ~30 subjects | Stress scores | CSV | `train_emotion_classifier.py` |
| Stress Level v2 | `~/Downloads/Stress_Level_v2.csv` | ~30 subjects | Stress scores | CSV | `train_emotion_classifier.py` |

### Text Emotion Datasets

| Dataset | Location | Records | Classes | Schema | Used By |
|---------|----------|---------|---------|--------|---------|
| GoEmotions (full) | `~/Downloads/go_emotions_dataset.csv` (28M) | 211K rows | 28 multi-label | text + multi-hot | `train_emotion_classifier.py --use-go-emotions` |
| GoEmotions (extracted) | `~/Downloads/goemotions_extracted.csv` (1.9M) | 24K rows | 13 Quantara-mapped | text + emotion | autoresearch `goemotions` |
| Tweet Emotions | `~/Downloads/text_emotion.csv` → symlink to `Quantara-Frontend/ml-training/datasets/` | 40K | 13 sentiments | text + emotion | autoresearch `text_emotion` |
| Text Emotion 6 | `~/Downloads/text.csv` (43M) | 417K (100K sampled) | 6 emotions | text + emotion | autoresearch `text_emotion_6` |
| Emotion Labels | `~/Downloads/archive (2) 2/emotion-labels-*.csv` | 7K (train+val+test) | 4 emotions | text + emotion | autoresearch `emotion_labels` |

### Audio Emotion Datasets

| Dataset | Location | Records | Features | Used By |
|---------|----------|---------|----------|---------|
| Speech Emotion (F) | `~/Downloads/archive (3) 2/Female_features.csv` (52M) | 49K | 58 audio features + 8 emotions | autoresearch `speech_emotion` |
| Speech Emotion (M) | `~/Downloads/archive (3) 2/Male_features.csv` (38M) | 36K | 58 audio features + 8 emotions | autoresearch `speech_emotion` |

### Prepared / Combined Datasets (data/external_datasets/)

All share schema: `text,emotion,hr,hrv,eda`

| File | Rows | Size | Source / How Built |
|------|------|------|--------------------|
| **all_combined_emotion_data.csv** | **198,711** | **25M** | Master merge of all below (`augment_balanced.py`) |
| wearable_stress_emotion_data.csv | 100,000 | 11M | `prepare_wearable_hf.py` — HuggingFace wearable/stress |
| augmented_emotion_data.csv | 69,242 | 9.4M | `augment_hard_cases.py` — hard-case augmentation |
| external_emotion_data.csv | 42,878 | 6.9M | `prepare.py` — external text emotions |
| chinese_emotion_data.csv | 25,591 | 4.4M | `prepare_chinese_emotion.py` — translated Chinese emotion corpus |
| setfit_emotion_data.csv | 20,000 | 2.4M | `prepare_setfit.py` — SetFit few-shot emotion |
| audio_emotion_data.csv | 10,242 | 890K | `prepare_audio.py` — audio emotion features as text |
| wearable_emotion_data.csv | 274 | 31K | `prepare_wearable_hf.py` — small wearable set |

### nanoGPT Tokenized Dataset

| File | Size | Built By |
|------|------|----------|
| `data/quantara_emotion/train.bin` | 6.5M | `data/quantara_emotion/prepare.py` |
| `data/quantara_emotion/val.bin` | 740K | `data/quantara_emotion/prepare.py` |

---

## Existing Trained Models

### Checkpoints (this repo)

| File | Size | Updated | Source |
|------|------|---------|--------|
| `checkpoints/emotion_fusion_head.pt` | 271K | Mar 16 2:48am | `train_emotion_classifier.py` |
| `checkpoints/ruview_calibration.pt` | 5.4K | Mar 15 4:50pm | `train_calibration_dreamer.py` |
| `out-quantara-emotion-fast/ckpt.pt` | 184M | Mar 14 9:10pm | nanoGPT fast config |
| `out-quantara-emotion-medium/ckpt.pt` | 344M | Mar 15 5:20am | nanoGPT medium config (still training) |

### Autoresearch Models (Quantara-Frontend/models/)

| Model | Size | Accuracy | Classes |
|-------|------|----------|---------|
| `autoresearch_text_emotion_6_voting_20260316_053820.pkl` | 3.1G | 68.6% | 6 |
| `autoresearch_speech_emotion_voting_20260315_093758.pkl` | 2.5G | 88.6% | 8 |
| `autoresearch_text_emotion_voting_embeddings_20260315_020705.pkl` | 2.5G | 35.7% | 13 |
| `autoresearch_goemotions_voting_20260315_124249.pkl` | 1.6G | 51.9% | 13 |
| `autoresearch_speech_emotion_voting_20260315_083002.pkl` | 665M | — | 8 (earlier run) |
| `autoresearch_emotion_labels_voting_20260315_102224.pkl` | 198M | 73.0% | 4 |

### Pre-existing Models (Quantara-Frontend/models/)

| Model | Size | Domain |
|-------|------|--------|
| `sleep_stage_model.pkl` | 39M | Sleep staging |
| `swell_eda_stress_model.pkl` | 17M | EDA stress detection |
| `wisdm_activity_model.pkl` | 13M | Activity recognition |
| `hrv_stress_model.pkl` | 4.6M | HRV stress |
| `bci_cognitive_model.pkl` | 2.8M | BCI cognitive load |
| `wesad_stress_model.pkl` | 2.1M | WESAD stress |
| `archive1_eeg_model.pkl` | 1.7M | EEG patterns |
| `dreamer_emotion_model.pkl` | 1.4M | DREAMER emotions |
| `mental_health_risk_model.pkl` | 1.4M | Mental health risk |
| `hr_emotion_model.pkl` | 936K | HR-based emotion |
| `text_emotion_model.pkl` | 921K | Text emotion |
| `amigos_emotion_model.pkl` | 517K | AMIGOS emotions |
| `archive8_motion_model.pkl` | 38K | Motion patterns |

---

## Training Architecture Summary

```
32-Emotion Taxonomy (Quantara)
├── FusionHead (416-dim input)
│   ├── SentenceTransformer embeddings (384-dim) — MiniLM-L6-v2
│   ├── BiometricEncoder (HR, HRV, EDA → 32-dim)
│   └── PoseEncoder (17 keypoints → 32-dim)
│
├── Two-stage hierarchical classification:
│   ├── Stage 1: Family classifier (9 families)
│   └── Stage 2: Sub-emotion classifier (32 emotions)
│
├── Calibration layer (RuView):
│   ├── WESAD (15 subjects, ECG/EDA/ACC)
│   └── DREAMER (23 subjects, ECG/valence/arousal)
│
└── nanoGPT emotion model (character-level, alternative to sentence-transformer):
    └── Medium: 6 layers, 6 heads, 384 embed, 5000 iters
```

### Emotion Families (9) → Sub-emotions (32)

| Family | Sub-emotions |
|--------|-------------|
| Joy | joy, excitement, enthusiasm, fun, gratitude, pride |
| Sadness | sadness, grief, boredom, nostalgia |
| Anger | anger, frustration, hate, contempt, disgust, jealousy |
| Fear | fear, anxiety, nervousness, panic |
| Surprise | surprise, shock, wonder |
| Love | love, caring, tenderness |
| Calm | calm, contentment, relief |
| Curiosity | curiosity, interest |
| Confusion | confusion, uncertainty |

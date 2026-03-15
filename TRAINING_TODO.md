# Training TODO

## Currently Running

| PID | Command | Started | Purpose | Status |
|-----|---------|---------|---------|--------|
| 54628 | `python train_emotion_classifier.py --use-go-emotions --use-hf-datasets --epochs 20 --batch-size 64 --device cpu` | Mar 15 ~1:30pm | Emotion classifier with GoEmotions dataset | Running (1576 min CPU, 4.2% MEM) |
| 72566 | `python3 train.py config/train_quantara_emotion_medium.py` | Mar 15 | nanoGPT medium emotion model | Running (321% CPU, 11.7% MEM) |

### Dead / Completed

| PID | Command | Status |
|-----|---------|--------|
| 93208 | `python train_emotion_classifier.py --use-sentence-transformer --external-data data/external_datasets/augmented_emotion_data.csv --epochs 20 --lr 0.0005` | Dead — check for checkpoint |

## Restart After Completion

These were killed as duplicates — restart if needed:

```bash
# Base emotion classifier with pose features (no external data)
# This is the one that trains the new FusionHead (416-dim) with PoseEncoder
python train_emotion_classifier.py --use-sentence-transformer --epochs 20

# Emotion classifier with external emotion data
python train_emotion_classifier.py --use-sentence-transformer --external-data data/external_datasets/external_emotion_data.csv --epochs 20
```

## Combine All Datasets — Full Retrain

After the current runs finish, retrain with the combined datasets:

```bash
# 1. Retrain calibration model on WESAD + DREAMER (checkpoint at checkpoints/ruview_calibration.pt)
#    8125 windows from 38 subjects. To retrain:
python train_calibration_dreamer.py --epochs 300

# 2. Retrain emotion classifier with ALL data sources combined:
#    - sentence-transformer embeddings
#    - augmented emotion data
#    - GoEmotions dataset
#    - pose features (from pose_encoder.py)
#    - WESAD-calibrated biometrics
python train_emotion_classifier.py --use-sentence-transformer --use-go-emotions --use-hf-datasets --external-data data/external_datasets/augmented_emotion_data.csv --epochs 30 --lr 0.0003

# 3. Once you have real WiFi paired data (from calibration_collector.py), retrain calibration:
python calibration_collector.py retrain --data-dir calibration_data/
```

## Commit & Deploy After Training

```bash
# Commit updated checkpoints
git add -f checkpoints/emotion_fusion_head.pt checkpoints/ruview_calibration.pt
git commit -m "feat: retrained emotion classifier and calibration with full datasets"
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

## Dataset Inventory

| Dataset | Location | Records | Signals | Used In |
|---------|----------|---------|---------|---------|
| WESAD | `Quantara-Backend/ml-pipeline/data/wesad/` | 2996 windows (15 subjects) | ECG→HR/HRV, Resp→BR, EDA, ACC | `train_calibration_wesad.py` |
| DREAMER | `data/dreamer/DREAMER.mat` | 5129 windows (23 subjects) | ECG→HR/HRV, valence/arousal labels | `train_calibration_dreamer.py` |
| HR Emotion | `~/Downloads/heart_rate_emotion_dataset.csv` | 100K readings | HR + emotion label | `train_emotion_classifier.py` |
| Stress Level | `~/Downloads/Stress_Level_v1.csv`, `v2.csv` | ~30 subjects | Stress scores | `train_emotion_classifier.py` |
| GoEmotions (full) | `~/Downloads/go_emotions_dataset.csv` | 211K rows | Text + 28 multi-label emotions | `train_emotion_classifier.py --use-go-emotions` |
| GoEmotions (extracted) | `~/Downloads/goemotions_extracted.csv` | 24K rows | Text + 13 Quantara-mapped emotions | autoresearch `goemotions` |
| Augmented Emotion | `data/external_datasets/augmented_emotion_data.csv` | ? | Text + emotion | `train_emotion_classifier.py --external-data` |
| External Emotion | `data/external_datasets/external_emotion_data.csv` | ? | Text + emotion | `train_emotion_classifier.py --external-data` |
| Tweet Emotions | `~/Downloads/text_emotion.csv` | 40K | Text + 13 sentiments | autoresearch `text_emotion` |
| Text Emotion 6 | `~/Downloads/text.csv` | 417K (100K sampled) | Text + 6 emotions | autoresearch `text_emotion_6` |
| Emotion Labels | `~/Downloads/archive (2) 2/emotion-labels-*.csv` | 7K (train+val+test) | Text + 4 emotions | autoresearch `emotion_labels` |
| Speech Emotion (F) | `~/Downloads/archive (3) 2/Female_features.csv` | 49K | 58 audio features + 8 emotions | autoresearch `speech_emotion` |
| Speech Emotion (M) | `~/Downloads/archive (3) 2/Male_features.csv` | 36K | 58 audio features + 8 emotions | autoresearch `speech_emotion` |

## Existing Trained Models (Quantara-Frontend/models/)

| Model | Size | Accuracy | Source |
|-------|------|----------|--------|
| `autoresearch_speech_emotion_voting_*.pkl` | 665M | 88.6% | Autoresearch |
| `autoresearch_text_emotion_6_voting_*.pkl` | ~2.5G | 68.6% | Autoresearch |
| `autoresearch_goemotions_voting_*.pkl` | ~1G | 51.9% | Autoresearch |
| `autoresearch_emotion_labels_voting_*.pkl` | ~500M | 73.0% | Autoresearch |
| `autoresearch_text_emotion_voting_embeddings_*.pkl` | 2.5G | 35.7% | Autoresearch |
| `sleep_stage_model.pkl` | 39M | — | Pre-existing |
| `swell_eda_stress_model.pkl` | 17M | — | Pre-existing |
| `wisdm_activity_model.pkl` | 13M | — | Pre-existing |
| `hrv_stress_model.pkl` | 4.6M | — | Pre-existing |
| `bci_cognitive_model.pkl` | 2.8M | — | Pre-existing |
| `wesad_stress_model.pkl` | 2.1M | — | Pre-existing |
| `archive1_eeg_model.pkl` | 1.7M | — | Pre-existing |
| `dreamer_emotion_model.pkl` | 1.4M | — | Pre-existing |
| `mental_health_risk_model.pkl` | 1.4M | — | Pre-existing |
| `hr_emotion_model.pkl` | 936K | — | Pre-existing |
| `text_emotion_model.pkl` | 921K | — | Pre-existing |
| `amigos_emotion_model.pkl` | 517K | — | Pre-existing |
| `archive8_motion_model.pkl` | 38K | — | Pre-existing |

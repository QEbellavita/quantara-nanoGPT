# Implementation Plan: Emotion Taxonomy Expansion (7 → 32 Emotions)

**Spec:** `docs/superpowers/specs/2026-03-14-emotion-taxonomy-expansion-design.md`
**Date:** 2026-03-14
**Status:** Ready for execution

## Context

Expand the Quantara nanoGPT emotion system from 7 flat emotions to 32 emotions organized into 9 families with two-stage classification. No new files — all changes modify existing files.

## Dependencies & Build Order

```
Step 1 (data layer)     → Step 2 (classifier)     → Step 3 (training)
     prepare.py              emotion_classifier.py      train_emotion_classifier.py
                                                        config/*.py

Step 4 (API layer)      → Step 5 (frontend/CLI)
     emotion_api_server.py   sample_emotion.py
     quantara_integration.py neural_ecosystem_connector.js
```

Steps 1-3 are sequential (each depends on prior). Steps 4-5 can run in parallel after Step 3.

---

## Step 1: Data Preparation — `data/quantara_emotion/prepare.py`

**File:** `data/quantara_emotion/prepare.py` (267 lines)
**Risk:** Medium — touches data pipeline, requires new dataset file paths
**Review checkpoint:** After step

### Changes:

1. **Add new dataset loaders** (after line 119, inside `load_emotion_datasets()`):
   - `heart_rate_emotion_dataset.csv` from Downloads — load disgust samples (14K), map `happy→joy`, `sad→sadness`
   - EEG cognitive dataset — load anxiety, calm, stressed (~500 each), map `Anxious→anxiety`
   - Tweet emotions (`text_emotion.csv`) already loaded — filter for boredom, enthusiasm, worry, hate, relief, fun (~3K each)

2. **Add Tier 2 reclassification function** `reclassify_derived_emotions()`:
   - frustration: anger samples with moderate arousal keywords ("frustrated", "annoyed", "irritated")
   - excitement: joy samples with high-energy keywords ("excited", "thrilled", "pumped")
   - grief: sadness samples with loss keywords ("lost", "died", "gone forever")
   - overwhelmed: stressed samples with overload keywords ("too much", "can't handle")
   - hope: positive valence + future keywords ("hope", "looking forward", "someday")
   - guilt: `ashamed` text with action-focus ("I did", "I shouldn't have")
   - shame: `ashamed` text with self-focus ("I am", "I'm worthless")

3. **Add Tier 3 synthetic generator** `generate_synthetic_samples()`:
   - Templates for: nostalgia, jealousy, contempt, pride, resilience, mindfulness, gratitude, compassion
   - ~500 base samples per emotion using template patterns from spec
   - Augmentation: synonym substitution + sentence restructuring to reach ~2,000

4. **Update emotion tag format** from `<emotion>text</emotion>` to `<family:emotion>text</family:emotion>`:
   - Add `EMOTION_FAMILIES` dict mapping each emotion to its family
   - Update tag generation: `<joy:excitement>text</joy:excitement>`
   - Keep backward-compatible: original 7 emotions use `<joy:joy>`, `<sadness:sadness>` etc.

5. **Add data balancing**:
   - Downsample original 7 emotions from ~100K to ~3,000 each
   - Oversample Tier 3 synthetic (~500→~2,000 via augmentation, then to ~3,000 via duplication)
   - Target: ~3,000 samples per emotion × 32 = ~96,000 total

6. **Update `emotion_tags` config** (line 250):
   - Expand from 10 tags to include all 32 emotions + families

### Verification:
- Run `python data/quantara_emotion/prepare.py` — should output sample counts per emotion
- Check `train.bin` and `val.bin` are regenerated
- Verify no emotion has < 1,000 samples

---

## Step 2: Classifier — `emotion_classifier.py`

**File:** `emotion_classifier.py` (479 lines)
**Risk:** Medium — core ML architecture change
**Review checkpoint:** After step

### Changes:

1. **Add `EMOTION_FAMILIES` dict** (after line 14):
   ```python
   EMOTION_FAMILIES = {
       'Joy': ['joy', 'excitement', 'enthusiasm', 'fun', 'gratitude', 'pride'],
       'Sadness': ['sadness', 'grief', 'boredom', 'nostalgia'],
       'Anger': ['anger', 'frustration', 'hate', 'contempt', 'disgust', 'jealousy'],
       'Fear': ['fear', 'anxiety', 'worry', 'overwhelmed', 'stressed'],
       'Love': ['love', 'compassion'],
       'Calm': ['calm', 'relief', 'mindfulness', 'resilience', 'hope'],
       'Self-Conscious': ['guilt', 'shame'],
       'Surprise': ['surprise'],
       'Neutral': ['neutral'],
   }
   ```

2. **Add `family_for_emotion()` function**:
   - Returns family name for any of the 32 emotions
   - O(1) lookup via pre-built reverse dict

3. **Expand `FusionHead.EMOTIONS`** (line 166):
   - From 7 to all 32 emotions in canonical order (family-grouped)

4. **Update `FusionHead.__init__`** (line 168):
   - Change `num_emotions` default from 7 to 32
   - Add `num_families=9` parameter
   - Add family classification head: `self.family_classifier = nn.Linear(hidden_dim // 2, num_families)`

5. **Add two-stage forward** in `FusionHead`:
   - `forward()` returns `(emotion_probs, family_probs)`
   - New `classify_with_fallback(text_emb, bio_emb, threshold=0.6)` method:
     - Stage 1: classify family (9-way)
     - Stage 2: classify sub-emotion within predicted family
     - If sub-emotion confidence < 0.6, return family root emotion
     - Return `{emotion, family, confidence, is_fallback}`

6. **Update `BiometricEncoder._extract_features`** (line 105):
   - Add derived features for new biometric discrimination:
     - `hrv_calm`: 1.0 if HRV > 65 (distinguishes Calm family)
     - `eda_overwhelm`: 1.0 if EDA > 7 (distinguishes overwhelmed)
   - Update input dim from 6 to 8 in `__init__` (line 92)

7. **Update `MultimodalEmotionAnalyzer.analyze`** (line 404):
   - Response now includes `family` and `is_fallback` fields
   - Use `classify_with_fallback()` instead of raw `forward()`

### Verification:
- `python -c "from emotion_classifier import EMOTION_FAMILIES, family_for_emotion; print(family_for_emotion('jealousy'))"`
- `python -c "from emotion_classifier import FusionHead; f = FusionHead(text_dim=384, num_emotions=32); print(len(f.EMOTIONS))"`

---

## Step 3: Training — `train_emotion_classifier.py` + configs

**File:** `train_emotion_classifier.py` (419 lines)
**Risk:** Low — training code, can be re-run
**Review checkpoint:** After step

### Changes to `train_emotion_classifier.py`:

1. **Expand `BIOMETRIC_RANGES`** (line 50):
   - Add all 32 emotions with ranges from spec (lines 109-175 of design doc)
   - Current: 7 entries → New: 32 entries

2. **Update `EMOTION_TO_IDX`** (line 60):
   - Will auto-update since it reads from `FusionHead.EMOTIONS`

3. **Update `load_emotion_data()`** (line 89):
   - Add loaders for heart_rate_emotion_dataset.csv, EEG data, tweet expanded emotions
   - Add label mapping for new dataset formats
   - Add Tier 2 reclassification logic
   - Add Tier 3 synthetic sample generation

4. **Update `FusionHead` instantiation** (line 297):
   - Change `num_emotions=7` to `num_emotions=32`

5. **Add two-stage training loss**:
   - Family-level cross-entropy (weighted 0.3)
   - Sub-emotion cross-entropy (weighted 0.7)
   - Combined: `loss = 0.3 * family_loss + 0.7 * emotion_loss`

6. **Add family-level accuracy logging**:
   - Track and print both family accuracy and sub-emotion accuracy per epoch

### Changes to `config/train_quantara_emotion.py`:

7. Update comment to mention 32 emotions (documentation only — no functional changes needed, `num_emotions` is set in training script)

### Changes to `config/train_quantara_emotion_fast.py`:

8. Same as above — comment update only

### Verification:
- `python -c "from train_emotion_classifier import BIOMETRIC_RANGES; print(len(BIOMETRIC_RANGES))"` → 32
- Dry-run training with `--epochs 1` to verify shape compatibility

---

## Step 4: API Server — `emotion_api_server.py` + `quantara_integration.py`

**File:** `emotion_api_server.py` (612 lines)
**Risk:** Medium — affects live API
**Review checkpoint:** After step

### Changes to `emotion_api_server.py`:

1. **Expand `EmotionGPTModel.EMOTIONS`** (line 68):
   - From 7 to 32 emotions

2. **Add `EMOTION_FAMILIES` dict** (after line 68):
   - Same structure as in emotion_classifier.py

3. **Expand `FAST_RESPONSES`** (line 226):
   - Add pre-written coaching responses for all 25 new emotions
   - Group by family for maintainability

4. **Expand `get_therapy_technique()`** (line 312):
   - Add all 32 emotions with techniques from spec (lines 177-235)
   - Add `transition` and `coaching_prompt` fields to response

5. **Add `TRANSITION_PATHWAYS` dict**:
   - All 32 emotion transition pathways from spec

6. **Add `COACHING_PROMPTS` dict**:
   - All 32 coaching prompts from spec

7. **Update `/api/emotion/analyze` response** (line 419):
   - Add `family`, `is_fallback` fields when multimodal analyzer is available

8. **Update `/api/emotion/therapy` response** (line 482):
   - Include `transition` and `coaching_prompt` in response

9. **Add new endpoint `/api/emotion/family`** (after line 500):
   - `GET /api/emotion/family` — list all families with their emotions
   - `GET /api/emotion/family/<name>` — get emotions in a specific family

10. **Add new endpoint `/api/neural/emotion-transition`** (after workflow endpoint):
    - `POST /api/neural/emotion-transition` — get transition pathway between two emotions
    - Request: `{from_emotion, to_emotion}`
    - Response: transition method + coaching prompt

11. **Update `neural_workflow` endpoint** (line 503):
    - Expand `workflow_actions` dict to cover all 9 families
    - Add family-aware workflow triggers

### Changes to `quantara_integration.py`:

12. **Update `self.emotions`** (line 59):
    - From 7 to 32 emotions

13. **Add family-aware analysis** in `analyze_emotion()`:
    - Return family grouping in response

### Verification:
- Syntax check: `python -c "import emotion_api_server"` (will fail at app creation without checkpoint, but import should work)
- Verify all 32 emotions have therapy techniques, transitions, coaching prompts

---

## Step 5: CLI & Frontend — `sample_emotion.py` + `neural_ecosystem_connector.js`

**File:** `sample_emotion.py` (206 lines), `neural_ecosystem_connector.js` (272 lines)
**Risk:** Low — user-facing, no data dependencies
**Review checkpoint:** After step

### Changes to `sample_emotion.py`:

1. **Update help text** (line 148):
   - Show 32 emotions grouped by family in interactive mode header
   - Format: `Joy: joy, excitement, enthusiasm, fun, gratitude, pride`

2. **Add `/family <name>` command** (after line 169):
   - Lists emotions in the specified family
   - Example: `/family Anger` → `anger, frustration, hate, contempt, disgust, jealousy`

3. **Update `--emotion` help** (line 107):
   - List all 32 emotions or say "see --list-emotions"

4. **Add `--list-emotions` flag**:
   - Prints all 32 emotions grouped by family and exits

### Changes to `neural_ecosystem_connector.js`:

5. **Add `EMOTION_FAMILIES` constant** (after line 21):
   - JavaScript object mirroring Python `EMOTION_FAMILIES`

6. **Update `processBiometricData()`** (line 193):
   - Expand biometric inference to use family-level detection
   - Add Calm family (low HR + high HRV), Self-Conscious (moderate HR + high EDA)

7. **Add `getEmotionFamily()` method** to `QuantaraEmotionAPI`:
   - Calls new `/api/emotion/family` endpoint

8. **Add `getEmotionTransition()` method** to `QuantaraEmotionAPI`:
   - Calls new `/api/neural/emotion-transition` endpoint

9. **Update `processUserMessage()`** (line 156):
   - Expand therapy trigger list from `['sadness', 'anger', 'fear']` to all negative-valence families
   - Include transition pathway in response

### Verification:
- `python sample_emotion.py --list-emotions` → prints 32 emotions
- `node -e "const {QuantaraEmotionAPI} = require('./neural_ecosystem_connector'); console.log('OK')"` → no syntax errors

---

## Summary

| Step | File(s) | Lines Changed (est.) | Risk |
|------|---------|---------------------|------|
| 1 | `data/quantara_emotion/prepare.py` | +200 | Medium |
| 2 | `emotion_classifier.py` | +120 | Medium |
| 3 | `train_emotion_classifier.py`, configs | +150 | Low |
| 4 | `emotion_api_server.py`, `quantara_integration.py` | +350 | Medium |
| 5 | `sample_emotion.py`, `neural_ecosystem_connector.js` | +80 | Low |

**Total estimated:** ~900 lines added/modified across 8 files
**No new files created**
**Backward compatible:** existing API fields preserved, new fields added

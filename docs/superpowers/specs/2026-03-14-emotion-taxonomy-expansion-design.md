# Emotion Taxonomy Expansion — 32 Emotions with Hierarchical Classification

**Date:** 2026-03-14
**Status:** Approved
**Scope:** Expand nanoGPT emotion system from 7 to 32 emotions using hierarchical taxonomy

## Overview

Expand the Quantara nanoGPT emotion system from 7 flat emotions to 32 emotions organized into 9 families with two-stage classification. Includes research-grounded biometric ranges, therapy techniques, transition pathways, and coaching prompts for each emotion.

## Taxonomy (9 Families, 32 Emotions)

```
Joy (family)
├── excitement    — high-energy positive anticipation
├── enthusiasm    — motivated engagement
├── fun           — playful enjoyment
├── gratitude     — thankful appreciation
└── pride         — accomplishment satisfaction

Sadness (family)
├── grief         — deep loss processing
├── boredom       — disengaged low-energy
└── nostalgia     — bittersweet past reflection

Anger (family)
├── frustration   — blocked goal irritation
├── hate          — intense aversion
├── contempt      — moral superiority dismissal
├── disgust       — revulsion/rejection
└── jealousy      — envy-driven resentment

Fear (family)
├── anxiety       — diffuse future-oriented worry
├── worry         — specific concern rumination
├── overwhelmed   — capacity exceeded
└── stressed      — pressure accumulation

Love (family)
├── compassion    — empathetic caring for others

Calm (new family)
├── relief        — tension release
├── mindfulness   — present-moment awareness
├── resilience    — bounce-back strength
└── hope          — positive future orientation

Self-Conscious (new family)
├── guilt         — action-specific regret
└── shame         — self-identity pain

Surprise (atomic)

Neutral (atomic)
```

## Data Strategy

### Tier 1 — Direct from existing datasets

| Emotion | Source | ~Samples |
|---|---|---|
| anxiety | EEG cognitive dataset | 500 |
| calm | EEG cognitive dataset | 500 |
| stressed | EEG cognitive dataset | 500 |
| disgust | heart_rate_emotion_dataset.csv | 14,227 |
| boredom | Tweet emotions (text_emotion.csv) | ~3,000 |
| enthusiasm | Tweet emotions | ~3,000 |
| worry | Tweet emotions | ~3,000 |
| hate | Tweet emotions | ~3,000 |
| relief | Tweet emotions | ~3,000 |
| fun | Tweet emotions | ~3,000 |

### Tier 2 — Derived via reclassification

| Emotion | Derivation method |
|---|---|
| frustration | Anger samples with moderate arousal + keyword filtering |
| excitement | Joy samples with high arousal (HR > 85) |
| grief | Sadness samples with high intensity + loss keywords |
| overwhelmed | Stressed + high EDA + negative valence |
| hope | Positive valence + future-oriented keywords |
| guilt | Psychological dataset `ashamed` score > 0.5 + action-focused text |
| shame | Psychological dataset `ashamed` score > 0.5 + self-focused text |

### Tier 3 — Synthetic generation

| Emotion | Approach |
|---|---|
| nostalgia | Template: "I remember when..." + bittersweet patterns, ~500 samples |
| jealousy | Template: competitive comparison patterns, ~500 samples |
| contempt | Derived from hate/disgust + moral judgment keywords, ~500 samples |
| pride | Template: achievement/accomplishment patterns, ~500 samples |
| resilience | Template: overcoming adversity patterns, ~500 samples |
| mindfulness | Template: present-moment awareness patterns, ~500 samples |
| gratitude | Template: thankfulness/appreciation patterns, ~500 samples |
| compassion | Template: empathy/caring-for-others patterns, ~500 samples |

### Original 7 emotions (carry-over)

The original emotions (joy, sadness, love, anger, fear, surprise, neutral) retain their existing training data from text.csv (~100K), tweet emotions (~40K), Emotion_classify_Data.csv (~6K), and archive datasets. These will be downsampled to ~3,000 per emotion to balance with new categories.

**Balancing:** Oversample low-count, undersample high-count. Target ~3,000 per emotion. For Tier 3 synthetic emotions (~500 base samples), use template augmentation (synonym substitution, sentence restructuring) to reach ~2,000 before oversampling to reduce overfitting risk.

## Biometric Ranges (Research-Grounded)

HR = heart rate (bpm), HRV = heart rate variability (ms), EDA = electrodermal activity (µS).

### Joy family
| Emotion | HR | HRV | EDA | Rationale |
|---|---|---|---|---|
| joy | 70-90 | 50-80 | 2-4 | existing |
| excitement | 85-110 | 40-60 | 4-7 | elevated arousal, moderate stress |
| enthusiasm | 75-95 | 45-70 | 3-5 | moderate activation |
| fun | 75-95 | 50-75 | 2-4 | relaxed positive arousal |
| gratitude | 65-80 | 60-85 | 1-3 | parasympathetic dominant, high HRV |
| pride | 70-90 | 45-65 | 3-5 | moderate arousal, slight EDA bump |

### Sadness family
| Emotion | HR | HRV | EDA | Rationale |
|---|---|---|---|---|
| sadness | 55-70 | 40-60 | 1-2 | existing |
| grief | 50-70 | 30-50 | 1-3 | lower HRV, vagal withdrawal |
| boredom | 55-65 | 55-75 | 0.5-1.5 | low arousal, high HRV |
| nostalgia | 60-75 | 45-65 | 1.5-3 | mild arousal, bittersweet activation |

### Anger family
| Emotion | HR | HRV | EDA | Rationale |
|---|---|---|---|---|
| anger | 85-110 | 20-40 | 5-8 | existing |
| frustration | 80-100 | 25-45 | 4-7 | moderate anger, less intense |
| hate | 85-105 | 20-35 | 5-9 | sustained sympathetic activation |
| contempt | 75-90 | 30-50 | 3-5 | controlled, lower arousal than anger |
| disgust | 70-90 | 35-55 | 4-7 | parasympathetic mixed, nausea response |
| jealousy | 80-100 | 25-45 | 5-8 | elevated HR + EDA from social threat (Harris 2004) |

### Fear family
| Emotion | HR | HRV | EDA | Rationale |
|---|---|---|---|---|
| fear | 80-105 | 25-45 | 6-10 | existing |
| anxiety | 75-100 | 20-40 | 5-9 | sustained elevated, low HRV (Chalmers 2014) |
| worry | 70-90 | 30-50 | 3-6 | cognitive, moderate somatic |
| overwhelmed | 85-110 | 15-35 | 7-12 | highest EDA, lowest HRV — system overload |
| stressed | 80-105 | 20-40 | 5-9 | cortisol-driven |

> **Note:** Anxiety and stressed have near-identical biometric ranges. Disambiguation will rely primarily on text features. The two-stage classifier handles this gracefully — both fall within the Fear family, so family-level classification remains accurate.

### Love family
| Emotion | HR | HRV | EDA | Rationale |
|---|---|---|---|---|
| love | 65-85 | 55-75 | 2-4 | existing |
| compassion | 60-80 | 60-80 | 1.5-3 | high vagal tone (Stellar 2015) |

### Calm family
| Emotion | HR | HRV | EDA | Rationale |
|---|---|---|---|---|
| calm | 55-70 | 65-90 | 0.5-2 | parasympathetic dominance |
| relief | 60-80 | 55-80 | 2-4 | transition state, EDA dropping |
| mindfulness | 55-68 | 70-95 | 0.5-1.5 | highest HRV — vagal meditation response |
| resilience | 60-75 | 60-85 | 1-3 | regulated, adaptive |
| hope | 65-80 | 55-75 | 1.5-3 | mild positive arousal |

### Self-Conscious family
| Emotion | HR | HRV | EDA | Rationale |
|---|---|---|---|---|
| guilt | 70-90 | 30-50 | 4-7 | moderate arousal (Dickerson 2004) |
| shame | 75-95 | 25-45 | 5-8 | higher arousal, cortisol spike |

> **Note:** Within the Joy family, enthusiasm/fun/pride have overlapping biometric ranges. These rely heavily on text features for sub-emotion classification, with family-level fallback when confidence is low.

### Atomic
| Emotion | HR | HRV | EDA |
|---|---|---|---|
| surprise | 75-100 | 35-55 | 4-7 |
| neutral | 60-80 | 50-70 | 1-3 |

## Therapy, Transitions & Coaching

### Joy family
| Emotion | Technique | Transition | Coaching Prompt |
|---|---|---|---|
| joy | Savoring | (existing) | (existing) |
| excitement | Mindful Savoring | excitement → focused calm via grounding | "That energy is powerful. Let's channel it — what's the one thing you most want to direct this toward?" |
| enthusiasm | Goal Anchoring | enthusiasm → sustained motivation via values alignment | "I love your drive. Let's connect it to your deeper goals so it carries you forward." |
| fun | Positive Journaling | fun → gratitude via reflection | "This joy matters. What about this moment would you want to remember?" |
| gratitude | Gratitude Letter | gratitude → compassion via loving-kindness | "That appreciation you feel — have you told them? Sometimes expressing it multiplies it." |
| pride | Achievement Integration | pride → resilience via strength inventory | "You earned this. Let's name the strengths that got you here — they'll serve you again." |

### Sadness family
| Emotion | Technique | Transition | Coaching Prompt |
|---|---|---|---|
| sadness | Behavioral Activation | (existing) | (existing) |
| grief | Grief Journaling + Continuing Bonds | grief → acceptance via meaning-making | "There's no timeline for this. What you're feeling honors what you lost. I'm here." |
| boredom | Values Exploration | boredom → engagement via micro-goals | "Boredom is often a signal. What would feel meaningful to you right now, even something small?" |
| nostalgia | Narrative Integration | nostalgia → present gratitude via bridging past-present | "Those memories shaped who you are. What from that time do you still carry with you today?" |

### Anger family
| Emotion | Technique | Transition | Coaching Prompt |
|---|---|---|---|
| anger | Cognitive Reframing | (existing) | (existing) |
| frustration | Problem-Solving Therapy | frustration → agency via action planning | "I hear the frustration. Let's separate what you can control from what you can't — and start there." |
| hate | Cognitive Defusion (ACT) | hate → understanding via perspective-taking | "That intensity is telling you something matters deeply. What need is underneath this?" |
| contempt | Empathy Building | contempt → curiosity via perspective shift | "What if there's a reason you haven't considered? Let's explore what might be driving their behavior." |
| disgust | Exposure Hierarchy | disgust → tolerance via graduated exposure | "That reaction is your boundaries speaking. Let's understand what's being violated and how to protect it." |
| jealousy | Self-Worth Inventory | jealousy → self-acceptance via comparing to past self | "That comparison is stealing your peace. What have you accomplished that you're not giving yourself credit for?" |

### Fear family
| Emotion | Technique | Transition | Coaching Prompt |
|---|---|---|---|
| fear | Grounding (5-4-3-2-1) | (existing) | (existing) |
| anxiety | Progressive Muscle Relaxation | anxiety → calm via body-down regulation | "Your mind is racing ahead. Let's bring you back to right now — what do you feel in your body?" |
| worry | Worry Time Scheduling | worry → problem-solving via containment | "Let's give this worry a boundary. What's the actual worst case, and how likely is it really?" |
| overwhelmed | Task Chunking + Triage | overwhelmed → manageable via smallest-next-step | "Everything at once is too much. What's the one smallest thing you could do in the next 5 minutes?" |
| stressed | Box Breathing (4-4-4-4) | stressed → regulated via autonomic reset | "Your system is running hot. Let's cool it down — breathe with me: in 4, hold 4, out 4, hold 4." |

### Calm family
| Emotion | Technique | Transition | Coaching Prompt |
|---|---|---|---|
| calm | Body Scan Meditation | calm → mindfulness via deepening awareness | "You're in a good space. Let's deepen it — notice how your body feels right now, without changing anything." |
| relief | Integration Practice | relief → gratitude via acknowledging the shift | "You made it through. Take a moment to notice — what helped you get here?" |
| mindfulness | Open Monitoring Meditation | mindfulness → insight via non-reactive observation | "Stay with this awareness. Whatever arises — thoughts, feelings — just notice without following." |
| resilience | Strength Spotting | resilience → hope via past-success recall | "You've weathered storms before. What got you through last time? That strength is still in you." |
| hope | Future Self Visualization | hope → motivation via concrete next steps | "Hold onto that vision. Now let's build a bridge to it — what's one step you can take this week?" |

### Self-Conscious family
| Emotion | Technique | Transition | Coaching Prompt |
|---|---|---|---|
| guilt | Self-Compassion Practice (Neff) | guilt → repair via amends planning | "Guilt means your values are intact. What would making it right look like?" |
| shame | Shame Resilience (Brene Brown) | shame → self-worth via separating action from identity | "What happened doesn't define who you are. Can you tell me what you're telling yourself about this?" |

### Atomic
| Emotion | Technique | Transition | Coaching Prompt |
|---|---|---|---|
| surprise | Mindful Observation | (existing) | (existing) |
| neutral | Values Clarification | (existing) | (existing) |

## Architecture Changes

### Files to modify (no new files)

1. **`emotion_classifier.py`** — Add `EMOTION_FAMILIES` dict, expand `FusionHead.EMOTIONS` to 32, add `family_for_emotion()`, update `BiometricEncoder` derived features, add two-stage classification with confidence fallback (threshold 0.6)

2. **`train_emotion_classifier.py`** — Expand `BIOMETRIC_RANGES` to 32 emotions, add new dataset loaders (EEG, heart rate, tweet sentiments), Tier 2 derivation logic, Tier 3 synthetic generator, data balancing, two-stage training

3. **`data/quantara_emotion/prepare.py`** — Add dataset loaders for heart_rate_emotion_dataset.csv and EEG data, label mapping (happy→joy, sad→sadness, Anxious→anxiety), Tier 2 reclassification, Tier 3 synthetic templates, family-aware tags `<joy:excitement>text</joy:excitement>`, expand `emotion_tags` config

4. **`emotion_api_server.py`** — Expand `EMOTIONS` to 32, add `EMOTION_FAMILIES`, `THERAPY_TECHNIQUES`, `TRANSITION_PATHWAYS`, `COACHING_PROMPTS` dicts, new endpoints (`/api/emotion/family`, `/api/neural/emotion-transition`), update `/api/emotion/analyze` response to include `{emotion, family, confidence, fallback}`, update `/api/emotion/therapy` to include transition + coaching

5. **`sample_emotion.py`** — Update help text with 32 emotions grouped by family, add `/family <name>` command

6. **`config/train_quantara_emotion.py`** & **`config/train_quantara_emotion_fast.py`** — Update output dimensions for 32 classes

7. **`neural_ecosystem_connector.js`** — Update emotion list, add family groupings

8. **`quantara_integration.py`** — Update emotion mappings

## Neural Ecosystem Integration

All 32 emotions connect to:
- Neural Workflow AI Engine
- AI Conversational Coach
- Emotion-Aware Training Engine
- Psychology Emotion Database
- Biometric Integration Engine
- Therapist Dashboard Engine
- Real-time Dashboard Data

## Confidence Fallback Behavior

Two-stage classification with confidence threshold of 0.6:
1. Classify into family (9 families) — high accuracy due to biometric separation
2. Classify sub-emotion within family — uses text features primarily
3. If sub-emotion confidence < 0.6, return family root emotion (e.g., "anger" instead of "frustration")
4. API response includes: `{emotion, family, confidence, is_fallback}`

## Migration & Backward Compatibility

- **Existing checkpoints** (7-emotion) are incompatible — full retraining required after data preparation
- **API response** adds new fields (`family`, `confidence`, `is_fallback`) but existing fields remain unchanged — backward compatible
- **Emotion tags** change from `<emotion>` to `<family:emotion>` format in training data, but the API accepts both old-style flat emotion names and new family-qualified names
- **Frontend** (`neural_ecosystem_connector.js`) needs update to handle family groupings — existing emotion references continue to work

## Testing Plan

- Unit tests for `family_for_emotion()` mapping and `EMOTION_FAMILIES` consistency
- Biometric range validation: all 32 emotions have valid, non-negative ranges
- Two-stage classifier accuracy: family-level > 85%, sub-emotion > 65%
- API integration tests for new endpoints and expanded response schema
- Sampling script: verify all 32 emotions generate coherent text
- Regression: original 7 emotions maintain quality after retraining

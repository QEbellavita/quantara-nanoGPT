"""
===============================================================================
QUANTARA - Emotion Classifier Benchmark (GoEmotions vs MiniLM vs nanoGPT)
===============================================================================
Compare accuracy across encoder types on edge-case test prompts and
dair-ai/emotion validation set.

Connected to:
- Neural Workflow AI Engine
- ML Training & Prediction Systems
- Emotion-Aware Training Engine

Usage:
    python benchmark_emotion.py --checkpoint checkpoints/emotion_fusion_head.pt
===============================================================================
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from emotion_classifier import (
    MultimodalEmotionAnalyzer, FusionHead, EMOTION_FAMILIES, FAMILY_NAMES,
    family_for_emotion,
)

# ─── Mapping from 32 model emotions → 6 dair-ai/emotion labels ──────────────
# dair-ai/emotion only has: sadness, joy, love, anger, fear, surprise
# Our model has 32 sub-emotions. This maps each back to the closest base label.

EMOTION_TO_BASE = {
    # Joy family → joy
    'joy': 'joy', 'excitement': 'joy', 'enthusiasm': 'joy', 'fun': 'joy',
    'gratitude': 'joy', 'pride': 'joy',
    # Sadness family → sadness
    'sadness': 'sadness', 'grief': 'sadness', 'boredom': 'sadness',
    'nostalgia': 'sadness',
    # Anger family → anger
    'anger': 'anger', 'frustration': 'anger', 'hate': 'anger',
    'contempt': 'anger', 'disgust': 'anger', 'jealousy': 'anger',
    # Fear family → fear
    'fear': 'fear', 'anxiety': 'fear', 'worry': 'fear',
    'overwhelmed': 'fear', 'stressed': 'fear',
    # Love family → love
    'love': 'love', 'compassion': 'love',
    # Calm family → joy (closest positive valence in 6-class)
    'calm': 'joy', 'relief': 'joy', 'mindfulness': 'joy',
    'resilience': 'joy', 'hope': 'joy',
    # Self-Conscious → sadness (closest negative valence in 6-class)
    'guilt': 'sadness', 'shame': 'sadness',
    # Direct mappings
    'surprise': 'surprise',
    'neutral': 'surprise',  # neutral has no match; surprise is least wrong
}

FAMILY_TO_BASE = {
    'Joy': 'joy', 'Sadness': 'sadness', 'Anger': 'anger',
    'Fear': 'fear', 'Love': 'love', 'Calm': 'joy',
    'Self-Conscious': 'sadness', 'Surprise': 'surprise', 'Neutral': 'surprise',
}

# ─── Edge-case test suite ────────────────────────────────────────────────────

EDGE_CASES = [
    # (text, expected_emotion, expected_family, description)
    # Clear emotions
    ("I'm so happy I could dance!", "joy", "Joy", "clear joy"),
    ("This is the worst day of my life", "sadness", "Sadness", "clear sadness"),
    ("I'm absolutely furious right now", "anger", "Anger", "clear anger"),
    ("I'm terrified of what might happen", "fear", "Fear", "clear fear"),
    ("I love you more than anything", "love", "Love", "clear love"),
    ("I feel so peaceful and at ease", "calm", "Calm", "clear calm"),
    ("I can't believe that just happened!", "surprise", "Surprise", "clear surprise"),

    # Subtle / nuanced
    ("I guess it's fine", "neutral", "Neutral", "passive-aggressive neutral"),
    ("Whatever, I don't care anymore", "boredom", "Sadness", "apathy/boredom"),
    ("I can't stop thinking about it", "anxiety", "Fear", "anxious rumination"),
    ("I should have done better", "guilt", "Self-Conscious", "self-blame/guilt"),
    ("Everyone is watching and judging me", "shame", "Self-Conscious", "shame/exposure"),

    # Sarcasm / mixed signals
    ("Oh great, another meeting", "frustration", "Anger", "sarcastic frustration"),
    ("Sure, that's just wonderful", "contempt", "Anger", "sarcastic contempt"),

    # Complex / mixed emotions
    ("I'm happy for you but I'll miss you", "nostalgia", "Sadness", "bittersweet"),
    ("I can't wait but I'm also nervous", "excitement", "Joy", "excited + anxious"),
    ("I did it! I didn't think I could", "pride", "Joy", "achievement pride"),
    ("Thank you so much, you saved me", "gratitude", "Joy", "deep gratitude"),

    # Biometric-relevant (where HR/EDA context matters)
    ("My heart is racing", "anxiety", "Fear", "physiological anxiety"),
    ("I feel numb inside", "sadness", "Sadness", "emotional numbness"),
    ("I have so much energy right now", "excitement", "Joy", "high arousal positive"),

    # Therapy-relevant
    ("I don't see the point anymore", "grief", "Sadness", "hopelessness"),
    ("I just want to be left alone", "overwhelmed", "Fear", "withdrawal"),
    ("Everything will work out eventually", "hope", "Calm", "optimism"),
    ("I need to take a deep breath", "mindfulness", "Calm", "grounding"),
]


def run_edge_cases(analyzer, verbose=True):
    """Run edge case tests and report accuracy."""
    emotion_correct = 0
    family_correct = 0
    mapped_family_correct = 0
    total = len(EDGE_CASES)
    results = []

    for text, expected_emotion, expected_family, description in EDGE_CASES:
        result = analyzer.analyze(text)
        predicted_emotion = result['dominant_emotion']
        predicted_family = result['family']

        e_match = predicted_emotion == expected_emotion
        f_match = predicted_family == expected_family
        emotion_correct += int(e_match)
        family_correct += int(f_match)

        # Mapped family: did the model at least get the right family?
        predicted_base_family = FAMILY_TO_BASE.get(predicted_family, predicted_family)
        expected_base_family = FAMILY_TO_BASE.get(expected_family, expected_family)
        mf_match = predicted_base_family == expected_base_family
        mapped_family_correct += int(mf_match)

        results.append({
            'text': text[:50],
            'expected': f"{expected_family}/{expected_emotion}",
            'predicted': f"{predicted_family}/{predicted_emotion}",
            'confidence': result['confidence'],
            'emotion_match': e_match,
            'family_match': f_match,
        })

        if verbose:
            status = "OK" if e_match else ("~" if f_match else ("m" if mf_match else "X"))
            print(f"  [{status}] {description:30s} | expected={expected_emotion:15s} | "
                  f"got={predicted_emotion:15s} ({result['confidence']:.2f})")

    emotion_acc = emotion_correct / total
    family_acc = family_correct / total
    mapped_family_acc = mapped_family_correct / total

    print(f"\n  Edge Case Results:")
    print(f"    Emotion accuracy:       {emotion_correct}/{total} ({emotion_acc:.1%})")
    print(f"    Family accuracy (9-cls): {family_correct}/{total} ({family_acc:.1%})")
    print(f"    Mapped family (6-cls):   {mapped_family_correct}/{total} ({mapped_family_acc:.1%})")

    return emotion_acc, family_acc, results


def run_validation_set(analyzer, max_samples=2000, verbose=True):
    """Run on dair-ai/emotion validation set."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [!] datasets not installed, skipping validation set")
        return None, None

    label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

    print(f"\n  Loading dair-ai/emotion validation set...")
    try:
        ds = load_dataset('dair-ai/emotion', 'split', split='validation')
    except Exception:
        ds = load_dataset('dair-ai/emotion', 'split', split='test')

    samples = list(ds)[:max_samples]
    print(f"  Evaluating {len(samples)} samples...")

    emotion_correct = 0
    family_correct = 0
    mapped_correct = 0
    total = 0
    start = time.time()

    for i, row in enumerate(samples):
        text = row['text']
        expected = label_map.get(row['label'], 'neutral')
        expected_family = family_for_emotion(expected)

        result = analyzer.analyze(text)
        predicted = result['dominant_emotion']
        predicted_family = result['family']

        # Raw: exact match (32-class predicted vs 6-class expected)
        if predicted == expected:
            emotion_correct += 1
        if predicted_family == expected_family:
            family_correct += 1

        # Mapped: collapse 32-class prediction to 6-class for fair comparison
        predicted_base = EMOTION_TO_BASE.get(predicted, predicted)
        if predicted_base == expected:
            mapped_correct += 1

        total += 1

        if verbose and (i + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"    {i+1}/{len(samples)} ({rate:.1f} samples/sec) "
                  f"raw={emotion_correct/total:.1%} mapped={mapped_correct/total:.1%} "
                  f"family={family_correct/total:.1%}")

    emotion_acc = emotion_correct / total
    family_acc = family_correct / total
    mapped_acc = mapped_correct / total
    elapsed = time.time() - start

    print(f"\n  Validation Set Results ({total} samples, {elapsed:.1f}s):")
    print(f"    Raw emotion accuracy:    {emotion_correct}/{total} ({emotion_acc:.1%})")
    print(f"    Mapped accuracy (6-cls): {mapped_correct}/{total} ({mapped_acc:.1%})")
    print(f"    Family accuracy:         {family_correct}/{total} ({family_acc:.1%})")

    return emotion_acc, family_acc, mapped_acc


def main():
    parser = argparse.ArgumentParser(description='Benchmark emotion classifier')
    parser.add_argument('--checkpoint', default='checkpoints/emotion_fusion_head.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--val-samples', type=int, default=2000)
    parser.add_argument('--skip-val', action='store_true', help='Skip validation set')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[!] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Detect encoder type from checkpoint
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    embedding_type = state.get('embedding_type', 'unknown')
    meta = state.get('meta', {})
    val_acc = state.get('val_emotion_acc', 0)
    val_fam = state.get('val_family_acc', 0)

    print("=" * 70)
    print("  QUANTARA EMOTION CLASSIFIER BENCHMARK")
    print("=" * 70)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Encoder: {embedding_type}")
    print(f"  Text dim: {meta.get('text_dim', '?')}")
    print(f"  Training val_emotion_acc: {val_acc:.4f}")
    print(f"  Training val_family_acc: {val_fam:.4f}")

    # Load analyzer with correct encoder
    use_go = embedding_type == 'go-emotions'
    go_combined = state.get('go_emotions_combined', True)

    print(f"\n  Loading analyzer ({'GoEmotions' if use_go else embedding_type})...")
    analyzer = MultimodalEmotionAnalyzer(
        classifier_checkpoint=str(checkpoint_path),
        device=args.device,
        use_go_emotions=use_go,
        go_emotions_combined=go_combined,
        use_sentence_transformer=not use_go,
    )

    # Edge cases
    print(f"\n{'─' * 70}")
    print("  EDGE CASE TESTS (25 challenging prompts)")
    print(f"{'─' * 70}")
    edge_emotion, edge_family, _ = run_edge_cases(analyzer, verbose=not args.quiet)

    # Validation set
    val_mapped = None
    if not args.skip_val:
        print(f"\n{'─' * 70}")
        print("  VALIDATION SET (dair-ai/emotion)")
        print(f"{'─' * 70}")
        val_emotion, val_family, val_mapped = run_validation_set(
            analyzer, max_samples=args.val_samples, verbose=not args.quiet
        )

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — {embedding_type}")
    print(f"{'=' * 70}")
    print(f"  Edge cases:  emotion={edge_emotion:.1%}  family={edge_family:.1%}")
    if not args.skip_val and val_emotion is not None:
        print(f"  Validation (raw 32-cls):   emotion={val_emotion:.1%}  family={val_family:.1%}")
        print(f"  Validation (mapped 6-cls): {val_mapped:.1%}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()

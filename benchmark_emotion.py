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

        results.append({
            'text': text[:50],
            'expected': f"{expected_family}/{expected_emotion}",
            'predicted': f"{predicted_family}/{predicted_emotion}",
            'confidence': result['confidence'],
            'emotion_match': e_match,
            'family_match': f_match,
        })

        if verbose:
            status = "OK" if e_match else ("~" if f_match else "X")
            print(f"  [{status}] {description:30s} | expected={expected_emotion:15s} | "
                  f"got={predicted_emotion:15s} ({result['confidence']:.2f})")

    emotion_acc = emotion_correct / total
    family_acc = family_correct / total

    print(f"\n  Edge Case Results:")
    print(f"    Emotion accuracy: {emotion_correct}/{total} ({emotion_acc:.1%})")
    print(f"    Family accuracy:  {family_correct}/{total} ({family_acc:.1%})")

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
    total = 0
    start = time.time()

    for i, row in enumerate(samples):
        text = row['text']
        expected = label_map.get(row['label'], 'neutral')
        expected_family = family_for_emotion(expected)

        result = analyzer.analyze(text)
        predicted = result['dominant_emotion']
        predicted_family = result['family']

        if predicted == expected:
            emotion_correct += 1
        if predicted_family == expected_family:
            family_correct += 1
        total += 1

        if verbose and (i + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"    {i+1}/{len(samples)} ({rate:.1f} samples/sec) "
                  f"emotion={emotion_correct/total:.1%} family={family_correct/total:.1%}")

    emotion_acc = emotion_correct / total
    family_acc = family_correct / total
    elapsed = time.time() - start

    print(f"\n  Validation Set Results ({total} samples, {elapsed:.1f}s):")
    print(f"    Emotion accuracy: {emotion_correct}/{total} ({emotion_acc:.1%})")
    print(f"    Family accuracy:  {family_correct}/{total} ({family_acc:.1%})")

    return emotion_acc, family_acc


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
    if not args.skip_val:
        print(f"\n{'─' * 70}")
        print("  VALIDATION SET (dair-ai/emotion)")
        print(f"{'─' * 70}")
        val_emotion, val_family = run_validation_set(
            analyzer, max_samples=args.val_samples, verbose=not args.quiet
        )

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — {embedding_type}")
    print(f"{'=' * 70}")
    print(f"  Edge cases:  emotion={edge_emotion:.1%}  family={edge_family:.1%}")
    if not args.skip_val and val_emotion is not None:
        print(f"  Validation:  emotion={val_emotion:.1%}  family={val_family:.1%}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()

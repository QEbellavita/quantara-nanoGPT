"""
===============================================================================
QUANTARA NEURAL ECOSYSTEM - Formal Evaluation Script
===============================================================================
Reproducible benchmarking against GoEmotions, SemEval, and held-out datasets.
Maps external emotion taxonomies to Quantara's 32-emotion system and computes
accuracy, weighted F1, family accuracy, and per-emotion breakdowns.

Integrates with:
- Neural Workflow AI Engine
- ML Training & Prediction Systems
- Emotion-Aware Training Engine
- All Dashboard Data Integration
===============================================================================
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from emotion_classifier import (
    EMOTION_FAMILIES,
    FusionHead,
    MultimodalEmotionAnalyzer,
    family_for_emotion,
)

# ─── GoEmotions → Quantara 32-Emotion Mapping (27 entries) ──────────────────
# Maps GoEmotions' emotion labels (excluding neutral) to Quantara taxonomy.
# Key design decisions documented inline.

GOEMOTIONS_MAPPING = {
    'admiration': 'gratitude',
    'amusement': 'fun',
    'anger': 'anger',
    'annoyance': 'frustration',       # annoyance is mild anger → frustration
    'approval': 'pride',
    'caring': 'compassion',
    'confusion': 'worry',
    'curiosity': 'enthusiasm',        # intellectual curiosity → enthusiasm
    'desire': 'love',                 # desire maps to love family
    'disappointment': 'sadness',
    'disapproval': 'contempt',
    'disgust': 'disgust',
    'embarrassment': 'shame',
    'excitement': 'excitement',
    'fear': 'fear',
    'gratitude': 'gratitude',
    'grief': 'grief',
    'joy': 'joy',
    'love': 'love',
    'nervousness': 'anxiety',         # nervousness → anxiety (fear family)
    'optimism': 'hope',
    'pride': 'pride',
    'realization': 'mindfulness',     # sudden awareness → mindfulness
    'relief': 'relief',
    'remorse': 'guilt',
    'sadness': 'sadness',
    'surprise': 'surprise',
}

# ─── SemEval → Quantara 32-Emotion Mapping (11 entries) ─────────────────────

SEMEVAL_MAPPING = {
    'anger': 'anger',
    'anticipation': 'enthusiasm',
    'disgust': 'disgust',
    'fear': 'fear',
    'joy': 'joy',
    'love': 'love',
    'optimism': 'hope',
    'pessimism': 'worry',
    'sadness': 'sadness',
    'surprise': 'surprise',
    'trust': 'calm',
}


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: list of ground-truth emotion labels
        y_pred: list of predicted emotion labels

    Returns:
        dict with accuracy, weighted_f1, family_accuracy, per_emotion,
        confusion_matrix, and labels
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Family-level accuracy
    family_true = [family_for_emotion(e) for e in y_true]
    family_pred = [family_for_emotion(e) for e in y_pred]
    family_accuracy = accuracy_score(family_true, family_pred)

    # Per-emotion breakdown
    unique_labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_labels, zero_division=0
    )

    per_emotion = {}
    for i, label in enumerate(unique_labels):
        per_emotion[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    return {
        'accuracy': float(accuracy),
        'weighted_f1': float(weighted_f1),
        'family_accuracy': float(family_accuracy),
        'per_emotion': per_emotion,
        'confusion_matrix': cm.tolist(),
        'labels': unique_labels,
    }


# ─── Dataset Loaders ─────────────────────────────────────────────────────────

def load_goemotions():
    """
    Load GoEmotions dataset from HuggingFace.
    Returns (texts, labels) mapped to Quantara taxonomy.
    Only includes single-label examples that map to a valid emotion.
    """
    from datasets import load_dataset

    print("[Eval] Loading GoEmotions from HuggingFace...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")

    # GoEmotions label names
    label_names = ds.features['labels'].feature.names

    texts = []
    labels = []

    for example in ds:
        # Only use single-label examples for clean evaluation
        if len(example['labels']) != 1:
            continue

        go_label = label_names[example['labels'][0]]

        # Skip neutral - not in our mapping
        if go_label == 'neutral':
            continue

        quantara_emotion = GOEMOTIONS_MAPPING.get(go_label)
        if quantara_emotion is not None:
            texts.append(example['text'])
            labels.append(quantara_emotion)

    print(f"[Eval] GoEmotions: {len(texts)} single-label mapped examples")
    return texts, labels


def load_semeval():
    """
    Load SemEval-2018 Task 1 (emotion classification) from HuggingFace.
    Returns (texts, labels) mapped to Quantara taxonomy.
    """
    from datasets import load_dataset

    print("[Eval] Loading SemEval from HuggingFace...")
    try:
        ds = load_dataset("sem_eval_2018_task_1", "subtask5.english", split="test")
    except Exception:
        # Try alternative name
        try:
            ds = load_dataset("tweet_eval", "emotion", split="test")
            # tweet_eval has different label set: anger, joy, optimism, sadness
            label_names = ['anger', 'joy', 'optimism', 'sadness']
            texts = []
            labels = []
            for example in ds:
                tweet_label = label_names[example['label']]
                quantara_emotion = SEMEVAL_MAPPING.get(tweet_label)
                if quantara_emotion:
                    texts.append(example['text'])
                    labels.append(quantara_emotion)
            print(f"[Eval] SemEval (tweet_eval): {len(texts)} mapped examples")
            return texts, labels
        except Exception as e:
            print(f"[Eval] Could not load SemEval dataset: {e}")
            return [], []

    texts = []
    labels = []

    for example in ds:
        # SemEval multi-label: take the dominant emotion
        emotion_cols = [k for k in example.keys() if k != 'ID' and k != 'Tweet']
        best_emotion = None
        best_score = -1

        for col in emotion_cols:
            if col.lower() in SEMEVAL_MAPPING and example[col] > best_score:
                best_score = example[col]
                best_emotion = col.lower()

        if best_emotion and best_score > 0:
            quantara_emotion = SEMEVAL_MAPPING[best_emotion]
            tweet_text = example.get('Tweet', example.get('text', ''))
            if tweet_text:
                texts.append(tweet_text)
                labels.append(quantara_emotion)

    print(f"[Eval] SemEval: {len(texts)} mapped examples")
    return texts, labels


def load_held_out():
    """
    Load held-out evaluation split from local CSV data.
    Takes the last 20% of the external_emotion_data.csv as held-out.
    """
    data_path = Path(__file__).parent / "data" / "external_datasets" / "external_emotion_data.csv"

    if not data_path.exists():
        print(f"[Eval] Held-out data not found at {data_path}")
        return [], []

    print(f"[Eval] Loading held-out data from {data_path}...")
    df = pd.read_csv(data_path)

    # Last 20% as held-out
    split_idx = int(len(df) * 0.8)
    held_out = df.iloc[split_idx:]

    texts = held_out['text'].tolist()
    labels = held_out['emotion'].tolist()

    # Filter to valid emotions only
    valid_emotions = set(FusionHead.EMOTIONS)
    filtered_texts = []
    filtered_labels = []
    for text, label in zip(texts, labels):
        if label in valid_emotions and isinstance(text, str) and len(text.strip()) > 0:
            filtered_texts.append(text)
            filtered_labels.append(label)

    print(f"[Eval] Held-out: {len(filtered_texts)} examples (last 20% of training data)")
    return filtered_texts, filtered_labels


# ─── Evaluation Runner ───────────────────────────────────────────────────────

def evaluate_dataset(analyzer, texts, labels, with_biometrics=False):
    """
    Run the analyzer on a dataset and collect predictions.

    Args:
        analyzer: MultimodalEmotionAnalyzer instance
        texts: list of input texts
        labels: list of ground-truth labels (unused, for signature consistency)
        with_biometrics: if True, generate synthetic biometric signals

    Returns:
        list of predicted emotion labels
    """
    predictions = []

    for i, text in enumerate(texts):
        biometrics = None
        if with_biometrics:
            # Generate plausible synthetic biometrics for evaluation
            biometrics = {
                'hr': 70 + np.random.normal(0, 10),
                'hrv': 50 + np.random.normal(0, 15),
                'eda': 2.0 + np.random.normal(0, 1.0),
            }

        try:
            result = analyzer.analyze(text, biometrics=biometrics)
            predictions.append(result['emotion'])
        except Exception as e:
            predictions.append('neutral')  # fallback on error

        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(texts)}] processed...")

    return predictions


# ─── Visualization ────────────────────────────────────────────────────────────

def save_confusion_matrix_plot(metrics, output_dir, dataset_name=""):
    """Save confusion matrix as PNG image."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[Eval] matplotlib/seaborn not installed, skipping confusion matrix plot")
        return

    cm = np.array(metrics['confusion_matrix'])
    labels = metrics['labels']

    # Only plot if reasonable size
    if len(labels) > 25:
        # Show top-15 most common labels
        support = [metrics['per_emotion'].get(l, {}).get('support', 0) for l in labels]
        top_indices = np.argsort(support)[-15:]
        cm = cm[np.ix_(top_indices, top_indices)]
        labels = [labels[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), max(8, len(labels) * 0.5)))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    title = f'Confusion Matrix'
    if dataset_name:
        title += f' - {dataset_name}'
    ax.set_title(title)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filename = f"confusion_matrix_{dataset_name.lower().replace(' ', '_')}.png" if dataset_name else "confusion_matrix.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"[Eval] Confusion matrix saved to {filepath}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantara Emotion Classifier - Formal Evaluation"
    )
    parser.add_argument(
        '--datasets', type=str, default='all',
        choices=['all', 'goemotions', 'semeval', 'held-out'],
        help='Which dataset(s) to evaluate on (default: all)'
    )
    parser.add_argument(
        '--output', type=str, default='results',
        help='Output directory for results (default: results)'
    )
    parser.add_argument(
        '--text-only', action='store_true',
        help='Evaluate with text-only (no biometrics)'
    )
    parser.add_argument(
        '--with-biometrics', action='store_true',
        help='Evaluate with synthetic biometric signals'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to classifier checkpoint'
    )

    args = parser.parse_args()

    # Determine which datasets to run
    if args.datasets == 'all':
        dataset_names = ['goemotions', 'semeval', 'held-out']
    else:
        dataset_names = [args.datasets]

    # Load analyzer
    print("=" * 70)
    print("QUANTARA EMOTION CLASSIFIER - FORMAL EVALUATION")
    print("=" * 70)

    analyzer_kwargs = {}
    if args.checkpoint:
        analyzer_kwargs['classifier_checkpoint'] = args.checkpoint

    print("\n[Eval] Loading MultimodalEmotionAnalyzer...")
    analyzer = MultimodalEmotionAnalyzer(**analyzer_kwargs)
    print("[Eval] Analyzer loaded.\n")

    # Results collection
    all_results = {}
    report = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': args.checkpoint,
        'datasets': {},
    }

    # Dataset loaders
    loaders = {
        'goemotions': ('GoEmotions', load_goemotions),
        'semeval': ('SemEval', load_semeval),
        'held-out': ('Held-Out', load_held_out),
    }

    for ds_key in dataset_names:
        ds_name, loader_fn = loaders[ds_key]
        print(f"\n{'─' * 50}")
        print(f"Evaluating: {ds_name}")
        print(f"{'─' * 50}")

        texts, labels = loader_fn()
        if not texts:
            print(f"[Eval] No data for {ds_name}, skipping.")
            continue

        # Text-only evaluation (default or --text-only)
        print(f"\n[Eval] Running text-only evaluation on {len(texts)} examples...")
        preds_text = evaluate_dataset(analyzer, texts, labels, with_biometrics=False)
        metrics_text = compute_metrics(labels, preds_text)

        all_results[ds_key] = {
            'text_only': metrics_text,
        }
        report['datasets'][ds_key] = {
            'name': ds_name,
            'n_examples': len(texts),
            'text_only': {
                'accuracy': metrics_text['accuracy'],
                'weighted_f1': metrics_text['weighted_f1'],
                'family_accuracy': metrics_text['family_accuracy'],
            },
        }

        # Print summary
        print(f"\n  {ds_name} (text-only):")
        print(f"    Accuracy:        {metrics_text['accuracy']:.4f}")
        print(f"    Weighted F1:     {metrics_text['weighted_f1']:.4f}")
        print(f"    Family Accuracy: {metrics_text['family_accuracy']:.4f}")

        # Save confusion matrix
        save_confusion_matrix_plot(metrics_text, args.output, ds_name)

        # Biometric evaluation (if requested and held-out)
        if args.with_biometrics and ds_key == 'held-out':
            print(f"\n[Eval] Running biometric evaluation on {len(texts)} examples...")
            preds_bio = evaluate_dataset(analyzer, texts, labels, with_biometrics=True)
            metrics_bio = compute_metrics(labels, preds_bio)

            all_results[ds_key]['with_biometrics'] = metrics_bio
            report['datasets'][ds_key]['with_biometrics'] = {
                'accuracy': metrics_bio['accuracy'],
                'weighted_f1': metrics_bio['weighted_f1'],
                'family_accuracy': metrics_bio['family_accuracy'],
            }

            # Compute fusion lift
            fusion_lift = {
                'accuracy_lift': metrics_bio['accuracy'] - metrics_text['accuracy'],
                'f1_lift': metrics_bio['weighted_f1'] - metrics_text['weighted_f1'],
                'family_accuracy_lift': metrics_bio['family_accuracy'] - metrics_text['family_accuracy'],
            }
            report['datasets'][ds_key]['fusion_lift'] = fusion_lift

            print(f"\n  {ds_name} (with biometrics):")
            print(f"    Accuracy:        {metrics_bio['accuracy']:.4f}")
            print(f"    Weighted F1:     {metrics_bio['weighted_f1']:.4f}")
            print(f"    Family Accuracy: {metrics_bio['family_accuracy']:.4f}")
            print(f"\n  Fusion Lift:")
            print(f"    Accuracy:        {fusion_lift['accuracy_lift']:+.4f}")
            print(f"    Weighted F1:     {fusion_lift['f1_lift']:+.4f}")
            print(f"    Family Accuracy: {fusion_lift['family_accuracy_lift']:+.4f}")

            save_confusion_matrix_plot(metrics_bio, args.output, f"{ds_name}_biometric")

    # Summary table
    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<20} {'Accuracy':>10} {'Weighted F1':>12} {'Family Acc':>12}")
    print(f"{'─' * 54}")

    for ds_key, ds_report in report['datasets'].items():
        name = ds_report['name']
        text_metrics = ds_report.get('text_only', {})
        print(f"{name:<20} {text_metrics.get('accuracy', 0):.4f}     {text_metrics.get('weighted_f1', 0):.4f}       {text_metrics.get('family_accuracy', 0):.4f}")

        if 'with_biometrics' in ds_report:
            bio_metrics = ds_report['with_biometrics']
            print(f"{'  + biometrics':<20} {bio_metrics.get('accuracy', 0):.4f}     {bio_metrics.get('weighted_f1', 0):.4f}       {bio_metrics.get('family_accuracy', 0):.4f}")

    # Save JSON report
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, f"eval_{datetime.now().strftime('%Y-%m-%d')}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[Eval] Report saved to {report_path}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Fold Pruning Evaluation

Tests different fold combinations to find optimal subset.
Strategy: Remove weakest performing folds to improve efficiency without sacrificing performance.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.metrics import f1_score

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration
RESULTS_DIR = project_root / "results" / "unified_v3"
OUTPUT_FILE = RESULTS_DIR / "fold_pruning_results.json"

# Load pre-computed predictions and thresholds
PREDICTIONS_FILE = RESULTS_DIR / "optimized_predictions.npy"
LABELS_FILE = RESULTS_DIR / "ground_truth.npy"
THRESHOLDS_FILE = RESULTS_DIR / "per_fold_thresholds.json"

CLASS_NAMES = [
    "Diabetic_Retinopathy",
    "Glaucoma",
    "Cataract",
    "AMD",
    "Hypertension",
    "Myopia",
    "Other_Diseases"
]

def print_section(title):
    print(f"\n{'='*80}")
    print(title)
    print('='*80)

def load_fold_data():
    """Load saved fold predictions from optimization run."""
    print("Loading fold predictions from previous evaluation...")
    
    # Check if individual fold predictions were saved
    fold_probs_files = []
    for i in range(5):
        fold_file = RESULTS_DIR / f"fold_{i}_probs.npy"
        if not fold_file.exists():
            print(f"⚠️  Fold {i} predictions not found. Need to re-run with save_fold_preds=True")
            return None
        fold_probs_files.append(fold_file)
    
    # Load fold probabilities
    fold_probs = [np.load(f) for f in fold_probs_files]
    
    # Load thresholds
    with open(THRESHOLDS_FILE, 'r') as f:
        threshold_data = json.load(f)
    
    fold_thresholds = np.array(threshold_data['fold_thresholds'])
    fold_f1_scores = threshold_data['fold_f1_scores']
    
    # Load labels
    labels = np.load(LABELS_FILE)
    
    return fold_probs, fold_thresholds, fold_f1_scores, labels

def compute_ensemble_predictions(fold_indices, fold_probs, fold_thresholds, fold_f1_scores):
    """
    Compute ensemble predictions using only specified folds.
    
    Args:
        fold_indices: List of fold indices to include (e.g., [0, 1, 4])
        fold_probs: List of probability arrays for all folds
        fold_thresholds: Array of thresholds for all folds
        fold_f1_scores: List of F1 scores for all folds (for weighting)
    
    Returns:
        Binary predictions array
    """
    n_samples, n_classes = fold_probs[0].shape
    
    # Get F1 scores for selected folds and normalize to weights
    selected_f1s = np.array([fold_f1_scores[i] for i in fold_indices])
    fold_weights = selected_f1s / selected_f1s.sum()
    
    # Apply per-fold thresholds
    fold_predictions = []
    for fold_idx in fold_indices:
        probs = fold_probs[fold_idx]
        thresholds = fold_thresholds[fold_idx]
        preds = (probs >= thresholds).astype(int)
        fold_predictions.append(preds)
    
    # Weighted voting
    weighted_predictions = np.zeros((n_samples, n_classes))
    for i, fold_idx in enumerate(fold_indices):
        weighted_predictions += fold_weights[i] * fold_predictions[i]
    
    # Final binary decision
    final_predictions = (weighted_predictions >= 0.5).astype(int)
    
    return final_predictions

def evaluate_fold_combination(fold_indices, fold_probs, fold_thresholds, fold_f1_scores, labels):
    """Evaluate a specific fold combination."""
    predictions = compute_ensemble_predictions(
        fold_indices, fold_probs, fold_thresholds, fold_f1_scores
    )
    
    # Calculate metrics
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    
    return {
        'fold_indices': fold_indices,
        'n_folds': len(fold_indices),
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'per_class_f1': [float(f) for f in per_class_f1],
        'fold_names': [f"Fold {i}" for i in fold_indices]
    }

def test_all_combinations(fold_probs, fold_thresholds, fold_f1_scores, labels):
    """Test all meaningful fold combinations."""
    results = []
    
    print("\nTesting fold combinations...")
    print(f"{'Combination':<30} {'N':<5} {'Macro F1':<12} {'Micro F1':<12}")
    print("-" * 65)
    
    # Test individual folds (for reference)
    for i in range(5):
        result = evaluate_fold_combination([i], fold_probs, fold_thresholds, fold_f1_scores, labels)
        results.append(result)
        print(f"{'Fold ' + str(i):<30} {1:<5} {result['macro_f1']:<12.4f} {result['micro_f1']:<12.4f}")
    
    # Test pairs
    for combo in combinations(range(5), 2):
        result = evaluate_fold_combination(list(combo), fold_probs, fold_thresholds, fold_f1_scores, labels)
        results.append(result)
        combo_str = ", ".join([f"Fold {i}" for i in combo])
        print(f"{combo_str:<30} {2:<5} {result['macro_f1']:<12.4f} {result['micro_f1']:<12.4f}")
    
    # Test triplets
    for combo in combinations(range(5), 3):
        result = evaluate_fold_combination(list(combo), fold_probs, fold_thresholds, fold_f1_scores, labels)
        results.append(result)
        combo_str = ", ".join([f"Fold {i}" for i in combo])
        print(f"{combo_str:<30} {3:<5} {result['macro_f1']:<12.4f} {result['micro_f1']:<12.4f}")
    
    # Test quadruplets
    for combo in combinations(range(5), 4):
        result = evaluate_fold_combination(list(combo), fold_probs, fold_thresholds, fold_f1_scores, labels)
        results.append(result)
        combo_str = ", ".join([f"Fold {i}" for i in combo])
        print(f"{combo_str:<30} {4:<5} {result['macro_f1']:<12.4f} {result['micro_f1']:<12.4f}")
    
    # Test all 5 (baseline)
    result = evaluate_fold_combination([0, 1, 2, 3, 4], fold_probs, fold_thresholds, fold_f1_scores, labels)
    results.append(result)
    print(f"{'All 5 folds (baseline)':<30} {5:<5} {result['macro_f1']:<12.4f} {result['micro_f1']:<12.4f}")
    
    return results

def analyze_results(results, fold_f1_scores):
    """Analyze and recommend best fold combinations."""
    print_section("ANALYSIS & RECOMMENDATIONS")
    
    # Find baseline (all 5 folds)
    baseline = next(r for r in results if r['n_folds'] == 5)
    baseline_f1 = baseline['macro_f1']
    
    print(f"\nBaseline (All 5 Folds): {baseline_f1:.4f} macro F1")
    print(f"\nFold Performance (Validation):")
    for i, f1 in enumerate(fold_f1_scores):
        print(f"  Fold {i}: {f1:.4f}")
    
    # Find best combination for each size
    print("\n" + "="*80)
    print("BEST COMBINATIONS BY SIZE")
    print("="*80)
    
    for n in range(1, 5):
        size_results = [r for r in results if r['n_folds'] == n]
        best = max(size_results, key=lambda x: x['macro_f1'])
        
        f1_diff = best['macro_f1'] - baseline_f1
        f1_diff_pct = (f1_diff / baseline_f1) * 100
        speedup = 5.0 / n
        
        print(f"\n{n} Fold(s): {', '.join(best['fold_names'])}")
        print(f"  Macro F1: {best['macro_f1']:.4f} ({f1_diff:+.4f}, {f1_diff_pct:+.2f}% vs baseline)")
        print(f"  Micro F1: {best['micro_f1']:.4f}")
        print(f"  Speedup:  {speedup:.1f}x faster inference")
        
        if abs(f1_diff) < 0.01:  # Within 1% of baseline
            print(f"  ⭐ RECOMMENDED: Maintains performance with {speedup:.1f}x speedup!")
    
    # Find best overall trade-off
    print("\n" + "="*80)
    print("OPTIMAL TRADE-OFF RECOMMENDATION")
    print("="*80)
    
    # Look for 3-fold combinations within 0.5% of baseline
    three_fold_results = [r for r in results if r['n_folds'] == 3]
    acceptable = [r for r in three_fold_results if (r['macro_f1'] - baseline_f1) >= -0.005]
    
    if acceptable:
        best_tradeoff = max(acceptable, key=lambda x: x['macro_f1'])
        print(f"\nRecommended: {', '.join(best_tradeoff['fold_names'])}")
        print(f"  Macro F1: {best_tradeoff['macro_f1']:.4f}")
        print(f"  Performance: {((best_tradeoff['macro_f1'] - baseline_f1) / baseline_f1) * 100:+.2f}% vs baseline")
        print(f"  Inference speedup: 1.67x")
        print(f"  Memory reduction: 40%")
    else:
        print("\nNo 3-fold combination maintains performance within 0.5%")
        print("Recommend using all 5 folds for maximum accuracy")
    
    return baseline, results

def main():
    print("="*80)
    print("FOLD PRUNING EVALUATION")
    print("Testing all fold combinations to optimize efficiency")
    print("="*80)
    
    # Load data
    print_section("Loading Data")
    data = load_fold_data()
    
    if data is None:
        print("\n⚠️  ERROR: Fold predictions not found!")
        print("\nThe optimization evaluation needs to save individual fold predictions.")
        print("Please re-run evaluate_optimized_memory_efficient.py first.")
        return
    
    fold_probs, fold_thresholds, fold_f1_scores, labels = data
    print(f"Loaded predictions for {len(fold_probs)} folds")
    print(f"Dataset: {labels.shape[0]} samples, {labels.shape[1]} classes")
    
    # Test all combinations
    print_section("Evaluating Combinations")
    results = test_all_combinations(fold_probs, fold_thresholds, fold_f1_scores, labels)
    
    # Analyze
    baseline, all_results = analyze_results(results, fold_f1_scores)
    
    # Save results
    output_data = {
        'baseline': baseline,
        'all_combinations': all_results,
        'fold_f1_scores': fold_f1_scores,
        'class_names': CLASS_NAMES
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_FILE}")
    
    print("\n" + "="*80)
    print("FOLD PRUNING EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

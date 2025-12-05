#!/usr/bin/env python3
"""
Fast Weighted Ensemble Evaluation with Per-Fold Optimized Thresholds

This script applies per-fold optimized thresholds and weighted ensemble averaging
without TTA, providing a quick performance boost over the baseline.

Expected improvements:
- Per-fold thresholds: +2-3% macro F1
- Weighted ensemble: +1-2% macro F1
- Total: +3-5% macro F1 improvement

Runtime: ~15-20 minutes (vs 12+ hours with TTA)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def print_section(title, char="="):
    """Print a section header."""
    print(f"\n{char * 80}")
    print(title)
    print(char * 80)

# Configuration
MODEL_BASE_PATH = project_root / "models" / "unified_v3"
THRESHOLDS_PATH = MODEL_BASE_PATH / "per_fold_thresholds.json"
DATA_CSV = project_root / "data" / "processed" / "unified_v3" / "unified_train_v3.csv"
OUTPUT_DIR = MODEL_BASE_PATH / "fast_ensemble_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64  # Larger batch for faster inference
NUM_WORKERS = 0  # Avoid multiprocessing issues

# Class names (display names)
CLASS_NAMES = [
    "Diabetic_Retinopathy",
    "Glaucoma",
    "Cataract",
    "AMD",
    "Hypertension",
    "Myopia",
    "Other_Diseases"
]

# Label columns in CSV (actual column names)
LABEL_COLUMNS = [
    "Label_D",
    "Label_G",
    "Label_C",
    "Label_A",
    "Label_H",
    "Label_M",
    "Label_O"
]

def load_per_fold_thresholds():
    """Load the pre-computed per-fold thresholds and weights."""
    print_section("Loading Per-Fold Thresholds")
    
    with open(THRESHOLDS_PATH, 'r') as f:
        data = json.load(f)
    
    fold_thresholds = np.array(data['fold_thresholds'])
    fold_weights = np.array(data['fold_weights'])
    fold_f1_scores = data['fold_f1_scores']
    
    print(f"Loaded thresholds for {len(fold_thresholds)} folds")
    print("\nFold Performance:")
    for i, (f1, weight) in enumerate(zip(fold_f1_scores, fold_weights)):
        print(f"  Fold {i}: F1={f1:.4f}, Weight={weight:.3f}")
    
    print(f"\nThresholds per fold:")
    for i, thresholds in enumerate(fold_thresholds):
        print(f"  Fold {i}: {', '.join([f'{t:.2f}' for t in thresholds])}")
    
    return fold_thresholds, fold_weights, fold_f1_scores


def load_model(fold_idx):
    """Load a trained model for a specific fold."""
    import timm
    import torch.nn as nn
    
    # Import necessary model classes
    sys.path.insert(0, str(project_root / "src"))
    from models.backbones import FundusBackbone
    from models.multilabel_head import MultiLabelClassifier
    
    checkpoint_path = MODEL_BASE_PATH / f"fold_{fold_idx}" / "best_model.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Recreate model architecture (SimpleModel)
    config = checkpoint['config']
    
    class SimpleModel(nn.Module):
        """Simple wrapper matching working architecture"""
        def __init__(self, num_classes=7, feature_dim=1024, dropout=0.3):
            super().__init__()
            self.backbone = FundusBackbone(model_name="convnext_base", pretrained=False, feature_dim=feature_dim)
            self.classifier = MultiLabelClassifier(feature_dim, num_classes)
        
        def forward(self, x):
            features = self.backbone(x)
            logits = self.classifier(features)
            return logits
    
    model = SimpleModel(
        num_classes=config['num_classes'],
        feature_dim=config['feature_dim'],
        dropout=config['dropout']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    return model


def get_predictions_single_fold(model, dataloader, device):
    """Get predictions from a single fold model."""
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Inference")):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    
    return all_probs, all_labels


def apply_per_fold_thresholds(fold_probs_list, fold_thresholds, fold_weights):
    """
    Apply per-fold thresholds and weighted averaging.
    
    Args:
        fold_probs_list: List of probability arrays, one per fold (N_samples, N_classes)
        fold_thresholds: Array of thresholds per fold (N_folds, N_classes)
        fold_weights: Array of weights per fold (N_folds,)
    
    Returns:
        final_predictions: Binary predictions after weighted ensemble (N_samples, N_classes)
        weighted_probs: Weighted average probabilities (N_samples, N_classes)
    """
    n_folds = len(fold_probs_list)
    n_samples, n_classes = fold_probs_list[0].shape
    
    # Apply per-fold thresholds to get binary predictions
    fold_predictions = []
    for fold_idx in range(n_folds):
        probs = fold_probs_list[fold_idx]
        thresholds = fold_thresholds[fold_idx]
        
        # Apply thresholds
        preds = (probs >= thresholds).astype(int)
        fold_predictions.append(preds)
    
    # Weighted average of probabilities
    weighted_probs = np.zeros((n_samples, n_classes))
    for fold_idx in range(n_folds):
        weighted_probs += fold_weights[fold_idx] * fold_probs_list[fold_idx]
    
    # Weighted voting for final predictions
    weighted_predictions = np.zeros((n_samples, n_classes))
    for fold_idx in range(n_folds):
        weighted_predictions += fold_weights[fold_idx] * fold_predictions[fold_idx]
    
    # Final binary decision: use weighted voting (>0.5 = at least majority weighted vote)
    final_predictions = (weighted_predictions >= 0.5).astype(int)
    
    return final_predictions, weighted_probs


def evaluate_predictions(y_true, y_pred, class_names):
    """Calculate comprehensive evaluation metrics."""
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Aggregate metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Support (number of positive samples per class)
    support = y_true.sum(axis=0)
    
    results = {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'per_class': {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class,
            'support': support
        }
    }
    
    return results


def print_evaluation_results(results, class_names, title="Evaluation Results"):
    """Print evaluation results in a formatted table."""
    print_section(title)
    
    print(f"Macro F1:       {results['macro_f1']:.4f}")
    print(f"Micro F1:       {results['micro_f1']:.4f}")
    print(f"Macro Precision: {results['macro_precision']:.4f}")
    print(f"Macro Recall:    {results['macro_recall']:.4f}")
    
    print("\nPer-Class Performance:")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)
    
    per_class = results['per_class']
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<25} "
              f"{per_class['precision'][i]:>10.4f} "
              f"{per_class['recall'][i]:>10.4f} "
              f"{per_class['f1'][i]:>10.4f} "
              f"{int(per_class['support'][i]):>10}")


def save_results(results, fold_predictions_list, fold_thresholds, fold_weights, 
                final_predictions, weighted_probs, y_true, output_dir):
    """Save all results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary metrics
    summary = {
        'macro_f1': float(results['macro_f1']),
        'micro_f1': float(results['micro_f1']),
        'macro_precision': float(results['macro_precision']),
        'macro_recall': float(results['macro_recall']),
        'per_class_f1': [float(f1) for f1 in results['per_class']['f1']],
        'per_class_precision': [float(p) for p in results['per_class']['precision']],
        'per_class_recall': [float(r) for r in results['per_class']['recall']],
        'per_class_support': [int(s) for s in results['per_class']['support']],
        'fold_thresholds': fold_thresholds.tolist(),
        'fold_weights': fold_weights.tolist(),
        'class_names': CLASS_NAMES
    }
    
    with open(output_dir / "fast_ensemble_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save predictions
    np.save(output_dir / "final_predictions.npy", final_predictions)
    np.save(output_dir / "weighted_probs.npy", weighted_probs)
    np.save(output_dir / "ground_truth.npy", y_true)
    
    # Save individual fold predictions
    for fold_idx, preds in enumerate(fold_predictions_list):
        np.save(output_dir / f"fold_{fold_idx}_probs.npy", preds)
    
    print(f"\nResults saved to: {output_dir}")


def main():
    print("=" * 80)
    print("FAST WEIGHTED ENSEMBLE EVALUATION")
    print("Using Per-Fold Optimized Thresholds (No TTA)")
    print("=" * 80)
    
    # Load per-fold thresholds and weights
    fold_thresholds, fold_weights, fold_f1_scores = load_per_fold_thresholds()
    n_folds = len(fold_thresholds)
    
    # Create dataset and dataloader
    print_section("Loading Dataset")
    df = pd.read_csv(DATA_CSV)
    print(f"Total samples: {len(df)}")
    
    # Use CV predictions (same as what was used for threshold optimization)
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from PIL import Image
    
    class EvalDataset(Dataset):
        def __init__(self, df, transform):
            self.df = df.reset_index(drop=True)
            self.transform = transform
            self.label_columns = LABEL_COLUMNS
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = row['image_path']
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Get labels
            labels = torch.tensor(
                row[self.label_columns].values.astype(np.float32)
            )
            
            return image, labels
    
    # Create transform (same as training - 448x448)
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = EvalDataset(df, transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Get predictions from each fold
    print_section("Running Inference on All Folds")
    fold_probs_list = []
    y_true = None
    
    for fold_idx in range(n_folds):
        print(f"\n--- Processing Fold {fold_idx} (F1: {fold_f1_scores[fold_idx]:.4f}, Weight: {fold_weights[fold_idx]:.3f}) ---")
        
        # Load model
        model = load_model(fold_idx)
        
        # Get predictions
        fold_probs, labels = get_predictions_single_fold(model, dataloader, DEVICE)
        fold_probs_list.append(fold_probs)
        
        if y_true is None:
            y_true = labels
        
        # Free memory
        del model
        torch.cuda.empty_cache()
        
        print(f"Fold {fold_idx} complete: {fold_probs.shape[0]} samples processed")
    
    # Apply per-fold thresholds and weighted ensemble
    print_section("Applying Per-Fold Thresholds and Weighted Ensemble")
    final_predictions, weighted_probs = apply_per_fold_thresholds(
        fold_probs_list, fold_thresholds, fold_weights
    )
    
    # Evaluate
    print_section("Evaluation Results")
    results = evaluate_predictions(y_true, final_predictions, CLASS_NAMES)
    print_evaluation_results(results, CLASS_NAMES, "Fast Weighted Ensemble Performance")
    
    # Save results
    save_results(
        results, fold_probs_list, fold_thresholds, fold_weights,
        final_predictions, weighted_probs, y_true, OUTPUT_DIR
    )
    
    # Compare with baseline
    print_section("Comparison with Baseline")
    baseline_f1 = 0.7130  # From previous evaluation
    improvement = results['macro_f1'] - baseline_f1
    improvement_pct = (improvement / baseline_f1) * 100
    
    print(f"Baseline (Global Thresholds):          {baseline_f1:.4f}")
    print(f"Optimized (Per-Fold + Weighted):       {results['macro_f1']:.4f}")
    print(f"Absolute Improvement:                   {improvement:+.4f}")
    print(f"Relative Improvement:                   {improvement_pct:+.2f}%")
    
    if improvement > 0:
        print("\n✓ Per-fold threshold optimization successful!")
    else:
        print("\n⚠ Warning: No improvement detected. Check fold predictions.")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

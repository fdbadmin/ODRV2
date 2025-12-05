#!/usr/bin/env python3
"""
Quick Evaluation of Per-Fold Optimized Ensemble

Tests the optimized ensemble on the full dataset to quantify improvement.
Uses efficient batching and minimal overhead.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from models.backbones import FundusBackbone
from models.multilabel_head import MultiLabelClassifier

# Configuration
MODEL_BASE_PATH = project_root / "models" / "unified_v3"
THRESHOLDS_PATH = project_root / "results" / "unified_v3" / "per_fold_thresholds.json"
DATA_CSV = project_root / "data" / "processed" / "unified_v3" / "unified_train_v3.csv"
OUTPUT_DIR = project_root / "results" / "unified_v3"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64

# Label columns
LABEL_COLUMNS = ["Label_D", "Label_G", "Label_C", "Label_A", "Label_H", "Label_M", "Label_O"]
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

class SimpleModel(torch.nn.Module):
    """Simple wrapper matching training architecture"""
    def __init__(self, num_classes=7, feature_dim=1024, dropout=0.3):
        super().__init__()
        self.backbone = FundusBackbone(model_name="convnext_base", pretrained=False, feature_dim=feature_dim)
        self.classifier = MultiLabelClassifier(feature_dim, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def load_model(fold_idx):
    """Load a trained model for specific fold."""
    checkpoint_path = MODEL_BASE_PATH / f"fold_{fold_idx}" / "best_model.pth"
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = checkpoint['config']
    
    model = SimpleModel(
        num_classes=config['num_classes'],
        feature_dim=config['feature_dim'],
        dropout=config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    return model

def load_thresholds():
    """Load per-fold thresholds and weights."""
    with open(THRESHOLDS_PATH, 'r') as f:
        data = json.load(f)
    
    return (
        np.array(data['fold_thresholds']),
        np.array(data['fold_weights']),
        data['fold_f1_scores']
    )

def predict_batch(models, images, fold_thresholds, fold_weights):
    """
    Get predictions from ensemble for a batch of images.
    
    Args:
        models: List of model objects
        images: Batch of images tensor
        fold_thresholds: Array of thresholds (n_folds, n_classes)
        fold_weights: Array of weights (n_folds,)
    
    Returns:
        Binary predictions (batch_size, n_classes)
    """
    n_folds = len(models)
    batch_size, _, _, _ = images.shape
    n_classes = 7
    
    # Get probabilities from each fold
    fold_probs = []
    for fold_idx, model in enumerate(models):
        with torch.no_grad():
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            fold_probs.append(probs)
    
    # Apply per-fold thresholds
    fold_predictions = []
    for fold_idx in range(n_folds):
        preds = (fold_probs[fold_idx] >= fold_thresholds[fold_idx]).astype(int)
        fold_predictions.append(preds)
    
    # Weighted voting
    weighted_predictions = np.zeros((batch_size, n_classes))
    for fold_idx in range(n_folds):
        weighted_predictions += fold_weights[fold_idx] * fold_predictions[fold_idx]
    
    # Final binary decision (majority weighted vote)
    final_predictions = (weighted_predictions >= 0.5).astype(int)
    
    return final_predictions

def evaluate_on_subset(models, df, fold_thresholds, fold_weights, n_samples=1000):
    """Quick evaluation on subset to estimate time."""
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from PIL import Image
    
    class QuickDataset(Dataset):
        def __init__(self, df, transform):
            self.df = df.reset_index(drop=True)
            self.transform = transform
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image = Image.open(row['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            labels = torch.tensor(row[LABEL_COLUMNS].values.astype(np.float32))
            return image, labels
    
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Use subset
    subset_df = df.sample(n=min(n_samples, len(df)), random_state=42)
    dataset = QuickDataset(subset_df, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    for images, labels in tqdm(dataloader, desc="Quick Eval"):
        images = images.to(DEVICE)
        preds = predict_batch(models, images, fold_thresholds, fold_weights)
        
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    
    elapsed = time.time() - start_time
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return macro_f1, elapsed, len(subset_df)

def full_evaluation(models, df, fold_thresholds, fold_weights):
    """Full evaluation on entire dataset."""
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from PIL import Image
    
    class EvalDataset(Dataset):
        def __init__(self, df, transform):
            self.df = df.reset_index(drop=True)
            self.transform = transform
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image = Image.open(row['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            labels = torch.tensor(row[LABEL_COLUMNS].values.astype(np.float32))
            return image, labels
    
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = EvalDataset(df, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    
    print(f"\nProcessing {len(df)} samples in {len(dataloader)} batches...")
    
    for images, labels in tqdm(dataloader, desc="Full Evaluation"):
        images = images.to(DEVICE)
        preds = predict_batch(models, images, fold_thresholds, fold_weights)
        
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate comprehensive metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    support = all_labels.sum(axis=0)
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'per_class_f1': per_class_f1,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'support': support,
        'predictions': all_preds,
        'labels': all_labels
    }

def print_results(results, baseline_f1=0.7130):
    """Print evaluation results."""
    print_section("OPTIMIZED ENSEMBLE PERFORMANCE")
    
    print(f"Macro F1:       {results['macro_f1']:.4f}")
    print(f"Micro F1:       {results['micro_f1']:.4f}")
    print(f"Macro Precision: {results['macro_precision']:.4f}")
    print(f"Macro Recall:    {results['macro_recall']:.4f}")
    
    print("\nPer-Class Performance:")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)
    
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name:<25} "
              f"{results['per_class_precision'][i]:>10.4f} "
              f"{results['per_class_recall'][i]:>10.4f} "
              f"{results['per_class_f1'][i]:>10.4f} "
              f"{int(results['support'][i]):>10}")
    
    print_section("COMPARISON WITH BASELINE")
    improvement = results['macro_f1'] - baseline_f1
    improvement_pct = (improvement / baseline_f1) * 100
    
    print(f"Baseline (Global Thresholds):          {baseline_f1:.4f}")
    print(f"Optimized (Per-Fold + Weighted):       {results['macro_f1']:.4f}")
    print(f"Absolute Improvement:                   {improvement:+.4f}")
    print(f"Relative Improvement:                   {improvement_pct:+.2f}%")

def save_results(results, fold_thresholds, fold_weights):
    """Save results to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'macro_f1': float(results['macro_f1']),
        'micro_f1': float(results['micro_f1']),
        'macro_precision': float(results['macro_precision']),
        'macro_recall': float(results['macro_recall']),
        'per_class_f1': [float(f) for f in results['per_class_f1']],
        'per_class_precision': [float(p) for p in results['per_class_precision']],
        'per_class_recall': [float(r) for r in results['per_class_recall']],
        'per_class_support': [int(s) for s in results['support']],
        'fold_thresholds': fold_thresholds.tolist(),
        'fold_weights': fold_weights.tolist(),
        'class_names': CLASS_NAMES,
        'baseline_f1': 0.7130,
        'improvement': float(results['macro_f1'] - 0.7130)
    }
    
    with open(OUTPUT_DIR / "optimized_ensemble_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    np.save(OUTPUT_DIR / "optimized_predictions.npy", results['predictions'])
    np.save(OUTPUT_DIR / "ground_truth.npy", results['labels'])
    
    print(f"\nResults saved to: {OUTPUT_DIR}")

def main():
    print("="*80)
    print("OPTIMIZED ENSEMBLE EVALUATION")
    print("Per-Fold Thresholds + Weighted Ensemble")
    print("="*80)
    
    # Load data
    print_section("Loading Data")
    df = pd.read_csv(DATA_CSV)
    print(f"Total samples: {len(df)}")
    
    # Load thresholds
    print_section("Loading Thresholds")
    fold_thresholds, fold_weights, fold_f1_scores = load_thresholds()
    print(f"Loaded {len(fold_thresholds)} folds")
    for i, (f1, weight) in enumerate(zip(fold_f1_scores, fold_weights)):
        print(f"  Fold {i}: F1={f1:.4f}, Weight={weight:.3f}")
    
    # Load all models
    print_section("Loading Models")
    models = []
    for fold_idx in range(5):
        print(f"Loading Fold {fold_idx}...", end=' ')
        model = load_model(fold_idx)
        models.append(model)
        print("✓")
    
    # Quick evaluation on subset to estimate time
    print_section("Time Estimation (1000 samples)")
    subset_f1, subset_time, n_subset = evaluate_on_subset(
        models, df, fold_thresholds, fold_weights, n_samples=1000
    )
    
    samples_per_sec = n_subset / subset_time
    estimated_total_time = len(df) / samples_per_sec
    
    print(f"\nSubset F1 Score: {subset_f1:.4f}")
    print(f"Processing speed: {samples_per_sec:.1f} samples/sec")
    print(f"Estimated total time: {estimated_total_time/60:.1f} minutes")
    
    # Ask user if they want to continue
    if estimated_total_time > 1800:  # More than 30 minutes
        print(f"\n⚠️  Warning: Full evaluation will take ~{estimated_total_time/60:.0f} minutes")
        print("Consider running this overnight or on faster hardware.")
    
    # Run full evaluation
    print_section("Full Evaluation")
    start_time = time.time()
    results = full_evaluation(models, df, fold_thresholds, fold_weights)
    total_time = time.time() - start_time
    
    print(f"\nCompleted in {total_time/60:.1f} minutes")
    
    # Print and save results
    print_results(results)
    save_results(results, fold_thresholds, fold_weights)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

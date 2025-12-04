#!/usr/bin/env python3
"""
Evaluate Unified V3 Ensemble Model
5-fold cross-validation ensemble evaluation with threshold optimization
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, multilabel_confusion_matrix
from tqdm import tqdm
import json

from src.data.dataset import FundusDataset
from src.models.backbones import FundusBackbone
from src.models.multilabel_head import MultiLabelClassifier

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

class SimpleModel(nn.Module):
    """Model architecture matching training"""
    def __init__(self, num_classes=7, feature_dim=1024, dropout=0.3):
        super().__init__()
        self.backbone = FundusBackbone(model_name="convnext_base", pretrained=False, feature_dim=feature_dim)
        self.classifier = MultiLabelClassifier(feature_dim, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

def load_ensemble(model_dir, device):
    """Load all 5 fold models"""
    models = []
    model_dir = Path(model_dir)
    
    print("Loading ensemble models...")
    for fold_idx in range(5):
        checkpoint_path = model_dir / f"fold_{fold_idx}" / "best_model.pth"
        if not checkpoint_path.exists():
            print(f"Warning: {checkpoint_path} not found!")
            continue
        
        print(f"  Loading fold {fold_idx}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        model = SimpleModel(num_classes=7, feature_dim=1024, dropout=0.3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        models.append(model)
    
    print(f"Loaded {len(models)} models")
    return models

def get_fold_indices(df, fold_idx, n_splits=5, seed=42):
    """Get validation indices for a specific fold (matching training logic)"""
    from sklearn.model_selection import StratifiedKFold
    
    # Group by patient
    patient_df = df.groupby('global_patient_id').first().reset_index()
    label_cols = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
    
    # Create multi-hot encoding for stratification
    patient_df['label_combo'] = patient_df[label_cols].apply(
        lambda row: ''.join(row.astype(int).astype(str)), axis=1
    )
    
    # Handle rare combinations by grouping them
    value_counts = patient_df['label_combo'].value_counts()
    rare_threshold = max(n_splits, 10)
    rare_combos = set(value_counts[value_counts < rare_threshold].index)
    patient_df['stratify_group'] = patient_df['label_combo'].apply(
        lambda x: 'RARE' if x in rare_combos else x
    )
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    patient_ids = patient_df['global_patient_id'].values
    stratify_labels = patient_df['stratify_group'].values
    
    for current_fold, (train_patient_idx, val_patient_idx) in enumerate(skf.split(patient_ids, stratify_labels)):
        if current_fold == fold_idx:
            val_patient_ids = set(patient_ids[val_patient_idx])
            val_indices = df[df['global_patient_id'].isin(val_patient_ids)].index.tolist()
            return val_indices
    
    return []

def evaluate_with_threshold(y_true, y_scores, threshold=0.5):
    """Evaluate predictions with a specific threshold"""
    y_pred = (y_scores >= threshold).astype(int)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_f1': macro_f1,
        'predictions': y_pred
    }

def optimize_thresholds(y_true, y_scores, class_names):
    """Find optimal threshold per class"""
    n_classes = y_true.shape[1]
    best_thresholds = []
    best_f1s = []
    
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION")
    print("="*80)
    
    for i in range(n_classes):
        y_true_class = y_true[:, i]
        y_scores_class = y_scores[:, i]
        
        best_t = 0.5
        best_f1 = 0.0
        
        # Search range
        for t in np.arange(0.05, 0.96, 0.05):
            y_pred_class = (y_scores_class >= t).astype(int)
            f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        
        best_thresholds.append(best_t)
        best_f1s.append(best_f1)
        
        # Get counts at optimal threshold
        y_pred_opt = (y_scores_class >= best_t).astype(int)
        tp = ((y_true_class == 1) & (y_pred_opt == 1)).sum()
        fn = ((y_true_class == 1) & (y_pred_opt == 0)).sum()
        fp = ((y_true_class == 0) & (y_pred_opt == 1)).sum()
        tn = ((y_true_class == 0) & (y_pred_opt == 0)).sum()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\n{class_names[i]} (Class {i}):")
        print(f"  Optimal Threshold: {best_t:.2f}")
        print(f"  Optimized F1: {best_f1:.4f}")
        print(f"  Sensitivity (Recall): {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Support: {y_true_class.sum():.0f} positive cases")
    
    macro_f1 = np.mean(best_f1s)
    print("\n" + "-"*80)
    print(f"Optimized Macro F1: {macro_f1:.4f}")
    print(f"Optimal Thresholds: {[f'{t:.2f}' for t in best_thresholds]}")
    
    return best_thresholds, best_f1s

def evaluate_ensemble_on_val_folds(models, df, device, batch_size=16):
    """
    Evaluate each model on its own validation fold
    This gives a true cross-validation estimate
    """
    class_names = ['Diabetic Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
    
    all_y_true = []
    all_y_scores = []
    
    print("\n" + "="*80)
    print("EVALUATING ENSEMBLE ON VALIDATION FOLDS")
    print("="*80)
    
    for fold_idx in range(len(models)):
        print(f"\n--- Fold {fold_idx} ---")
        
        # Get validation indices for this fold
        val_indices = get_fold_indices(df, fold_idx)
        val_df = df.loc[val_indices].reset_index(drop=True)
        
        print(f"Validation set size: {len(val_df)} samples")
        
        # Create dataset and loader
        val_dataset = FundusDataset(
            df=val_df,
            image_size=(448, 448),
            augment=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Run inference
        model = models[fold_idx]
        fold_y_true = []
        fold_y_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold_idx} inference"):
                images = batch['image'].to(device)
                labels = batch['labels'].cpu().numpy()
                
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                fold_y_true.append(labels)
                fold_y_scores.append(probs)
        
        fold_y_true = np.vstack(fold_y_true)
        fold_y_scores = np.vstack(fold_y_scores)
        
        all_y_true.append(fold_y_true)
        all_y_scores.append(fold_y_scores)
        
        # Quick fold summary
        default_metrics = evaluate_with_threshold(fold_y_true, fold_y_scores, threshold=0.5)
        print(f"Fold {fold_idx} Macro F1 (thresh=0.5): {default_metrics['macro_f1']:.4f}")
    
    # Concatenate all folds
    y_true = np.vstack(all_y_true)
    y_scores = np.vstack(all_y_scores)
    
    print(f"\nTotal samples evaluated: {len(y_true)}")
    
    return y_true, y_scores, class_names

def evaluate_full_ensemble(models, df, device, batch_size=16, thresholds=None):
    """
    Evaluate ensemble by averaging predictions from all models
    Use this on a held-out test set or for final evaluation
    """
    class_names = ['Diabetic Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
    
    print("\n" + "="*80)
    print("FULL ENSEMBLE EVALUATION (Average of all 5 models)")
    print("="*80)
    
    # Create dataset
    dataset = FundusDataset(
        df=df,
        image_size=(448, 448),
        augment=False
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    y_true = []
    y_scores_all_models = [[] for _ in range(len(models))]
    
    print(f"Running inference on {len(df)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Ensemble inference"):
            images = batch['image'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            # Get predictions from each model
            for model_idx, model in enumerate(models):
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                y_scores_all_models[model_idx].append(probs)
            
            y_true.append(labels)
    
    y_true = np.vstack(y_true)
    
    # Average predictions across models
    y_scores_per_model = [np.vstack(scores) for scores in y_scores_all_models]
    y_scores = np.mean(y_scores_per_model, axis=0)
    
    print(f"\nEvaluated {len(y_true)} samples")
    
    # Evaluate with default threshold
    if thresholds is None:
        thresholds = [0.5] * len(class_names)
    
    y_pred = np.zeros_like(y_scores)
    for i, thresh in enumerate(thresholds):
        y_pred[:, i] = (y_scores[:, i] >= thresh).astype(int)
    
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(f"Thresholds: {[f'{t:.2f}' for t in thresholds]}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # Confusion matrices
    print("\n" + "="*80)
    print("CONFUSION MATRICES")
    print("="*80)
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    for i, class_name in enumerate(class_names):
        tn, fp, fn, tp = mcm[i].ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        print(f"\n{class_name}:")
        print(f"  TP: {tp:5d}  FN: {fn:5d}")
        print(f"  FP: {fp:5d}  TN: {tn:5d}")
        print(f"  Sensitivity (Recall): {sensitivity:.4f}")
        print(f"  Specificity:          {specificity:.4f}")
        print(f"  PPV (Precision):      {ppv:.4f}")
        print(f"  NPV:                  {npv:.4f}")
    
    return y_true, y_scores, class_names

def main():
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    csv_path = Path('data/processed/unified_v3/unified_train_v3.csv')
    print(f"\nLoading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Prepare patient IDs (required for fold splitting)
    print("Preparing patient IDs...")
    df['patient_id'] = df['filename'].apply(lambda x: x.split('_')[0])
    df['global_patient_id'] = df['source_dataset'].astype(str) + '_' + df['patient_id'].astype(str)
    
    print(f"Total samples: {len(df)}")
    print(f"Unique patients: {df['global_patient_id'].nunique()}")
    
    # Load ensemble
    models = load_ensemble('models/unified_v3', device)
    
    if len(models) == 0:
        print("No models loaded! Exiting.")
        return
    
    # ========================================
    # OPTION 1: Cross-validation evaluation
    # ========================================
    print("\n" + "#"*80)
    print("# CROSS-VALIDATION EVALUATION")
    print("# Each model evaluated on its own validation fold")
    print("#"*80)
    
    y_true_cv, y_scores_cv, class_names = evaluate_ensemble_on_val_folds(models, df, device)
    
    # Optimize thresholds on CV predictions
    optimal_thresholds, optimal_f1s = optimize_thresholds(y_true_cv, y_scores_cv, class_names)
    
    # Evaluate with optimized thresholds
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS WITH OPTIMIZED THRESHOLDS")
    print("="*80)
    optimized_results = evaluate_with_threshold(y_true_cv, y_scores_cv, threshold=0.5)
    
    # Apply per-class thresholds
    y_pred_optimized = np.zeros_like(y_scores_cv)
    for i, thresh in enumerate(optimal_thresholds):
        y_pred_optimized[:, i] = (y_scores_cv[:, i] >= thresh).astype(int)
    
    print(classification_report(y_true_cv, y_pred_optimized, target_names=class_names, zero_division=0))
    
    # ========================================
    # OPTION 2: Full ensemble evaluation
    # ========================================
    # Uncomment this to evaluate ensemble on entire dataset or a test set
    # y_true_full, y_scores_full, _ = evaluate_full_ensemble(
    #     models, df, device, thresholds=optimal_thresholds
    # )
    
    # Save optimal thresholds
    threshold_path = Path('models/unified_v3/optimal_thresholds.json')
    threshold_data = {
        'thresholds': optimal_thresholds,
        'class_names': class_names,
        'cv_macro_f1': np.mean(optimal_f1s)
    }
    with open(threshold_path, 'w') as f:
        json.dump(threshold_data, f, indent=2)
    print(f"\nSaved optimal thresholds to {threshold_path}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

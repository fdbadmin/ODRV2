#!/usr/bin/env python3
"""
Threshold optimization specifically for pruned 3-fold ensemble (folds 0, 1, 4).
Optimizes thresholds on CV validation data, then evaluates on holdout test set.
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
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import json

from src.data.dataset import FundusDataset
from src.models.backbones import FundusBackbone
from src.models.multilabel_head import MultiLabelClassifier

torch.set_num_threads(4)

class SimpleModel(nn.Module):
    def __init__(self, num_classes=7, feature_dim=1024, dropout=0.3):
        super().__init__()
        self.backbone = FundusBackbone(model_name="convnext_base", pretrained=False, feature_dim=feature_dim)
        self.classifier = MultiLabelClassifier(feature_dim, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

FOLDS_TO_USE = [0, 1, 4]
CLASS_NAMES = ['Diabetic Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
LABEL_COLS = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']

def get_fold_val_indices(df, fold_idx, n_splits=5, seed=42):
    """Get validation indices for a specific fold"""
    patient_df = df.groupby('global_patient_id').first().reset_index()
    
    patient_df['label_combo'] = patient_df[LABEL_COLS].apply(
        lambda row: ''.join(row.astype(int).astype(str)), axis=1
    )
    
    value_counts = patient_df['label_combo'].value_counts()
    rare_combos = set(value_counts[value_counts < n_splits].index)
    patient_df['stratify_group'] = patient_df['label_combo'].apply(
        lambda x: 'RARE' if x in rare_combos else x
    )
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    patient_ids = patient_df['global_patient_id'].values
    stratify_labels = patient_df['stratify_group'].values
    
    for current_fold, (_, val_patient_idx) in enumerate(skf.split(patient_ids, stratify_labels)):
        if current_fold == fold_idx:
            val_patient_ids = set(patient_ids[val_patient_idx])
            val_indices = df[df['global_patient_id'].isin(val_patient_ids)].index.tolist()
            return val_indices
    return []

def optimize_thresholds(y_true, y_scores):
    """Find optimal threshold per class"""
    n_classes = y_true.shape[1]
    best_thresholds = []
    
    for i in range(n_classes):
        y_true_class = y_true[:, i]
        y_scores_class = y_scores[:, i]
        
        best_t = 0.5
        best_f1 = 0.0
        
        # Fine-grained search
        for t in np.arange(0.05, 0.96, 0.02):
            y_pred_class = (y_scores_class >= t).astype(int)
            f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        
        best_thresholds.append(best_t)
        print(f"  {CLASS_NAMES[i]}: threshold={best_t:.2f}, F1={best_f1:.4f}")
    
    return best_thresholds

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION FOR PRUNED ENSEMBLE (FOLDS 0, 1, 4)")
    print("="*80)
    
    # Load train data for CV
    train_csv = Path('data/processed/unified_v3/train_split.csv')
    print(f"\nLoading training data from {train_csv}...")
    train_df = pd.read_csv(train_csv)
    train_df['patient_id'] = train_df['filename'].apply(lambda x: x.split('_')[0])
    train_df['global_patient_id'] = train_df['source_dataset'].astype(str) + '_' + train_df['patient_id'].astype(str)
    print(f"Train samples: {len(train_df)}")
    
    # Load models
    print("\nLoading pruned ensemble models...")
    models = {}
    model_dir = Path('models/unified_v3_retrain')
    for fold_idx in FOLDS_TO_USE:
        checkpoint_path = model_dir / f"fold_{fold_idx}" / "best_model.pth"
        print(f"  Loading fold {fold_idx}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = SimpleModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        models[fold_idx] = model
    
    # Step 1: Collect CV predictions from each fold's validation set
    print("\n" + "="*80)
    print("STEP 1: COLLECTING CV PREDICTIONS")
    print("="*80)
    
    all_y_true = []
    all_y_scores = []
    
    for fold_idx in FOLDS_TO_USE:
        print(f"\n--- Fold {fold_idx} ---")
        val_indices = get_fold_val_indices(train_df, fold_idx)
        val_df = train_df.loc[val_indices].reset_index(drop=True)
        print(f"Validation samples: {len(val_df)}")
        
        val_dataset = FundusDataset(df=val_df, image_size=(448, 448), augment=False)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        model = models[fold_idx]
        fold_y_true = []
        fold_y_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold_idx}"):
                images = batch['image'].to(device)
                labels = batch['labels'].cpu().numpy()
                
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                fold_y_true.append(labels)
                fold_y_scores.append(probs)
        
        all_y_true.append(np.vstack(fold_y_true))
        all_y_scores.append(np.vstack(fold_y_scores))
    
    # Combine all CV predictions
    y_true_cv = np.vstack(all_y_true)
    y_scores_cv = np.vstack(all_y_scores)
    print(f"\nTotal CV samples: {len(y_true_cv)}")
    
    # Step 2: Optimize thresholds on CV data
    print("\n" + "="*80)
    print("STEP 2: OPTIMIZING THRESHOLDS")
    print("="*80)
    
    optimal_thresholds = optimize_thresholds(y_true_cv, y_scores_cv)
    
    # Check CV performance with optimal thresholds
    y_pred_cv = np.zeros_like(y_scores_cv)
    for i, t in enumerate(optimal_thresholds):
        y_pred_cv[:, i] = (y_scores_cv[:, i] >= t).astype(int)
    
    cv_macro_f1 = f1_score(y_true_cv, y_pred_cv, average='macro', zero_division=0)
    print(f"\nCV Macro F1 with optimized thresholds: {cv_macro_f1:.4f}")
    
    # Step 3: Evaluate on test set
    print("\n" + "="*80)
    print("STEP 3: EVALUATING ON HOLDOUT TEST SET")
    print("="*80)
    
    test_csv = Path('data/processed/unified_v3/test_split.csv')
    test_df = pd.read_csv(test_csv)
    print(f"Test samples: {len(test_df)}")
    
    test_dataset = FundusDataset(df=test_df, image_size=(448, 448), augment=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    test_y_true = []
    test_y_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test inference"):
            images = batch['image'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            # Average predictions from all 3 models
            all_probs = []
            for fold_idx in FOLDS_TO_USE:
                logits = models[fold_idx](images)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
            
            ensemble_probs = np.mean(all_probs, axis=0)
            
            test_y_true.append(labels)
            test_y_scores.append(ensemble_probs)
    
    y_true_test = np.vstack(test_y_true)
    y_scores_test = np.vstack(test_y_scores)
    
    # Apply optimized thresholds
    y_pred_test = np.zeros_like(y_scores_test)
    for i, t in enumerate(optimal_thresholds):
        y_pred_test[:, i] = (y_scores_test[:, i] >= t).astype(int)
    
    # Results
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT (OPTIMIZED THRESHOLDS)")
    print("="*80)
    print(classification_report(y_true_test, y_pred_test, target_names=CLASS_NAMES, zero_division=0))
    
    macro_f1 = f1_score(y_true_test, y_pred_test, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true_test, y_pred_test, average='micro', zero_division=0)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Optimized Thresholds: {[f'{t:.2f}' for t in optimal_thresholds]}")
    print(f"\nTest Set Performance:")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Micro F1: {micro_f1:.4f}")
    print(f"\nComparison:")
    print(f"  Previous (avg thresholds): 0.8189")
    print(f"  Optimized thresholds:      {macro_f1:.4f}")
    print(f"  Difference:                {macro_f1 - 0.8189:+.4f}")
    
    # Save results
    results = {
        'folds_used': FOLDS_TO_USE,
        'optimal_thresholds': optimal_thresholds,
        'class_names': CLASS_NAMES,
        'cv_macro_f1': cv_macro_f1,
        'test_macro_f1': macro_f1,
        'test_micro_f1': micro_f1,
        'baseline': 0.8189
    }
    
    output_path = Path('models/unified_v3_retrain/pruned_optimized_thresholds.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()

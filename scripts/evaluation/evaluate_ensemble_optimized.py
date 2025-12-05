#!/usr/bin/env python3
"""
Optimized Ensemble Evaluation with Quick Wins:
1. Per-fold threshold optimization
2. Test-Time Augmentation (TTA)
3. Weighted ensemble based on validation performance
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, multilabel_confusion_matrix
from tqdm import tqdm
import json
from PIL import Image
import torchvision.transforms as transforms

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

class TTADataset(torch.utils.data.Dataset):
    """Dataset with Test-Time Augmentation"""
    def __init__(self, base_dataset, num_augmentations=5):
        self.base_dataset = base_dataset
        self.num_augmentations = num_augmentations
        
        # Define TTA transforms
        self.tta_transforms = [
            lambda x: x,  # Original
            lambda x: transforms.functional.hflip(x),  # Horizontal flip
            lambda x: transforms.functional.vflip(x),  # Vertical flip
            lambda x: transforms.functional.rotate(x, 90),  # 90° rotation
            lambda x: transforms.functional.adjust_brightness(x, 1.1),  # Brightness
        ]
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original sample
        sample = self.base_dataset[idx]
        image = sample['image']
        
        # Apply TTA transforms
        augmented_images = []
        for transform in self.tta_transforms[:self.num_augmentations]:
            aug_img = transform(image)
            augmented_images.append(aug_img)
        
        # Stack augmented images
        sample['images_tta'] = torch.stack(augmented_images)
        sample['image'] = image  # Keep original for compatibility
        
        return sample

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
    """Get validation indices for a specific fold"""
    from sklearn.model_selection import StratifiedKFold
    
    patient_df = df.groupby('global_patient_id').first().reset_index()
    label_cols = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
    
    patient_df['label_combo'] = patient_df[label_cols].apply(
        lambda row: ''.join(row.astype(int).astype(str)), axis=1
    )
    
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

def optimize_thresholds_per_fold(y_true, y_scores, class_names, fold_idx):
    """Find optimal threshold per class for a specific fold"""
    n_classes = y_true.shape[1]
    best_thresholds = []
    best_f1s = []
    
    for i in range(n_classes):
        y_true_class = y_true[:, i]
        y_scores_class = y_scores[:, i]
        
        best_t = 0.5
        best_f1 = 0.0
        
        for t in np.arange(0.05, 0.96, 0.05):
            y_pred_class = (y_scores_class >= t).astype(int)
            f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        
        best_thresholds.append(best_t)
        best_f1s.append(best_f1)
    
    return best_thresholds, best_f1s

def evaluate_with_tta_and_weights(models, df, device, batch_size=16, use_tta=True):
    """
    Evaluate ensemble with:
    1. Per-fold threshold optimization
    2. Test-Time Augmentation
    3. Weighted ensemble
    """
    class_names = ['Diabetic Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
    
    # Step 1: Optimize thresholds per fold
    print("\n" + "="*80)
    print("STEP 1: PER-FOLD THRESHOLD OPTIMIZATION")
    print("="*80)
    
    fold_thresholds = []
    fold_weights = []
    
    for fold_idx in range(len(models)):
        print(f"\n--- Optimizing Fold {fold_idx} ---")
        
        val_indices = get_fold_indices(df, fold_idx)
        val_df = df.loc[val_indices].reset_index(drop=True)
        
        val_dataset = FundusDataset(df=val_df, image_size=(448, 448), augment=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
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
        
        fold_y_true = np.vstack(fold_y_true)
        fold_y_scores = np.vstack(fold_y_scores)
        
        # Optimize thresholds for this fold
        thresholds, f1s = optimize_thresholds_per_fold(fold_y_true, fold_y_scores, class_names, fold_idx)
        macro_f1 = np.mean(f1s)
        
        fold_thresholds.append(thresholds)
        fold_weights.append(macro_f1)  # Weight by performance
        
        print(f"Fold {fold_idx}: Macro F1 = {macro_f1:.4f}")
        print(f"Thresholds: {[f'{t:.2f}' for t in thresholds]}")
    
    # Normalize weights
    total_weight = sum(fold_weights)
    fold_weights = [w / total_weight for w in fold_weights]
    
    print(f"\n" + "-"*80)
    print(f"Fold Weights (normalized): {[f'{w:.3f}' for w in fold_weights]}")
    print(f"Best Fold: {np.argmax(fold_weights)} (weight: {max(fold_weights):.3f})")
    
    # Step 2: Evaluate with TTA and weighted ensemble
    print("\n" + "="*80)
    print("STEP 2: WEIGHTED ENSEMBLE WITH TTA")
    print("="*80)
    
    # Use all data for final evaluation
    if use_tta:
        print("Using Test-Time Augmentation with 5 transforms")
        base_dataset = FundusDataset(df=df, image_size=(448, 448), augment=False)
        dataset = TTADataset(base_dataset, num_augmentations=5)
    else:
        print("TTA disabled, using single inference")
        dataset = FundusDataset(df=df, image_size=(448, 448), augment=False)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_y_true = []
    all_y_scores_weighted = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Ensemble inference"):
            labels = batch['labels'].cpu().numpy()
            
            # Get predictions from all models
            fold_predictions = []
            
            for model_idx, model in enumerate(models):
                if use_tta:
                    # TTA: predict on all augmented versions
                    images_tta = batch['images_tta'].to(device)  # [B, N_aug, C, H, W]
                    batch_size_actual, n_aug = images_tta.shape[0], images_tta.shape[1]
                    
                    # Reshape to [B*N_aug, C, H, W]
                    images_flat = images_tta.view(-1, *images_tta.shape[2:])
                    
                    # Predict
                    logits = model(images_flat)
                    probs = torch.sigmoid(logits)
                    
                    # Reshape back to [B, N_aug, num_classes] and average
                    probs = probs.view(batch_size_actual, n_aug, -1)
                    probs = probs.mean(dim=1).cpu().numpy()  # Average over augmentations
                else:
                    # Single inference
                    images = batch['image'].to(device)
                    logits = model(images)
                    probs = torch.sigmoid(logits).cpu().numpy()
                
                # Apply fold-specific thresholds
                fold_predictions.append(probs)
            
            # Weighted average across folds
            weighted_probs = np.zeros_like(fold_predictions[0])
            for fold_idx, (probs, weight) in enumerate(zip(fold_predictions, fold_weights)):
                weighted_probs += weight * probs
            
            all_y_true.append(labels)
            all_y_scores_weighted.append(weighted_probs)
    
    y_true = np.vstack(all_y_true)
    y_scores = np.vstack(all_y_scores_weighted)
    
    # Step 3: Apply averaged thresholds
    print("\n" + "="*80)
    print("STEP 3: FINAL EVALUATION WITH OPTIMIZED THRESHOLDS")
    print("="*80)
    
    # Use weighted average of fold thresholds
    final_thresholds = np.average(fold_thresholds, axis=0, weights=fold_weights)
    
    print(f"\nFinal Thresholds (weighted average):")
    for i, (name, thresh) in enumerate(zip(class_names, final_thresholds)):
        print(f"  {name}: {thresh:.2f}")
    
    # Apply thresholds
    y_pred = np.zeros_like(y_scores)
    for i, thresh in enumerate(final_thresholds):
        y_pred[:, i] = (y_scores[:, i] >= thresh).astype(int)
    
    # Classification report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
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
        print(f"  Sensitivity: {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  PPV: {ppv:.4f}")
        print(f"  NPV: {npv:.4f}")
    
    # Save results
    results = {
        'fold_thresholds': [t for t in fold_thresholds],
        'fold_weights': fold_weights,
        'final_thresholds': final_thresholds.tolist(),
        'class_names': class_names,
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0)
    }
    
    output_path = Path('models/unified_v3/optimized_ensemble_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    csv_path = Path('data/processed/unified_v3/unified_train_v3.csv')
    print(f"\nLoading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Prepare patient IDs
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
    
    # Run optimized evaluation
    results = evaluate_with_tta_and_weights(
        models, df, device, 
        batch_size=8,  # Reduced for TTA (5x more memory)
        use_tta=True
    )
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print(f"Micro F1: {results['micro_f1']:.4f}")

if __name__ == "__main__":
    main()

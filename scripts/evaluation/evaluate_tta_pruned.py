#!/usr/bin/env python3
"""
TTA evaluation on pruned ensemble (folds 0, 1, 4) on holdout test set.
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
from tqdm import tqdm
import json
import torchvision.transforms as transforms

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

# Pruned ensemble - best 3 folds
FOLDS_TO_USE = [0, 1, 4]

# Thresholds for folds 0, 1, 4 (from earlier optimization)
FOLD_THRESHOLDS = {
    0: [0.45, 0.45, 0.45, 0.55, 0.20, 0.45, 0.50],
    1: [0.50, 0.50, 0.50, 0.30, 0.80, 0.35, 0.45],
    4: [0.45, 0.65, 0.55, 0.75, 0.30, 0.55, 0.50],
}

CLASS_NAMES = ['Diabetic Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
LABEL_COLS = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']

def apply_tta(image):
    """Apply TTA transforms and return list of augmented images"""
    augmented = [
        image,  # Original
        transforms.functional.hflip(image),  # Horizontal flip
        transforms.functional.vflip(image),  # Vertical flip
        transforms.functional.rotate(image, 90),  # 90° rotation
        transforms.functional.rotate(image, -90),  # -90° rotation
    ]
    return torch.stack(augmented)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("TTA EVALUATION ON PRUNED ENSEMBLE (FOLDS 0, 1, 4)")
    print("="*80)
    
    # Load test data
    csv_path = Path('data/processed/unified_v3/test_split.csv')
    print(f"\nLoading HOLDOUT TEST data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Test samples: {len(df)}")
    
    # Load models
    print("\nLoading pruned ensemble models...")
    models = []
    model_dir = Path('models/unified_v3_retrain')
    for fold_idx in FOLDS_TO_USE:
        checkpoint_path = model_dir / f"fold_{fold_idx}" / "best_model.pth"
        print(f"  Loading fold {fold_idx}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = SimpleModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        models.append(model)
    print(f"Loaded {len(models)} models")
    
    # Compute average thresholds for pruned folds
    avg_thresholds = np.mean([FOLD_THRESHOLDS[f] for f in FOLDS_TO_USE], axis=0)
    print(f"\nAverage thresholds: {[f'{t:.2f}' for t in avg_thresholds]}")
    
    # Create dataset
    dataset = FundusDataset(df=df, image_size=(448, 448), augment=False)
    
    # Run TTA inference
    print("\n" + "="*80)
    print("TTA INFERENCE (5 augmentations x 3 folds)")
    print("="*80)
    
    all_y_true = []
    all_y_scores = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="TTA inference"):
            sample = dataset[idx]
            image = sample['image']  # [C, H, W]
            labels = sample['labels'].numpy()
            
            # Apply TTA - get 5 augmented versions
            tta_images = apply_tta(image).to(device)  # [5, C, H, W]
            
            # Get predictions from all models on all TTA versions
            all_probs = []
            for model in models:
                logits = model(tta_images)  # [5, num_classes]
                probs = torch.sigmoid(logits).cpu().numpy()  # [5, num_classes]
                avg_probs = probs.mean(axis=0)  # Average over TTA [num_classes]
                all_probs.append(avg_probs)
            
            # Average across models
            ensemble_probs = np.mean(all_probs, axis=0)
            
            all_y_true.append(labels)
            all_y_scores.append(ensemble_probs)
    
    y_true = np.vstack(all_y_true)
    y_scores = np.vstack(all_y_scores)
    
    # Apply thresholds
    y_pred = np.zeros_like(y_scores)
    for i, thresh in enumerate(avg_thresholds):
        y_pred[:, i] = (y_scores[:, i] >= thresh).astype(int)
    
    # Results
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT (WITH TTA)")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"\nComparison:")
    print(f"  Without TTA: 0.8189")
    print(f"  With TTA:    {macro_f1:.4f}")
    print(f"  Difference:  {macro_f1 - 0.8189:+.4f}")
    
    # Save results
    results = {
        'folds_used': FOLDS_TO_USE,
        'tta_augmentations': 5,
        'thresholds': avg_thresholds.tolist(),
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'baseline_no_tta': 0.8189
    }
    
    output_path = Path('models/unified_v3_retrain/tta_ensemble_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()

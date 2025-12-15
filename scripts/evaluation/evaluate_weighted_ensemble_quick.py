#!/usr/bin/env python3
"""
Quick weighted ensemble evaluation using pre-computed thresholds from Step 1.
Skips threshold optimization, just runs weighted ensemble inference.
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
from sklearn.metrics import classification_report, f1_score, multilabel_confusion_matrix
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

# Pre-computed results from Step 1
FOLD_THRESHOLDS = [
    [0.45, 0.45, 0.45, 0.55, 0.20, 0.45, 0.50],  # Fold 0: F1=0.7811
    [0.50, 0.50, 0.50, 0.30, 0.80, 0.35, 0.45],  # Fold 1: F1=0.7618
    [0.40, 0.45, 0.50, 0.45, 0.20, 0.45, 0.50],  # Fold 2: F1=0.7203
    [0.50, 0.50, 0.45, 0.50, 0.60, 0.45, 0.40],  # Fold 3: F1=0.7443
    [0.45, 0.65, 0.55, 0.75, 0.30, 0.55, 0.50],  # Fold 4: F1=0.7728
]
FOLD_F1S = [0.7811, 0.7618, 0.7203, 0.7443, 0.7728]
FOLD_WEIGHTS = [f1 / sum(FOLD_F1S) for f1 in FOLD_F1S]

CLASS_NAMES = ['Diabetic Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Print pre-computed weights
    print("\n" + "="*80)
    print("USING PRE-COMPUTED FOLD WEIGHTS AND THRESHOLDS")
    print("="*80)
    print(f"Fold Weights: {[f'{w:.3f}' for w in FOLD_WEIGHTS]}")
    print(f"Best Fold: {np.argmax(FOLD_WEIGHTS)} (weight: {max(FOLD_WEIGHTS):.3f})")
    
    # Compute weighted average thresholds
    final_thresholds = np.average(FOLD_THRESHOLDS, axis=0, weights=FOLD_WEIGHTS)
    print(f"\nFinal Thresholds (weighted average):")
    for name, thresh in zip(CLASS_NAMES, final_thresholds):
        print(f"  {name}: {thresh:.2f}")
    
    # Load HOLDOUT TEST data
    csv_path = Path('data/processed/unified_v3/test_split.csv')
    print(f"\nLoading HOLDOUT TEST data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['patient_id'] = df['filename'].apply(lambda x: x.split('_')[0])
    df['global_patient_id'] = df['source_dataset'].astype(str) + '_' + df['patient_id'].astype(str)
    print(f"Total test samples: {len(df)}")
    
    # Load models
    print("\nLoading ensemble models...")
    models = []
    model_dir = Path('models/unified_v3_retrain')
    for fold_idx in range(5):
        checkpoint_path = model_dir / f"fold_{fold_idx}" / "best_model.pth"
        print(f"  Loading fold {fold_idx}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = SimpleModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        models.append(model)
    print(f"Loaded {len(models)} models")
    
    # Create dataset and loader
    dataset = FundusDataset(df=df, image_size=(448, 448), augment=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Run weighted ensemble inference
    print("\n" + "="*80)
    print("WEIGHTED ENSEMBLE INFERENCE (NO TTA)")
    print("="*80)
    
    all_y_true = []
    all_y_scores = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Ensemble inference"):
            labels = batch['labels'].cpu().numpy()
            images = batch['image'].to(device)
            
            # Get predictions from all models
            fold_predictions = []
            for model in models:
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_predictions.append(probs)
            
            # Weighted average across folds
            weighted_probs = np.zeros_like(fold_predictions[0])
            for probs, weight in zip(fold_predictions, FOLD_WEIGHTS):
                weighted_probs += weight * probs
            
            all_y_true.append(labels)
            all_y_scores.append(weighted_probs)
    
    y_true = np.vstack(all_y_true)
    y_scores = np.vstack(all_y_scores)
    
    # Apply thresholds
    y_pred = np.zeros_like(y_scores)
    for i, thresh in enumerate(final_thresholds):
        y_pred[:, i] = (y_scores[:, i] >= thresh).astype(int)
    
    # Results
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    
    # Save results
    results = {
        'fold_thresholds': FOLD_THRESHOLDS,
        'fold_weights': FOLD_WEIGHTS,
        'fold_f1_scores': FOLD_F1S,
        'final_thresholds': final_thresholds.tolist(),
        'class_names': CLASS_NAMES,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'use_tta': False
    }
    
    output_path = Path('models/unified_v3_retrain/weighted_ensemble_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()

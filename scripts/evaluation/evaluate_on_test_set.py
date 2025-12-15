#!/usr/bin/env python3
"""
Evaluate trained models on HELD-OUT TEST SET.

This evaluation uses:
- Test set: test_split.csv (6,282 images from 4,058 patients)
- Models: Folds 0, 1, 4 (pruned ensemble)
- Thresholds: Per-fold optimized thresholds

CRITICAL: Test set was NOT used during training or threshold optimization.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from tqdm import tqdm

# Import model components
from src.models.backbones import FundusBackbone
from src.models.multilabel_head import MultiLabelClassifier

# Determine device
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

print(f"Using device: {DEVICE}")

# Configuration
MODEL_DIR = Path('models/unified_v3')
RESULTS_DIR = Path('results/unified_v3')
DATA_DIR = Path('data/processed/unified_v3')
IMAGE_BASE_DIR = Path('data/raw')

# Use HELD-OUT TEST SET
TEST_CSV = DATA_DIR / 'test_split.csv'

LABEL_COLS = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
CLASS_NAMES = ['Diabetic_Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']

# Pruned ensemble - top 3 folds by CV performance
# Fold 0: 0.7811, Fold 1: 0.7618, Fold 4: 0.7728
FOLDS_TO_USE = [0, 1, 4]

BATCH_SIZE = 64
NUM_WORKERS = 4
IMG_SIZE = 384

class SimpleModel(nn.Module):
    """Simple wrapper matching training architecture"""
    def __init__(self, num_classes=7, feature_dim=1024, dropout=0.3):
        super().__init__()
        self.backbone = FundusBackbone(model_name="convnext_base", pretrained=False, feature_dim=feature_dim)
        self.classifier = MultiLabelClassifier(feature_dim, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

class EyeDiseaseDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Use the full image path from the dataframe
        img_path = row['image_path']
        
        # Load and transform image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # Get labels
        labels = torch.tensor(row[LABEL_COLS].values.astype(np.float32))
        
        return img, labels

def load_model(fold: int):
    """Load a trained model for a specific fold."""
    model_path = MODEL_DIR / f'fold_{fold}' / 'best_model.pth'
    
    model = SimpleModel(num_classes=7, feature_dim=1024, dropout=0.3)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    return model

def get_predictions(model, dataloader):
    """Get predictions from a single model."""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Predicting", leave=False):
            images = images.to(DEVICE)
            outputs = torch.sigmoid(model(images))
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.vstack(all_preds), np.vstack(all_labels)

def compute_metrics(y_true, y_pred, y_prob, class_names):
    """Compute comprehensive metrics."""
    results = {}
    
    # Per-class metrics
    per_class = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    results['per_class'] = {}
    for i, class_name in enumerate(class_names):
        class_metrics = per_class[class_name]
        
        # Add AUC-ROC and AUC-PR
        try:
            auc_roc = roc_auc_score(y_true[:, i], y_prob[:, i])
        except:
            auc_roc = 0.0
        
        try:
            auc_pr = average_precision_score(y_true[:, i], y_prob[:, i])
        except:
            auc_pr = 0.0
        
        results['per_class'][class_name] = {
            'precision': class_metrics['precision'],
            'recall': class_metrics['recall'],
            'f1-score': class_metrics['f1-score'],
            'support': int(class_metrics['support']),
            'auc_roc': auc_roc,
            'auc_pr': auc_pr
        }
    
    # Macro averages
    results['macro_avg'] = {
        'precision': per_class['macro avg']['precision'],
        'recall': per_class['macro avg']['recall'],
        'f1-score': per_class['macro avg']['f1-score'],
        'auc_roc': np.mean([results['per_class'][c]['auc_roc'] for c in class_names]),
        'auc_pr': np.mean([results['per_class'][c]['auc_pr'] for c in class_names])
    }
    
    # Weighted averages
    results['weighted_avg'] = {
        'precision': per_class['weighted avg']['precision'],
        'recall': per_class['weighted avg']['recall'],
        'f1-score': per_class['weighted avg']['f1-score']
    }
    
    return results

def main():
    print("="*80)
    print("EVALUATING ON HELD-OUT TEST SET")
    print("="*80)
    
    # Verify test set exists
    if not TEST_CSV.exists():
        print(f"\n✗ ERROR: Test set not found at {TEST_CSV}")
        print("Run create_train_test_split.py first!")
        return
    
    print(f"\nTest Set: {TEST_CSV}")
    print(f"Folds to use: {FOLDS_TO_USE}")
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(TEST_CSV)
    print(f"Test images: {len(test_df)}")
    print(f"Test patients: {test_df['global_patient_id'].nunique()}")
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = EyeDiseaseDataset(test_df, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Load per-fold thresholds
    threshold_path = RESULTS_DIR / 'per_fold_thresholds.json'
    with open(threshold_path) as f:
        threshold_data = json.load(f)
    
    # Get predictions from each fold
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    fold_predictions = []
    
    for fold in FOLDS_TO_USE:
        print(f"\nFold {fold}:")
        print(f"  Loading model...")
        model = load_model(fold)
        
        print(f"  Generating predictions...")
        preds, labels = get_predictions(model, test_loader)
        fold_predictions.append(preds)
        
        # Free memory
        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        elif DEVICE == 'mps':
            torch.mps.empty_cache()
    
    # Average predictions across folds
    print("\n" + "="*80)
    print("ENSEMBLE PREDICTIONS")
    print("="*80)
    
    print("\nAveraging predictions across folds...")
    avg_probs = np.mean(fold_predictions, axis=0)
    
    # Apply per-fold thresholds (use average of fold thresholds)
    print("Applying optimized thresholds...")
    fold_thresholds_array = np.array(threshold_data['fold_thresholds'])
    thresholds = fold_thresholds_array[FOLDS_TO_USE]
    avg_thresholds = np.mean(thresholds, axis=0)
    
    print("\nAverage thresholds per class:")
    for i, (class_name, thresh) in enumerate(zip(CLASS_NAMES, avg_thresholds)):
        print(f"  {class_name:<25} {thresh:.4f}")
    
    # Apply thresholds
    predictions = (avg_probs >= avg_thresholds).astype(int)
    
    # Compute metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    results = compute_metrics(labels, predictions, avg_probs, CLASS_NAMES)
    
    # Print results
    print("\nPer-Class Metrics:")
    print("-" * 95)
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10} {'Support':>10}")
    print("-" * 95)
    
    for class_name in CLASS_NAMES:
        metrics = results['per_class'][class_name]
        print(f"{class_name:<25} "
              f"{metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} "
              f"{metrics['f1-score']:>10.4f} "
              f"{metrics['auc_roc']:>10.4f} "
              f"{metrics['support']:>10}")
    
    print("-" * 95)
    print(f"{'Macro Average':<25} "
          f"{results['macro_avg']['precision']:>10.4f} "
          f"{results['macro_avg']['recall']:>10.4f} "
          f"{results['macro_avg']['f1-score']:>10.4f} "
          f"{results['macro_avg']['auc_roc']:>10.4f}")
    print("-" * 95)
    
    # Save results
    output_path = RESULTS_DIR / 'test_set_evaluation.json'
    results['metadata'] = {
        'test_csv': str(TEST_CSV),
        'test_images': len(test_df),
        'test_patients': int(test_df['global_patient_id'].nunique()),
        'folds_used': FOLDS_TO_USE,
        'avg_thresholds': avg_thresholds.tolist()
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Compare with training CV results
    print("\n" + "="*80)
    print("COMPARISON WITH TRAINING CV")
    print("="*80)
    
    print("\nTraining CV Performance (from logs):")
    cv_f1_scores = {
        'Fold 0': 0.7441,
        'Fold 1': 0.7790,
        'Fold 2': 0.6259,
        'Fold 3': 0.6995,
        'Fold 4': 0.7290
    }
    
    for fold, f1 in cv_f1_scores.items():
        print(f"  {fold}: {f1:.4f}")
    
    cv_avg = np.mean(list(cv_f1_scores.values()))
    print(f"  Average: {cv_avg:.4f}")
    
    test_f1 = results['macro_avg']['f1-score']
    print(f"\nTest Set Performance:")
    print(f"  Macro F1: {test_f1:.4f}")
    
    diff = test_f1 - cv_avg
    print(f"\nDifference: {diff:+.4f} ({diff/cv_avg*100:+.1f}%)")
    
    if abs(diff) < 0.05:
        print("✓ Test performance matches CV (model generalizes well)")
    elif diff < 0:
        print("⚠ Test performance lower than CV (possible overfitting)")
    else:
        print("⚠ Test performance higher than CV (unusual, check for issues)")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

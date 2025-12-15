#!/usr/bin/env python3
"""
Evaluate the pruned ensemble on the 215 "missing" HYGD images
that were never included in training or test sets.
This is a true external validation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from tqdm import tqdm

# Configuration
ROOT_DIR = Path(__file__).parent.parent.parent
HYGD_DIR = ROOT_DIR / "external_data" / "glaucoma_standard"
MODEL_DIR = ROOT_DIR / "models" / "unified_v3_retrain"
FOLDS_TO_USE = [0, 1, 4]  # Pruned ensemble

# Thresholds from final model config
THRESHOLDS = {
    'Label_D': 0.48,
    'Label_G': 0.62,
    'Label_C': 0.55,
    'Label_A': 0.50,
    'Label_H': 0.43,
    'Label_M': 0.62,
    'Label_O': 0.43
}

def load_model(fold_path):
    """Load a trained model using the actual project architecture"""
    import timm
    
    # Match the actual architecture from src/models/
    class FundusBackbone(torch.nn.Module):
        def __init__(self, model_name='convnext_base', pretrained=False, feature_dim=1024):
            super().__init__()
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
            self.projection = torch.nn.Linear(self.backbone.num_features, feature_dim)
        
        def forward(self, image):
            features = self.backbone(image)
            return self.projection(features)
    
    class MultiLabelClassifier(torch.nn.Module):
        def __init__(self, feature_dim=1024, num_classes=7):
            super().__init__()
            self.dropout = torch.nn.Dropout(p=0.3)
            self.head = torch.nn.Linear(feature_dim, num_classes)
        
        def forward(self, features):
            return self.head(self.dropout(features))
    
    class FullModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = FundusBackbone(pretrained=False)
            self.classifier = MultiLabelClassifier()
        
        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)
    
    model = FullModel()
    
    # Load weights
    checkpoint = torch.load(fold_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model

def get_missing_images():
    """Find images not in train or test splits"""
    # Load HYGD labels
    labels = pd.read_csv(HYGD_DIR / "Labels.csv")
    labels.columns = labels.columns.str.strip()
    
    # Load used images
    train = pd.read_csv(ROOT_DIR / "data" / "processed" / "unified_v3" / "train_split.csv")
    test = pd.read_csv(ROOT_DIR / "data" / "processed" / "unified_v3" / "test_split.csv")
    
    used_train = set(train[train['source_dataset'] == 'HYGD']['filename'].values)
    used_test = set(test[test['source_dataset'] == 'HYGD']['filename'].values)
    used_all = used_train | used_test
    
    # Find missing
    labels['in_dataset'] = labels['Image Name'].isin(used_all)
    missing = labels[~labels['in_dataset']].copy()
    
    print(f"Found {len(missing)} images never seen during training/testing")
    print(f"  GON+: {(missing['Label'] == 'GON+').sum()}")
    print(f"  GON-: {(missing['Label'] == 'GON-').sum()}")
    
    return missing

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get missing images
    missing_df = get_missing_images()
    
    # Load models
    print(f"\nLoading models from folds {FOLDS_TO_USE}...")
    models = []
    for fold in FOLDS_TO_USE:
        model_path = MODEL_DIR / f"fold_{fold}" / "best_model.pth"
        if not model_path.exists():
            print(f"  Warning: {model_path} not found, skipping")
            continue
        model = load_model(model_path)
        model.to(device)
        model.eval()
        models.append(model)
        print(f"  Loaded fold {fold}")
    
    if not models:
        print("No models found!")
        return
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Evaluate
    print(f"\nEvaluating on {len(missing_df)} images...")
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for _, row in tqdm(missing_df.iterrows(), total=len(missing_df)):
            img_path = HYGD_DIR / "Images" / row['Image Name']
            
            if not img_path.exists():
                continue
            
            # Load and preprocess
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Ensemble prediction
            fold_preds = []
            for model in models:
                logits = model(img_tensor)
                probs = torch.sigmoid(logits)
                fold_preds.append(probs.cpu().numpy())
            
            avg_probs = np.mean(fold_preds, axis=0)[0]
            all_probs.append(avg_probs)
            
            # Ground truth (only glaucoma label available)
            # GON+ = glaucoma positive, GON- = normal
            is_glaucoma = 1 if row['Label'] == 'GON+' else 0
            all_labels.append(is_glaucoma)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Extract glaucoma predictions (index 1 = Label_G)
    glaucoma_probs = all_probs[:, 1]
    glaucoma_preds = (glaucoma_probs >= THRESHOLDS['Label_G']).astype(int)
    
    # Metrics
    print("\n" + "="*60)
    print("EXTERNAL VALIDATION: Missing HYGD Images")
    print("="*60)
    print(f"Total images: {len(all_labels)}")
    print(f"Glaucoma positive: {all_labels.sum()}")
    print(f"Normal: {(1 - all_labels).sum()}")
    print()
    
    # AUC-ROC
    auc = roc_auc_score(all_labels, glaucoma_probs)
    print(f"AUC-ROC: {auc:.4f}")
    
    # Classification report
    print(f"\nUsing threshold: {THRESHOLDS['Label_G']}")
    print(classification_report(all_labels, glaucoma_preds, 
                                target_names=['Normal (GON-)', 'Glaucoma (GON+)']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, glaucoma_preds)
    print("Confusion Matrix:")
    print(f"                 Predicted")
    print(f"              Normal  Glaucoma")
    print(f"Actual Normal    {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"Actual Glaucoma  {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    # Also show other predictions (even though we don't have labels)
    print("\n" + "-"*60)
    print("Other disease predictions (no ground truth available):")
    disease_names = ['DR', 'Glaucoma', 'Cataract', 'AMD', 'HTN', 'Myopia', 'Other']
    for i, name in enumerate(disease_names):
        if name == 'Glaucoma':
            continue
        pred_rate = (all_probs[:, i] >= list(THRESHOLDS.values())[i]).mean() * 100
        avg_prob = all_probs[:, i].mean() * 100
        print(f"  {name}: {pred_rate:.1f}% predicted positive (avg prob: {avg_prob:.1f}%)")

if __name__ == "__main__":
    main()

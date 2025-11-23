#!/usr/bin/env python3
"""
Main Training Script for Unified V3
Optimized for Apple Silicon (MPS)
"""

import sys
from pathlib import Path
# Add project root to path (3 levels up: scripts/training/train.py -> scripts/training -> scripts -> root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import time
import json

from src.data.dataset import FundusDataset
from src.models.backbones import FundusBackbone
from src.models.multilabel_head import MultiLabelClassifier

# Optimize for Apple Silicon 4 performance cores
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification with enhanced rare disease focus."""
    def __init__(self, alpha=None, gamma=2.0, gamma_rare=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.gamma_rare = gamma_rare
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        # Indices for rare diseases: Cataract(2), AMD(3), HTN(4), Myopia(5)
        self.rare_indices = {2, 3, 4, 5}
    
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        
        # Apply different gamma for rare diseases
        focal_loss = torch.zeros_like(bce_loss)
        for class_idx in range(logits.shape[1]):
            if class_idx in self.rare_indices:
                focal_loss[:, class_idx] = (1 - pt[:, class_idx]) ** self.gamma_rare * bce_loss[:, class_idx]
            else:
                focal_loss[:, class_idx] = (1 - pt[:, class_idx]) ** self.gamma * bce_loss[:, class_idx]
        
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        
        return focal_loss.mean()

class SimpleModel(nn.Module):
    """Simple wrapper matching working architecture"""
    def __init__(self, num_classes=7, feature_dim=1024, dropout=0.3):
        super().__init__()
        self.backbone = FundusBackbone(model_name="convnext_base", pretrained=True, feature_dim=feature_dim)
        # Dropout is already included in MultiLabelClassifier
        self.classifier = MultiLabelClassifier(feature_dim, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

def prepare_patient_ids(df):
    """
    Extract patient IDs from filenames and create global IDs.
    Must be called before splitting or logging patient counts.
    """
    print("Refining patient IDs from filenames...")
    # EyePACS: "10_left.jpeg" -> "10"
    # ODIR: "0_right.jpg" -> "0"
    df['patient_id'] = df['filename'].apply(lambda x: x.split('_')[0])

    # Create globally unique patient ID
    df['global_patient_id'] = df['source_dataset'].astype(str) + '_' + df['patient_id'].astype(str)
    return df

def create_folds(df, n_splits=5, seed=42):
    """Create patient-level folds with handling for rare combinations"""
    
    # Ensure global_patient_id exists
    if 'global_patient_id' not in df.columns:
        df = prepare_patient_ids(df)
    
    # Group by GLOBAL patient ID, not the image ID
    patient_df = df.groupby('global_patient_id').first().reset_index()
    
    label_cols = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
    
    # Create combination key
    patient_df['stratify_key'] = patient_df[label_cols].astype(str).agg('_'.join, axis=1)
    
    # Handle rare combinations to avoid StratifiedKFold warning
    # If a combination has fewer samples than n_splits, StratifiedKFold can't split it evenly.
    # We group these rare combinations into a single 'rare_combination' bucket.
    key_counts = patient_df['stratify_key'].value_counts()
    rare_keys = key_counts[key_counts < n_splits].index
    
    if len(rare_keys) > 0:
        print(f"Grouping {len(rare_keys)} rare label combinations into 'rare_combination' bucket for stratification.")
        patient_df.loc[patient_df['stratify_key'].isin(rare_keys), 'stratify_key'] = 'rare_combination'
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    patient_df['fold'] = -1
    
    for fold_idx, (_, val_idx) in enumerate(skf.split(patient_df, patient_df['stratify_key'])):
        patient_df.loc[val_idx, 'fold'] = fold_idx
    
    # Merge back to original dataframe using global_patient_id
    df = df.merge(patient_df[['global_patient_id', 'fold']], on='global_patient_id', how='left')
    return df

def calculate_metrics(outputs, targets):
    """Calculate per-class metrics"""
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()
    
    # Per-class accuracy
    correct = (preds == targets).float()
    acc_per_class = correct.mean(dim=0)
    
    # F1 scores
    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'accuracy': acc_per_class.mean().item(),
        'f1_macro': f1.mean().item(),
        'f1_per_class': f1.cpu().numpy().tolist()
    }

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        all_outputs.append(outputs.detach())
        all_targets.append(labels.detach())
        
        if batch_idx % 10 == 0:
            pbar.set_postfix({'loss': f'{running_loss/(batch_idx+1):.4f}'})
    
    # Calculate metrics
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_outputs, all_targets)
    metrics['loss'] = running_loss / len(loader)
    
    return metrics

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            all_outputs.append(outputs)
            all_targets.append(labels)
    
    # Calculate metrics
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_outputs, all_targets)
    metrics['loss'] = running_loss / len(loader)
    
    return metrics

def train_fold(fold, train_df, val_df, config):
    """Train a single fold"""
    print(f"\n{'='*80}")
    print(f"Training Fold {fold}")
    print(f"{'='*80}")
    print(f"Train: {len(train_df)} samples ({len(train_df['ID'].unique())} patients)")
    print(f"Val: {len(val_df)} samples ({len(val_df['ID'].unique())} patients)")
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Datasets
    train_dataset = FundusDataset(train_df, image_size=config['image_size'], augment=True)
    val_dataset = FundusDataset(val_df, image_size=config['image_size'], augment=False)
    
    # Dataloaders - optimized for 4 cores
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,  # 2 workers for I/O
        pin_memory=False,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    model = SimpleModel(
        num_classes=config['num_classes'],
        feature_dim=config['feature_dim'],
        dropout=config['dropout']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Loss with class weights
    criterion = FocalLoss(alpha=torch.tensor(config['class_weights']).to(device))
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )
    
    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    history = []
    
    fold_dir = config['model_dir'] / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log
        print(f"\nEpoch {epoch+1}/{config['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1_macro']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1_macro']:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time
        })
        
        # Save best model
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': config
            }
            torch.save(checkpoint, fold_dir / 'best_model.pth')
            print(f"  ✓ New best model saved! (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\n  Early stopping triggered (patience={config['patience']})")
                break
    
    # Save history
    with open(fold_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nFold {fold} complete! Best Val F1: {best_f1:.4f}")
    return best_f1

def main():
    print("="*80)
    print("Pure PyTorch Training - Unified V3 (Optimized for Apple Silicon)")
    print("="*80)
    
    # Config
    config = {
        'csv_path': Path('data/processed/unified_v3/unified_train_v3.csv'),
        'model_dir': Path('models/unified_v3'),
        'num_folds': 5,
        'num_classes': 7,
        'feature_dim': 1024,  # ConvNeXt-base
        'dropout': 0.3,
        'image_size': (448, 448),
        'batch_size': 8,
        'epochs': 30,
        'learning_rate': 2e-4,
        'weight_decay': 1e-2,
        'class_weights': [1.0, 5.0, 8.0, 60.0, 150.0, 60.0, 1.0],  # D, G, C, A, H, M, O
        'patience': 5
    }
    
    config['model_dir'].mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading {config['csv_path']}...")
    df = pd.read_csv(config['csv_path'])
    
    # Prepare IDs immediately to get correct patient count
    df = prepare_patient_ids(df)
    print(f"  {len(df)} samples from {len(df['global_patient_id'].unique())} unique patients")
    
    # Create folds
    print(f"\nCreating {config['num_folds']}-fold splits...")
    df = create_folds(df, n_splits=config['num_folds'], seed=42)
    
    # Train each fold
    fold_results = []
    for fold in range(config['num_folds']):
        train_df = df[df['fold'] != fold].reset_index(drop=True)
        val_df = df[df['fold'] == fold].reset_index(drop=True)
        
        best_f1 = train_fold(fold, train_df, val_df, config)
        fold_results.append(best_f1)
    
    # Summary
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print("\nFold Results:")
    for i, f1 in enumerate(fold_results):
        print(f"  Fold {i}: {f1:.4f}")
    print(f"\nMean F1: {sum(fold_results)/len(fold_results):.4f} ± {torch.tensor(fold_results).std().item():.4f}")
    print(f"\nModels saved in: {config['model_dir']}")

if __name__ == "__main__":
    main()

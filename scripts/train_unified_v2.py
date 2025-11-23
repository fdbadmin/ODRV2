#!/usr/bin/env python3
"""
Train ensemble on Unified Dataset V2 (44,673 images)

This script trains a 5-fold ensemble on the expanded unified dataset with
proper class imbalance handling using:
1. Focal Loss with class-specific alpha weights
2. Weighted sampling during training
3. Class-balanced metrics

Expected training time: 24-36 hours on GPU
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from src.training.loop import TrainingConfig, FundusLightningModule
from src.data.dataset import FundusDataset


def calculate_class_weights(df: pd.DataFrame, method: str = 'effective_number') -> list[float]:
    """
    Calculate class weights for handling imbalance.
    
    Args:
        df: Training dataframe
        method: 'inverse_freq', 'balanced', or 'effective_number'
    
    Returns:
        List of weights for each class
    """
    disease_labels = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
    total = len(df)
    
    weights = []
    for label in disease_labels:
        positive = df[label].sum()
        negative = total - positive
        
        if method == 'inverse_freq':
            # sklearn style: total / (n_classes * n_samples_for_class)
            weight = total / (len(disease_labels) * positive) if positive > 0 else 1.0
        elif method == 'balanced':
            # Simple neg/pos ratio
            weight = negative / positive if positive > 0 else 1.0
        elif method == 'effective_number':
            # Effective number of samples (better for extreme imbalance)
            beta = 0.9999
            weight = (1 - beta**total) / (1 - beta**positive) if positive > 0 else 1.0
        else:
            weight = 1.0
        
        weights.append(weight)
    
    return weights


def calculate_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate per-sample weights for WeightedRandomSampler.
    
    Samples with rare diseases get higher weights to ensure balanced sampling.
    """
    disease_labels = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
    
    # Calculate inverse frequency for each class
    class_counts = df[disease_labels].sum(axis=0)
    class_weights = len(df) / (class_counts + 1)  # +1 to avoid division by zero
    
    # For each sample, use the max weight of its positive labels
    sample_weights = []
    class_weights_list = class_weights.tolist()  # Convert to list to avoid indexing warning
    
    for _, row in df.iterrows():
        # Get weights for all positive labels in this sample
        positive_labels = [i for i, label in enumerate(disease_labels) if row[label] == 1]
        
        if positive_labels:
            # Use max weight of positive labels
            weight = max(class_weights_list[i] for i in positive_labels)
        else:
            # Normal sample (no disease)
            weight = 1.0
        
        sample_weights.append(weight)
    
    return np.array(sample_weights)


def create_stratified_folds(df: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified k-folds based on multi-label distribution.
    
    For multi-label, we create a stratification key based on the most severe disease.
    """
    disease_labels = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
    
    # Create stratification key: most severe disease present
    # Priority: H > C > A > M > G > D > O
    priority_order = ['Label_H', 'Label_C', 'Label_A', 'Label_M', 'Label_G', 'Label_D', 'Label_O']
    
    strat_keys = []
    for _, row in df.iterrows():
        # Find highest priority disease
        for i, label in enumerate(priority_order):
            if row[label] == 1:
                strat_keys.append(i)
                break
        else:
            # No disease
            strat_keys.append(len(priority_order))
    
    strat_keys = np.array(strat_keys)
    
    # Create folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(len(df)), strat_keys))
    
    return folds


def train_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: TrainingConfig,
    output_dir: Path,
    use_weighted_sampling: bool = True
):
    """Train a single fold of the ensemble."""
    
    print(f"\n{'='*80}")
    print(f"Training Fold {fold_idx + 1}/5")
    print(f"{'='*80}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_df, method='effective_number')
    print(f"\nClass weights (Effective Number):")
    disease_names = ['DR', 'Glaucoma', 'Cataract', 'AMD', 'HTN', 'Myopia', 'Other']
    for name, weight in zip(disease_names, class_weights):
        print(f"  {name:12s}: {weight:.4f}")
    
    # Update config with class weights
    config.pos_weight = class_weights
    
    # Create datasets
    train_dataset = FundusDataset(
        df=train_df,
        image_size=config.image_size or (448, 448),
        augment=True
    )
    
    val_dataset = FundusDataset(
        df=val_df,
        image_size=config.image_size or (448, 448),
        augment=False
    )
    
    # Create data loaders with weighted sampling
    if use_weighted_sampling:
        sample_weights = calculate_sample_weights(train_df)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = FundusLightningModule(config)
    
    # Setup callbacks
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=fold_dir,
        filename="best-{epoch:02d}-{val_macro_f1:.4f}",
        monitor="val_macro_f1",
        mode="max",
        save_top_k=1
    )
    
    callbacks = [checkpoint_callback]
    
    if config.early_stopping_patience:
        early_stop = EarlyStopping(
            monitor="val_macro_f1",
            patience=config.early_stopping_patience,
            mode="max",
            verbose=True
        )
        callbacks.append(early_stop)
    
    # Create logger
    logger = CSVLogger(save_dir=fold_dir, name="logs")
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=config.epochs,
        accelerator=config.accelerator or "auto",
        devices=config.devices or "auto",
        precision=config.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.log_every_n_steps,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=1.0,  # Clip gradients to prevent exploding gradients
        deterministic=False  # For speed, set True for reproducibility
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\nFold {fold_idx + 1} training complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_macro_f1: {checkpoint_callback.best_model_score:.4f}")
    
    return checkpoint_callback.best_model_path


def main():
    """Main training function."""
    
    # Configuration
    base_dir = Path("/Users/fdb/VSCode/ODRV2")
    train_csv = base_dir / "data/processed/unified_v2/unified_train_v2.csv"
    output_dir = base_dir / "models/unified_v2_ensemble"
    
    print("="*80)
    print("Training Ensemble on Unified Dataset V2")
    print("="*80)
    print(f"Training data: {train_csv}")
    print(f"Output directory: {output_dir}")
    
    # Load training data
    train_df = pd.read_csv(train_csv)
    print(f"\nLoaded {len(train_df)} training samples")
    
    # Print class distribution
    disease_labels = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
    disease_names = ['DR', 'Glaucoma', 'Cataract', 'AMD', 'HTN', 'Myopia', 'Other']
    print("\nClass distribution:")
    for label, name in zip(disease_labels, disease_names):
        count = train_df[label].sum()
        pct = 100 * count / len(train_df)
        print(f"  {name:12s}: {int(count):>6} ({pct:>5.2f}%)")
    
    # Create stratified folds
    print("\nCreating 5-fold stratified splits...")
    folds = create_stratified_folds(train_df, n_splits=5)
    
    # Training configuration
    config = TrainingConfig(
        num_classes=7,
        feature_dim=512,
        learning_rate=1e-4,
        weight_decay=1e-5,
        epochs=30,
        batch_size=32,
        model_dir=output_dir,
        precision="16-mixed",  # Use mixed precision for faster training
        accumulate_grad_batches=2,  # Effective batch size = 64
        early_stopping_patience=5,
        image_size=(448, 448)
    )
    
    print("\nTraining configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size} (effective: {config.batch_size * config.accumulate_grad_batches})")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Precision: {config.precision}")
    print(f"  Early stopping patience: {config.early_stopping_patience}")
    print(f"  Class weighting: Focal Loss with Effective Number weights")
    print(f"  Sample weighting: WeightedRandomSampler for balanced batches")
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠ No GPU detected - training will be VERY slow!")
        response = input("Continue with CPU training? (yes/no): ")
        if response.lower() != 'yes':
            print("Training cancelled.")
            return
    
    # Train each fold
    best_models = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        fold_train_df = train_df.iloc[train_indices].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_indices].reset_index(drop=True)
        
        best_model_path = train_fold(
            fold_idx=fold_idx,
            train_df=fold_train_df,
            val_df=fold_val_df,
            config=config,
            output_dir=output_dir,
            use_weighted_sampling=True
        )
        
        best_models.append(best_model_path)
    
    # Save ensemble info
    ensemble_info = {
        'num_folds': len(folds),
        'model_paths': [str(p) for p in best_models],
        'config': config.__dict__,
        'training_samples': len(train_df),
        'class_distribution': {
            name: int(train_df[label].sum())
            for label, name in zip(disease_labels, disease_names)
        }
    }
    
    import json
    with open(output_dir / 'ensemble_info.json', 'w') as f:
        json.dump(ensemble_info, f, indent=2)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nEnsemble saved to: {output_dir}")
    print(f"Model paths:")
    for i, path in enumerate(best_models, 1):
        print(f"  Fold {i}: {path}")
    
    print("\nNext steps:")
    print("1. Evaluate on validation set: unified_val_v2.csv")
    print("2. Evaluate on test set: unified_test_v2.csv")
    print("3. Re-evaluate on HYGD external dataset (expect major improvement)")


if __name__ == "__main__":
    main()

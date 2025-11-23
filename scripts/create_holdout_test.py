"""Create a holdout test set from ODIR for final model validation.

This script creates a proper test set from ODIR data that was NOT used during
training/validation. This is the gold standard for model evaluation.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


def create_holdout_test_set():
    """Create a stratified holdout test set from ODIR."""
    
    print("📊 Creating Holdout Test Set from ODIR")
    print("="*60)
    
    # Load full dataset
    df_path = Path("data/processed/odir_eye_labels.csv")
    if not df_path.exists():
        print(f"❌ Dataset not found: {df_path}")
        return False
    
    df = pd.read_csv(df_path)
    print(f"\n✓ Loaded {len(df)} samples from ODIR")
    
    # Check if test set already exists
    test_path = Path("data/processed/odir_holdout_test.csv")
    if test_path.exists():
        print(f"\n⚠️  Holdout test set already exists at: {test_path}")
        df_test = pd.read_csv(test_path)
        print(f"   Contains {len(df_test)} samples")
        
        response = input("\nOverwrite existing test set? (yes/no): ")
        if response.lower() != 'yes':
            print("Keeping existing test set.")
            return True
    
    # Create stratified split (20% for final testing)
    # Stratify by most common disease (Diabetes) to ensure balance
    label_cols = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
    
    # Split using simple stratification on Diabetes (most samples)
    df_train_val, df_test = train_test_split(
        df,
        test_size=0.2,
        stratify=df['Label_D'],  # Stratify on most common disease
        random_state=42
    )
    
    # Save test set
    df_test.to_csv(test_path, index=False)
    
    # Save updated train/val set
    train_val_path = Path("data/processed/odir_train_val.csv")
    df_train_val.to_csv(train_val_path, index=False)
    
    print(f"\n✅ Created holdout test set!")
    print(f"\nDataset split:")
    print(f"  Train/Val: {len(df_train_val)} samples ({len(df_train_val)/len(df)*100:.1f}%)")
    print(f"  Test:      {len(df_test)} samples ({len(df_test)/len(df)*100:.1f}%)")
    
    print(f"\n📈 Test set disease distribution:")
    for col in label_cols:
        disease_name = col.replace('Label_', '')
        n_positive = int(df_test[col].sum())
        print(f"  {disease_name:12s}: {n_positive:4d} positive ({n_positive/len(df_test)*100:5.1f}%)")
    
    print(f"\n✓ Saved to:")
    print(f"  Test set:      {test_path}")
    print(f"  Train/Val set: {train_val_path}")
    
    print(f"\n🎯 Next steps:")
    print(f"  1. Re-train your models using ONLY odir_train_val.csv")
    print(f"  2. After training, evaluate on odir_holdout_test.csv")
    print(f"  3. This gives unbiased performance estimates")
    
    print(f"\n⚠️  IMPORTANT: Never look at test set metrics during development!")
    print(f"   Only evaluate once at the very end.")
    
    return True


if __name__ == "__main__":
    create_holdout_test_set()

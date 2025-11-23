#!/usr/bin/env python3
"""
Create patient-level stratified splits for Unified Dataset V3

Ensures zero patient overlap between train/val/test splits
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def extract_patient_id(row: pd.Series) -> str:
    """Extract patient ID from different dataset formats"""
    dataset = row['source_dataset']
    source_id = str(row['source_id'])
    
    if dataset == 'ODIR':
        # ODIR format: "123_left" -> patient_id = "123"
        return source_id.split('_')[0]
    elif dataset in ['HYGD', 'PAPILA', 'PALM', 'ADAM', 'Cataract_Kaggle']:
        # Use source_id directly (each image from different patient or no patient info)
        return source_id
    elif dataset == 'RFMID1':
        # RFMID format: each image is unique
        return source_id
    elif dataset == 'EyePACS':
        # EyePACS format: "patient_side" -> extract patient
        parts = source_id.split('_')
        return parts[0] if len(parts) > 1 else source_id
    else:
        return source_id


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    random_state: int = 42
):
    """Create patient-level stratified splits"""
    
    print("\n" + "="*80)
    print("CREATING PATIENT-LEVEL STRATIFIED SPLITS FOR V3")
    print("="*80)
    
    # Extract patient IDs
    print("\nExtracting patient IDs...")
    df['patient_id'] = df.apply(extract_patient_id, axis=1)
    
    # Group by patient
    print("Grouping by patient...")
    patient_groups = df.groupby('patient_id').agg({
        'Label_D': 'max',
        'Label_G': 'max',
        'Label_C': 'max',
        'Label_A': 'max',
        'Label_H': 'max',
        'Label_M': 'max',
        'Label_O': 'max',
    }).reset_index()
    
    n_patients = len(patient_groups)
    print(f"Total unique patients: {n_patients:,}")
    
    # Create stratification key based on rare diseases first (priority: H > C > A > M > G > D > O)
    priority_labels = ['Label_H', 'Label_C', 'Label_A', 'Label_M', 'Label_G', 'Label_D', 'Label_O']
    
    strat_keys = []
    for _, patient in patient_groups.iterrows():
        # Find highest priority disease
        for i, label in enumerate(priority_labels):
            if patient[label] == 1:
                strat_keys.append(i)
                break
        else:
            # No disease
            strat_keys.append(len(priority_labels))
    
    patient_groups['strat_key'] = strat_keys
    
    # Print stratification distribution
    print("\nStratification by disease priority:")
    disease_names = ['HTN (highest)', 'Cataract', 'AMD', 'Myopia', 'Glaucoma', 'DR', 'Other', 'Normal']
    for i, name in enumerate(disease_names):
        count = (patient_groups['strat_key'] == i).sum()
        pct = count / len(patient_groups) * 100
        print(f"  {name:20s}: {count:6,} patients ({pct:5.2f}%)")
    
    # First split: 80% train+val, 20% test
    np.random.seed(random_state)
    indices = np.arange(len(patient_groups))
    
    from sklearn.model_selection import train_test_split
    
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=patient_groups['strat_key'],
        random_state=random_state
    )
    
    # Second split: from train+val, split into train and val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_adjusted,
        stratify=patient_groups.iloc[train_val_idx]['strat_key'],
        random_state=random_state
    )
    
    # Get patient IDs for each split
    train_patients = set(patient_groups.iloc[train_idx]['patient_id'])
    val_patients = set(patient_groups.iloc[val_idx]['patient_id'])
    test_patients = set(patient_groups.iloc[test_idx]['patient_id'])
    
    # Verify no overlap
    print(f"\nVerifying patient-level separation...")
    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients
    
    print(f"  Train-Val overlap:  {len(train_val_overlap)} patients")
    print(f"  Train-Test overlap: {len(train_test_overlap)} patients")
    print(f"  Val-Test overlap:   {len(val_test_overlap)} patients")
    
    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print("  ✅ NO PATIENT OVERLAP - Perfect separation!")
    else:
        print("  ⚠️  WARNING: Patient overlap detected!")
    
    # Create dataframe splits
    train_df = df[df['patient_id'].isin(train_patients)].copy()
    val_df = df[df['patient_id'].isin(val_patients)].copy()
    test_df = df[df['patient_id'].isin(test_patients)].copy()
    
    # Remove patient_id column (temporary)
    train_df = train_df.drop(columns=['patient_id'])
    val_df = val_df.drop(columns=['patient_id'])
    test_df = test_df.drop(columns=['patient_id'])
    
    # Print split statistics
    print(f"\n{'='*80}")
    print("SPLIT STATISTICS")
    print("="*80)
    print(f"\nTrain: {len(train_df):,} images from {len(train_patients):,} patients ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val_df):,} images from {len(val_patients):,} patients ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test_df):,} images from {len(test_patients):,} patients ({len(test_df)/len(df)*100:.1f}%)")
    
    # Disease distribution per split
    print(f"\nDisease distribution by split:")
    disease_labels = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
    disease_names = ['DR', 'Glaucoma', 'Cataract', 'AMD', 'HTN', 'Myopia', 'Other']
    
    for label, name in zip(disease_labels, disease_names):
        train_count = train_df[label].sum()
        val_count = val_df[label].sum()
        test_count = test_df[label].sum()
        total = train_count + val_count + test_count
        
        print(f"\n  {name}:")
        print(f"    Train: {train_count:6,.0f} ({train_count/total*100:5.1f}%)")
        print(f"    Val:   {val_count:6,.0f} ({val_count/total*100:5.1f}%)")
        print(f"    Test:  {test_count:6,.0f} ({test_count/total*100:5.1f}%)")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Load unified V3
    base_path = Path("/Users/fdb/VSCode/ODRV2")
    v3_path = base_path / "data" / "processed" / "unified_v3" / "unified_dataset_v3.csv"
    
    if not v3_path.exists():
        print(f"❌ Unified V3 not found at {v3_path}")
        print("Run: python scripts/create_unified_dataset_v3.py first")
        sys.exit(1)
    
    print(f"Loading {v3_path}...")
    df = pd.read_csv(v3_path)
    print(f"Loaded {len(df):,} images")
    
    # Create splits
    train_df, val_df, test_df = create_stratified_splits(df)
    
    # Save splits
    output_dir = base_path / "data" / "processed" / "unified_v3"
    
    train_path = output_dir / "unified_train_v3.csv"
    val_path = output_dir / "unified_val_v3.csv"
    test_path = output_dir / "unified_test_v3.csv"
    
    print(f"\nSaving splits...")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  ✅ Train: {train_path}")
    print(f"  ✅ Val:   {val_path}")
    print(f"  ✅ Test:  {test_path}")
    
    print(f"\n{'='*80}")
    print("NEXT STEP")
    print("="*80)
    print("\nReady to train with enhanced rare disease handling:")
    print("  python scripts/train_unified_v3.py")
    print("="*80)

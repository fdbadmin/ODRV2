#!/usr/bin/env python3
"""
Create proper train/test split with patient-level separation.

This ensures no patient appears in both train and test sets.
Test set: 20% of patients (stratified by disease labels)
Train set: 80% of patients (will be used for 5-fold CV)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = Path('data/processed/unified_v3')
INPUT_CSV = DATA_DIR / 'unified_train_v3.csv'
OUTPUT_TRAIN_CSV = DATA_DIR / 'train_split.csv'
OUTPUT_TEST_CSV = DATA_DIR / 'test_split.csv'

LABEL_COLS = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
TEST_SIZE = 0.20
RANDOM_STATE = 42

def main():
    print("="*80)
    print("CREATING PROPER TRAIN/TEST SPLIT")
    print("="*80)
    
    # Load data
    print(f"\nLoading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total images: {len(df)}")
    
    # Create global patient IDs
    print("\nCreating patient IDs...")
    df['global_patient_id'] = df['source_dataset'].astype(str) + '_' + df['filename'].apply(lambda x: x.split('_')[0])
    
    unique_patients = df['global_patient_id'].nunique()
    print(f"Unique patients: {unique_patients}")
    
    # Aggregate at patient level
    print("\nAggregating to patient level...")
    patient_df = df.groupby('global_patient_id').first().reset_index()
    
    # Create stratification key from label combinations
    patient_df['stratify_key'] = patient_df[LABEL_COLS].astype(str).agg('_'.join, axis=1)
    
    # Handle rare combinations
    key_counts = patient_df['stratify_key'].value_counts()
    rare_keys = key_counts[key_counts < 10].index  # Combinations with <10 patients
    
    if len(rare_keys) > 0:
        print(f"Grouping {len(rare_keys)} rare combinations for stratification...")
        patient_df.loc[patient_df['stratify_key'].isin(rare_keys), 'stratify_key'] = 'RARE_COMBO'
    
    # Split patients into train and test
    print(f"\nSplitting patients: {int((1-TEST_SIZE)*100)}% train, {int(TEST_SIZE*100)}% test...")
    
    train_patients, test_patients = train_test_split(
        patient_df['global_patient_id'].values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=patient_df['stratify_key'].values
    )
    
    train_patients_set = set(train_patients)
    test_patients_set = set(test_patients)
    
    # Verify no overlap
    overlap = train_patients_set.intersection(test_patients_set)
    assert len(overlap) == 0, f"ERROR: {len(overlap)} patients in both train and test!"
    
    # Split images
    train_df = df[df['global_patient_id'].isin(train_patients_set)].copy()
    test_df = df[df['global_patient_id'].isin(test_patients_set)].copy()
    
    # Statistics
    print("\n" + "="*80)
    print("SPLIT STATISTICS")
    print("="*80)
    
    print(f"\nTrain Set:")
    print(f"  Patients: {len(train_patients)} ({len(train_patients)/unique_patients*100:.1f}%)")
    print(f"  Images: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    
    print(f"\nTest Set:")
    print(f"  Patients: {len(test_patients)} ({len(test_patients)/unique_patients*100:.1f}%)")
    print(f"  Images: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Per-class distribution
    print(f"\nLabel Distribution:")
    print(f"{'Class':<25} {'Train':>10} {'Test':>10} {'Test %':>10}")
    print("-" * 60)
    
    class_names = ['Diabetic_Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
    for col, name in zip(LABEL_COLS, class_names):
        train_count = train_df[col].sum()
        test_count = test_df[col].sum()
        total = train_count + test_count
        test_pct = (test_count / total * 100) if total > 0 else 0
        print(f"{name:<25} {int(train_count):>10} {int(test_count):>10} {test_pct:>9.1f}%")
    
    # Save splits
    print(f"\nSaving splits...")
    train_df.to_csv(OUTPUT_TRAIN_CSV, index=False)
    test_df.to_csv(OUTPUT_TEST_CSV, index=False)
    
    print(f"  Train: {OUTPUT_TRAIN_CSV}")
    print(f"  Test: {OUTPUT_TEST_CSV}")
    
    # Verification
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    # Check patient overlap
    train_patient_check = set(train_df['global_patient_id'].unique())
    test_patient_check = set(test_df['global_patient_id'].unique())
    overlap_check = train_patient_check.intersection(test_patient_check)
    
    if len(overlap_check) == 0:
        print("✓ No patient overlap between train and test sets")
    else:
        print(f"✗ ERROR: {len(overlap_check)} patients in both sets!")
    
    # Check totals
    total_check = len(train_df) + len(test_df)
    if total_check == len(df):
        print(f"✓ All {len(df)} images accounted for")
    else:
        print(f"✗ ERROR: Missing {len(df) - total_check} images!")
    
    print("\n" + "="*80)
    print("SPLIT CREATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Train models using train_split.csv")
    print("2. Evaluate on test_split.csv for unbiased results")

if __name__ == "__main__":
    main()

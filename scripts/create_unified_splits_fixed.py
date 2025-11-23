#!/usr/bin/env python3
"""
Fix unified dataset splits to prevent patient-level leakage.

Ensures that both eyes from the same patient stay in the same split.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict

def extract_patient_id(row):
    """Extract patient ID from each dataset"""
    if row['source_dataset'] == 'ODIR':
        # ODIR filenames: 1234_left.jpg or 1234_right.jpg
        match = re.match(r'(\d+)_(left|right)', row['filename'])
        if match:
            return f"ODIR_{match.group(1)}"
    elif row['source_dataset'] == 'HYGD':
        # HYGD: Patient_ImageName
        patient = row['source_id'].split('_')[0]
        return f"HYGD_{patient}"
    elif row['source_dataset'] == 'PAPILA':
        # PAPILA: RET002_OD.jpg -> patient RET002
        match = re.match(r'(RET\d+)', row['filename'])
        if match:
            return f"PAPILA_{match.group(1)}"
    elif row['source_dataset'] == 'RFMID1':
        # RFMID1: Each image is separate patient
        return f"RFMID1_{row['source_id']}"
    elif row['source_dataset'] == 'EyePACS':
        # EyePACS: Each image is separate patient (single eye per patient)
        return f"EyePACS_{row['source_id']}"
    
    return f"{row['source_dataset']}_{row['source_id']}"

def create_stratified_patient_splits(df, disease_labels, test_size=0.2, val_size=0.125):
    """
    Create train/val/test splits at the patient level with stratification.
    
    Args:
        df: DataFrame with patient_id column
        disease_labels: List of disease label columns
        test_size: Proportion for test set (default 0.2 = 20%)
        val_size: Proportion of remaining for val set (default 0.125 = 10% of total)
    
    Returns:
        train_df, val_df, test_df
    """
    # Group by patient and aggregate labels
    patient_groups = df.groupby('patient_id').agg({
        **{label: 'max' for label in disease_labels},  # Patient has disease if any eye has it
        'ID': 'first'  # Keep track of indices
    }).reset_index()
    
    # Create stratification label based on most severe disease
    # Priority: Glaucoma > AMD > Cataract > Hypertension > Myopia > DR > Other
    def get_stratification_label(row):
        if row['Label_G'] == 1:
            return 'Glaucoma'
        elif row['Label_A'] == 1:
            return 'AMD'
        elif row['Label_C'] == 1:
            return 'Cataract'
        elif row['Label_H'] == 1:
            return 'Hypertension'
        elif row['Label_M'] == 1:
            return 'Myopia'
        elif row['Label_D'] == 1:
            return 'DR'
        elif row['Label_O'] == 1:
            return 'Other'
        else:
            return 'Normal'
    
    patient_groups['stratify_label'] = patient_groups.apply(get_stratification_label, axis=1)
    
    # First split: train+val vs test (80/20)
    train_val_patients, test_patients = train_test_split(
        patient_groups,
        test_size=test_size,
        stratify=patient_groups['stratify_label'],
        random_state=42
    )
    
    # Second split: train vs val (90/10 of remaining = 72/8 of total)
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=val_size,
        stratify=train_val_patients['stratify_label'],
        random_state=42
    )
    
    # Get patient IDs for each split
    train_patient_ids = set(train_patients['patient_id'])
    val_patient_ids = set(val_patients['patient_id'])
    test_patient_ids = set(test_patients['patient_id'])
    
    # Split original dataframe by patient_id
    train_df = df[df['patient_id'].isin(train_patient_ids)].reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_patient_ids)].reset_index(drop=True)
    test_df = df[df['patient_id'].isin(test_patient_ids)].reset_index(drop=True)
    
    return train_df, val_df, test_df

def main():
    base_path = Path("/Users/fdb/VSCode/ODRV2")
    
    print("="*80)
    print("Fixing Unified Dataset V2 Splits - Patient-Level Separation")
    print("="*80)
    print()
    
    # Load unified dataset
    unified_path = base_path / "data" / "processed" / "unified_v2" / "unified_dataset_v2.csv"
    print(f"Loading: {unified_path}")
    df = pd.read_csv(unified_path)
    print(f"Total images: {len(df)}")
    print()
    
    # Extract patient IDs
    print("Extracting patient IDs...")
    df['patient_id'] = df.apply(extract_patient_id, axis=1)
    
    unique_patients = df['patient_id'].nunique()
    print(f"Unique patients: {unique_patients}")
    print()
    
    # Show patient statistics by dataset
    print("Patients per dataset:")
    for dataset in df['source_dataset'].unique():
        ds_df = df[df['source_dataset'] == dataset]
        n_patients = ds_df['patient_id'].nunique()
        n_images = len(ds_df)
        avg_images = n_images / n_patients
        print(f"  {dataset:10s}: {n_patients:>5} patients, {n_images:>5} images (avg {avg_images:.1f} images/patient)")
    print()
    
    # Disease labels
    disease_labels = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
    
    # Create splits at patient level
    print("Creating patient-level stratified splits...")
    train_df, val_df, test_df = create_stratified_patient_splits(df, disease_labels)
    
    print(f"  Train: {len(train_df):>5} images from {train_df['patient_id'].nunique():>5} patients")
    print(f"  Val:   {len(val_df):>5} images from {val_df['patient_id'].nunique():>5} patients")
    print(f"  Test:  {len(test_df):>5} images from {test_df['patient_id'].nunique():>5} patients")
    print()
    
    # Verify no patient overlap
    train_patients = set(train_df['patient_id'])
    val_patients = set(val_df['patient_id'])
    test_patients = set(test_df['patient_id'])
    
    overlap_train_val = train_patients & val_patients
    overlap_train_test = train_patients & test_patients
    overlap_val_test = val_patients & test_patients
    
    print("Verification:")
    print(f"  Train-Val patient overlap:  {len(overlap_train_val)}")
    print(f"  Train-Test patient overlap: {len(overlap_train_test)}")
    print(f"  Val-Test patient overlap:   {len(overlap_val_test)}")
    
    if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
        print("  ✅ NO PATIENT-LEVEL LEAKAGE")
    else:
        print("  ❌ PATIENT LEAKAGE STILL EXISTS!")
        return
    print()
    
    # Show disease distribution in each split
    print("Disease distribution by split:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{split_name}:")
        for label in disease_labels:
            count = split_df[label].sum()
            pct = 100 * count / len(split_df)
            disease_name = {
                'Label_D': 'DR',
                'Label_G': 'Glaucoma',
                'Label_C': 'Cataract',
                'Label_A': 'AMD',
                'Label_H': 'HTN',
                'Label_M': 'Myopia',
                'Label_O': 'Other'
            }[label]
            print(f"  {disease_name:10s}: {count:>5} ({pct:>5.1f}%)")
    
    # Save splits (remove patient_id column)
    output_dir = base_path / "data" / "processed" / "unified_v2"
    
    train_df_save = train_df.drop(columns=['patient_id'])
    val_df_save = val_df.drop(columns=['patient_id'])
    test_df_save = test_df.drop(columns=['patient_id'])
    
    train_path = output_dir / "unified_train_v2.csv"
    val_path = output_dir / "unified_val_v2.csv"
    test_path = output_dir / "unified_test_v2.csv"
    
    print()
    print("Saving fixed splits...")
    train_df_save.to_csv(train_path, index=False)
    val_df_save.to_csv(val_path, index=False)
    test_df_save.to_csv(test_path, index=False)
    
    print(f"  ✅ {train_path}")
    print(f"  ✅ {val_path}")
    print(f"  ✅ {test_path}")
    
    print()
    print("="*80)
    print("Fixed splits saved! No patient-level leakage.")
    print("="*80)

if __name__ == "__main__":
    main()

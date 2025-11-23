import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

def audit_pipeline():
    print("="*80)
    print("PIPELINE AUDIT: Unified V3 Training Setup")
    print("="*80)
    
    # 1. Data Loading
    csv_path = Path('data/processed/unified_v3/unified_train_v3.csv')
    print(f"\n[1] Data Loading Check")
    if not csv_path.exists():
        print(f"❌ CRITICAL: CSV file not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows.")
    
    # Check for critical columns
    required_cols = ['filename', 'source_dataset', 'image_path'] + [f'Label_{c}' for c in 'DGCAHMO']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"❌ CRITICAL: Missing columns: {missing_cols}")
        return
    print("✅ All required columns present.")

    # 2. Image Existence Check (Sample)
    print(f"\n[2] Image Integrity Check (Sample 100)")
    sample_df = df.sample(min(100, len(df)), random_state=42)
    missing_images = []
    for _, row in sample_df.iterrows():
        if not Path(row['image_path']).exists():
            missing_images.append(row['image_path'])
    
    if missing_images:
        print(f"❌ CRITICAL: Found {len(missing_images)} missing images in sample!")
        print(f"   Example: {missing_images[0]}")
    else:
        print("✅ All sampled image paths exist.")

    # 3. Patient Grouping Logic (The "Fix")
    print(f"\n[3] Patient Grouping Logic Audit")
    
    # Replicate the logic from train_unified_v3_pure_pytorch.py
    print("   Applying extraction logic: df['patient_id'] = df['filename'].apply(lambda x: x.split('_')[0])")
    df['patient_id_derived'] = df['filename'].apply(lambda x: x.split('_')[0])
    df['global_patient_id'] = df['source_dataset'].astype(str) + '_' + df['patient_id_derived'].astype(str)
    
    # Specific check for 13701
    patient_13701 = df[df['filename'].str.contains('13701_')]
    if len(patient_13701) > 0:
        unique_ids = patient_13701['global_patient_id'].unique()
        print(f"   Patient 13701 (EyePACS) maps to global IDs: {unique_ids}")
        if len(unique_ids) == 1:
            print("✅ Patient 13701 correctly grouped.")
        else:
            print(f"❌ CRITICAL: Patient 13701 split across IDs: {unique_ids}")
    else:
        print("⚠️ Patient 13701 not found in dataset (might be okay if filtered out).")

    # Check for ODIR patient 0
    patient_0 = df[(df['source_dataset'] == 'ODIR') & (df['filename'].str.startswith('0_'))]
    if len(patient_0) > 0:
        unique_ids_0 = patient_0['global_patient_id'].unique()
        print(f"   Patient 0 (ODIR) maps to global IDs: {unique_ids_0}")
        if len(unique_ids_0) == 1:
            print("✅ Patient 0 correctly grouped.")
        else:
            print(f"❌ CRITICAL: Patient 0 split across IDs: {unique_ids_0}")

    # 4. Stratification & Leakage Check
    print(f"\n[4] Stratification & Leakage Audit")
    
    # Group by patient
    patient_df = df.groupby('global_patient_id').first().reset_index()
    label_cols = [f'Label_{c}' for c in 'DGCAHMO']
    patient_df['stratify_key'] = patient_df[label_cols].astype(str).agg('_'.join, axis=1)
    
    # Handle rare combinations
    key_counts = patient_df['stratify_key'].value_counts()
    rare_keys = key_counts[key_counts < 5].index
    print(f"   Found {len(rare_keys)} rare label combinations (<5 samples).")
    patient_df.loc[patient_df['stratify_key'].isin(rare_keys), 'stratify_key'] = 'rare_combination'
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_leakage_found = False
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_df, patient_df['stratify_key'])):
        train_patients = set(patient_df.iloc[train_idx]['global_patient_id'])
        val_patients = set(patient_df.iloc[val_idx]['global_patient_id'])
        
        # LEAKAGE CHECK
        intersection = train_patients.intersection(val_patients)
        if intersection:
            print(f"❌ CRITICAL: DATA LEAKAGE in Fold {fold_idx}!")
            print(f"   {len(intersection)} patients appear in both Train and Val.")
            fold_leakage_found = True
        
        # Distribution Check
        val_subset = patient_df.iloc[val_idx]
        # Just check one label for brevity
        pos_rate = val_subset['Label_D'].mean()
        print(f"   Fold {fold_idx}: Val Size {len(val_patients)} | Leakage: {'YES' if intersection else 'NO'} | Label_D Rate: {pos_rate:.3f}")

    if not fold_leakage_found:
        print("✅ No data leakage detected across any fold.")

    # 5. Class Weight Check
    print(f"\n[5] Class Weight Logic Check")
    # Calculate actual positive rates
    total_samples = len(df)
    pos_counts = df[label_cols].sum()
    print("   Actual Positive Counts:")
    print(pos_counts)
    
    # Configured weights in script: [1.0, 5.0, 8.0, 60.0, 150.0, 60.0, 1.0]
    # D, G, C, A, H, M, O
    # Let's see if they make sense inversely
    suggested_weights = total_samples / (pos_counts + 1e-6)
    # Normalize to D=1.0 roughly
    suggested_weights = suggested_weights / suggested_weights['Label_D']
    print("   Suggested Weights (Inverse Frequency, normalized to D=1):")
    print(suggested_weights.round(1))
    
    print("   Current Config Weights: [1.0, 5.0, 8.0, 60.0, 150.0, 60.0, 1.0]")
    print("   (Visual check: Do these align with the rarity shown above?)")

    print("\n" + "="*80)
    print("AUDIT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    audit_pipeline()

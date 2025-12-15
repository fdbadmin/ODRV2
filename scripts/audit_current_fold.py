
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

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

def audit_fold_1():
    csv_path = 'data/processed/unified_v3/unified_train_v3.csv'
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("Recreating folds...")
    df = create_folds(df, n_splits=5, seed=42)
    
    fold = 1
    train_df = df[df['fold'] != fold]
    val_df = df[df['fold'] == fold]
    
    print(f"\n--- Audit for Fold {fold} ---")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    train_patients = set(train_df['global_patient_id'])
    val_patients = set(val_df['global_patient_id'])
    
    intersection = train_patients.intersection(val_patients)
    
    print(f"Unique patients in Train: {len(train_patients)}")
    print(f"Unique patients in Val: {len(val_patients)}")
    print(f"Overlapping patients: {len(intersection)}")
    
    if len(intersection) > 0:
        print("❌ CRITICAL LEAKAGE DETECTED!")
        print(f"Leaked Patient IDs: {list(intersection)[:5]}...")
    else:
        print("✅ No patient leakage detected.")
        
    # Check for image overlap (should be impossible if patients are unique, but good to check)
    train_images = set(train_df['filename'])
    val_images = set(val_df['filename'])
    image_intersection = train_images.intersection(val_images)
    
    print(f"Overlapping images: {len(image_intersection)}")
    if len(image_intersection) > 0:
         print("❌ IMAGE LEAKAGE DETECTED!")
    else:
         print("✅ No image leakage detected.")

if __name__ == "__main__":
    audit_fold_1()

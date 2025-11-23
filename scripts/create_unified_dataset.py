"""
Create unified multi-dataset CSV for training

Combines ODIR, HYGD, RFMID1, and RFMID2 datasets with proper label mapping.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


def create_unified_dataset():
    """Merge all datasets into unified format"""
    
    print("="*80)
    print("CREATING UNIFIED MULTI-DATASET FOR TRAINING")
    print("="*80)
    
    # Define output paths
    output_dir = Path("data/processed/unified")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_rows = []
    
    # =========================================================================
    # 1. ODIR DATA (baseline)
    # =========================================================================
    print("\n1. Processing ODIR dataset...")
    odir_df = pd.read_csv('data/processed/odir_eye_labels.csv')
    
    for _, row in tqdm(odir_df.iterrows(), total=len(odir_df), desc="ODIR"):
        all_rows.append({
            'ID': len(all_rows),
            'source_dataset': 'ODIR',
            'source_id': row['ID'],
            'filename': row['filename'],
            'image_path': f"RAW DATA FULL/{row['filename']}",
            'Patient Age': row['Patient Age'],
            'Patient Sex': row['Patient Sex'],
            'Label_D': int(row['Label_D']),
            'Label_G': int(row['Label_G']),
            'Label_C': int(row['Label_C']),
            'Label_A': int(row['Label_A']),
            'Label_H': int(row['Label_H']),
            'Label_M': int(row['Label_M']),
            'Label_O': int(row['Label_O']),
        })
    
    print(f"  Added {len(odir_df)} ODIR images")
    
    # =========================================================================
    # 2. HYGD DATA (glaucoma-focused)
    # =========================================================================
    print("\n2. Processing HYGD dataset...")
    hygd_df = pd.read_csv('external_data/glaucoma_standard/Labels.csv')
    
    # Default age/sex for HYGD (not provided in dataset)
    default_age = 60.0  # Typical glaucoma screening age
    default_sex = 'Female'  # Balanced assumption
    
    for _, row in tqdm(hygd_df.iterrows(), total=len(hygd_df), desc="HYGD"):
        is_glaucoma = 1 if row['Label'] == 'GON+' else 0
        
        all_rows.append({
            'ID': len(all_rows),
            'source_dataset': 'HYGD',
            'source_id': row['Patient'],
            'filename': row['Image Name'],
            'image_path': f"external_data/glaucoma_standard/Images/{row['Image Name']}",
            'Patient Age': default_age,
            'Patient Sex': default_sex,
            'Label_D': 0,  # Binary glaucoma dataset - no DR
            'Label_G': is_glaucoma,
            'Label_C': 0,
            'Label_A': 0,
            'Label_H': 0,
            'Label_M': 0,
            'Label_O': 0,  # Pure glaucoma dataset
        })
    
    print(f"  Added {len(hygd_df)} HYGD images")
    print(f"    - Glaucoma: {hygd_df[hygd_df['Label'] == 'GON+'].shape[0]}")
    print(f"    - Normal: {hygd_df[hygd_df['Label'] == 'GON-'].shape[0]}")
    
    # =========================================================================
    # 3. RFMID1 DATA (multi-disease)
    # =========================================================================
    print("\n3. Processing RFMID1 dataset...")
    rfmid1_df = pd.read_csv('external_data/RFMID1/RFMiD_Training_Labels.csv')
    
    # Default demographics for RFMID1
    default_age = 55.0
    default_sex = 'Female'
    
    for _, row in tqdm(rfmid1_df.iterrows(), total=len(rfmid1_df), desc="RFMID1"):
        # Map RFMID diseases to ODIR categories
        label_d = int(row.get('DR', 0))  # Diabetic Retinopathy
        label_a = int(row.get('ARMD', 0))  # AMD
        label_m = int(row.get('MYA', 0))  # Myopia
        
        # Glaucoma: Use optic disc abnormalities as proxy
        # ODC (Optic Disc Cupping), ODP (Optic Disc Pallor), ODE (Optic Disc Edema)
        label_g = int(max(row.get('ODC', 0), row.get('ODP', 0), row.get('ODE', 0)))
        
        # Cataract: Not clearly mapped in RFMID1
        label_c = 0
        
        # Hypertension: Not in RFMID1
        label_h = 0
        
        # Other: If disease risk but no mapped category
        has_mapped_disease = any([label_d, label_g, label_a, label_m])
        label_o = int(row.get('Disease_Risk', 0)) if not has_mapped_disease else 0
        
        filename = f"{row['ID']}.png"
        
        all_rows.append({
            'ID': len(all_rows),
            'source_dataset': 'RFMID1',
            'source_id': row['ID'],
            'filename': filename,
            'image_path': f"external_data/RFMID1/Training/{filename}",
            'Patient Age': default_age,
            'Patient Sex': default_sex,
            'Label_D': label_d,
            'Label_G': label_g,
            'Label_C': label_c,
            'Label_A': label_a,
            'Label_H': label_h,
            'Label_M': label_m,
            'Label_O': label_o,
        })
    
    print(f"  Added {len(rfmid1_df)} RFMID1 images")
    rfmid1_stats = pd.DataFrame([r for r in all_rows if r['source_dataset'] == 'RFMID1'])
    print(f"    - DR (D): {rfmid1_stats['Label_D'].sum()}")
    print(f"    - Glaucoma (G): {rfmid1_stats['Label_G'].sum()}")
    print(f"    - AMD (A): {rfmid1_stats['Label_A'].sum()}")
    print(f"    - Myopia (M): {rfmid1_stats['Label_M'].sum()}")
    
    # =========================================================================
    # 4. RFMID2 DATA (multi-disease)
    # =========================================================================
    print("\n4. Processing RFMID2 dataset...")
    rfmid2_df = pd.read_csv('external_data/RFMID2/Training_set/RFMiD_2_Training_labels.csv', 
                            encoding='latin1')
    
    for _, row in tqdm(rfmid2_df.iterrows(), total=len(rfmid2_df), desc="RFMID2"):
        # Map RFMID2 diseases to ODIR categories
        label_d = int(row.get('DR', 0))
        label_a = int(row.get('ARMD', 0))
        label_m = int(row.get('MYA', 0))
        label_h = int(row.get('HTN', 0))  # Hypertension available in RFMID2!
        
        # Glaucoma from optic disc abnormalities
        label_g = int(max(
            row.get('GRT', 0),  # Glaucomatous Retinopathy
            row.get('ODC', 0),
            row.get('ODP', 0),
            row.get('ODE', 0)
        ))
        
        label_c = 0  # Cataract not mapped
        
        # Other: If not WNL (Within Normal Limits) and no mapped disease
        has_mapped_disease = any([label_d, label_g, label_a, label_m, label_h])
        label_o = int(not row.get('WNL', 0)) if not has_mapped_disease else 0
        
        filename = f"{row['ID']}.jpg"
        
        all_rows.append({
            'ID': len(all_rows),
            'source_dataset': 'RFMID2',
            'source_id': row['ID'],
            'filename': filename,
            'image_path': f"external_data/RFMID2/Training_set/{filename}",
            'Patient Age': default_age,
            'Patient Sex': default_sex,
            'Label_D': label_d,
            'Label_G': label_g,
            'Label_C': label_c,
            'Label_A': label_a,
            'Label_H': label_h,
            'Label_M': label_m,
            'Label_O': label_o,
        })
    
    print(f"  Added {len(rfmid2_df)} RFMID2 images")
    rfmid2_stats = pd.DataFrame([r for r in all_rows if r['source_dataset'] == 'RFMID2'])
    print(f"    - DR (D): {rfmid2_stats['Label_D'].sum()}")
    print(f"    - Glaucoma (G): {rfmid2_stats['Label_G'].sum()}")
    print(f"    - AMD (A): {rfmid2_stats['Label_A'].sum()}")
    print(f"    - Myopia (M): {rfmid2_stats['Label_M'].sum()}")
    print(f"    - Hypertension (H): {rfmid2_stats['Label_H'].sum()}")
    
    # =========================================================================
    # CREATE UNIFIED DATAFRAME
    # =========================================================================
    print("\n" + "="*80)
    print("CREATING UNIFIED DATAFRAME")
    print("="*80)
    
    unified_df = pd.DataFrame(all_rows)
    
    # Verify all image paths exist
    print("\nVerifying image paths...")
    missing_count = 0
    for idx, row in unified_df.iterrows():
        if not Path(row['image_path']).exists():
            missing_count += 1
            if missing_count <= 5:  # Show first 5 missing
                print(f"  Warning: Missing {row['image_path']}")
    
    if missing_count > 0:
        print(f"\n  Total missing images: {missing_count}")
        print(f"  Removing missing images from dataset...")
        unified_df = unified_df[unified_df['image_path'].apply(lambda x: Path(x).exists())]
    
    # Statistics
    print("\n" + "="*80)
    print("UNIFIED DATASET STATISTICS")
    print("="*80)
    
    print(f"\nTotal images: {len(unified_df)}")
    print(f"\nBy source:")
    print(unified_df['source_dataset'].value_counts())
    
    print(f"\nDisease prevalence (combined):")
    for disease_code, disease_name in [
        ('D', 'Diabetic Retinopathy'),
        ('G', 'Glaucoma'),
        ('C', 'Cataract'),
        ('A', 'Age-related Macular Degeneration'),
        ('H', 'Hypertension'),
        ('M', 'Myopia'),
        ('O', 'Other diseases')
    ]:
        count = unified_df[f'Label_{disease_code}'].sum()
        pct = count / len(unified_df) * 100
        print(f"  {disease_code} ({disease_name}): {int(count)} ({pct:.1f}%)")
    
    # Comparison with ODIR-only
    print(f"\nImprovement over ODIR-only training:")
    odir_counts = {
        'D': 2252, 'G': 397, 'C': 402, 'A': 319, 'H': 204, 'M': 293, 'O': 1527
    }
    for code, odir_count in odir_counts.items():
        new_count = int(unified_df[f'Label_{code}'].sum())
        improvement = ((new_count - odir_count) / odir_count * 100) if odir_count > 0 else 0
        print(f"  {code}: {odir_count} → {new_count} ({improvement:+.1f}%)")
    
    # Save unified dataset
    output_path = output_dir / "unified_dataset.csv"
    unified_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved unified dataset to: {output_path}")
    
    # Create train/val/test splits (80/10/10)
    print("\nCreating train/val/test splits...")
    
    # Shuffle with fixed seed
    unified_df = unified_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n_total = len(unified_df)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    train_df = unified_df[:n_train]
    val_df = unified_df[n_train:n_train+n_val]
    test_df = unified_df[n_train+n_val:]
    
    train_df.to_csv(output_dir / "unified_train.csv", index=False)
    val_df.to_csv(output_dir / "unified_val.csv", index=False)
    test_df.to_csv(output_dir / "unified_test.csv", index=False)
    
    print(f"  Train: {len(train_df)} images")
    print(f"  Val:   {len(val_df)} images")
    print(f"  Test:  {len(test_df)} images")
    
    print("\n" + "="*80)
    print("✓ DATASET INTEGRATION COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {output_dir}/unified_dataset.csv")
    print(f"  - {output_dir}/unified_train.csv")
    print(f"  - {output_dir}/unified_val.csv")
    print(f"  - {output_dir}/unified_test.csv")
    
    print("\nNext steps:")
    print("  1. Review the unified dataset statistics")
    print("  2. Update training config to use unified_train.csv")
    print("  3. Retrain models with expanded dataset")
    print("  4. Evaluate on unified_test.csv and external datasets")
    
    return unified_df


if __name__ == "__main__":
    create_unified_dataset()

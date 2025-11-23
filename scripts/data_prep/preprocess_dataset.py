#!/usr/bin/env python3
"""
Create Unified Dataset V3: V2 + Cataract + PALM + ADAM

This script extends Unified V2 by adding rare disease datasets:
- Cataract Dataset (Kaggle): +1,038 images
- PALM (Myopia): +1,200 images
- ADAM (AMD): +800 images

Total V3: ~47,700 images (up from 44,673)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class UnifiedDatasetV3Creator:
    def __init__(self):
        self.base_path = Path("/Users/fdb/VSCode/ODRV2")
        self.output_dir = self.base_path / "data" / "processed" / "unified_v3"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Disease labels
        self.disease_labels = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
        
    def load_unified_v2(self) -> pd.DataFrame:
        """Load existing Unified V2 dataset as base"""
        print("Loading Unified V2 as base...")
        v2_path = self.base_path / "data" / "processed" / "unified_v2" / "unified_dataset_v2.csv"
        
        if not v2_path.exists():
            raise FileNotFoundError(
                f"Unified V2 not found at {v2_path}\n"
                "Please run scripts/create_unified_dataset_v2.py first"
            )
        
        df = pd.read_csv(v2_path)
        print(f"  Loaded {len(df):,} images from V2")
        return df
    
    def load_cataract_kaggle(self) -> pd.DataFrame:
        """Load Cataract Dataset from Kaggle"""
        print("\nLoading Cataract Dataset (Kaggle)...")
        cataract_dir = self.base_path / "external_data" / "cataract_kaggle"
        
        # Check for different possible structures
        possible_dirs = [
            cataract_dir / "train",
            cataract_dir / "dataset" / "train",
            cataract_dir,
        ]
        
        image_dir = None
        for d in possible_dirs:
            if d.exists() and any(d.glob("**/*.jpg")) or any(d.glob("**/*.png")):
                image_dir = d
                break
        
        if image_dir is None:
            print(f"  ⚠️  Cataract dataset not found in {cataract_dir}")
            print(f"  Run: ./scripts/download_rare_disease_datasets.sh")
            return pd.DataFrame()
        
        # Find all cataract and normal images
        cataract_images = list(image_dir.glob("**/cataract/*.jpg")) + list(image_dir.glob("**/cataract/*.png"))
        normal_images = list(image_dir.glob("**/normal/*.jpg")) + list(image_dir.glob("**/normal/*.png"))
        
        # If not in subdirectories, look for naming patterns
        if not cataract_images and not normal_images:
            all_images = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.png"))
            cataract_images = [img for img in all_images if 'cataract' in str(img).lower()]
            normal_images = [img for img in all_images if 'normal' in str(img).lower()]
        
        data = []
        
        # Process cataract images
        for img_path in cataract_images:
            data.append({
                'source_dataset': 'Cataract_Kaggle',
                'source_id': img_path.stem,
                'filename': img_path.name,
                'image_path': str(img_path),
                'Patient Age': np.nan,
                'Patient Sex': np.nan,
                'Label_D': 0,
                'Label_G': 0,
                'Label_C': 1,  # Cataract positive
                'Label_A': 0,
                'Label_H': 0,
                'Label_M': 0,
                'Label_O': 0
            })
        
        # Process normal images
        for img_path in normal_images:
            data.append({
                'source_dataset': 'Cataract_Kaggle',
                'source_id': img_path.stem,
                'filename': img_path.name,
                'image_path': str(img_path),
                'Patient Age': np.nan,
                'Patient Sex': np.nan,
                'Label_D': 0,
                'Label_G': 0,
                'Label_C': 0,  # Normal (no cataract)
                'Label_A': 0,
                'Label_H': 0,
                'Label_M': 0,
                'Label_O': 0
            })
        
        df = pd.DataFrame(data)
        print(f"  Loaded {len(df):,} images ({df['Label_C'].sum():.0f} cataract, {(df['Label_C']==0).sum():.0f} normal)")
        return df
    
    def load_palm(self) -> pd.DataFrame:
        """Load PALM Myopia Dataset"""
        print("\nLoading PALM (Myopia) Dataset...")
        palm_dir = self.base_path / "external_data" / "PALM"
        
        if not palm_dir.exists():
            print(f"  ⚠️  PALM dataset not found at {palm_dir}")
            print(f"  Download from: https://palm.grand-challenge.org/")
            return pd.DataFrame()
        
        data = []
        
        # Check Training set
        training_dirs = [
            palm_dir / "Training" / "Training400" / "fundus_images",
            palm_dir / "Training400" / "fundus_images",
            palm_dir / "Training" / "fundus_images",
        ]
        
        for train_dir in training_dirs:
            if train_dir.exists():
                images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
                for img_path in images:
                    data.append({
                        'source_dataset': 'PALM',
                        'source_id': img_path.stem,
                        'filename': img_path.name,
                        'image_path': str(img_path),
                        'Patient Age': np.nan,
                        'Patient Sex': np.nan,
                        'Label_D': 0,
                        'Label_G': 0,
                        'Label_C': 0,
                        'Label_A': 0,
                        'Label_H': 0,
                        'Label_M': 1,  # Pathologic myopia
                        'Label_O': 0
                    })
                print(f"  Found {len(images):,} images in {train_dir.name}")
                break
        
        # Check Validation set
        val_dirs = [
            palm_dir / "PALM-Validation400",
            palm_dir / "Validation400",
            palm_dir / "Validation" / "fundus_images",
        ]
        
        for val_dir in val_dirs:
            if val_dir.exists():
                images = list(val_dir.glob("**/*.jpg")) + list(val_dir.glob("**/*.png"))
                for img_path in images:
                    if 'fundus' in str(img_path).lower() or img_path.suffix in ['.jpg', '.png']:
                        data.append({
                            'source_dataset': 'PALM',
                            'source_id': img_path.stem,
                            'filename': img_path.name,
                            'image_path': str(img_path),
                            'Patient Age': np.nan,
                            'Patient Sex': np.nan,
                            'Label_D': 0,
                            'Label_G': 0,
                            'Label_C': 0,
                            'Label_A': 0,
                            'Label_H': 0,
                            'Label_M': 1,  # Pathologic myopia
                            'Label_O': 0
                        })
                print(f"  Found {len(images):,} images in validation set")
                break
        
        df = pd.DataFrame(data)
        if len(df) > 0:
            print(f"  Total loaded: {len(df):,} images (all myopia positive)")
        return df
    
    def load_adam(self) -> pd.DataFrame:
        """Load ADAM AMD Dataset"""
        print("\nLoading ADAM (AMD) Dataset...")
        adam_dir = self.base_path / "external_data" / "ADAM"
        
        if not adam_dir.exists():
            print(f"  ⚠️  ADAM dataset not found at {adam_dir}")
            print(f"  Download from: https://amd.grand-challenge.org/")
            return pd.DataFrame()
        
        data = []
        
        # Training set
        training_dirs = [
            adam_dir / "Training400",
            adam_dir / "Training",
        ]
        
        for train_dir in training_dirs:
            if train_dir.exists():
                # AMD positive images
                amd_dir = train_dir / "AMD"
                if amd_dir.exists():
                    images = list(amd_dir.glob("*.jpg")) + list(amd_dir.glob("*.png"))
                    for img_path in images:
                        data.append({
                            'source_dataset': 'ADAM',
                            'source_id': img_path.stem,
                            'filename': img_path.name,
                            'image_path': str(img_path),
                            'Patient Age': np.nan,
                            'Patient Sex': np.nan,
                            'Label_D': 0,
                            'Label_G': 0,
                            'Label_C': 0,
                            'Label_A': 1,  # AMD positive
                            'Label_H': 0,
                            'Label_M': 0,
                            'Label_O': 0
                        })
                    print(f"  Found {len(images):,} AMD positive images")
                
                # Non-AMD images
                non_amd_dir = train_dir / "Non-AMD"
                if non_amd_dir.exists():
                    images = list(non_amd_dir.glob("*.jpg")) + list(non_amd_dir.glob("*.png"))
                    for img_path in images:
                        data.append({
                            'source_dataset': 'ADAM',
                            'source_id': img_path.stem,
                            'filename': img_path.name,
                            'image_path': str(img_path),
                            'Patient Age': np.nan,
                            'Patient Sex': np.nan,
                            'Label_D': 0,
                            'Label_G': 0,
                            'Label_C': 0,
                            'Label_A': 0,  # AMD negative
                            'Label_H': 0,
                            'Label_M': 0,
                            'Label_O': 0
                        })
                    print(f"  Found {len(images):,} non-AMD images")
                break
        
        df = pd.DataFrame(data)
        if len(df) > 0:
            print(f"  Total loaded: {len(df):,} images ({df['Label_A'].sum():.0f} AMD, {(df['Label_A']==0).sum():.0f} normal)")
        return df
    
    def load_hrf(self) -> pd.DataFrame:
        """Load HRF Dataset (Hypertensive Retinopathy)"""
        print("\nLoading HRF Dataset (HTN Retinopathy)...")
        hrf_dir = self.base_path / "external_data" / "HRF"
        
        if not hrf_dir.exists():
            print(f"  ⚠️  HRF dataset not found at {hrf_dir}")
            return pd.DataFrame()
        
        data = []
        
        # HRF has images named like: 01_h.jpg (healthy), 01_dr.jpg (DR), 01_g.jpg (glaucoma)
        # Some images show hypertensive retinopathy features
        all_images = list(hrf_dir.glob("**/*.jpg")) + list(hrf_dir.glob("**/*.png")) + list(hrf_dir.glob("**/*.JPG"))
        
        for img_path in all_images:
            filename = img_path.name.lower()
            
            # Determine labels based on filename patterns
            # HRF doesn't have explicit HTN labels, but we mark healthy ones for general training
            label_dr = 1 if '_dr' in filename else 0
            label_g = 1 if '_g' in filename else 0
            label_h = 1 if '_h' in filename else 0  # healthy
            
            data.append({
                'source_dataset': 'HRF',
                'source_id': img_path.stem,
                'filename': img_path.name,
                'image_path': str(img_path),
                'Patient Age': np.nan,
                'Patient Sex': np.nan,
                'Label_D': label_dr,
                'Label_G': label_g,
                'Label_C': 0,
                'Label_A': 0,
                'Label_H': 0,  # HRF doesn't have HTN labels, just high quality images
                'Label_M': 0,
                'Label_O': label_h
            })
        
        df = pd.DataFrame(data)
        if len(df) > 0:
            print(f"  Total loaded: {len(df):,} images")
            print(f"    DR: {df['Label_D'].sum():.0f}, Glaucoma: {df['Label_G'].sum():.0f}, Healthy: {df['Label_O'].sum():.0f}")
        return df
    
    def create_unified_v3(self):
        """Combine V2 + rare disease datasets"""
        print("\n" + "="*80)
        print("Creating Unified Dataset V3")
        print("="*80 + "\n")
        
        # Load base V2
        v2_df = self.load_unified_v2()
        
        # Load rare disease datasets
        cataract_df = self.load_cataract_kaggle()
        palm_df = self.load_palm()
        adam_df = self.load_adam()
        hrf_df = self.load_hrf()
        
        # Combine all
        dfs_to_combine = [v2_df]
        
        if len(cataract_df) > 0:
            dfs_to_combine.append(cataract_df)
        if len(palm_df) > 0:
            dfs_to_combine.append(palm_df)
        if len(adam_df) > 0:
            dfs_to_combine.append(adam_df)
        if len(hrf_df) > 0:
            dfs_to_combine.append(hrf_df)
        
        unified_df = pd.concat(dfs_to_combine, ignore_index=True)
        
        # Reset ID
        unified_df['ID'] = range(len(unified_df))
        
        # Reorder columns
        cols = ['ID', 'source_dataset', 'source_id', 'filename', 'image_path', 'Patient Age', 'Patient Sex'] + self.disease_labels
        unified_df = unified_df[cols]
        
        # Print statistics
        print(f"\n{'='*80}")
        print("UNIFIED V3 STATISTICS")
        print("="*80)
        print(f"\nTotal images: {len(unified_df):,}")
        print(f"\nBy dataset:")
        for dataset in unified_df['source_dataset'].unique():
            count = (unified_df['source_dataset'] == dataset).sum()
            print(f"  {dataset:20s}: {count:>7,} images")
        
        print(f"\nDisease distribution:")
        disease_names = ['DR', 'Glaucoma', 'Cataract', 'AMD', 'HTN', 'Myopia', 'Other']
        for label, name in zip(self.disease_labels, disease_names):
            count = unified_df[label].sum()
            pct = 100 * count / len(unified_df)
            old_count_v2 = v2_df[label].sum()
            improvement = count - old_count_v2
            improvement_pct = (improvement / old_count_v2 * 100) if old_count_v2 > 0 else 0
            
            if improvement > 0:
                print(f"  {name:15s}: {count:>6,.0f} ({pct:>5.2f}%)  [+{improvement:>4,.0f}, +{improvement_pct:>5.1f}%] 🔥")
            else:
                print(f"  {name:15s}: {count:>6,.0f} ({pct:>5.2f}%)")
        
        # Save
        output_path = self.output_dir / "unified_dataset_v3.csv"
        unified_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved unified V3 to: {output_path}")
        
        # Next steps
        print(f"\n{'='*80}")
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Create patient-level stratified splits:")
        print("   python scripts/create_unified_splits_v3.py")
        print("\n2. Train with enhanced rare disease focus:")
        print("   python scripts/train_unified_v3.py")
        print("="*80)
        
        return unified_df


if __name__ == "__main__":
    creator = UnifiedDatasetV3Creator()
    unified_df = creator.create_unified_v3()

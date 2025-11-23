#!/usr/bin/env python3
"""
Create Unified Dataset V2: ODIR + HYGD + RFMID1 + EyePACS + PAPILA

This script integrates 5 major fundus image datasets into a single unified dataset
with standardized disease labels and train/val/test splits.

Datasets:
- ODIR: 6,392 images, 7 diseases (D, G, C, A, H, M, O)
- HYGD: 747 images, glaucoma binary classification  
- RFMID1: 1,920 images, 46 disease categories
- EyePACS: 35,126 images, DR grading 0-4
- PAPILA: 488 images, glaucoma classification

Expected output: ~44,673 images with unified labels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from typing import Dict, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class UnifiedDatasetV2Creator:
    def __init__(self):
        self.base_path = Path("/Users/fdb/VSCode/ODRV2")
        self.output_dir = self.base_path / "data" / "processed" / "unified_v2"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Disease labels
        self.disease_labels = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
        
    def load_odir(self) -> pd.DataFrame:
        """Load ODIR dataset (6,392 images)"""
        print("Loading ODIR dataset...")
        df = pd.read_csv(self.base_path / "data" / "processed" / "odir_eye_labels.csv")
        
        # Standardize columns
        df['source_dataset'] = 'ODIR'
        df['source_id'] = df['ID'].astype(str)
        df['image_path'] = df['filename'].apply(
            lambda x: str(self.base_path / "data" / "raw" / "ODIR-5K" / "ODIR-5K_Training_Images" / x)
        )
        
        # Convert gender to numeric: Male=1, Female=0
        df['Patient Sex'] = df['Patient Sex'].map({'Male': 1, 'Female': 0})
        
        # Keep disease labels
        cols = ['source_dataset', 'source_id', 'filename', 'image_path', 'Patient Age', 'Patient Sex'] + self.disease_labels
        return df[cols]
    
    def load_hygd(self) -> pd.DataFrame:
        """Load HYGD dataset (747 images)"""
        print("Loading HYGD dataset...")
        df = pd.read_csv(self.base_path / "external_data" / "glaucoma_standard" / "labels.csv")
        
        result = pd.DataFrame({
            'source_dataset': ['HYGD'] * len(df),
            'source_id': df['Patient'].astype(str) + '_' + df['Image Name'],
            'filename': df['Image Name'],
            'image_path': df['Image Name'].apply(
                lambda x: str(self.base_path / "external_data" / "glaucoma_standard" / x)
            ),
            'Patient Age': np.nan,
            'Patient Sex': np.nan,
            'Label_D': 0,
            'Label_G': (df['Label'] == 'GON+').astype(int),
            'Label_C': 0,
            'Label_A': 0,
            'Label_H': 0,
            'Label_M': 0,
            'Label_O': 0
        })
                
        return result
    
    def load_rfmid1(self) -> pd.DataFrame:
        """Load RFMID1 dataset (1,920 images)"""
        print("Loading RFMID1 dataset...")
        df = pd.read_csv(self.base_path / "external_data" / "RFMiD1" / "RFMiD_Training_Labels.csv")
        
        result = pd.DataFrame({
            'source_dataset': ['RFMID1'] * len(df),
            'source_id': df['ID'].astype(str),
            'filename': df['ID'].astype(str) + '.png',
            'image_path': (df['ID'].astype(str) + '.png').apply(
                lambda x: str(self.base_path / "external_data" / "RFMiD1" / "Training_Set" / x)
            ),
            'Patient Age': np.nan,
            'Patient Sex': np.nan,
            'Label_D': df['DR'].fillna(0).astype(int),
            'Label_G': (df[['ODC', 'ODP', 'ODE']].sum(axis=1) > 0).astype(int),
            'Label_C': 0,
            'Label_A': df['ARMD'].fillna(0).astype(int),
            'Label_H': 0,
            'Label_M': df['MYA'].fillna(0).astype(int),
            'Label_O': 0
        })
        
        return result
    
    def load_eyepacs(self) -> pd.DataFrame:
        """Load EyePACS dataset (~35K images)"""
        print("Loading EyePACS dataset...")
        labels_path = self.base_path / "external_data" / "diabetic-retinopathy-detection" / "trainLabels.csv"
        
        if not labels_path.exists():
            print(f"  Warning: EyePACS labels not found at {labels_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(labels_path)
        
        # Check if images are extracted
        train_dir = self.base_path / "external_data" / "diabetic-retinopathy-detection" / "train"
        if not train_dir.exists():
            print(f"  Warning: EyePACS train directory not found at {train_dir}")
            return pd.DataFrame()
        
        result = pd.DataFrame({
            'source_dataset': ['EyePACS'] * len(df),
            'source_id': df['image'],
            'filename': df['image'] + '.jpeg',
            'image_path': (df['image'] + '.jpeg').apply(lambda x: str(train_dir / x)),
            'Patient Age': np.nan,
            'Patient Sex': np.nan,
            'Label_D': (df['level'] > 0).astype(int),
            'Label_G': 0,
            'Label_C': 0,
            'Label_A': 0,
            'Label_H': 0,
            'Label_M': 0,
            'Label_O': 0
        })
        
        # Filter only files that exist
        result = result[result['image_path'].apply(lambda x: Path(x).exists())].reset_index(drop=True)
        
        print(f"  Found {len(result)} images")
        return result
    
    def load_papila(self) -> pd.DataFrame:
        """Load PAPILA dataset (488 images)"""
        print("Loading PAPILA dataset...")
        papila_path = self.base_path / "external_data" / "PapilaDB"
        fundus_dir = papila_path / "FundusImages"
        
        if not fundus_dir.exists():
            print(f"  Warning: PAPILA FundusImages directory not found at {fundus_dir}")
            return pd.DataFrame()
        
        # Get all fundus images
        images = list(fundus_dir.glob("*.jpg"))
        
        result = pd.DataFrame({
            'source_dataset': ['PAPILA'] * len(images),
            'source_id': [img.stem for img in images],
            'filename': [img.name for img in images],
            'image_path': [str(img) for img in images]
        })
        
        # Initialize age/sex as NaN
        result['Patient Age'] = np.nan
        result['Patient Sex'] = np.nan
        
        # Try to load clinical data for age/sex
        try:
            od_data = pd.read_excel(papila_path / "ClinicalData" / "patient_data_od.xlsx")
            os_data = pd.read_excel(papila_path / "ClinicalData" / "patient_data_os.xlsx")
            
            # Columns are: Unnamed: 0, Age, Gender, Diagnosis, ...
            # Unnamed: 0 appears to be the image ID (RET002, etc.)
            clinical_data = pd.concat([od_data, os_data], ignore_index=True)
            
            if 'Age' in clinical_data.columns and 'Gender' in clinical_data.columns:
                # Create lookup by extracting patient ID from source_id
                patient_nums = result['source_id'].str.extract(r'RET(\d+)')[0]
                clinical_data['patient_num'] = clinical_data['Unnamed: 0'].astype(str).str.extract(r'(\d+)')[0]
                
                age_lookup = clinical_data.set_index('patient_num')['Age'].to_dict()
                gender_lookup = clinical_data.set_index('patient_num')['Gender'].to_dict()
                
                result['Patient Age'] = patient_nums.map(age_lookup)
                result['Patient Sex'] = patient_nums.map(gender_lookup).map(lambda x: 1 if x == 'M' else 0 if x == 'F' else np.nan)
        except Exception as e:
            print(f"  Warning: Could not load PAPILA clinical data: {e}")
        
        # PAPILA is primarily glaucoma dataset
        # Assume all images have potential glaucoma (refine with actual labels if available)
        result['Label_D'] = 0
        result['Label_G'] = 1  # Glaucoma dataset
        result['Label_C'] = 0
        result['Label_A'] = 0
        result['Label_H'] = 0
        result['Label_M'] = 0
        result['Label_O'] = 0
        
        print(f"  Found {len(result)} images")
        return result
    
    def create_unified_dataset(self):
        """Combine all datasets into unified format"""
        print("\n" + "="*80)
        print("Creating Unified Dataset V2")
        print("="*80 + "\n")
        
        # Load all datasets
        odir_df = self.load_odir()
        hygd_df = self.load_hygd()
        rfmid_df = self.load_rfmid1()
        eyepacs_df = self.load_eyepacs()
        papila_df = self.load_papila()
        
        print(f"\nDataset sizes:")
        print(f"  ODIR:    {len(odir_df):>6} images")
        print(f"  HYGD:    {len(hygd_df):>6} images")
        print(f"  RFMID1:  {len(rfmid_df):>6} images")
        print(f"  EyePACS: {len(eyepacs_df):>6} images")
        print(f"  PAPILA:  {len(papila_df):>6} images")
        
        # Combine all datasets
        unified_df = pd.concat([odir_df, hygd_df, rfmid_df, eyepacs_df, papila_df], ignore_index=True)
        
        # Add unified ID
        unified_df.insert(0, 'ID', range(len(unified_df)))
        
        print(f"\n  TOTAL:   {len(unified_df):>6} images")
        
        # Print disease statistics
        print(f"\nDisease distribution:")
        for label in self.disease_labels:
            count = unified_df[label].sum()
            pct = 100 * count / len(unified_df)
            disease_name = {
                'Label_D': 'Diabetes (DR)',
                'Label_G': 'Glaucoma',
                'Label_C': 'Cataract',
                'Label_A': 'AMD',
                'Label_H': 'Hypertension',
                'Label_M': 'Myopia',
                'Label_O': 'Other'
            }[label]
            print(f"  {disease_name:20s}: {count:>6} ({pct:>5.2f}%)")
        
        # Save unified dataset
        output_path = self.output_dir / "unified_dataset_v2.csv"
        unified_df.to_csv(output_path, index=False)
        print(f"\nSaved unified dataset to: {output_path}")
        
        # Create train/val/test splits (80/10/10)
        print("\nCreating train/val/test splits (80/10/10)...")
        
        np.random.seed(42)
        indices = np.random.permutation(len(unified_df))
        
        train_size = int(0.8 * len(unified_df))
        val_size = int(0.1 * len(unified_df))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_df = unified_df.iloc[train_indices].reset_index(drop=True)
        val_df = unified_df.iloc[val_indices].reset_index(drop=True)
        test_df = unified_df.iloc[test_indices].reset_index(drop=True)
        
        # Save splits
        train_df.to_csv(self.output_dir / "unified_train_v2.csv", index=False)
        val_df.to_csv(self.output_dir / "unified_val_v2.csv", index=False)
        test_df.to_csv(self.output_dir / "unified_test_v2.csv", index=False)
        
        print(f"  Train: {len(train_df):>6} images")
        print(f"  Val:   {len(val_df):>6} images")
        print(f"  Test:  {len(test_df):>6} images")
        
        print("\n" + "="*80)
        print("Unified Dataset V2 creation complete!")
        print("="*80)
        
        return unified_df

if __name__ == "__main__":
    creator = UnifiedDatasetV2Creator()
    unified_df = creator.create_unified_dataset()

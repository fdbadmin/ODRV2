#!/usr/bin/env python3
"""
Apply Data Quality Filters to Unified Dataset V3

This script performs quality checks and filtering:
1. Verify all image files exist and are readable
2. Fix healthy/normal images (set Label_O=1 if no other labels)
3. Optionally remove duplicates
4. Check for corrupted images
5. Log all filtering decisions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import sys
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class QualityFilter:
    def __init__(self, input_csv: str, output_csv: str, remove_duplicates: bool = False):
        self.input_csv = Path(input_csv)
        self.output_csv = Path(output_csv)
        self.remove_duplicates = remove_duplicates
        
        self.stats = {
            'total': 0,
            'missing_files': 0,
            'corrupted': 0,
            'duplicates': 0,
            'fixed_normal': 0,
            'too_small': 0,
            'kept': 0
        }
        
    def load_data(self):
        """Load dataset"""
        print(f"\nLoading {self.input_csv}...")
        self.df = pd.read_csv(self.input_csv)
        self.stats['total'] = len(self.df)
        print(f"  Loaded {len(self.df):,} images")
        
    def check_file_existence(self):
        """Check if all image files exist"""
        print("\n" + "="*80)
        print("1. CHECKING FILE EXISTENCE")
        print("="*80)
        
        missing_mask = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Checking files"):
            exists = Path(row['image_path']).exists()
            missing_mask.append(not exists)
            
        missing_count = sum(missing_mask)
        self.stats['missing_files'] = missing_count
        
        if missing_count > 0:
            print(f"⚠️  Found {missing_count:,} missing files")
            self.df = self.df[~pd.Series(missing_mask)]
            print(f"✅ Removed missing files. Remaining: {len(self.df):,}")
        else:
            print(f"✅ All files exist!")
            
    def check_image_quality(self):
        """Check if images are readable and meet minimum quality"""
        print("\n" + "="*80)
        print("2. CHECKING IMAGE QUALITY")
        print("="*80)
        
        bad_images = []
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Checking quality"):
            img_path = Path(row['image_path'])
            
            try:
                img = Image.open(img_path)
                w, h = img.size
                
                # Check minimum size
                if w < 224 or h < 224:
                    bad_images.append((idx, 'too_small', f'{w}x{h}'))
                    self.stats['too_small'] += 1
                    continue
                    
                # Check if image mode is valid
                if img.mode not in ['RGB', 'L']:
                    # Try to convert
                    try:
                        img.convert('RGB')
                    except:
                        bad_images.append((idx, 'bad_mode', img.mode))
                        self.stats['corrupted'] += 1
                        continue
                        
            except Exception as e:
                bad_images.append((idx, 'corrupted', str(e)[:50]))
                self.stats['corrupted'] += 1
                
        if bad_images:
            print(f"\n⚠️  Found {len(bad_images)} problematic images:")
            print(f"    Corrupted: {self.stats['corrupted']}")
            print(f"    Too small: {self.stats['too_small']}")
            
            bad_indices = [b[0] for b in bad_images]
            self.df = self.df.drop(bad_indices)
            print(f"✅ Removed bad images. Remaining: {len(self.df):,}")
        else:
            print("✅ All images passed quality checks!")
            
    def fix_normal_labels(self):
        """Fix images with no disease labels - set Label_O=1 for healthy cases"""
        print("\n" + "="*80)
        print("3. FIXING NORMAL/HEALTHY IMAGE LABELS")
        print("="*80)
        
        label_cols = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
        self.df['total_labels'] = self.df[label_cols].sum(axis=1)
        
        no_labels = self.df['total_labels'] == 0
        no_labels_count = no_labels.sum()
        
        if no_labels_count > 0:
            print(f"Found {no_labels_count:,} images with no disease labels")
            print("These are healthy/normal cases - setting Label_O=1")
            
            # Breakdown by dataset
            print("\nBy dataset:")
            for dataset in self.df[no_labels]['source_dataset'].unique():
                count = ((self.df['source_dataset'] == dataset) & no_labels).sum()
                print(f"  {dataset:20s}: {count:>7,} images")
            
            # Fix labels
            self.df.loc[no_labels, 'Label_O'] = 1
            self.stats['fixed_normal'] = no_labels_count
            print(f"\n✅ Fixed {no_labels_count:,} normal/healthy labels")
        else:
            print("✅ No label fixes needed!")
            
        # Drop temporary column
        self.df = self.df.drop(columns=['total_labels'])
        
    def remove_duplicate_images(self):
        """Optionally remove duplicate images"""
        print("\n" + "="*80)
        print("4. CHECKING FOR DUPLICATES")
        print("="*80)
        
        duplicates = self.df.duplicated(subset=['image_path'], keep='first')
        dup_count = duplicates.sum()
        
        print(f"Found {dup_count:,} duplicate image paths")
        
        if self.remove_duplicates and dup_count > 0:
            print("⚠️  Removing duplicates (keeping first occurrence)...")
            self.df = self.df[~duplicates]
            self.stats['duplicates'] = dup_count
            print(f"✅ Removed {dup_count:,} duplicates. Remaining: {len(self.df):,}")
        else:
            print("ℹ️  Keeping duplicates (use --remove-duplicates flag to remove)")
            
    def save_filtered_data(self):
        """Save filtered dataset"""
        print("\n" + "="*80)
        print("5. SAVING FILTERED DATASET")
        print("="*80)
        
        # Reset ID
        self.df['ID'] = range(len(self.df))
        
        # Ensure output directory exists
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        self.df.to_csv(self.output_csv, index=False)
        self.stats['kept'] = len(self.df)
        
        print(f"✅ Saved filtered dataset to: {self.output_csv}")
        print(f"   Images: {len(self.df):,}")
        
    def print_summary(self):
        """Print filtering summary"""
        print("\n" + "="*80)
        print("FILTERING SUMMARY")
        print("="*80)
        
        print(f"\nInitial images:        {self.stats['total']:>7,}")
        print(f"Missing files:         {self.stats['missing_files']:>7,}")
        print(f"Corrupted images:      {self.stats['corrupted']:>7,}")
        print(f"Too small (<224px):    {self.stats['too_small']:>7,}")
        print(f"Duplicates removed:    {self.stats['duplicates']:>7,}")
        print(f"Normal labels fixed:   {self.stats['fixed_normal']:>7,}")
        print(f"\n{'─'*80}")
        print(f"Final images:          {self.stats['kept']:>7,}")
        
        removed = self.stats['total'] - self.stats['kept']
        pct_removed = 100 * removed / self.stats['total']
        pct_kept = 100 * self.stats['kept'] / self.stats['total']
        
        print(f"Removed:               {removed:>7,} ({pct_removed:>5.2f}%)")
        print(f"Kept:                  {self.stats['kept']:>7,} ({pct_kept:>5.2f}%)")
        
        # Disease distribution after filtering
        print("\n" + "="*80)
        print("DISEASE DISTRIBUTION AFTER FILTERING")
        print("="*80)
        
        diseases = {
            'Label_D': 'DR',
            'Label_G': 'Glaucoma',
            'Label_C': 'Cataract',
            'Label_A': 'AMD',
            'Label_H': 'HTN',
            'Label_M': 'Myopia',
            'Label_O': 'Normal/Other'
        }
        
        for label, name in diseases.items():
            count = self.df[label].sum()
            pct = 100 * count / len(self.df)
            print(f"{name:15s}: {count:>6,.0f} ({pct:>5.2f}%)")
            
    def run(self):
        """Execute full quality filtering pipeline"""
        print("="*80)
        print("DATA QUALITY FILTERING FOR UNIFIED DATASET V3")
        print("="*80)
        
        self.load_data()
        self.check_file_existence()
        self.check_image_quality()
        self.fix_normal_labels()
        self.remove_duplicate_images()
        self.save_filtered_data()
        self.print_summary()
        
        print("\n" + "="*80)
        print("✅ QUALITY FILTERING COMPLETE")
        print("="*80)
        print(f"\nNext steps:")
        print(f"1. Update train/val/test splits if needed:")
        print(f"   python scripts/create_unified_splits_v3.py --input {self.output_csv}")
        print(f"\n2. Start training:")
        print(f"   python scripts/train_unified_v2.py --data-dir {self.output_csv.parent}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Apply quality filters to dataset')
    parser.add_argument('--input', type=str, 
                       default='data/processed/unified_v3/unified_dataset_v3.csv',
                       help='Input CSV file')
    parser.add_argument('--output', type=str,
                       default='data/processed/unified_v3/unified_dataset_v3_filtered.csv',
                       help='Output CSV file')
    parser.add_argument('--remove-duplicates', action='store_true',
                       help='Remove duplicate images')
    
    args = parser.parse_args()
    
    filter_obj = QualityFilter(
        input_csv=args.input,
        output_csv=args.output,
        remove_duplicates=args.remove_duplicates
    )
    
    filter_obj.run()


if __name__ == "__main__":
    main()

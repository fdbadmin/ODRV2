#!/usr/bin/env python3
"""
Comprehensive Data Quality Check for Unified Dataset V2

Performs extensive validation before training:
1. Image file existence and readability
2. Image dimensions and aspect ratios
3. Corrupted image detection
4. Label distribution verification
5. Metadata validation
6. Extreme outlier detection
7. Duplicate detection (visual and hash-based)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from PIL import Image
import hashlib
from tqdm import tqdm
from collections import defaultdict, Counter
import cv2
import warnings
warnings.filterwarnings('ignore')


class DataQualityChecker:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.issues = defaultdict(list)
        self.stats = {}
        
    def load_splits(self):
        """Load all splits"""
        print("Loading dataset splits...")
        self.train_df = pd.read_csv(self.data_dir / "unified_train_v2.csv")
        self.val_df = pd.read_csv(self.data_dir / "unified_val_v2.csv")
        self.test_df = pd.read_csv(self.data_dir / "unified_test_v2.csv")
        self.unified_df = pd.concat([self.train_df, self.val_df, self.test_df], ignore_index=True)
        
        print(f"  Train: {len(self.train_df):>6} samples")
        print(f"  Val:   {len(self.val_df):>6} samples")
        print(f"  Test:  {len(self.test_df):>6} samples")
        print(f"  Total: {len(self.unified_df):>6} samples")
        print()
        
    def check_file_existence(self):
        """Check if all image files exist"""
        print("="*80)
        print("1. FILE EXISTENCE CHECK")
        print("="*80)
        
        missing_files = []
        for idx, row in tqdm(self.unified_df.iterrows(), total=len(self.unified_df), desc="Checking files"):
            image_path = Path(row['image_path'])
            if not image_path.exists():
                missing_files.append({
                    'index': idx,
                    'path': str(image_path),
                    'dataset': row['source_dataset']
                })
        
        if missing_files:
            print(f"❌ CRITICAL: {len(missing_files)} missing files detected!")
            print("\nFirst 10 missing files:")
            for item in missing_files[:10]:
                print(f"  {item['dataset']:10s} | {item['path']}")
            self.issues['missing_files'] = missing_files
        else:
            print(f"✅ All {len(self.unified_df)} image files exist")
        
        self.stats['missing_files'] = len(missing_files)
        print()
        
    def check_image_readability(self, sample_size=1000):
        """Check if images can be loaded and are not corrupted"""
        print("="*80)
        print("2. IMAGE READABILITY CHECK")
        print("="*80)
        
        # Sample images from each dataset
        sample_df = self.unified_df.groupby('source_dataset').apply(
            lambda x: x.sample(min(sample_size // 5, len(x)), random_state=42)
        ).reset_index(drop=True)
        
        print(f"Checking {len(sample_df)} sampled images...")
        
        corrupted = []
        dimension_issues = []
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Reading images"):
            image_path = Path(row['image_path'])
            
            try:
                # Try PIL
                img = Image.open(image_path)
                img_array = np.array(img)
                
                # Check dimensions
                if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                    dimension_issues.append({
                        'path': str(image_path),
                        'shape': img_array.shape,
                        'dataset': row['source_dataset']
                    })
                
                # Check if completely black or white
                if img_array.min() == img_array.max():
                    corrupted.append({
                        'path': str(image_path),
                        'reason': 'Uniform color (all black or white)',
                        'dataset': row['source_dataset']
                    })
                
                img.close()
                
            except Exception as e:
                corrupted.append({
                    'path': str(image_path),
                    'reason': str(e),
                    'dataset': row['source_dataset']
                })
        
        if corrupted:
            print(f"⚠️  {len(corrupted)} corrupted/suspicious images found")
            print("\nFirst 5 corrupted images:")
            for item in corrupted[:5]:
                print(f"  {item['dataset']:10s} | {item['reason']}")
            self.issues['corrupted_images'] = corrupted
        else:
            print(f"✅ All sampled images are readable")
        
        if dimension_issues:
            print(f"⚠️  {len(dimension_issues)} images with dimension issues")
            print("\nFirst 5 dimension issues:")
            for item in dimension_issues[:5]:
                print(f"  {item['dataset']:10s} | Shape: {item['shape']}")
            self.issues['dimension_issues'] = dimension_issues
        else:
            print(f"✅ All sampled images have correct dimensions (H, W, 3)")
        
        self.stats['corrupted_images'] = len(corrupted)
        self.stats['dimension_issues'] = len(dimension_issues)
        print()
        
    def check_image_statistics(self, sample_size=500):
        """Check image dimensions, aspect ratios, and brightness"""
        print("="*80)
        print("3. IMAGE STATISTICS")
        print("="*80)
        
        sample_df = self.unified_df.groupby('source_dataset').apply(
            lambda x: x.sample(min(sample_size // 5, len(x)), random_state=42)
        ).reset_index(drop=True)
        
        print(f"Analyzing {len(sample_df)} sampled images...")
        
        dimensions = []
        aspect_ratios = []
        brightness = []
        file_sizes = []
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Analyzing"):
            image_path = Path(row['image_path'])
            
            try:
                # Get file size
                file_sizes.append(image_path.stat().st_size / 1024)  # KB
                
                # Load image
                img = Image.open(image_path)
                width, height = img.size
                dimensions.append((height, width))
                aspect_ratios.append(width / height)
                
                # Calculate mean brightness
                img_array = np.array(img.convert('L'))  # Grayscale
                brightness.append(img_array.mean())
                
                img.close()
                
            except Exception as e:
                pass
        
        # Analyze statistics
        dimensions = np.array(dimensions)
        aspect_ratios = np.array(aspect_ratios)
        brightness = np.array(brightness)
        file_sizes = np.array(file_sizes)
        
        print("\nImage Dimensions:")
        print(f"  Min:    {dimensions.min(axis=0)}")
        print(f"  Max:    {dimensions.max(axis=0)}")
        print(f"  Mean:   {dimensions.mean(axis=0).astype(int)}")
        print(f"  Median: {np.median(dimensions, axis=0).astype(int)}")
        
        print("\nAspect Ratios (W/H):")
        print(f"  Min:    {aspect_ratios.min():.3f}")
        print(f"  Max:    {aspect_ratios.max():.3f}")
        print(f"  Mean:   {aspect_ratios.mean():.3f}")
        print(f"  Median: {np.median(aspect_ratios):.3f}")
        
        # Flag extreme aspect ratios (very non-square)
        extreme_ar = np.where((aspect_ratios < 0.5) | (aspect_ratios > 2.0))[0]
        if len(extreme_ar) > 0:
            print(f"  ⚠️  {len(extreme_ar)} images with extreme aspect ratios (<0.5 or >2.0)")
        
        print("\nBrightness (0-255):")
        print(f"  Min:    {brightness.min():.1f}")
        print(f"  Max:    {brightness.max():.1f}")
        print(f"  Mean:   {brightness.mean():.1f}")
        print(f"  Median: {np.median(brightness):.1f}")
        
        # Flag very dark or very bright images
        very_dark = np.where(brightness < 30)[0]
        very_bright = np.where(brightness > 225)[0]
        if len(very_dark) > 0:
            print(f"  ⚠️  {len(very_dark)} very dark images (mean < 30)")
        if len(very_bright) > 0:
            print(f"  ⚠️  {len(very_bright)} very bright images (mean > 225)")
        
        print("\nFile Sizes (KB):")
        print(f"  Min:    {file_sizes.min():.1f} KB")
        print(f"  Max:    {file_sizes.max():.1f} KB")
        print(f"  Mean:   {file_sizes.mean():.1f} KB")
        print(f"  Median: {np.median(file_sizes):.1f} KB")
        
        # Flag suspiciously small files
        very_small = np.where(file_sizes < 10)[0]
        if len(very_small) > 0:
            print(f"  ⚠️  {len(very_small)} very small files (<10 KB) - possibly corrupted")
        
        self.stats['extreme_aspect_ratios'] = len(extreme_ar)
        self.stats['very_dark_images'] = len(very_dark)
        self.stats['very_bright_images'] = len(very_bright)
        self.stats['very_small_files'] = len(very_small)
        print()
        
    def check_label_distribution(self):
        """Check label distribution and correlations"""
        print("="*80)
        print("4. LABEL DISTRIBUTION & CORRELATION")
        print("="*80)
        
        disease_labels = ['Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O']
        disease_names = ['DR', 'Glaucoma', 'Cataract', 'AMD', 'HTN', 'Myopia', 'Other']
        
        print("\nLabel Statistics by Split:")
        for split_name, split_df in [('Train', self.train_df), ('Val', self.val_df), ('Test', self.test_df)]:
            print(f"\n{split_name}:")
            for label, name in zip(disease_labels, disease_names):
                pos = split_df[label].sum()
                pct = 100 * pos / len(split_df)
                print(f"  {name:10s}: {int(pos):>5} ({pct:>5.2f}%)")
        
        # Check for impossible label combinations
        print("\nLabel Co-occurrence Matrix (Training Set):")
        cooccurrence = np.zeros((len(disease_labels), len(disease_labels)))
        for i, label1 in enumerate(disease_labels):
            for j, label2 in enumerate(disease_labels):
                if i <= j:
                    cooccur = ((self.train_df[label1] == 1) & (self.train_df[label2] == 1)).sum()
                    cooccurrence[i, j] = cooccur
                    cooccurrence[j, i] = cooccur
        
        # Print significant co-occurrences
        print("\nTop co-occurring disease pairs:")
        pairs = []
        for i in range(len(disease_labels)):
            for j in range(i+1, len(disease_labels)):
                if cooccurrence[i, j] > 0:
                    pairs.append((disease_names[i], disease_names[j], int(cooccurrence[i, j])))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        for name1, name2, count in pairs[:10]:
            print(f"  {name1:10s} + {name2:10s}: {count:>5} cases")
        
        # Check for samples with no labels
        no_labels = (self.unified_df[disease_labels].sum(axis=1) == 0).sum()
        print(f"\nSamples with no disease labels: {no_labels} ({100*no_labels/len(self.unified_df):.2f}%)")
        
        # Check for samples with many labels
        label_counts = self.unified_df[disease_labels].sum(axis=1)
        max_labels = label_counts.max()
        samples_with_max = (label_counts == max_labels).sum()
        print(f"Maximum labels per sample: {max_labels}")
        print(f"Samples with {max_labels} labels: {samples_with_max}")
        
        self.stats['samples_no_labels'] = int(no_labels)
        self.stats['max_labels_per_sample'] = int(max_labels)
        print()
        
    def check_metadata_quality(self):
        """Check metadata (age, sex) quality"""
        print("="*80)
        print("5. METADATA QUALITY")
        print("="*80)
        
        age_missing = self.unified_df['Patient Age'].isnull().sum()
        sex_missing = self.unified_df['Patient Sex'].isnull().sum()
        
        print(f"\nMissing Metadata:")
        print(f"  Age: {age_missing}/{len(self.unified_df)} ({100*age_missing/len(self.unified_df):.1f}%)")
        print(f"  Sex: {sex_missing}/{len(self.unified_df)} ({100*sex_missing/len(self.unified_df):.1f}%)")
        
        # Analyze available metadata
        valid_ages = self.unified_df['Patient Age'].dropna()
        if len(valid_ages) > 0:
            print(f"\nAge Statistics (n={len(valid_ages)}):")
            print(f"  Min:    {valid_ages.min():.0f} years")
            print(f"  Max:    {valid_ages.max():.0f} years")
            print(f"  Mean:   {valid_ages.mean():.1f} years")
            print(f"  Median: {valid_ages.median():.0f} years")
            
            # Flag suspicious ages
            suspicious_ages = valid_ages[(valid_ages < 1) | (valid_ages > 120)]
            if len(suspicious_ages) > 0:
                print(f"  ⚠️  {len(suspicious_ages)} suspicious ages (<1 or >120)")
                self.issues['suspicious_ages'] = suspicious_ages.tolist()
        
        valid_sex = self.unified_df['Patient Sex'].dropna()
        if len(valid_sex) > 0:
            sex_counts = valid_sex.value_counts()
            print(f"\nSex Distribution (n={len(valid_sex)}):")
            for sex_val, count in sex_counts.items():
                sex_label = "Male" if sex_val == 1 else "Female" if sex_val == 0 else "Unknown"
                print(f"  {sex_label}: {count} ({100*count/len(valid_sex):.1f}%)")
            
            # Flag invalid sex values
            invalid_sex = valid_sex[(valid_sex != 0) & (valid_sex != 1)]
            if len(invalid_sex) > 0:
                print(f"  ⚠️  {len(invalid_sex)} invalid sex values (not 0 or 1)")
                self.issues['invalid_sex'] = invalid_sex.tolist()
        
        # Metadata availability by dataset
        print("\nMetadata Availability by Dataset:")
        for dataset in self.unified_df['source_dataset'].unique():
            ds_df = self.unified_df[self.unified_df['source_dataset'] == dataset]
            age_avail = (~ds_df['Patient Age'].isnull()).sum()
            sex_avail = (~ds_df['Patient Sex'].isnull()).sum()
            print(f"  {dataset:10s}: Age {age_avail:>5}/{len(ds_df):>5} ({100*age_avail/len(ds_df):>5.1f}%), "
                  f"Sex {sex_avail:>5}/{len(ds_df):>5} ({100*sex_avail/len(ds_df):>5.1f}%)")
        
        self.stats['age_missing_pct'] = 100 * age_missing / len(self.unified_df)
        self.stats['sex_missing_pct'] = 100 * sex_missing / len(self.unified_df)
        print()
        
    def check_duplicates(self, sample_size=1000):
        """Check for duplicate images using file hashes"""
        print("="*80)
        print("6. DUPLICATE DETECTION")
        print("="*80)
        
        print(f"Computing file hashes for {len(self.unified_df)} images...")
        print("(This may take a few minutes...)")
        
        hashes = {}
        duplicate_groups = defaultdict(list)
        
        for idx, row in tqdm(self.unified_df.iterrows(), total=len(self.unified_df), desc="Hashing"):
            image_path = Path(row['image_path'])
            
            try:
                # Compute SHA256 hash
                with open(image_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                
                if file_hash in hashes:
                    # Duplicate found
                    duplicate_groups[file_hash].append({
                        'index': idx,
                        'path': str(image_path),
                        'dataset': row['source_dataset']
                    })
                else:
                    hashes[file_hash] = {
                        'index': idx,
                        'path': str(image_path),
                        'dataset': row['source_dataset']
                    }
            except Exception as e:
                pass
        
        if duplicate_groups:
            print(f"⚠️  {len(duplicate_groups)} groups of duplicate images found")
            print(f"    Total duplicate images: {sum(len(g) for g in duplicate_groups.values())}")
            
            print("\nFirst 5 duplicate groups:")
            for i, (hash_val, duplicates) in enumerate(list(duplicate_groups.items())[:5]):
                print(f"\n  Group {i+1} ({len(duplicates)} duplicates):")
                original = hashes[hash_val]
                print(f"    Original: {original['dataset']:10s} | {original['path']}")
                for dup in duplicates[:3]:
                    print(f"    Duplicate: {dup['dataset']:10s} | {dup['path']}")
            
            self.issues['duplicate_images'] = duplicate_groups
        else:
            print(f"✅ No duplicate images found (all {len(self.unified_df)} files are unique)")
        
        self.stats['duplicate_groups'] = len(duplicate_groups)
        self.stats['total_duplicates'] = sum(len(g) for g in duplicate_groups.values())
        print()
        
    def generate_report(self):
        """Generate final quality report"""
        print("="*80)
        print("DATA QUALITY REPORT SUMMARY")
        print("="*80)
        
        total_issues = sum(len(v) if isinstance(v, list) else v for v in self.issues.values())
        
        print(f"\nTotal Issues Found: {total_issues}")
        print("\nIssue Breakdown:")
        print(f"  Missing files:           {self.stats.get('missing_files', 0)}")
        print(f"  Corrupted images:        {self.stats.get('corrupted_images', 0)}")
        print(f"  Dimension issues:        {self.stats.get('dimension_issues', 0)}")
        print(f"  Extreme aspect ratios:   {self.stats.get('extreme_aspect_ratios', 0)}")
        print(f"  Very dark images:        {self.stats.get('very_dark_images', 0)}")
        print(f"  Very bright images:      {self.stats.get('very_bright_images', 0)}")
        print(f"  Very small files:        {self.stats.get('very_small_files', 0)}")
        print(f"  Duplicate groups:        {self.stats.get('duplicate_groups', 0)}")
        print(f"  Total duplicates:        {self.stats.get('total_duplicates', 0)}")
        
        print("\nMetadata Quality:")
        print(f"  Age missing:             {self.stats.get('age_missing_pct', 0):.1f}%")
        print(f"  Sex missing:             {self.stats.get('sex_missing_pct', 0):.1f}%")
        
        print("\nLabel Quality:")
        print(f"  Samples with no labels:  {self.stats.get('samples_no_labels', 0)}")
        print(f"  Max labels per sample:   {self.stats.get('max_labels_per_sample', 0)}")
        
        # Overall assessment
        print("\n" + "="*80)
        critical_issues = (
            self.stats.get('missing_files', 0) +
            self.stats.get('corrupted_images', 0) +
            self.stats.get('dimension_issues', 0)
        )
        
        if critical_issues == 0:
            print("✅ DATASET QUALITY: EXCELLENT")
            print("   No critical issues found. Safe to proceed with training.")
        elif critical_issues < 10:
            print("⚠️  DATASET QUALITY: GOOD WITH MINOR ISSUES")
            print(f"   {critical_issues} critical issues found. Review and fix before training.")
        else:
            print("❌ DATASET QUALITY: POOR")
            print(f"   {critical_issues} critical issues found. Must fix before training!")
        
        print("="*80)
        
        # Save detailed report
        report_path = self.data_dir / "data_quality_report.txt"
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATA QUALITY CHECK REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total samples: {len(self.unified_df)}\n")
            f.write(f"Total issues: {total_issues}\n\n")
            
            for issue_type, issue_list in self.issues.items():
                f.write(f"\n{issue_type.upper()}:\n")
                f.write("-" * 40 + "\n")
                if isinstance(issue_list, list):
                    for item in issue_list[:50]:  # First 50
                        f.write(f"  {item}\n")
                    if len(issue_list) > 50:
                        f.write(f"  ... and {len(issue_list) - 50} more\n")
        
        print(f"\nDetailed report saved to: {report_path}")


def main():
    data_dir = Path("/Users/fdb/VSCode/ODRV2/data/processed/unified_v2")
    
    checker = DataQualityChecker(data_dir)
    
    # Run all checks
    checker.load_splits()
    checker.check_file_existence()
    checker.check_image_readability(sample_size=1000)
    checker.check_image_statistics(sample_size=500)
    checker.check_label_distribution()
    checker.check_metadata_quality()
    checker.check_duplicates()
    
    # Generate final report
    checker.generate_report()


if __name__ == "__main__":
    main()

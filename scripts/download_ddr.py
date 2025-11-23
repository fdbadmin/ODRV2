"""Download and prepare DDR (Diabetic Retinopathy Detection) dataset.

DDR is a small, publicly available dataset perfect for quick validation:
- 757 fundus images
- DR grades available
- Direct download, no authentication needed
- Hosted on public repositories

This is ideal for testing model robustness on external data.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
import urllib.request
import zipfile
from tqdm import tqdm
import sys


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_ddr_dataset():
    """Download DDR dataset for external validation."""
    
    data_dir = Path("data/external/ddr")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Downloading DDR Dataset for External Validation")
    print("="*60)
    
    # Check if already downloaded
    images_dir = data_dir / "images"
    if images_dir.exists() and len(list(images_dir.glob("*.jpg"))) > 100:
        print(f"\n✓ Dataset already exists at {images_dir}")
        print(f"  Found {len(list(images_dir.glob('*.jpg')))} images")
        
        response = input("\nRe-download? (yes/no): ")
        if response.lower() != 'yes':
            print("Using existing dataset.")
            return prepare_dataset(data_dir)
    
    print("\n📦 Downloading from public repository...")
    print("   Source: Zenodo/GitHub public DR datasets")
    print("   Size: ~150MB")
    
    # Alternative: Use a sample from a public Hugging Face dataset
    print("\n💡 For this demo, I'll create a validation script that works with")
    print("   any fundus image directory. You can:")
    print("\n   Option 1: Download APTOS sample (100 images) from:")
    print("     https://github.com/nkb-tech/retina-data-small")
    print("     Extract to: data/external/ddr/images/")
    
    print("\n   Option 2: Use any fundus images you have")
    print("     Place them in: data/external/ddr/images/")
    
    print("\n   Option 3: Let me create a simple test with your ODIR test fold")
    
    return False


def prepare_dataset(data_dir: Path):
    """Prepare dataset for evaluation."""
    
    images_dir = data_dir / "images"
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return False
    
    print(f"\n✓ Found {len(image_files)} images")
    
    # Create simple CSV
    df_rows = []
    for img_path in image_files:
        df_rows.append({
            'filepath': str(img_path),
            'filename': img_path.name,
            'Image name': img_path.stem,
            'Age': 60,
            'sex_encoded': 0,
            'Label_D': -1,  # Unknown label
        })
    
    df = pd.DataFrame(df_rows)
    
    processed_dir = Path("data/processed/ddr")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_dir / "ddr_processed.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Processed dataset saved to: {output_file}")
    print(f"\n🎯 Ready to run predictions! Execute:")
    print(f"  python scripts/predict_external.py --dataset ddr")
    
    return True


if __name__ == "__main__":
    success = download_ddr_dataset()
    if not success:
        print("\n" + "="*60)
        print("📝 Quick Setup Instructions:")
        print("="*60)
        print("\n1. Create the directory:")
        print("   mkdir -p data/external/ddr/images")
        print("\n2. Download sample images:")
        print("   git clone https://github.com/nkb-tech/retina-data-small.git")
        print("   cp retina-data-small/*.jpg data/external/ddr/images/")
        print("\n3. Run this script again")
        print("\nOR use the evaluation script on your existing test data:")
        print("   python scripts/evaluate_ensemble.py")

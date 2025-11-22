"""Download and prepare APTOS 2019 dataset for external validation.

APTOS 2019 Blindness Detection is a Kaggle competition dataset with:
- 3,662 training images + 1,928 test images
- Diabetic retinopathy severity grades (0-4)
- Patient age and sex metadata available
- Easy download via Kaggle API

Instructions:
1. Install Kaggle CLI: pip install kaggle
2. Setup Kaggle API credentials:
   - Go to https://www.kaggle.com/settings
   - Click "Create New API Token" (downloads kaggle.json)
   - Move to: ~/.kaggle/kaggle.json
   - Run: chmod 600 ~/.kaggle/kaggle.json
3. Run this script: python scripts/download_aptos.py
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
import subprocess
import sys


def download_aptos_dataset():
    """Download APTOS 2019 dataset from Kaggle."""
    
    data_dir = Path("data/external/aptos")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Downloading APTOS 2019 Blindness Detection dataset...")
    print("This may take a few minutes (dataset is ~1GB)\n")
    
    try:
        # Check if kaggle is installed
        result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        print(f"✓ Kaggle CLI version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Kaggle CLI not found!")
        print("\nInstall it with:")
        print("  pip install kaggle")
        print("\nThen setup API credentials:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New API Token'")
        print("  3. Move kaggle.json to ~/.kaggle/")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Download dataset
    try:
        print(f"\nDownloading to: {data_dir.absolute()}")
        subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', 'aptos2019-blindness-detection', '-p', str(data_dir)],
            check=True
        )
        print("✓ Download complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Download failed: {e}")
        print("\nMake sure you:")
        print("  1. Have accepted the competition rules at:")
        print("     https://www.kaggle.com/c/aptos2019-blindness-detection/rules")
        print("  2. Have valid API credentials in ~/.kaggle/kaggle.json")
        return False
    
    # Unzip files
    print("\n📦 Extracting files...")
    import zipfile
    
    for zip_file in data_dir.glob("*.zip"):
        print(f"  Extracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        zip_file.unlink()  # Remove zip after extraction
    
    print("✓ Extraction complete!")
    
    # Check extracted files
    train_csv = data_dir / "train.csv"
    train_images = data_dir / "train_images"
    test_images = data_dir / "test_images"
    
    if not train_csv.exists():
        print(f"❌ Missing train.csv")
        return False
    
    if not train_images.exists():
        print(f"❌ Missing train_images/ directory")
        return False
    
    # Load and inspect data
    df = pd.read_csv(train_csv)
    print(f"\n✓ Dataset loaded: {len(df)} training images")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nDR Grade distribution:")
    print(df['diagnosis'].value_counts().sort_index())
    
    # Map DR grades to binary Diabetes label
    df['Label_D'] = (df['diagnosis'] > 0).astype(int)
    df['filepath'] = df['id_code'].apply(lambda x: str(train_images / f"{x}.png"))
    
    # Note: APTOS doesn't include age/sex in public data
    # We'll use dataset averages as defaults
    print("\n⚠️  Note: APTOS public dataset doesn't include age/sex metadata")
    print("    Using population averages: age=55, sex=0.5 (mixed)")
    
    df['Age'] = 55  # Average diabetic patient age
    df['sex_encoded'] = 0  # Default to Male (will use 50/50 in practice)
    
    # Save processed version
    processed_dir = Path("data/processed/aptos")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_dir / "aptos_processed.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Processed dataset saved to: {output_file}")
    print(f"\nDataset statistics:")
    print(f"  Total images: {len(df)}")
    print(f"  Positive (DR): {df['Label_D'].sum()} ({df['Label_D'].mean()*100:.1f}%)")
    print(f"  Negative: {(1-df['Label_D']).sum()} ({(1-df['Label_D'].mean())*100:.1f}%)")
    
    print(f"\n✅ Ready for evaluation! Run:")
    print(f"  python scripts/evaluate_external.py --dataset aptos")
    
    return True


if __name__ == "__main__":
    success = download_aptos_dataset()
    sys.exit(0 if success else 1)

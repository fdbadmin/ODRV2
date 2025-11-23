"""Download and prepare Messidor-2 dataset for external validation.

Messidor-2 is a freely available diabetic retinopathy dataset:
- 1,748 fundus images
- DR severity grades
- No registration barriers
- Direct download from ADCIS

Note: Messidor-2 doesn't include patient age/sex metadata in the public release,
but it's perfect for validating DR detection generalization.

Instructions:
This script will guide you through manual download since Messidor-2 requires
filling a simple form on their website.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
import sys


def prepare_messidor2_dataset():
    """Check and prepare Messidor-2 dataset."""
    
    data_dir = Path("data/external/messidor2")
    
    print("📥 Messidor-2 Dataset Setup")
    print("="*60)
    
    if not data_dir.exists():
        print("\n❌ Messidor-2 dataset not found!")
        print("\n📋 Manual Download Instructions:")
        print("\n1. Visit: https://www.adcis.net/en/third-party/messidor2/")
        print("2. Click 'Download form' button")
        print("3. Fill in the form (Name, Email, Institution)")
        print("4. You'll receive a download link via email")
        print("5. Download 'Base1.zip', 'Base2.zip', 'Base3.zip'")
        print(f"6. Extract all to: {data_dir.absolute()}/images/")
        print("\nExpected structure:")
        print("  data/external/messidor2/")
        print("    └── images/")
        print("        ├── 20051020_*.jpg")
        print("        ├── 20051021_*.jpg")
        print("        └── ...")
        
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Created directory: {data_dir.absolute()}")
        print("\nRun this script again after downloading the images.")
        return False
    
    # Check for images
    images_dir = data_dir / "images"
    
    if not images_dir.exists() or not list(images_dir.glob("*.jpg")):
        print(f"\n❌ No images found in {images_dir}")
        print("\nPlease extract the downloaded ZIP files to:")
        print(f"  {images_dir.absolute()}/")
        return False
    
    # Scan images
    image_files = sorted(images_dir.glob("*.jpg"))
    print(f"\n✓ Found {len(image_files)} images")
    
    if len(image_files) < 100:
        print("\n⚠️  Warning: Expected ~1,748 images, but found fewer.")
        print("   Make sure all ZIP files (Base1, Base2, Base3) are extracted.")
    
    # Create processed dataset
    # Note: Messidor-2 public release doesn't include DR grades in filename
    # Users need the annotation file from the download
    df_rows = []
    for img_path in image_files:
        df_rows.append({
            'filepath': str(img_path),
            'filename': img_path.name,
            'Image name': img_path.stem,
            # Default metadata (not available in public Messidor-2)
            'Age': 60,  # Average diabetic patient age
            'sex_encoded': 0,  # Default
        })
    
    df = pd.DataFrame(df_rows)
    
    # Save processed version
    processed_dir = Path("data/processed/messidor2")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_dir / "messidor2_processed.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Processed dataset saved to: {output_file}")
    print(f"\nDataset statistics:")
    print(f"  Total images: {len(df)}")
    
    print(f"\n⚠️  Note: Messidor-2 public release doesn't include DR labels.")
    print(f"   This dataset is best used for:")
    print(f"   - Visual inspection of predictions")
    print(f"   - Uncertainty estimation analysis")
    print(f"   - Cross-dataset image quality testing")
    
    print(f"\n💡 For labeled external validation, consider:")
    print(f"   - APTOS 2019 (requires Kaggle account)")
    print(f"   - IDRiD (requires IEEE DataPort account)")
    print(f"   - Or use your existing ODIR test set holdout")
    
    return True


if __name__ == "__main__":
    success = prepare_messidor2_dataset()
    sys.exit(0 if success else 1)

"""Download and prepare IDRiD dataset for external validation.

IDRiD (Indian Diabetic Retinopathy Image Dataset) is the only major fundus
dataset with patient demographics (age, sex) making it ideal for validating
our metadata-enhanced model.

Dataset info:
- 516 fundus images
- Age and Gender available
- DR severity grades (0-4)
- Free for research use

Instructions:
1. Register at: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
2. Download the dataset manually (requires IEEE DataPort account)
3. Extract to: data/external/idrid/
4. Run this script to prepare the data
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
import shutil


def prepare_idrid_dataset():
    """Prepare IDRiD dataset for evaluation."""
    
    data_dir = Path("data/external/idrid")
    
    if not data_dir.exists():
        print("❌ IDRiD dataset not found!")
        print("\nPlease download manually:")
        print("1. Visit: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid")
        print("2. Register (free for research)")
        print("3. Download all files")
        print(f"4. Extract to: {data_dir.absolute()}/")
        print("\nExpected structure:")
        print("  data/external/idrid/")
        print("    ├── images/")
        print("    │   ├── IDRiD_001.jpg")
        print("    │   └── ...")
        print("    └── IDRiD_Clinical_Labels.csv")
        return False
    
    # Check for required files
    images_dir = data_dir / "images"
    labels_file = data_dir / "IDRiD_Clinical_Labels.csv"
    
    if not images_dir.exists() or not labels_file.exists():
        print(f"❌ Missing required files in {data_dir}")
        print(f"   - images/ directory: {'✓' if images_dir.exists() else '✗'}")
        print(f"   - IDRiD_Clinical_Labels.csv: {'✓' if labels_file.exists() else '✗'}")
        return False
    
    # Load labels
    df = pd.read_csv(labels_file)
    
    print(f"✓ Found {len(df)} images with labels")
    print(f"\nDataset structure:")
    print(f"  Images: {len(list(images_dir.glob('*.jpg')))} files")
    print(f"  Labels: {labels_file.name}")
    
    if 'Age' in df.columns and 'Gender' in df.columns:
        print(f"\n✓ Patient metadata available:")
        print(f"  - Age range: {df['Age'].min()}-{df['Age'].max()} years")
        print(f"  - Gender: {df['Gender'].value_counts().to_dict()}")
    
    if 'Retinopathy grade' in df.columns:
        print(f"\n✓ DR severity grades:")
        print(df['Retinopathy grade'].value_counts().sort_index())
    
    # Create processed version
    processed_dir = Path("data/processed/idrid")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Map DR grades to our binary Diabetes label
    df_processed = df.copy()
    df_processed['Label_D'] = (df_processed['Retinopathy grade'] > 0).astype(int)
    df_processed['filepath'] = df_processed['Image name'].apply(
        lambda x: str(images_dir / f"{x}.jpg")
    )
    
    # Encode sex: Male=0, Female=1 (matching ODIR)
    df_processed['sex_encoded'] = df_processed['Gender'].map({'Male': 0, 'Female': 1})
    
    output_file = processed_dir / "idrid_processed.csv"
    df_processed.to_csv(output_file, index=False)
    
    print(f"\n✓ Processed dataset saved to: {output_file}")
    print(f"\nColumns: {list(df_processed.columns)}")
    print(f"\nReady for evaluation! Run:")
    print(f"  python scripts/evaluate_external.py --dataset idrid")
    
    return True


if __name__ == "__main__":
    prepare_idrid_dataset()

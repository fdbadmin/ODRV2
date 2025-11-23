#!/usr/bin/env python3
"""
Extract EyePACS multi-part zip archives
"""

import os
import zipfile
from pathlib import Path

def extract_multipart_zip(base_path, prefix="train.zip"):
    """Extract multi-part zip files"""
    parts = sorted([f for f in os.listdir(base_path) if f.startswith(prefix) and f.endswith(('.001', '.002', '.003', '.004', '.005'))])
    
    if not parts:
        print(f"No parts found for {prefix}")
        return
    
    print(f"Found {len(parts)} parts for {prefix}")
    
    # Combine parts
    combined_path = base_path / f"{prefix.replace('.zip', '')}_combined.zip"
    print(f"Combining parts into {combined_path}...")
    
    with open(combined_path, 'wb') as outfile:
        for part in parts:
            part_path = base_path / part
            print(f"  Adding {part}...")
            with open(part_path, 'rb') as infile:
                outfile.write(infile.read())
    
    print(f"Extracting {combined_path}...")
    with zipfile.ZipFile(combined_path, 'r') as zip_ref:
        zip_ref.extractall(base_path)
    
    print(f"Extraction complete! Files extracted to {base_path}")
    
    # Count files
    extract_dir = base_path / prefix.replace('.zip', '')
    if extract_dir.exists():
        file_count = len(list(extract_dir.glob('*.jpeg')))
        print(f"Extracted {file_count} JPEG images")

if __name__ == "__main__":
    eyepacs_path = Path("/Users/fdb/VSCode/ODRV2/external_data/diabetic-retinopathy-detection")
    
    print("Extracting EyePACS training images...")
    extract_multipart_zip(eyepacs_path, "train.zip")
    
    print("\nDone!")

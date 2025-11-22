
import pandas as pd
import numpy as np
from src.datamodules.lightning import FundusDataModule
from pathlib import Path

def check_val_distribution():
    csv_path = Path("data/processed/odir_eye_labels.csv")
    dm = FundusDataModule(
        csv_path=csv_path,
        image_root=Path("RAW DATA FULL"),
        image_size=(512, 512),
        batch_size=16,
        val_fold=0
    )
    dm.setup()
    
    val_df = dm.val_df
    print(f"Validation set size (patients): {len(val_df)}")
    
    # Expand to eyes
    # We need to simulate what ODIRDataset does
    # It takes Left and Right for each patient
    
    # Let's count the labels in val_df for Left and Right eyes
    class_codes = ["D", "G", "C", "A", "H", "M", "O"]
    
    counts = {c: 0 for c in class_codes}
    
    for _, row in val_df.iterrows():
        # Left eye
        for c in class_codes:
            col = f"Left_{c}"
            if col in row and row[col] == 1:
                counts[c] += 1
        # Right eye
        for c in class_codes:
            col = f"Right_{c}"
            if col in row and row[col] == 1:
                counts[c] += 1
                
    print("Class counts in Validation Set (Fold 0):")
    print(counts)
    
    total_eyes = len(val_df) * 2
    print(f"Total eyes: {total_eyes}")
    
    for c in class_codes:
        print(f"{c}: {counts[c]} ({counts[c]/total_eyes:.2%})")

if __name__ == "__main__":
    check_val_distribution()

"""
Universal fundus dataset for unified multi-dataset training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.label_parser import CLASS_CODES


class FundusDataset(Dataset):
    """
    Universal fundus dataset compatible with unified CSV format.
    
    Expected CSV columns:
    - image_path: Full path to fundus image
    - Label_D, Label_G, Label_C, Label_A, Label_H, Label_M, Label_O: Disease labels (0/1)
    - Patient Age: Optional age metadata
    - Patient Sex: Optional gender metadata (0=Female, 1=Male)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: tuple[int, int] = (448, 448),
        augment: bool = True,
        use_metadata: bool = False
    ):
        """
        Args:
            df: DataFrame with image_path and label columns
            image_size: Target image size (height, width)
            augment: Whether to apply data augmentation
            use_metadata: Whether to include age/sex metadata
        """
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.use_metadata = use_metadata
        
        # Define augmentation pipeline
        if augment:
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        
        # Load image
        image_path = Path(row['image_path'])
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image_np = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        
        # Apply transformations
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image']
        
        # Get labels
        label_cols = [f'Label_{c}' for c in CLASS_CODES]
        labels = torch.tensor(row[label_cols].values.astype('float32'))
        
        # Get metadata
        if self.use_metadata:
            age = row.get('Patient Age', 50.0)  # Default to 50 if missing
            sex = row.get('Patient Sex', 0.0)   # Default to Female if missing
            
            # Normalize age to [0, 1]
            age_normalized = (age - 1.0) / 90.0 if not pd.isna(age) else 0.55
            sex_normalized = float(sex) if not pd.isna(sex) else 0.0
        else:
            # Use dummy metadata
            age_normalized = 0.55  # ~50 years old
            sex_normalized = 0.0   # Female
        
        age_tensor = torch.tensor([age_normalized], dtype=torch.float32)
        sex_tensor = torch.tensor([sex_normalized], dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'labels': labels,
            'age': age_tensor,
            'sex': sex_tensor
        }

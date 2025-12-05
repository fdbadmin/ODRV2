#!/usr/bin/env python3
"""
Save Individual Fold Predictions for Pruning Analysis

Quickly generates and saves predictions from each fold separately.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from models.backbones import FundusBackbone
from models.multilabel_head import MultiLabelClassifier

# Configuration
MODEL_BASE_PATH = project_root / "models" / "unified_v3"
DATA_CSV = project_root / "data" / "processed" / "unified_v3" / "unified_train_v3.csv"
OUTPUT_DIR = project_root / "results" / "unified_v3"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
LABEL_COLUMNS = ["Label_D", "Label_G", "Label_C", "Label_A", "Label_H", "Label_M", "Label_O"]

class SimpleModel(torch.nn.Module):
    def __init__(self, num_classes=7, feature_dim=1024, dropout=0.3):
        super().__init__()
        self.backbone = FundusBackbone(model_name="convnext_base", pretrained=False, feature_dim=feature_dim)
        self.classifier = MultiLabelClassifier(feature_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(self.backbone(x))

def load_model(fold_idx):
    checkpoint_path = MODEL_BASE_PATH / f"fold_{fold_idx}" / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = checkpoint['config']
    
    model = SimpleModel(config['num_classes'], config['feature_dim'], config['dropout'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model

def main():
    print("="*80)
    print("GENERATING FOLD PREDICTIONS FOR PRUNING")
    print("="*80)
    
    # Load data
    print("\nLoading dataset...")
    df = pd.read_csv(DATA_CSV)
    print(f"Total samples: {len(df)}")
    
    # Create dataloader
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from PIL import Image
    
    class EvalDataset(Dataset):
        def __init__(self, df, transform):
            self.df = df.reset_index(drop=True)
            self.transform = transform
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image = Image.open(row['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            labels = torch.tensor(row[LABEL_COLUMNS].values.astype(np.float32))
            return image, labels
    
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = EvalDataset(df, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Save labels once
    labels = df[LABEL_COLUMNS].values
    np.save(OUTPUT_DIR / "ground_truth.npy", labels)
    print(f"✓ Saved ground truth labels")
    
    # Process each fold
    for fold_idx in range(5):
        print(f"\n{'='*80}")
        print(f"Processing Fold {fold_idx}")
        print('='*80)
        
        # Check if already exists
        output_file = OUTPUT_DIR / f"fold_{fold_idx}_probs.npy"
        if output_file.exists():
            print(f"  Fold {fold_idx} predictions already exist, skipping...")
            continue
        
        # Load model
        print(f"  Loading model...")
        model = load_model(fold_idx)
        
        # Get predictions
        all_probs = []
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc=f"  Inference"):
                images = images.to(DEVICE)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
        
        all_probs = np.vstack(all_probs)
        
        # Save
        np.save(output_file, all_probs)
        print(f"  ✓ Saved fold {fold_idx} predictions: {all_probs.shape}")
        
        # Free memory
        del model
        torch.mps.empty_cache() if torch.backends.mps.is_available() else torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("COMPLETE - Ready for fold pruning analysis")
    print("="*80)
    print(f"\nRun: python scripts/evaluation/evaluate_fold_pruning.py")

if __name__ == "__main__":
    main()

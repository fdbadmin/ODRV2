import torch
import pandas as pd
import numpy as np
from pathlib import Path
from src.inference.service import load_ensemble, _normalize, _load_image
from src.training.loop import FundusLightningModule

def demo_ensemble():
    # Setup paths
    root_dir = Path(".")
    data_path = root_dir / "data/processed/odir_eye_labels.csv"
    img_dir = root_dir / "RAW DATA FULL"
    models_dir = root_dir / "models"
    config_path = root_dir / "configs/training.yaml"
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Load ensemble
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        models, config = load_ensemble(models_dir, config_path, device)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Optimized thresholds
    THRESHOLDS = [0.45, 0.45, 0.40, 0.75, 0.15, 0.30, 0.25]
    CLASS_NAMES = ['D', 'G', 'C', 'A', 'H', 'M', 'O']
    
    # Pick 5 random samples
    samples = df.sample(5)
    
    print("\n=== Ensemble Prediction Demo ===")
    print(f"Loaded {len(models)} models.")
    print("-" * 50)
    
    for _, row in samples.iterrows():
        img_path = img_dir / row['filename']
        if not img_path.exists():
            continue
            
        # Prepare input
        img_tensor = _load_image(img_path, config.image_size).to(device)
        img_tensor = _normalize(img_tensor)
        
        sex_val = 0 if str(row['Patient Sex']).lower().startswith("m") else 1
        
        batch = {
            "image": img_tensor.unsqueeze(0),
            "age": torch.tensor([row['Patient Age']], dtype=torch.float32, device=device),
            "sex": torch.tensor([sex_val], dtype=torch.float32, device=device),
        }
        
        # Inference
        with torch.no_grad():
            all_probs = []
            for model in models:
                logits = model(batch)
                probs = torch.sigmoid(logits)[0]
                all_probs.append(probs)
            
            avg_probs = torch.stack(all_probs).mean(dim=0).cpu().numpy()
            
        # Display results
        print(f"\nImage: {row['filename']}")
        print(f"Ground Truth: ", end="")
        gt_labels = []
        for cls in CLASS_NAMES:
            if row[f'Label_{cls}'] == 1:
                gt_labels.append(cls)
        print(gt_labels if gt_labels else ["Normal"])
        
        print("Prediction:   ", end="")
        pred_labels = []
        for i, (prob, thresh) in enumerate(zip(avg_probs, THRESHOLDS)):
            if prob >= thresh:
                pred_labels.append(f"{CLASS_NAMES[i]} ({prob:.2f})")
        print(pred_labels if pred_labels else ["Normal"])
        
        # Show raw probs for rare classes
        print(f"Rare Class Probs: H={avg_probs[4]:.3f}, M={avg_probs[5]:.3f}, O={avg_probs[6]:.3f}")

if __name__ == "__main__":
    demo_ensemble()

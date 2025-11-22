import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score, hamming_loss
from src.inference.service import load_ensemble, _normalize, _load_image
from src.utils.label_parser import CLASS_CODES

def evaluate_ensemble():
    # Setup paths
    root_dir = Path(".")
    data_path = root_dir / "data/processed/odir_eye_labels.csv"
    img_dir = root_dir / "RAW DATA FULL"
    models_dir = root_dir / "models"
    config_path = root_dir / "configs/training.yaml"
    
    # Load data
    print(f"Loading data from {data_path}...")
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
    
    # Select a subset for evaluation (e.g., 1000 samples) to keep it quick
    # Use a fixed seed for reproducibility
    eval_df = df.sample(n=1000, random_state=42)
    print(f"Evaluating on {len(eval_df)} random samples...")
    
    y_true = []
    y_pred = []
    y_prob = []
    
    print("Running inference...")
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        img_path = img_dir / row['filename']
        if not img_path.exists():
            continue
            
        # Prepare input
        try:
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
                
            # Get Ground Truth
            gt_labels = []
            for cls in CLASS_NAMES:
                gt_labels.append(row[f'Label_{cls}'])
            
            # Get Predictions
            pred_labels = [1 if p >= t else 0 for p, t in zip(avg_probs, THRESHOLDS)]
            
            y_true.append(gt_labels)
            y_pred.append(pred_labels)
            y_prob.append(avg_probs)
            
        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")
            continue

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    print("\n" + "="*60)
    print("FINAL ENSEMBLE EVALUATION REPORT")
    print("="*60)
    
    print("\n1. Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    print("\n2. Overall Metrics:")
    print(f"Subset Accuracy (Exact Match): {accuracy_score(y_true, y_pred):.4f}")
    print(f"Hamming Loss (Fraction of wrong labels): {hamming_loss(y_true, y_pred):.4f}")
    
    print("\n3. Confusion Matrices per Class (TN, FP, FN, TP):")
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    for i, class_name in enumerate(CLASS_NAMES):
        tn, fp, fn, tp = mcm[i].ravel()
        print(f"\nClass {class_name}:")
        print(f"  TP: {tp:<5} FN: {fn}")
        print(f"  FP: {fp:<5} TN: {tn}")
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"  Sensitivity (Recall): {sensitivity:.3f}")
        print(f"  Specificity:          {specificity:.3f}")

if __name__ == "__main__":
    evaluate_ensemble()

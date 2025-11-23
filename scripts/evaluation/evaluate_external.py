"""Evaluate ensemble on external datasets for cross-dataset validation.

This script tests the trained ensemble on external fundus datasets to measure
generalization performance and robustness across different data distributions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from src.inference.service import load_ensemble


class ExternalDataset(Dataset):
    """Dataset for external validation."""
    
    def __init__(self, df: pd.DataFrame, image_size: tuple[int, int] = (448, 448)):
        self.df = df
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        
        # Load and transform image
        image = Image.open(row['filepath']).convert('RGB')
        image_tensor = self.transform(image)
        
        # Get metadata
        age = float(row.get('Age', 50))  # Default to 50 if missing
        sex = float(row.get('sex_encoded', 0))  # Default to Male if missing
        
        # Get label (for Diabetes detection)
        label = int(row.get('Label_D', 0))
        
        return {
            'image': image_tensor,
            'age': torch.tensor([age], dtype=torch.float32),
            'sex': torch.tensor([sex], dtype=torch.float32),
            'label': torch.tensor([label], dtype=torch.long),
            'image_name': row.get('Image name', row.get('filename', 'unknown'))
        }


def evaluate_external_dataset(
    dataset_name: str,
    models: list,
    device: torch.device,
    use_tta: bool = True,
    threshold: float = 0.45,
) -> dict[str, Any]:
    """Evaluate ensemble on external dataset."""
    
    # Load processed dataset
    data_path = Path(f"data/processed/{dataset_name}/{dataset_name}_processed.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"Run: python scripts/download_{dataset_name}.py first"
        )
    
    df = pd.read_csv(data_path)
    print(f"\nLoading {dataset_name.upper()} dataset: {len(df)} images")
    
    # Create dataset and dataloader
    dataset = ExternalDataset(df)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Collect predictions
    all_probs = []
    all_labels = []
    all_names = []
    
    print(f"Evaluating with {'TTA' if use_tta else 'single'} inference...")
    
    with torch.no_grad():
        for batch in loader:
            batch_probs = []
            
            # TTA transforms if enabled
            tta_transforms = [
                lambda x: x,  # Original
                lambda x: torch.flip(x, dims=[3]),  # Horizontal flip
                lambda x: torch.flip(x, dims=[2]),  # Vertical flip
                lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # 90° rotation
                lambda x: torch.rot90(x, k=3, dims=[2, 3]),  # 270° rotation
            ] if use_tta else [lambda x: x]
            
            # Average predictions across TTA and ensemble
            for transform in tta_transforms:
                image_transformed = transform(batch['image']).to(device)
                
                for model in models:
                    model_batch = {
                        'image': image_transformed,
                        'age': batch['age'].to(device),
                        'sex': batch['sex'].to(device),
                    }
                    logits = model(model_batch)
                    probs = torch.sigmoid(logits)[:, 1]  # Diabetes probability (index 1)
                    batch_probs.append(probs.cpu())
            
            # Average across all predictions
            avg_probs = torch.stack(batch_probs).mean(dim=0)
            all_probs.extend(avg_probs.numpy())
            all_labels.extend(batch['label'].numpy())
            all_names.extend(batch['image_name'])
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'dataset': dataset_name.upper(),
        'n_samples': len(all_labels),
        'n_positive': int(all_labels.sum()),
        'threshold': threshold,
        'use_tta': use_tta,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1_score': f1_score(all_labels, all_preds, zero_division=0),
        'specificity': recall_score(1 - all_labels, 1 - all_preds, zero_division=0),
    }
    
    # ROC-AUC if both classes present
    if len(np.unique(all_labels)) > 1:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_probs)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics, all_probs, all_labels, all_names


def print_results(metrics: dict[str, Any]):
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print(f"📊 External Validation Results: {metrics['dataset']}")
    print(f"{'='*60}")
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {metrics['n_samples']}")
    print(f"  Positive cases: {metrics['n_positive']} ({metrics['n_positive']/metrics['n_samples']*100:.1f}%)")
    print(f"  Negative cases: {metrics['n_samples'] - metrics['n_positive']}")
    
    print(f"\nModel Configuration:")
    print(f"  Threshold: {metrics['threshold']}")
    print(f"  TTA enabled: {metrics['use_tta']}")
    
    print(f"\n🎯 Performance Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1 Score:    {metrics['f1_score']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    
    print(f"\n📈 Confusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"                Predicted")
    print(f"              Neg    Pos")
    print(f"  Actual Neg  {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"         Pos  {cm[1,0]:4d}   {cm[1,1]:4d}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on external datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['idrid', 'aptos', 'messidor', 'refuge'],
        help="External dataset name"
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable test-time augmentation"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Classification threshold for Diabetes (default: 0.45)"
    )
    
    args = parser.parse_args()
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load ensemble
    model_dir = Path("models")
    config_path = Path("configs/training.yaml")
    
    print(f"\nLoading ensemble from {model_dir}...")
    models, config = load_ensemble(model_dir, config_path, device)
    print(f"✓ Loaded {len(models)} models")
    
    # Evaluate
    metrics, probs, labels, names = evaluate_external_dataset(
        dataset_name=args.dataset,
        models=models,
        device=device,
        use_tta=not args.no_tta,
        threshold=args.threshold,
    )
    
    # Print results
    print_results(metrics)
    
    # Save detailed results
    results_dir = Path("results/external_validation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    results_df = pd.DataFrame({
        'image_name': names,
        'true_label': labels,
        'predicted_prob': probs,
        'predicted_label': (probs >= args.threshold).astype(int),
    })
    
    output_file = results_dir / f"{args.dataset}_predictions.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Detailed predictions saved to: {output_file}")
    
    # Save metrics
    import json
    metrics_file = results_dir / f"{args.dataset}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()

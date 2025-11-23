
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.training.loop import FundusLightningModule, TrainingConfig
from src.datamodules.lightning import FundusDataModule
from src.training.cli import _load_with_defaults
from sklearn.metrics import f1_score

def optimize_thresholds():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Config and Data
    config_path = Path("configs/training.yaml")
    cfg = _load_with_defaults(config_path)
    
    # Reconstruct TrainingConfig (same as in service.py)
    training_cfg = TrainingConfig(
        num_classes=cfg.data.num_classes,
        feature_dim=cfg.model.feature_dim,
        learning_rate=cfg.model.optimizer.lr,
        weight_decay=cfg.model.optimizer.weight_decay,
        epochs=cfg.trainer.max_epochs,
        batch_size=cfg.datamodule.batch_size,
        model_dir=Path(cfg.paths.models),
        pos_weight=cfg.datamodule.get("class_weights", None),
        image_size=tuple(cfg.datamodule.image_size),
    )

    # 2. Load Model
    model_path = Path("models/model.ckpt")
    if not model_path.exists():
        print("Model checkpoint not found at models/model.ckpt")
        return

    print("Loading model...")
    model = FundusLightningModule.load_from_checkpoint(str(model_path), config=training_cfg)
    model.to(device)
    model.eval()

    # 3. Prepare Validation Data
    print("Preparing validation data...")
    datamodule = FundusDataModule(
        csv_path=Path(cfg.paths.processed_data) / "odir_eye_labels.csv",
        image_root=Path(cfg.paths.raw_data),
        image_size=tuple(cfg.datamodule.image_size),
        batch_size=16, # Larger batch for inference
        num_workers=4,
        val_fold=0
    )
    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    # 4. Collect Predictions
    print("Running inference on validation set...")
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            # Move batch to device
            batch["image"] = batch["image"].to(device)
            batch["age"] = batch["age"].to(device)
            batch["sex"] = batch["sex"].to(device)
            
            logits = model(batch)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(batch["labels"].numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    # 5. Optimize Thresholds per Class
    class_names = ["Diabetes", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia", "Other"]
    best_thresholds = []
    best_f1s = []

    print("\n--- Optimization Results ---")
    
    for i, name in enumerate(class_names):
        y_true = all_targets[:, i]
        y_scores = all_probs[:, i]
        
        best_t = 0.5
        best_f1 = 0.0
        
        # Search range 0.1 to 0.9
        for t in np.arange(0.1, 0.95, 0.05):
            y_pred = (y_scores >= t).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t
        
        best_thresholds.append(best_t)
        best_f1s.append(best_f1)
        print(f"{name:<12} | Best Threshold: {best_t:.2f} | F1 Score: {best_f1:.4f}")

    macro_f1 = np.mean(best_f1s)
    print("-" * 40)
    print(f"Optimized Macro F1: {macro_f1:.4f}")
    print(f"Recommended Thresholds: {best_thresholds}")

if __name__ == "__main__":
    optimize_thresholds()

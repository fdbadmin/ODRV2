
import os
from pathlib import Path
from omegaconf import OmegaConf
from src.training.cli import _load_with_defaults
from src.training.loop import TrainingConfig, train_model
from src.datamodules.lightning import FundusDataModule

def train_ensemble():
    config_path = Path("configs/training.yaml")
    base_cfg = _load_with_defaults(config_path)
    
    # Ensure model directory exists
    model_dir = Path(base_cfg.paths.models)
    model_dir.mkdir(parents=True, exist_ok=True)

    num_folds = base_cfg.datamodule.get("num_folds", 5)
    
    print(f"Starting ensemble training for {num_folds} folds...")

    for fold in range(num_folds):
        print(f"\n=== Training Fold {fold}/{num_folds - 1} ===")
        
        # Create fold-specific config
        cfg = base_cfg.copy()
        
        # Update validation fold
        cfg.datamodule.val_fold = fold
        
        # Construct TrainingConfig
        training_cfg = TrainingConfig(
            num_classes=cfg.data.num_classes,
            feature_dim=cfg.model.feature_dim,
            learning_rate=cfg.model.optimizer.lr,
            weight_decay=cfg.model.optimizer.weight_decay,
            epochs=cfg.trainer.max_epochs,
            batch_size=cfg.datamodule.batch_size,
            model_dir=Path(cfg.paths.models),
            pos_weight=cfg.datamodule.get("class_weights", None),
            precision=str(cfg.trainer.get("precision", "32")),
            accelerator=cfg.trainer.get("accelerator", None),
            devices=cfg.trainer.get("devices", None),
            accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
            log_every_n_steps=cfg.logging.get("log_every_n_steps", 20),
            num_sanity_val_steps=cfg.trainer.get("num_sanity_val_steps", 2),
            early_stopping_patience=cfg.trainer.get("early_stopping_patience", None),
            image_size=tuple(cfg.datamodule.image_size),
            ckpt_name=f"model_fold_{fold}.ckpt"
        )

        # Setup DataModule
        metadata_csv = cfg.datamodule.get("metadata_csv", None)
        if metadata_csv is None:
            metadata_csv_path = Path(cfg.paths.processed_data) / "odir_eye_labels.csv"
        else:
            metadata_csv_path = Path(metadata_csv)
            
        datamodule = FundusDataModule(
            csv_path=metadata_csv_path,
            image_root=Path(cfg.paths.raw_data),
            image_size=tuple(cfg.datamodule.image_size),
            batch_size=cfg.datamodule.batch_size,
            num_workers=cfg.datamodule.get("num_workers", 8),
            seed=cfg.project.seed,
            num_folds=cfg.datamodule.get("num_folds", 5),
            val_fold=fold, # Explicitly set the fold
            pin_memory=cfg.datamodule.get("pin_memory", True),
            persistent_workers=cfg.datamodule.get("persistent_workers", False),
            single_eye=cfg.datamodule.get("single_eye", True),
        )

        # Train
        train_model(training_cfg, datamodule)
        print(f"Fold {fold} completed. Model saved to {training_cfg.ckpt_name}")

if __name__ == "__main__":
    train_ensemble()

from __future__ import annotations

import os
from pathlib import Path

# Silence Albumentations network version check warnings before importing Albumentations consumers.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from omegaconf import DictConfig, OmegaConf  # type: ignore[import]

from src.datamodules.lightning import FundusDataModule
from src.training.loop import TrainingConfig, train_model


def _load_with_defaults(config_path: Path) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    defaults = cfg.get("defaults", [])
    if defaults:
        config_dir = config_path.parent
        merged = OmegaConf.create()
        for default in defaults:
            if isinstance(default, str):
                default_cfg = OmegaConf.load(config_dir / f"{default}.yaml")
                merged = OmegaConf.merge(merged, default_cfg)
        cfg = OmegaConf.merge(merged, cfg)
        if "defaults" in cfg:
            del cfg["defaults"]
    return cfg


def run_from_config(config_path: Path) -> None:
    cfg = _load_with_defaults(config_path)
    training_cfg = TrainingConfig(
        num_classes=cfg.data.num_classes,
        feature_dim=cfg.model.feature_dim,
        learning_rate=cfg.model.optimizer.lr,
        weight_decay=cfg.model.optimizer.weight_decay,
        epochs=cfg.trainer.max_epochs,
        batch_size=cfg.datamodule.batch_size,
        model_dir=Path(cfg.paths.models),
        pos_weight=cfg.datamodule.get("class_weights", None),
    )
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
        val_fold=cfg.datamodule.get("val_fold", 0),
        pin_memory=cfg.datamodule.get("pin_memory", True),
        persistent_workers=cfg.datamodule.get("persistent_workers", False),
    )
    train_model(training_cfg, datamodule)


if __name__ == "__main__":
    run_from_config(Path("configs/training.yaml"))

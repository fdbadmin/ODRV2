from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig  # type: ignore[import]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training.cli import _load_with_defaults
from src.training.loop import FundusLightningModule, TrainingConfig
from src.datamodules.lightning import FundusDataModule


def _build_training_config(cfg: DictConfig) -> TrainingConfig:
    return TrainingConfig(
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
    )


def _load_datamodule(cfg: DictConfig) -> FundusDataModule:
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
        single_eye=cfg.datamodule.get("single_eye", True),
    )
    datamodule.setup("validate")
    return datamodule


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def evaluate(checkpoint_path: Path, config_path: Path) -> None:
    cfg = _load_with_defaults(config_path)
    training_cfg = _build_training_config(cfg)
    datamodule = _load_datamodule(cfg)

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    model = FundusLightningModule.load_from_checkpoint(
        str(checkpoint_path),
        config=training_cfg,
        map_location=device,
    )
    model.eval()
    model.to(device)

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    val_loader = datamodule.val_dataloader()
    pos_weight = None
    if training_cfg.pos_weight is not None:
        pos_weight = torch.tensor(training_cfg.pos_weight, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    running_loss = 0.0
    batches = 0

    with torch.no_grad():
        for batch in val_loader:
            device_batch = _move_batch_to_device(batch, device)
            logits = model.forward(device_batch)
            labels = device_batch["labels"]
            loss = criterion(logits, labels.to(logits.device))
            running_loss += float(loss.cpu().item())
            batches += 1
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())

    if not all_logits:
        print("No validation batches were processed.")
        return

    stacked_logits = torch.cat(all_logits)
    stacked_labels = torch.cat(all_labels)
    metrics = model._compute_f1_metrics(stacked_logits, stacked_labels)  # type: ignore[attr-defined]
    macro_f1 = metrics["macro_f1"]
    per_class_f1 = metrics["per_class_f1"]
    val_loss = running_loss / max(1, batches)

    print("Evaluation results")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Per-class F1: " + ", ".join(f"{score:.4f}" for score in per_class_f1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the validation split.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/model.ckpt"), help="Path to the model checkpoint")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to the training config used for the run",
    )
    args = parser.parse_args()
    evaluate(args.checkpoint, args.config)


if __name__ == "__main__":
    main()

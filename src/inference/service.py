from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch  # type: ignore[import]
from fastapi import FastAPI, UploadFile  # type: ignore[import]

from src.preprocessing.pipeline import preprocess_batch
from src.training.loop import FundusLightningModule, TrainingConfig
from src.training.cli import _load_with_defaults

_IMG_MEAN = (0.485, 0.456, 0.406)
_IMG_STD = (0.229, 0.224, 0.225)


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to a CHW tensor."""
    mean = torch.tensor(_IMG_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(_IMG_STD, device=tensor.device).view(3, 1, 1)
    return (tensor - mean) / std


def load_ensemble(model_dir: Path, config_path: Path, device: torch.device) -> tuple[list[FundusLightningModule], TrainingConfig]:
    """Load all trained fold models for ensembling."""
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
        precision=str(cfg.trainer.get("precision", "32")),
        accelerator=cfg.trainer.get("accelerator", None),
        devices=cfg.trainer.get("devices", None),
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
        log_every_n_steps=cfg.logging.get("log_every_n_steps", 20),
        num_sanity_val_steps=cfg.trainer.get("num_sanity_val_steps", 2),
        image_size=tuple(cfg.datamodule.image_size),
    )
    
    models = []
    # Look for model_fold_0.ckpt, model_fold_1.ckpt, etc.
    # Also include model.ckpt if it exists (as a fallback or part of ensemble)
    checkpoint_files = sorted(model_dir.glob("model_fold_*.ckpt"))
    if not checkpoint_files and (model_dir / "model.ckpt").exists():
        checkpoint_files = [model_dir / "model.ckpt"]
        
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")

    print(f"Loading {len(checkpoint_files)} models for ensemble...")
    for ckpt in checkpoint_files:
        model = FundusLightningModule.load_from_checkpoint(str(ckpt), config=training_cfg)
        model.to(device)
        model.eval()
        models.append(model)
        
    return models, training_cfg


def _write_temp_image(data: bytes) -> Path:
    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(data)
        return Path(tmp.name)


def _load_image(path: Path, size: tuple[int, int]) -> torch.Tensor:
    image = preprocess_batch([path], size)[0]
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    return tensor


def create_app(models: list[FundusLightningModule], config: TrainingConfig, device: torch.device) -> FastAPI:
    """Create FastAPI inference service with ensemble support."""
    app = FastAPI()
    size = config.image_size or (448, 448)
    
    # Optimized thresholds from validation set
    # D, G, C, A, H, M, O
    THRESHOLDS = [0.45, 0.45, 0.40, 0.75, 0.15, 0.30, 0.25]

    @app.post("/predict")
    async def predict(file: UploadFile, age: float, sex: int) -> dict[str, Any]:
        image_bytes = await file.read()
        image_path = _write_temp_image(image_bytes)

        try:
            image_tensor = _load_image(image_path, size).to(device)
        finally:
            image_path.unlink(missing_ok=True)

        image_tensor = _normalize(image_tensor)
        batch = {
            "image": image_tensor.unsqueeze(0),
            "age": torch.tensor([age], dtype=torch.float32, device=device),
            "sex": torch.tensor([sex], dtype=torch.float32, device=device),
        }

        with torch.no_grad():
            # Ensemble averaging
            all_probs = []
            for model in models:
                logits = model(batch)
                probs = torch.sigmoid(logits)[0]
                all_probs.append(probs)
            
            # Stack and mean
            avg_probs = torch.stack(all_probs).mean(dim=0).cpu().tolist()
            
        predictions = {
            "probabilities": avg_probs,
            "labels": [p >= t for p, t in zip(avg_probs, THRESHOLDS)],
            "thresholds": THRESHOLDS
        }
        return predictions

    return app

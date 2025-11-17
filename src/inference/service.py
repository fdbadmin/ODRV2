from __future__ import annotations

def load_checkpoint(model_path: Path, device: torch.device) -> torch.nn.Module:
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch  # type: ignore[import]
from fastapi import FastAPI, UploadFile  # type: ignore[import]
from omegaconf import OmegaConf  # type: ignore[import]

from src.preprocessing.pipeline import preprocess_batch
from src.training.loop import FundusLightningModule, TrainingConfig


def load_checkpoint(model_path: Path, config_path: Path, device: torch.device) -> FundusLightningModule:
    """Load the trained Lightning checkpoint for inference."""
    cfg = OmegaConf.load(config_path)
    training_cfg = TrainingConfig(
        num_classes=cfg.data.num_classes,
        feature_dim=cfg.model.feature_dim,
        learning_rate=cfg.model.optimizer.lr,
        weight_decay=cfg.model.optimizer.weight_decay,
        epochs=cfg.trainer.max_epochs,
        batch_size=cfg.datamodule.batch_size,
        model_dir=Path(cfg.paths.models),
    )
    model = FundusLightningModule.load_from_checkpoint(str(model_path), config=training_cfg)
    model.to(device)
    model.eval()
    return model


def create_app(model: torch.nn.Module, device: torch.device) -> FastAPI:
    """Create FastAPI inference service."""
    app = FastAPI()

    @app.post("/predict")
    async def predict(file_left: UploadFile, file_right: UploadFile, age: float, sex: int) -> dict[str, Any]:
        left_bytes = await file_left.read()
        right_bytes = await file_right.read()
        with NamedTemporaryFile(suffix=".jpg", delete=False) as left_tmp, NamedTemporaryFile(suffix=".jpg", delete=False) as right_tmp:
            left_tmp.write(left_bytes)
            right_tmp.write(right_bytes)
            left_path = Path(left_tmp.name)
            right_path = Path(right_tmp.name)
        images = preprocess_batch([left_path, right_path], (512, 512))
        left_path.unlink(missing_ok=True)
        right_path.unlink(missing_ok=True)
        left_tensor = torch.from_numpy(images[0]).permute(2, 0, 1).to(device)
        right_tensor = torch.from_numpy(images[1]).permute(2, 0, 1).to(device)
        age_tensor = torch.tensor([age], device=device)
        sex_tensor = torch.tensor([sex], device=device)
        with torch.no_grad():
            logits = model({
                "left_image": left_tensor.unsqueeze(0),
                "right_image": right_tensor.unsqueeze(0),
                "age": age_tensor,
                "sex": sex_tensor,
            })
            probabilities = torch.sigmoid(logits)[0].cpu().tolist()
        return {"probabilities": probabilities}

    return app

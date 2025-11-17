from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
from lightning import LightningModule, Trainer  # type: ignore[import]
from lightning.pytorch.loggers import MLFlowLogger  # type: ignore[import]

from src.models.backbones import DualFundusBackbone
from src.models.multilabel_head import MetadataConditioner, MultiLabelClassifier


@dataclass
class TrainingConfig:
    num_classes: int
    feature_dim: int
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    model_dir: Path
    pos_weight: list[float] | None = None


class FundusLightningModule(LightningModule):
    """Lightning module orchestrating the fundus classifier."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.save_hyperparameters(config.__dict__)
        self.backbone = DualFundusBackbone(feature_dim=config.feature_dim)
        self.conditioner = MetadataConditioner(config.feature_dim)
        self.classifier = MultiLabelClassifier(config.feature_dim, config.num_classes)
        pos_weight = getattr(config, "pos_weight", None)
        if pos_weight is not None:
            weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
        else:
            weight_tensor = None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:  # type: ignore[override]
        features = self.backbone(batch["left_image"], batch["right_image"])
        conditioned = self.conditioner(features, batch["age"], batch["sex"])
        return self.classifier(conditioned)

    def training_step(self, batch: dict[str, Any], _: int) -> torch.Tensor:  # type: ignore[override]
        logits = self.forward(batch)
        loss = self.criterion(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], _: int) -> None:  # type: ignore[override]
        logits = self.forward(batch)
        loss = self.criterion(logits, batch["labels"])
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def train_model(config: TrainingConfig, datamodule: Any) -> None:
    """Train the model with MLflow logging enabled."""
    logger = MLFlowLogger(experiment_name="fundus-odir")
    config.model_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        max_epochs=config.epochs,
        logger=logger,
        precision="16-mixed",
        log_every_n_steps=20,
    )
    model = FundusLightningModule(config)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.save_checkpoint(str(config.model_dir / "model.ckpt"))

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
from lightning import LightningModule, Trainer  # type: ignore[import]
from lightning.pytorch.callbacks import EarlyStopping  # type: ignore[import]
from lightning.pytorch.loggers import MLFlowLogger  # type: ignore[import]

from src.models.backbones import FundusBackbone
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
    precision: str = "32"
    accelerator: str | None = None
    devices: Any | None = None
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 20
    num_sanity_val_steps: int = 2
    image_size: tuple[int, int] | None = None
    early_stopping_patience: int | None = None
    ckpt_name: str = "model.ckpt"


import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, gamma_rare: float = 3.0, reduction: str = 'mean'):
        """
        Focal Loss for multi-label classification with enhanced rare disease focus.
        
        Args:
            alpha: Class weights (from pos_weight)
            gamma: Base focal parameter for common diseases
            gamma_rare: Higher focal parameter for rare diseases (indices 2,3,4,5: C,A,H,M)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.gamma_rare = gamma_rare
        self.reduction = reduction
        # Indices for rare diseases: Cataract(2), AMD(3), HTN(4), Myopia(5)
        self.rare_indices = {2, 3, 4, 5}

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch, num_classes] logits
            targets: [batch, num_classes] labels
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # Apply different gamma for rare diseases
        # focal_loss shape: [batch, num_classes]
        focal_loss = torch.zeros_like(bce_loss)
        for class_idx in range(inputs.shape[1]):
            if class_idx in self.rare_indices:
                # Use higher gamma for rare diseases
                focal_loss[:, class_idx] = (1 - pt[:, class_idx]) ** self.gamma_rare * bce_loss[:, class_idx]
            else:
                # Use base gamma for common diseases
                focal_loss[:, class_idx] = (1 - pt[:, class_idx]) ** self.gamma * bce_loss[:, class_idx]
        
        # Apply alpha weights
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # Multiply each class by its corresponding alpha weight
            focal_loss = focal_loss * self.alpha
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FundusLightningModule(LightningModule):
    """Lightning module orchestrating the fundus classifier."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.save_hyperparameters(config.__dict__)
        self.backbone = FundusBackbone(feature_dim=config.feature_dim)
        # Metadata conditioner removed - not using age/sex
        self.classifier = MultiLabelClassifier(config.feature_dim, config.num_classes)
        pos_weight = getattr(config, "pos_weight", None)
        if pos_weight is not None:
            weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
        else:
            weight_tensor = None
        # Use enhanced focal loss with higher gamma for rare diseases
        # gamma=2.0 for common (DR, Glaucoma, Other)
        # gamma_rare=3.0 for rare (Cataract, AMD, HTN, Myopia)
        self.criterion = FocalLoss(alpha=weight_tensor, gamma=2.0, gamma_rare=3.0)
        self._val_logits: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []
        self._val_losses: list[torch.Tensor] = []

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:  # type: ignore[override]
        features = self.backbone(batch["image"])
        # Skip metadata conditioning - use features directly
        return self.classifier(features)

    def training_step(self, batch: dict[str, Any], _: int) -> torch.Tensor:  # type: ignore[override]
        logits = self.forward(batch)
        loss = self.criterion(logits, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch["labels"].size(0))
        return loss

    def validation_step(self, batch: dict[str, Any], _: int) -> None:  # type: ignore[override]
        logits = self.forward(batch)
        loss = self.criterion(logits, batch["labels"])
        self._val_logits.append(logits.detach().cpu())
        self._val_labels.append(batch["labels"].detach().cpu())
        self._val_losses.append(loss.detach().cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch["labels"].size(0))

    def on_validation_epoch_start(self) -> None:  # type: ignore[override]
        self._val_logits.clear()
        self._val_labels.clear()
        self._val_losses.clear()

    def on_validation_epoch_end(self) -> None:  # type: ignore[override]
        if not self._val_logits:
            return
        logits = torch.cat(self._val_logits)
        targets = torch.cat(self._val_labels)
        mean_loss = torch.stack(self._val_losses).mean().item()
        metrics = self._compute_f1_metrics(logits, targets)
        macro_f1 = float(metrics["macro_f1"])
        per_class_f1 = metrics["per_class_f1"]
        self.log("val_macro_f1", macro_f1, prog_bar=True, on_epoch=True)
        for idx, score in enumerate(per_class_f1):
            self.log(f"val_f1_class_{idx}", float(score), prog_bar=False, on_epoch=True)
        per_class_str = ", ".join(f"{score:.3f}" for score in per_class_f1)
        self.print(
            f"Epoch {self.current_epoch}: val_loss={mean_loss:.4f} val_macro_f1={macro_f1:.4f} per_class_f1=[{per_class_str}]"
        )

    @staticmethod
    def _compute_f1_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).to(dtype=torch.int32)
        labels = targets.to(dtype=torch.int32)
        tp = (preds & labels).sum(dim=0).to(dtype=torch.float32)
        fp = (preds & (1 - labels)).sum(dim=0).to(dtype=torch.float32)
        fn = ((1 - preds) & labels).sum(dim=0).to(dtype=torch.float32)
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        macro_f1 = f1.mean()
        return {"macro_f1": macro_f1.item(), "per_class_f1": f1.tolist()}

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def train_model(config: TrainingConfig, datamodule: Any) -> None:
    """Train the model with MLflow logging enabled."""
    logger = MLFlowLogger(experiment_name="fundus-odir")
    config.model_dir.mkdir(parents=True, exist_ok=True)
    callbacks = []
    if config.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_macro_f1",
                patience=config.early_stopping_patience,
                mode="max",
                min_delta=1e-4,
            )
        )

    trainer = Trainer(
        max_epochs=config.epochs,
        logger=logger,
        accelerator=config.accelerator,
        devices=config.devices,
        precision=config.precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=config.log_every_n_steps,
        num_sanity_val_steps=config.num_sanity_val_steps,
        callbacks=callbacks,
    )
    model = FundusLightningModule(config)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.save_checkpoint(str(config.model_dir / config.ckpt_name))

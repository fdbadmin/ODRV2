from __future__ import annotations

from typing import Any

import timm  # type: ignore[import]
import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]


class FundusBackbone(nn.Module):
    """Backbone extracting features from a single fundus image."""

    def __init__(self, model_name: str = "convnext_base", pretrained: bool = True, feature_dim: int = 1024) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.projection = nn.Linear(self.backbone.num_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        return self.projection(features)

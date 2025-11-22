from __future__ import annotations

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]


class MetadataConditioner(nn.Module):
    """Fuse age and sex metadata with visual embeddings."""

    def __init__(self, in_features: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_features),
            nn.Sigmoid(),
        )

    def forward(self, embedding: torch.Tensor, age: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        metadata = torch.stack([age, sex], dim=1)
        gamma = self.mlp(metadata)
        return embedding * (1 + gamma)


class MultiLabelClassifier(nn.Module):
    """Predict disease probabilities for seven ocular classes."""

    def __init__(self, feature_dim: int, num_classes: int = 7) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(features))

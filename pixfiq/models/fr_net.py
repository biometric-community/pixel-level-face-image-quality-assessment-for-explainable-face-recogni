"""ResNet-18 face embedder for model-specific FIQ."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FaceEmbedder(nn.Module):
    def __init__(self, n_classes: int, emb_dim: int = 128, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-1])
        # SER-FIQ (Eq. 1) needs dropout before the embedding layer
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.classifier = nn.Linear(emb_dim, n_classes)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(x), 1)

    def head(self, features: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.fc(self.dropout(features)))
        return F.normalize(x, p=2, dim=1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def forward(self, x: torch.Tensor):
        emb = self.embed(x)
        return self.classifier(emb), emb

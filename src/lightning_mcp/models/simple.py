from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleClassifier(pl.LightningModule):
    """A minimal LightningModule for testing MCP integration.

    This model is intentionally simple:
    - linear layer
    - synthetic data
    - CPU/GPU agnostic
    """

    def __init__(self, input_dim: int = 4, num_classes: int = 3, lr: float = 1e-3):  # noqa: ARG002
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Linear(input_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, _batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        x = torch.randn(64, self.hparams.input_dim)
        y = torch.randint(0, self.hparams.num_classes, (64,))
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=8)

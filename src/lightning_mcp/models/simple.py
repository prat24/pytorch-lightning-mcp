"""Simple classifier model for testing MCP integration."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleClassifier(pl.LightningModule):
    """A minimal LightningModule for testing MCP integration.

    This model is intentionally simple:
    - Single linear layer
    - Synthetic data generation
    - CPU/GPU/MPS agnostic
    - Supports all Lightning operations: train, validate, test, predict
    """

    def __init__(self, input_dim: int = 4, num_classes: int = 3, lr: float = 1e-3):  # noqa: ARG002
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Linear(input_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # type: ignore[no-any-return]

    def training_step(self, batch, _batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, _batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", acc, prog_bar=False)
        return loss

    def test_step(self, batch, _batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=False)
        self.log("test_acc", acc, prog_bar=False)
        return loss

    def predict_step(self, batch, _batch_idx: int):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)  # type: ignore[attr-defined]

    def _make_dataset(self, n_samples: int = 64) -> TensorDataset:
        """Create synthetic dataset for training/eval."""
        x = torch.randn(n_samples, self.hparams.input_dim)  # type: ignore[attr-defined]
        y = torch.randint(0, self.hparams.num_classes, (n_samples,))  # type: ignore[attr-defined]
        return TensorDataset(x, y)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._make_dataset(64), batch_size=8)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._make_dataset(32), batch_size=8)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._make_dataset(32), batch_size=8)

    def predict_dataloader(self) -> DataLoader:
        # For prediction, we only need inputs (no labels)
        x = torch.randn(16, self.hparams.input_dim)  # type: ignore[attr-defined]
        return DataLoader(TensorDataset(x), batch_size=8)

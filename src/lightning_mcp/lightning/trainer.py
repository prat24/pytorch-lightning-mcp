from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning import Trainer


class LightningTrainerService:
    """Thin, explicit wrapper around PyTorch Lightning Trainer.

    This layer exists to:
    - isolate third-party APIs
    - centralize Trainer configuration
    - provide a stable interface for MCP handlers
    """

    def __init__(self, **trainer_kwargs: Any) -> None:
        self._trainer = Trainer(**trainer_kwargs)

    @property
    def trainer(self) -> Trainer:
        """Expose the underlying Trainer when needed (read-only)."""
        return self._trainer

    def fit(self, model: pl.LightningModule) -> None:
        """Run training."""
        self._trainer.fit(model)

    def validate(self, model: pl.LightningModule) -> None:
        """Run validation."""
        self._trainer.validate(model)

    def test(self, model: pl.LightningModule) -> None:
        """Run testing."""
        self._trainer.test(model)

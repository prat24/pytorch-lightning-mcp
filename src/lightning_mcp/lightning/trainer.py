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

    Note:
        Progress bar and logger are disabled by default to prevent
        polluting stdout when used in MCP server context.
    """

    def __init__(self, **trainer_kwargs: Any) -> None:
        # Disable progress bar and logger by default for MCP server use
        # These can be overridden by explicit user config if needed
        defaults = {
            "enable_progress_bar": False,
            "logger": False,
            "enable_model_summary": False,
        }
        # User-provided kwargs take precedence
        merged_kwargs = {**defaults, **trainer_kwargs}
        self._trainer = Trainer(**merged_kwargs)

    @property
    def trainer(self) -> Trainer:
        """Expose the underlying Trainer when needed (read-only)."""
        return self._trainer

    def fit(self, model: pl.LightningModule) -> None:
        """Run training."""
        self._trainer.fit(model)

    def validate(self, model: pl.LightningModule) -> list[dict[str, Any]]:
        """Run validation."""
        return self._trainer.validate(model, verbose=False)

    def test(self, model: pl.LightningModule) -> list[dict[str, Any]]:
        """Run testing."""
        return self._trainer.test(model, verbose=False)

    def predict(self, model: pl.LightningModule, dataloaders: Any = None) -> list[Any]:
        """Run prediction."""
        return self._trainer.predict(model, dataloaders=dataloaders)

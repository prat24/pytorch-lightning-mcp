from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from lightning_mcp.handlers.train import _load_model  # noqa: F401


class BaseHandler:
    """Shared utilities for MCP handlers."""

    def load_model(self, params: dict[str, Any]) -> pl.LightningModule:
        return _load_model(params)

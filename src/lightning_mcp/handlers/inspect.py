from __future__ import annotations
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary

from lightning_mcp.protocol import MCPRequest, MCPResponse
from lightning_mcp.handlers.train import TrainHandler


class InspectHandler:
    """Handle MCP inspection requests.

    Inspection is read-only and has no side effects.
    """

    def handle(self, request: MCPRequest) -> MCPResponse:
        params = request.params
        what = params.get("what")

        if what is None:
            raise ValueError("Missing 'what' field in inspect params")

        # Reuse the SAME model-loading logic as training
        model = self._load_model(params)

        if what == "model_summary":
            result = self._model_summary(model)
        elif what == "num_parameters":
            result = self._num_parameters(model)
        elif what == "hyperparameters":
            result = self._hyperparameters(model)
        else:
            raise ValueError(f"Unknown inspect target '{what}'")

        return MCPResponse(
            id=request.id,
            result=result,
        )

    # --- helpers ---

    def _load_model(self, params: dict[str, Any]) -> pl.LightningModule:
        """Reuse TrainHandler model loading logic."""
        return TrainHandler()._load_model(params)

    def _model_summary(self, model: pl.LightningModule) -> dict[str, Any]:
        summary = ModelSummary(model, max_depth=2)
        return {
            "summary": str(summary),
        }

    def _num_parameters(self, model: pl.LightningModule) -> dict[str, int]:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "total": total,
            "trainable": trainable,
        }

    def _hyperparameters(self, model: pl.LightningModule) -> dict[str, Any]:
        return dict(model.hparams)
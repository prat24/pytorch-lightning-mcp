from __future__ import annotations
import sys
from typing import Any
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.model_summary import ModelSummary

from lightning_mcp.protocol import MCPRequest, MCPResponse
from lightning_mcp.handlers.train import _load_model

class InspectHandler:
    """Production-grade inspection handler (read-only)."""

    def handle(self, request: MCPRequest) -> MCPResponse:
        params = request.params
        what = params.get("what")

        if not isinstance(what, str):
            raise ValueError("Inspect requires 'what' field")

        if what == "model":
            result = self._inspect_model(params)
        elif what == "environment":
            result = self._inspect_environment()
        elif what == "summary":
            result = self._inspect_summary(params)
        else:
            raise ValueError(f"Unknown inspect target '{what}'")

        return MCPResponse(
            id=request.id,
            result=result,
        )

    def _inspect_model(self, params: dict[str, Any]) -> dict[str, Any]:
        model = _load_model(params)
        return {
            "class": model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "hyperparameters": dict(model.hparams),
        }

    def _inspect_summary(self, params: dict[str, Any]) -> dict[str, str]:
        model = _load_model(params)
        summary = ModelSummary(model, max_depth=2)
        return {"summary": str(summary)}

    def _inspect_environment(self) -> dict[str, Any]:
        return {
            "python": sys.version,
            "torch": torch.__version__,
            "lightning": pl.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        }
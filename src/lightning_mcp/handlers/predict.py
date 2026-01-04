"""Predict handler for PyTorch Lightning models."""

from __future__ import annotations

from typing import Any

import torch

from lightning_mcp.handlers.base import build_tool_response, load_model, suppress_output
from lightning_mcp.lightning.trainer import LightningTrainerService
from lightning_mcp.protocol import MCPRequest, MCPResponse


class PredictHandler:
    """Handler for model prediction/inference."""

    def handle(self, request: MCPRequest) -> MCPResponse:
        params = request.params
        model = load_model(params)
        trainer_service = self._load_trainer(params)

        with suppress_output():
            predictions = trainer_service.predict(model)

        # Convert predictions to serializable format
        serialized = self._serialize_predictions(predictions)

        result = {
            "status": "completed",
            "model": {
                "class": model.__class__.__name__,
            },
            "predictions": serialized,
            "num_batches": len(predictions) if predictions else 0,
        }

        return build_tool_response(request.id, result)

    def _load_trainer(self, params: dict[str, Any]) -> LightningTrainerService:
        cfg = params.get("trainer", {})
        if not isinstance(cfg, dict):
            raise TypeError("'trainer' must be a dict")
        return LightningTrainerService(**cfg)

    def _serialize_predictions(self, predictions: list[Any] | None) -> list[Any]:
        """Convert predictions to JSON-serializable format."""
        if not predictions:
            return []

        result = []
        for batch in predictions:
            if isinstance(batch, torch.Tensor):
                result.append(batch.tolist())
            elif isinstance(batch, list):
                result.append([
                    t.tolist() if isinstance(t, torch.Tensor) else t
                    for t in batch
                ])
            else:
                result.append(batch)
        return result

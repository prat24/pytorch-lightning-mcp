from __future__ import annotations

from typing import Any

from lightning_mcp.handlers.base import build_tool_response, load_model, suppress_output
from lightning_mcp.lightning.trainer import LightningTrainerService
from lightning_mcp.protocol import MCPRequest, MCPResponse


class TrainHandler:
    """Production-grade Lightning training handler."""

    def handle(self, request: MCPRequest) -> MCPResponse:
        params = request.params
        model = load_model(params)
        trainer_service = self._load_trainer(params)

        with suppress_output():
            trainer_service.fit(model)

        trainer = trainer_service.trainer

        # Extract metrics
        metrics = {}
        for k, v in trainer.callback_metrics.items():
            if hasattr(v, "item"):
                metrics[k] = float(v.item())
            elif isinstance(v, (int, float)):
                metrics[k] = float(v)

        result = {
            "status": "completed",
            "model": {
                "class": model.__class__.__name__,
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "hyperparameters": dict(model.hparams) if hasattr(model, "hparams") else {},
            },
            "trainer": {
                "max_epochs": trainer.max_epochs,
                "accelerator": trainer.accelerator.__class__.__name__,
                "devices": trainer.num_devices,
            },
            "metrics": metrics,
        }

        return build_tool_response(request.id, result)

    def _load_trainer(self, params: dict[str, Any]) -> LightningTrainerService:
        cfg = params.get("trainer", {})
        if not isinstance(cfg, dict):
            raise TypeError("'trainer' must be a dict")

        return LightningTrainerService(**cfg)

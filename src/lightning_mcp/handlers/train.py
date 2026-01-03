from __future__ import annotations

import importlib
from typing import Any

import pytorch_lightning as pl

from lightning_mcp.lightning.trainer import LightningTrainerService
from lightning_mcp.protocol import MCPRequest, MCPResponse


def _load_model(params: dict[str, Any]) -> pl.LightningModule:
    if "model" not in params:
        raise ValueError("Missing 'model' configuration")

    cfg = params["model"]
    if not isinstance(cfg, dict):
        raise TypeError("'model' must be a dict")

    target = cfg.get("_target_")
    if not isinstance(target, str):
        raise ValueError("'model._target_' must be a string")

    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if not isinstance(cls, type):
        raise TypeError(f"{target} is not a class")

    if not issubclass(cls, pl.LightningModule):
        raise TypeError(f"{target} is not a LightningModule")

    kwargs = {k: v for k, v in cfg.items() if k != "_target_"}
    return cls(**kwargs)

class TrainHandler:
    """Production-grade Lightning training handler."""

    def handle(self, request: MCPRequest) -> MCPResponse:
        params = request.params

        model = _load_model(params)
        trainer_service = self._load_trainer(params)

        trainer_service.fit(model)
        trainer = trainer_service.trainer

        metrics = {
            k: float(v)
            for k, v in trainer.callback_metrics.items()
            if hasattr(v, "item")
        }

        result = {
            "status": "completed",
            "model": {
                "class": model.__class__.__name__,
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "hyperparameters": dict(model.hparams),
            },
            "trainer": {
                "max_epochs": trainer.max_epochs,
                "accelerator": trainer.accelerator.__class__.__name__,
                "devices": trainer.num_devices,
            },
            "metrics": metrics,
        }

        return MCPResponse(
            id=request.id,
            result=result,
        )

    def _load_trainer(self, params: dict[str, Any]) -> LightningTrainerService:
        cfg = params.get("trainer", {})
        if not isinstance(cfg, dict):
            raise TypeError("'trainer' must be a dict")

        return LightningTrainerService(**cfg)

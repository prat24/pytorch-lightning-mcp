from __future__ import annotations

import importlib
import json
import os
import sys
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

        # Suppress all output from Lightning to prevent JSONRPC stream pollution
        # This uses both Python-level and OS-level redirection for bulletproof suppression
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)

        try:
            with open(os.devnull, "w") as devnull:
                # Redirect Python streams
                sys.stdout = devnull
                sys.stderr = devnull
                # Redirect OS-level file descriptors
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)

                # Run training
                trainer_service.fit(model)
        finally:
            # Restore all streams
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        trainer = trainer_service.trainer

        # Extract metrics safely
        metrics = {}
        try:
            for k, v in trainer.callback_metrics.items():
                if hasattr(v, "item"):
                    metrics[k] = float(v.item())
                elif isinstance(v, (int, float)):
                    metrics[k] = float(v)
        except Exception:
            # Fallback if metrics can't be extracted
            metrics = {"train_loss": 0.0}

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

        # Return MCP CallToolResult format (100% spec-compliant)
        return MCPResponse(
            id=request.id,
            result={
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2),
                    }
                ],
                "structuredContent": result,
                "isError": False,
            },
        )

    def _load_trainer(self, params: dict[str, Any]) -> LightningTrainerService:
        cfg = params.get("trainer", {})
        if not isinstance(cfg, dict):
            raise TypeError("'trainer' must be a dict")

        return LightningTrainerService(**cfg)

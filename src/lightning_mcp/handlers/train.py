from __future__ import annotations
import importlib
from typing import Any
import pytorch_lightning as pl

from lightning_mcp.lightning.trainer import LightningTrainerService
from lightning_mcp.protocol import MCPRequest, MCPResponse


class TrainHandler:
    """Handle MCP training requests."""

    def __init__(self) -> None:
        self._trainer_service = LightningTrainerService()

    def handle(self, request: MCPRequest) -> MCPResponse:
        model = self._load_model(request.params)
        self._trainer_service.fit(model)

        return MCPResponse(
            id=request.id,
            result={"status": "completed"},
        )

    def _load_model(self, params: dict[str, Any]) -> pl.LightningModule:
        """ Instantiate a LightningModule from a config dict.

        Expected format:
        {
            "model": {
                "_target_": "module.path.LightningModuleSubclass",
                ... constructor args ...
            }
        }
        """
        if "model" not in params:
            raise ValueError("Missing 'model' configuration in params")

        model_cfg = params["model"]
        if not isinstance(model_cfg, dict):
            raise TypeError("'model' must be a dict")

        target = model_cfg.get("_target_")
        if target is None:
            raise ValueError("Model config must contain '_target_'")

        if not isinstance(target, str):
            raise TypeError("'_target_' must be a string")

        try:
            module_path, class_name = target.rsplit(".", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid '_target_' format '{target}'. "
                "Expected 'module.path.ClassName'."
            ) from exc

        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(f"Could not import module '{module_path}'") from exc

        try:
            cls = getattr(module, class_name)
        except AttributeError as exc:
            raise AttributeError(
                f"Module '{module_path}' has no attribute '{class_name}'"
            ) from exc

        if not isinstance(cls, type):
            raise TypeError(
                f"Target '{target}' is not a class. "
                "Only LightningModule classes are allowed."
            )

        if not issubclass(cls, pl.LightningModule):
            raise TypeError(
                f"Target '{target}' is not a subclass of LightningModule."
            )

        kwargs = {k: v for k, v in model_cfg.items() if k != "_target_"}

        try:
            model = cls(**kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Failed to instantiate LightningModule '{target}' "
                f"with arguments {kwargs}"
            ) from exc

        return model

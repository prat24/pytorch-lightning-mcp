"""Checkpoint handler for PyTorch Lightning models.

Provides save, load, and list operations for model checkpoints.
All operations suppress stdout/stderr to avoid polluting MCP JSON-RPC stream.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from lightning_mcp.handlers.base import build_tool_response, load_model, suppress_output
from lightning_mcp.protocol import MCPRequest, MCPResponse


class CheckpointHandler:
    """Handler for checkpoint operations: save, load, list."""

    def handle(self, request: MCPRequest) -> MCPResponse:
        params = request.params
        action = params.get("action")

        if not isinstance(action, str):
            raise ValueError("'action' is required (save, load, list)")

        if action == "save":
            result = self._save(params)
        elif action == "load":
            result = self._load(params)
        elif action == "list":
            result = self._list(params)
        else:
            raise ValueError(f"Unknown action: {action}")

        return build_tool_response(request.id, result)

    def _save(self, params: dict[str, Any]) -> dict[str, Any]:
        """Save model checkpoint.

        Args:
            params: Must contain 'path' and 'model' configuration.

        Returns:
            Dict with action, path, model_class, and num_parameters.
        """
        path = params.get("path")
        if not isinstance(path, str):
            raise ValueError("'path' is required for save")

        with suppress_output():
            model = load_model(params)

            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), path)

        return {
            "action": "save",
            "path": path,
            "model_class": model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
        }

    def _load(self, params: dict[str, Any]) -> dict[str, Any]:
        """Load model from checkpoint.

        Args:
            params: Must contain 'path' and 'model' configuration.

        Returns:
            Dict with action, path, model_class, and num_parameters.
        """
        path = params.get("path")
        if not isinstance(path, str):
            raise ValueError("'path' is required for load")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with suppress_output():
            model = load_model(params)
            state_dict = torch.load(path, weights_only=True)
            model.load_state_dict(state_dict)

        return {
            "action": "load",
            "path": path,
            "model_class": model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
        }

    def _list(self, params: dict[str, Any]) -> dict[str, Any]:
        """List checkpoints in directory."""
        directory = params.get("directory", ".")

        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Not a directory: {directory}")

        checkpoints = []
        for f in os.listdir(directory):
            if f.endswith((".ckpt", ".pt", ".pth")):
                full_path = os.path.join(directory, f)
                checkpoints.append({
                    "name": f,
                    "path": full_path,
                    "size_bytes": os.path.getsize(full_path),
                })

        return {
            "action": "list",
            "directory": directory,
            "checkpoints": checkpoints,
            "count": len(checkpoints),
        }

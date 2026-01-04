"""Base utilities for Lightning MCP handlers.

Shared code for model loading and MCP response building.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from contextlib import contextmanager
from typing import Any, Generator

import pytorch_lightning as pl

from lightning_mcp.protocol import MCPRequest, MCPResponse


def load_model(params: dict[str, Any]) -> pl.LightningModule:
    """Load a LightningModule from params.

    Args:
        params: Must contain 'model' dict with '_target_' key.

    Returns:
        Instantiated LightningModule.

    Raises:
        ValueError: If model config is missing or invalid.
        TypeError: If target is not a LightningModule.
    """
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


def build_tool_response(request_id: str, result: dict[str, Any]) -> MCPResponse:
    """Build MCP CallToolResult response.
    """
    return MCPResponse(
        id=request_id,
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


@contextmanager
def suppress_output() -> Generator[None, None, None]:
    """Suppress stdout/stderr to prevent polluting JSON-RPC stream.

    Uses both Python-level and OS-level redirection.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        sys.stdout = old_stdout
        sys.stderr = old_stderr

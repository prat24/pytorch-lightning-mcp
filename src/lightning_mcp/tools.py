from __future__ import annotations

from typing import Any


def list_tools() -> list[dict[str, Any]]:
    """
    Return the list of tools supported by this MCP server.

    This is a declarative description only.
    Execution is handled by existing MCP handlers.
    """

    return [
        {
            "name": "lightning.train",
            "description": "Train a PyTorch Lightning model with explicit configuration.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "object",
                        "description": "Model configuration (_target_ + kwargs).",
                    },
                    "trainer": {
                        "type": "object",
                        "description": "Trainer configuration.",
                    },
                },
                "required": ["model"],
            },
        },
        {
            "name": "lightning.inspect",
            "description": "Inspect models or runtime environment.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "what": {
                        "type": "string",
                        "description": "Inspection target (model, environment, summary).",
                    },
                    "model": {
                        "type": "object",
                        "description": "Model configuration (required for model inspection).",
                    },
                },
                "required": ["what"],
            },
        },
        {
            "name": "lightning.validate",
            "description": "Validate a PyTorch Lightning model.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "object",
                        "description": "Model configuration (_target_ + kwargs).",
                    },
                    "trainer": {
                        "type": "object",
                        "description": "Trainer configuration.",
                    },
                },
                "required": ["model"],
            },
        },
        {
            "name": "lightning.test",
            "description": "Test a PyTorch Lightning model.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "object",
                        "description": "Model configuration (_target_ + kwargs).",
                    },
                    "trainer": {
                        "type": "object",
                        "description": "Trainer configuration.",
                    },
                },
                "required": ["model"],
            },
        },
    ]

import pytest

from lightning_mcp.handlers.inspect import InspectHandler
from lightning_mcp.protocol import MCPRequest


def test_inspect_unknown_target():
    """
    InspectHandler must reject unknown inspection targets.
    """

    handler = InspectHandler()

    request = MCPRequest(
        id="inspect-bad",
        method="lightning.inspect",
        params={
            "what": "does-not-exist",
        },
    )

    with pytest.raises(ValueError):
        handler.handle(request)


def test_inspect_model_missing_config():
    """
    InspectHandler must fail if model inspection is requested without model config.
    """

    handler = InspectHandler()

    request = MCPRequest(
        id="inspect-missing-model",
        method="lightning.inspect",
        params={
            "what": "model",
        },
    )

    with pytest.raises(ValueError):
        handler.handle(request)
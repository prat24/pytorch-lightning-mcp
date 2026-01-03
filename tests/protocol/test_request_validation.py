import pytest
from pydantic import ValidationError

from lightning_mcp.protocol import MCPRequest


def test_valid_request_minimal():
    request = MCPRequest(
        id="1",
        method="lightning.inspect",
        params={},
    )

    assert request.id == "1"
    assert request.method == "lightning.inspect"
    assert request.params == {}


def test_request_requires_id():
    with pytest.raises(ValidationError):
        MCPRequest(
            method="lightning.inspect",
            params={},
        )


def test_request_requires_method():
    with pytest.raises(ValidationError):
        MCPRequest(
            id="1",
            params={},
        )


def test_request_params_defaults_to_empty_dict():
    request = MCPRequest(
        id="1",
        method="lightning.inspect",
    )

    assert isinstance(request.params, dict)
    assert request.params == {}

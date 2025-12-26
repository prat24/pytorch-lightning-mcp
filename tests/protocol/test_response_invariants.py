import pytest
from pydantic import ValidationError

from lightning_mcp.protocol import MCPResponse, MCPError


def test_response_with_result_only():
    response = MCPResponse(
        id="1",
        result={"ok": True},
    )

    assert response.id == "1"
    assert response.result == {"ok": True}
    assert response.error is None


def test_response_with_error_only():
    error = MCPError(
        code=400,
        message="bad request",
    )

    response = MCPResponse(
        id="2",
        error=error,
    )

    assert response.id == "2"
    assert response.error == error
    assert response.result is None


def test_response_never_has_result_and_error():
    """
    Protocol invariant:
    A response must contain either `result` or `error`, never both.
    """

    error = MCPError(code=500, message="boom")

    with pytest.raises(ValidationError):
        MCPResponse(
            id="3",
            result={"ok": True},
            error=error,
        )
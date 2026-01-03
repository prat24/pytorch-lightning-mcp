from lightning_mcp.handlers.inspect import InspectHandler
from lightning_mcp.protocol import MCPRequest


def test_inspect_model_metadata():
    """
    InspectHandler should return structured model metadata.
    """

    handler = InspectHandler()

    request = MCPRequest(
        id="inspect-model",
        method="lightning.inspect",
        params={
            "what": "model",
            "model": {
                "_target_": "lightning_mcp.models.simple.SimpleClassifier",
                "input_dim": 4,
                "num_classes": 3,
            },
        },
    )

    response = handler.handle(request)

    assert response.id == "inspect-model"
    assert response.error is None

    # Extract from CallToolResult format
    result = response.result
    assert "content" in result
    assert "structuredContent" in result
    assert "isError" in result
    assert result["isError"] is False

    structured = result["structuredContent"]
    assert structured["class"] == "SimpleClassifier"
    assert structured["num_parameters"] > 0
    assert "trainable_parameters" in structured
    assert "hyperparameters" in structured


def test_inspect_environment():
    """
    InspectHandler should return environment information.
    """

    handler = InspectHandler()

    request = MCPRequest(
        id="inspect-env",
        method="lightning.inspect",
        params={
            "what": "environment",
        },
    )

    response = handler.handle(request)

    assert response.id == "inspect-env"
    assert response.error is None

    # Extract from CallToolResult format
    result = response.result
    assert "content" in result
    assert "structuredContent" in result
    assert "isError" in result
    assert result["isError"] is False

    structured = result["structuredContent"]
    assert "python" in structured
    assert "torch" in structured
    assert "lightning" in structured
    assert isinstance(structured["cuda_available"], bool)

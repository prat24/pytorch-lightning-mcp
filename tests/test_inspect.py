from lightning_mcp.handlers.inspect import InspectHandler
from lightning_mcp.protocol import MCPRequest


def test_inspect_model_metadata():
    handler = InspectHandler()

    request = MCPRequest(
        id="inspect-model",
        method="lightning/inspect",
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

    assert response.error is None
    result = response.result

    assert result["class"] == "SimpleClassifier"
    assert result["num_parameters"] > 0
    assert "trainable_parameters" in result
    assert "hyperparameters" in result


def test_inspect_environment():
    handler = InspectHandler()

    request = MCPRequest(
        id="inspect-env",
        method="lightning/inspect",
        params={"what": "environment"},
    )

    response = handler.handle(request)

    assert response.error is None
    result = response.result

    assert "python" in result
    assert "torch" in result
    assert "lightning" in result
    assert isinstance(result["cuda_available"], bool)


def test_inspect_summary():
    handler = InspectHandler()

    request = MCPRequest(
        id="inspect-summary",
        method="lightning/inspect",
        params={
            "what": "summary",
            "model": {
                "_target_": "lightning_mcp.models.simple.SimpleClassifier",
            },
        },
    )

    response = handler.handle(request)

    assert response.error is None
    assert "summary" in response.result
    assert isinstance(response.result["summary"], str)
import pytest

from lightning_mcp.handlers.train import TrainHandler
from lightning_mcp.protocol import MCPRequest


def test_train_simple_model():
    handler = TrainHandler()

    request = MCPRequest(
        id="test-train",
        method="lightning/train",
        params={
            "model": {
                "_target_": "lightning_mcp.models.simple.SimpleClassifier",
                "input_dim": 4,
                "num_classes": 3,
            },
            "trainer": {
                "max_epochs": 1,
                "accelerator": "cpu",
            },
        },
    )

    response = handler.handle(request)

    assert response.id == "test-train"
    assert response.error is None

    result = response.result
    assert result["status"] == "completed"

    # Model metadata
    assert result["model"]["class"] == "SimpleClassifier"
    assert result["model"]["num_parameters"] > 0
    assert "hyperparameters" in result["model"]

    # Trainer metadata
    assert result["trainer"]["max_epochs"] == 1
    assert "accelerator" in result["trainer"]

    # Metrics exist (may be empty but must exist)
    assert "metrics" in result
    assert isinstance(result["metrics"], dict)


def test_train_invalid_model_target():
    handler = TrainHandler()

    request = MCPRequest(
        id="bad-train",
        method="lightning/train",
        params={
            "model": {
                "_target_": "math.sqrt",
            }
        },
    )

    with pytest.raises(TypeError):
        handler.handle(request)
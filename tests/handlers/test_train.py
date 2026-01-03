from lightning_mcp.handlers.train import TrainHandler
from lightning_mcp.protocol import MCPRequest


def test_train_simple_model_cpu():
    """
    Happy-path test for TrainHandler.

    This test verifies that:
    - a valid LightningModule can be instantiated
    - training executes without error
    - a structured MCPResponse is returned
    """

    handler = TrainHandler()

    request = MCPRequest(
        id="train-test-1",
        method="lightning.train",
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

    assert response.id == "train-test-1"
    assert response.error is None

    # Extract from CallToolResult format
    result = response.result
    assert "content" in result
    assert "structuredContent" in result
    assert "isError" in result
    assert result["isError"] is False

    structured = result["structuredContent"]
    assert structured["status"] == "completed"

    # Model metadata
    assert structured["model"]["class"] == "SimpleClassifier"
    assert structured["model"]["num_parameters"] > 0
    assert "hyperparameters" in structured["model"]

    # Trainer metadata
    assert structured["trainer"]["max_epochs"] == 1
    assert structured["trainer"]["devices"] == 1

    # Metrics must exist (values may vary)
    assert "metrics" in structured
    assert isinstance(structured["metrics"], dict)

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

    result = response.result
    assert result["status"] == "completed"

    # Model metadata
    assert result["model"]["class"] == "SimpleClassifier"
    assert result["model"]["num_parameters"] > 0
    assert "hyperparameters" in result["model"]

    # Trainer metadata
    assert result["trainer"]["max_epochs"] == 1
    assert result["trainer"]["devices"] == 1

    # Metrics must exist (values may vary)
    assert "metrics" in result
    assert isinstance(result["metrics"], dict)
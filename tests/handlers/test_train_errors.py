import pytest

from lightning_mcp.handlers.train import TrainHandler
from lightning_mcp.protocol import MCPRequest


def test_train_rejects_non_lightning_module():
    """
    TrainHandler must reject targets that are not LightningModule subclasses.
    """

    handler = TrainHandler()

    request = MCPRequest(
        id="bad-model",
        method="lightning/train",
        params={
            "model": {
                "_target_": "math.sqrt"
            }
        },
    )

    with pytest.raises(TypeError):
        handler.handle(request)


def test_train_missing_model_config():
    """
    TrainHandler must fail fast if model config is missing.
    """

    handler = TrainHandler()

    request = MCPRequest(
        id="missing-model",
        method="lightning/train",
        params={},
    )

    with pytest.raises(ValueError):
        handler.handle(request)
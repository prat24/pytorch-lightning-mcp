"""
Pytest configuration and shared fixtures for the test suite.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning_mcp.models.simple import SimpleClassifier
from lightning_mcp.protocol import MCPRequest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataset():
    """Create a small synthetic dataset for testing."""
    X = torch.randn(100, 4)
    y = torch.randint(0, 3, (100,))
    dataset = TensorDataset(X, y)
    return dataset


@pytest.fixture
def sample_dataloader(sample_dataset):
    """Create a dataloader from the sample dataset."""
    return DataLoader(sample_dataset, batch_size=16, shuffle=True)


@pytest.fixture
def simple_model():
    """Create an instance of SimpleClassifier for testing."""
    return SimpleClassifier(input_dim=4, num_classes=3)


@pytest.fixture
def trainer_config():
    """Provide a standard trainer configuration for tests."""
    return {
        "max_epochs": 1,
        "accelerator": "cpu",
        "devices": 1,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }


@pytest.fixture
def mcp_request_factory():
    """Factory fixture for creating MCPRequest objects."""
    def _create_request(
        method: str,
        params: dict,
        request_id: str = "test-request-1",
    ) -> MCPRequest:
        return MCPRequest(
            id=request_id,
            method=method,
            params=params,
        )
    return _create_request


@pytest.fixture
def train_request_factory(mcp_request_factory):
    """Factory for creating train requests."""
    def _create_train_request(
        model_config: dict = None,
        trainer_config: dict = None,
        request_id: str = "train-test-1",
    ) -> MCPRequest:
        if model_config is None:
            model_config = {
                "_target_": "lightning_mcp.models.simple.SimpleClassifier",
                "input_dim": 4,
                "num_classes": 3,
            }
        if trainer_config is None:
            trainer_config = {
                "max_epochs": 1,
                "accelerator": "cpu",
            }

        return mcp_request_factory(
            method="lightning.train",
            params={
                "model": model_config,
                "trainer": trainer_config,
            },
            request_id=request_id,
        )
    return _create_train_request


@pytest.fixture
def inspect_request_factory(mcp_request_factory):
    """Factory for creating inspect requests."""
    def _create_inspect_request(
        model_config: dict = None,
        request_id: str = "inspect-test-1",
    ) -> MCPRequest:
        if model_config is None:
            model_config = {
                "_target_": "lightning_mcp.models.simple.SimpleClassifier",
                "input_dim": 4,
                "num_classes": 3,
            }

        return mcp_request_factory(
            method="lightning.inspect",
            params={"model": model_config},
            request_id=request_id,
        )
    return _create_inspect_request

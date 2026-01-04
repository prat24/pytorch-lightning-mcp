"""CLI integration tests - tests the actual server process."""

import json
import subprocess
import sys


def run_mcp_command(request: dict) -> dict:
    """Run a single MCP request through the CLI."""
    result = subprocess.run(
        [sys.executable, "-m", "lightning_mcp.cli"],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
    )

    stdout = result.stdout.strip()

    # Find JSON object in output
    start = stdout.find("{")
    end = stdout.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"No valid JSON object in stdout.\n"
            f"stdout: {stdout!r}\n"
            f"stderr: {result.stderr!r}\n"
            f"returncode: {result.returncode}"
        )

    json_str = stdout[start : end + 1]
    return json.loads(json_str)

def test_cli_initialize():
    """Test initialize via actual CLI process."""
    response = run_mcp_command({
        "id": "cli-init-1",
        "method": "initialize",
        "params": {},
    })

    assert response["id"] == "cli-init-1"
    assert "error" not in response
    assert response["result"]["protocolVersion"] == "2024-11-05"
    assert response["result"]["serverInfo"]["name"] == "lightning-mcp"


def test_cli_tools_list():
    """Test tools/list via actual CLI process."""
    response = run_mcp_command({
        "id": "cli-tools-1",
        "method": "tools/list",
        "params": {},
    })

    assert response["id"] == "cli-tools-1"
    assert "error" not in response

    tools = response["result"]["tools"]
    names = {tool["name"] for tool in tools}

    assert "lightning.train" in names
    assert "lightning.validate" in names
    assert "lightning.test" in names
    assert "lightning.predict" in names
    assert "lightning.inspect" in names


def test_cli_inspect_environment():
    """Test lightning.inspect environment via CLI."""
    response = run_mcp_command({
        "id": "cli-inspect-1",
        "method": "lightning.inspect",
        "params": {"what": "environment"},
    })

    assert response["id"] == "cli-inspect-1"
    assert "error" not in response
    assert response["result"]["isError"] is False

    content = response["result"]["structuredContent"]
    assert "torch" in content
    assert "lightning" in content


def test_cli_train_simple_model():
    """Test lightning.train via CLI."""
    response = run_mcp_command({
        "id": "cli-train-1",
        "method": "lightning.train",
        "params": {
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
    })

    assert response["id"] == "cli-train-1"
    assert "error" not in response
    assert response["result"]["isError"] is False

    content = response["result"]["structuredContent"]
    assert content["status"] == "completed"
    assert content["model"]["class"] == "SimpleClassifier"
    assert "train_loss" in content["metrics"]

# PyTorch Lightning MCP (Model Context Protocol)

An integration layer that exposes **PyTorch Lightning training and inspection** through a structured, machine-readable API.

Intended for programmatic use by tools, agents, and orchestration systems.

## Features

* Structured training and inspection APIs
* Real PyTorch Lightning execution (no mocks)
* Explicit, config-driven behavior
* Safe model instantiation
* Stdio and HTTP servers
* Fully tested core logic
* Clean separation between protocol, capabilities, and transport

## Project Structure

```
src/lightning_mcp/
├── protocol.py          # Request / response schema
├── handlers/
│   ├── train.py         # Training capability
│   └── inspect.py       # Inspection capability
├── lightning/
│   └── trainer.py       # Lightning integration boundary
├── server.py            # Stdio server
├── http_server.py       # HTTP server (FastAPI)
├── models/
│   └── simple.py        # Example LightningModule
tests/
├── test_train.py
└── test_inspect.py
```

## Requirements

* Python 3.10 – 3.12
* PyTorch Lightning (compatible versions)
* uv (recommended)

## Installation (using uv)

### 1. Install uv (if not already installed)

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Restart your shell after installation.

Verify:

```bash
uv --version
```

### 2. Clone the repository

```bash
git clone https://github.com/<your-org>/lightning-mcp.git
cd lightning-mcp
```

### 3. Install dependencies

To install all dependencies (including server extras):

```bash
uv sync --all-extras
```

This will:

* create a local virtual environment
* install PyTorch Lightning and dependencies
* install HTTP server dependencies (FastAPI, Uvicorn)

No manual venv management is required.

## Usage

### Training (in-process)

```python
from lightning_mcp.handlers.train import TrainHandler
from lightning_mcp.protocol import MCPRequest

handler = TrainHandler()

request = MCPRequest(
    id="train-1",
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
```

### Inspection (in-process)

```python
from lightning_mcp.handlers.inspect import InspectHandler
from lightning_mcp.protocol import MCPRequest

handler = InspectHandler()

request = MCPRequest(
    id="inspect-1",
    method="lightning/inspect",
    params={
        "what": "environment"
    },
)

response = handler.handle(request)
```

## Stdio Server

The stdio server reads one JSON request per line from stdin and writes one JSON response per line to stdout.

### Run

```bash
uv run python -m lightning_mcp.server
```

### Example

```bash
echo '{"id":"1","method":"lightning/inspect","params":{"what":"environment"}}' \
| uv run python -m lightning_mcp.server
```

## HTTP Server

The HTTP server exposes a single MCP endpoint.

### Run

```bash
uv run uvicorn lightning_mcp.http_server:app --host 0.0.0.0 --port 3333
```

### Endpoint

```
POST /mcp
```

### Example (curl)

```bash
curl -X POST http://localhost:3333/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "id": "train-http-1",
    "method": "lightning/train",
    "params": {
      "model": {
        "_target_": "lightning_mcp.models.simple.SimpleClassifier",
        "input_dim": 4,
        "num_classes": 3
      },
      "trainer": {
        "max_epochs": 1,
        "accelerator": "cpu"
      }
    }
  }'
```

## Testing

Run the full test suite:

```bash
uv run pytest
```

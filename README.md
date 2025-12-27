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

### Docker (recommended)

You can run the MCP server using Docker

<pre class="overflow-visible! px-0!" data-start="1989" data-end="2178"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"mcpServers"</span><span>:</span><span></span><span>{</span><span>
    </span><span>"Lightning"</span><span>:</span><span></span><span>{</span><span>
      </span><span>"command"</span><span>:</span><span></span><span>"docker"</span><span>,</span><span>
      </span><span>"args"</span><span>:</span><span></span><span>[</span><span>
        </span><span>"run"</span><span>,</span><span>
        </span><span>"--rm"</span><span>,</span><span>
        </span><span>"-i"</span><span>,</span><span>
        </span><span>"lightning-mcp:latest"</span><span>
      </span><span>]</span><span>
    </span><span>}</span><span>
  </span><span>}</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

## MCP Tools

The MCP server exposes a small, explicit set of tools that agents can discover and invoke dynamically.

Tool discovery is available via the standard MCP method:

```json
{
  "method": "tools/list"
}
```

This returns a machine-readable description of all supported tools, including their input schemas.

### Available Tools

#### `lightning.train`

Train a PyTorch Lightning model using an explicit configuration.

This tool allows an agent to:

* instantiate a LightningModule
* configure a Trainer
* execute training in-process
* receive structured training metadata

**Input schema (simplified):**

```json
{
  "model": {
    "_target_": "string",
    "...": "model-specific arguments"
  },
  "trainer": {
    "...": "trainer configuration (optional)"
  }
}
```

The `model` field is required and must reference a valid LightningModule class.

#### `lightning.inspect`

Inspect a model or the runtime environment without performing training.

This tool can be used to:

* inspect model architecture and parameter counts
* inspect the execution environment (Python, Torch, Lightning versions, device availability)

**Input schema (simplified):**

```json
{
  "what": "model | environment",
  "model": {
    "_target_": "string",
    "...": "model-specific arguments (required for model inspection)"
  }
}
```

### Tool Discovery Example

Using MCP stdio:

```bash
echo '{"id":"1","method":"tools/list","params":{}}' \
| uv run python -m lightning_mcp.server
```

Using Docker:

```bash
echo '{"id":"1","method":"tools/list","params":{}}' \
| docker run --rm -i lightning-mcp:latest
```

The response contains a list of tools with their names, descriptions, and input schemas.

## Demo

Below is a quick example of an agent-driven interaction with the MCP server using OpenAI tool calling. The agent decides which MCP tool to use, executes exactly one action, and then summarizes the result.

### Running the demo

Start the MCP server:

<pre class="overflow-visible! px-0!" data-start="536" data-end="597"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>uvicorn lightning_mcp.http_server:app --port 8000
</span></span></code></div></div></pre>

```
INFO:     Started server process [15724]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Run the demo on an agent of choice
For example: running using an **OpenAI-based agent** that adapts MCP for use with OpenAI’s tool-calling interface, we may get:

```
Discovered MCP tools:
  OpenAI: lightning_train  ->  MCP: lightning.train
  OpenAI: lightning_inspect  ->  MCP: lightning.inspect

Agent → MCP: lightning.inspect
Args:
{
  "what": "environment"
}

MCP Result:
{
  "python": "3.11.14 (main, Oct 31 2025, 23:15:22) [Clang 21.1.4 ]",
  "torch": "2.9.1",
  "lightning": "2.6.0",
  "cuda_available": false,
  "mps_available": true
}

The inspection of the environment reveals the following setup:

- Python version: 3.11.14
- PyTorch version: 2.9.1
- PyTorch Lightning version: 2.6.0
- CUDA support: Not available
- Apple MPS (Metal Performance Shaders) support: Available

This setup indicates that the system is equipped for machine learning tasks using PyTorch and PyTorch Lightning on Apple hardware with MPS support for accelerated computing, but without CUDA support.
```

# PyTorch Lightning MCP

An **MCP (Model Context Protocol) server** that exposes **PyTorch Lightning** training and inspection capabilities through a structured, machine-readable interface.

## Features

* MCP-compliant server
* Stdio transport (primary)
* HTTP transport (optional)
* Explicit, config-driven execution
* Safe model instantiation
* Tool discovery via `tools/list`
* Clean separation between protocol, tools, and transport


## Project Structure

```
src/lightning_mcp/
├── protocol.py          # MCP request / response models
├── handlers/
│   ├── train.py         # lightning.train tool
│   └── inspect.py       # lightning.inspect tool
├── lightning/
│   └── trainer.py       # Lightning execution boundary
├── tools.py             # Tool registry
├── server.py            # MCP stdio server
├── http_server.py       # MCP HTTP server (FastAPI)
├── cli.py               # CLI entrypoint
tests/
```

## Requirements

* Python 3.10 – 3.12
* PyTorch ≥ 2.0
* PyTorch Lightning ≥ 2.x
* `uv` (recommended)

## Installation

We recommend that you use uv - an extremely fast Python package and project manager, written in Rust.
Install it using curl: ``` curl -LsSf https://astral.sh/uv/install.sh | sh```

```bash
git clone https://github.com/prat24/pytorch-lightning-mcp
cd pytorch-lightning-mcp
uv sync --all-extras
```

## Running the MCP Server

### Stdio

This is the **primary MCP transport** and the one expected by MCP clients.

```bash
uv run lightning-mcp
```

The server:

* reads one MCP request per line from **stdin**
* writes one MCP response per line to **stdout**

### Example

```bash
echo '{"id":"1","method":"tools/list","params":{}}' \
| uv run lightning-mcp
```

## HTTP Server

The HTTP server exposes the **same MCP interface** over HTTP.

### Run

```bash
uv run lightning-mcp --http --port 3333
```

### Endpoint

```
POST /mcp
```

### Example

```bash
curl -X POST http://localhost:3333/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "id": "inspect-1",
    "method": "lightning.inspect",
    "params": { "what": "environment" }
  }'
```

## MCP Tools

Tool discovery is available via the standard MCP method:

```json
{
  "id": "tools-1",
  "method": "tools/list",
  "params": {}
}
```

The response contains a machine-readable description of all supported tools and their input schemas.


### `lightning.train`

Train a PyTorch Lightning model using an explicit configuration.

**Purpose**

* Instantiate a `LightningModule`
* Configure a `Trainer`
* Execute training in-process
* Return structured metadata

**Input (simplified)**

```json
{
  "model": {
    "_target_": "string",
    "...": "model arguments"
  },
  "trainer": {
    "...": "trainer configuration (optional)"
  }
}
```

### `lightning.inspect`

Inspect a model or the runtime environment without training.

**Supported inspections**

* Environment (Python, Torch, Lightning, devices)
* Model structure and parameters

**Input (simplified)**

```json
{
  "what": "model | environment",
  "model": {
    "_target_": "string",
    "...": "model arguments (required for model inspection)"
  }
}
```

## Docker

The container runs the MCP stdio server.

It:

* reads MCP requests from **stdin**
* writes MCP responses to **stdout**

Example MCP client configuration:

```json
{
  "mcpServers": {
    "Lightning": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "lightning-mcp:latest"
      ]
    }
  }
}
```

## Testing

Run the test suite:

```bash
uv run pytest
```

## Demo

Below is a quick example of an agent-driven interaction with the MCP server using OpenAI tool calling. The agent decides which MCP tool to use, executes exactly one action, and then summarizes the result.

### Running the demo

Start the MCP server:



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

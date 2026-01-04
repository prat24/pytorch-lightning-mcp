# PyTorch Lightning MCP Server

A minimal integration layer exposing PyTorch Lightning via a structured, machine-readable API for tools, agents, and orchestration systems.

## Features

- Structured APIs for training, inspecting, validating, testing, predicting, and checkpointing models
- PyTorch Lightning execution
- Stdio and HTTP server modes

## Requirements

- Python 3.10–3.12
- PyTorch Lightning (compatible version)
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

## Installation

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
git clone https://github.com/<your-org>/lightning-mcp.git
cd lightning-mcp
uv sync --all-extras
```

## Usage

### CLI

You can run the MCP server via CLI:

```bash
# Stdio server (default)
uv run lightning-mcp

# HTTP server
uv run lightning-mcp --http --host 0.0.0.0 --port 3333
```

### Stdio Example

```bash
echo '{"id":"1","method":"lightning.inspect","params":{"what":"environment"}}' | uv run lightning-mcp
```

### HTTP Example

```bash
curl -X POST http://localhost:3333/mcp \
  -H "Content-Type: application/json" \
  -d '{"id":"1","method":"lightning.inspect","params":{"what":"environment"}}'
```

## Available Tools

The MCP server exposes the following tools (methods):

### `lightning.train`

Train a PyTorch Lightning model with explicit configuration.

**Input schema:**

```json
{
  "model": {"_target_": "string", ...},
  "trainer": { ... }
}
```

### `lightning.inspect`

Inspect a model or the runtime environment.

**Input schema:**

```json
{
  "what": "model | environment | summary",
  "model": {"_target_": "string", ...} // required for model inspection
}
```

### `lightning.validate`

Validate a PyTorch Lightning model.

**Input schema:**

```json
{
  "model": {"_target_": "string", ...},
  "trainer": { ... }
}
```

### `lightning.test`

Test a PyTorch Lightning model.

**Input schema:**

```json
{
  "model": {"_target_": "string", ...},
  "trainer": { ... }
}
```

### `lightning.predict`

Run prediction/inference with a PyTorch Lightning model.

**Input schema:**

```json
{
  "model": {"_target_": "string", ...},
  "trainer": { ... }
}
```

### `lightning.checkpoint`

Manage model checkpoints: save, load, or list.

**Input schema:**

```json
{
  "action": "save | load | list",
  "path": "string",         // for save/load
  "directory": "string",    // for list
  "model": { ... }           // for save/load
}
```

## Tool Discovery

To list all available tools and their schemas at runtime:

```bash
echo '{"id":"1","method":"tools/list","params":{}}' | uv run lightning-mcp
```

## Testing

```bash
uv run pytest
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [DEVELOPMENT.md](DEVELOPMENT.md).

## License

Apache 2.0

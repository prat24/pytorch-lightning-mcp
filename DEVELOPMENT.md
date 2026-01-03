# Development Guide

This guide covers the development workflow for PyTorch Lightning MCP.

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **uv** - Fast Python package manager

### Installing uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

### 1. Clone and Enter Project

```bash
git clone https://github.com/pratyushmishra/pytorch-lightning-mcp.git
cd pytorch-lightning-mcp
```

### 2. Install Dependencies

```bash
uv sync --all-extras
```

This installs:
- Core dependencies (torch, pytorch-lightning, pydantic, etc.)
- Dev dependencies (pytest, ruff, mypy)
- Optional server dependencies (fastapi, uvicorn)

### 3. Activate Virtual Environment

```bash
source .venv/bin/activate
```

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/handlers/test_train.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src/lightning_mcp --cov-report=html
```

### Run Specific Test

```bash
pytest tests/handlers/test_train.py::test_train_simple_model_cpu -v
```

## Code Quality

### Check Code Style

```bash
ruff check src/ tests/
```

### Auto-fix Code Style

```bash
ruff check src/ tests/ --fix
```

### Type Checking

```bash
mypy src/lightning_mcp --ignore-missing-imports
```

### Pre-commit Hooks

```bash
pre-commit run --all-files
```

Install the hooks to run automatically on commit:

```bash
pre-commit install
```

## Project Structure

```
src/lightning_mcp/
├── __init__.py              # Package exports
├── protocol.py              # Request/response schema
├── cli.py                   # CLI interface
├── server.py                # Stdio MCP server
├── http_server.py           # HTTP server (FastAPI)
├── tools.py                 # Tool definitions
├── handlers/
│   ├── base.py             # Base handler class
│   ├── train.py            # Training handler
│   └── inspect.py          # Inspection handler
├── lightning/
│   └── trainer.py          # Lightning integration
└── models/
    └── simple.py           # Example LightningModule

tests/
├── conftest.py             # Pytest fixtures
├── handlers/               # Handler tests
├── protocol/               # Protocol tests
├── server/                 # Server tests
└── tools/                  # Tool tests
```

## Making Changes

### 1. Create a Feature Branch

```bash
git checkout -b feature/my-feature
```

### 2. Make Changes

- Keep changes focused and atomic
- Write tests for new functionality
- Update docstrings and comments
- Run tests: `pytest tests/ -v`

### 3. Check Code Quality

```bash
ruff check src/ tests/ --fix
mypy src/lightning_mcp --ignore-missing-imports
```

### 4. Commit

```bash
git add .
git commit -m "Add: clear description of change"
```

Use conventional commit prefixes:
- `add:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation update
- `test:` - Test addition/modification
- `refactor:` - Code refactoring
- `perf:` - Performance improvement

### 5. Push and Open PR

```bash
git push origin feature/my-feature
```

Open a Pull Request on GitHub with:
- Clear description of changes
- References to related issues
- Mention of any breaking changes

## Testing Best Practices

- **Write descriptive test names** - should explain what is being tested
- **Use fixtures** - see `tests/conftest.py` for available fixtures
- **Test both happy and error paths** - include edge cases
- **Keep tests isolated** - avoid dependencies between tests
- **Use mocks for external dependencies**

Example test structure:

```python
def test_handler_returns_valid_response(train_request_factory):
    """Verify handler returns properly structured response."""
    handler = TrainHandler()
    request = train_request_factory()
    
    response = handler.handle(request)
    
    assert response.id == request.id
    assert response.error is None
    assert response.result is not None
```

## Common Tasks

### Adding a New Handler

1. Create `src/lightning_mcp/handlers/my_handler.py`
2. Implement handler class inheriting from `BaseHandler`
3. Add tests in `tests/handlers/test_my_handler.py`
4. Register in `tools.py`

### Adding a New Protocol Message Type

1. Define in `src/lightning_mcp/protocol.py`
2. Add request/response validation
3. Add corresponding tests

### Updating Dependencies

```bash
uv add package-name
```

Update dev dependencies:

```bash
uv add --dev package-name
```

## Debugging

### Run Tests with Print Debugging

```bash
pytest tests/ -v -s
```

The `-s` flag captures print statements.

### Run with IPython Shell

```bash
# In your test, add:
import ipdb; ipdb.set_trace()

# Then run:
pytest tests/your_test.py -v -s
```

### Check Test Coverage

```bash
pytest tests/ --cov=src/lightning_mcp --cov-report=term-missing
```

Shows which lines aren't covered by tests.

## Documentation

- Keep READMEs current
- Add docstrings to all public functions
- Document complex logic with comments
- Update CHANGELOG for notable changes

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` (if present)
3. Create commit: `git commit -m "Release: v0.x.x"`
4. Create tag: `git tag v0.x.x`
5. Push: `git push origin --tags`
6. GitHub Actions will build and publish

## Getting Help

- Check [CONTRIBUTING.md](CONTRIBUTING.md)
- Review existing tests for patterns
- Open an issue for questions
- Read docstrings and comments in source code

Happy coding! 🚀

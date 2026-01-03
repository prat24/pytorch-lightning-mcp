# Contributing to PyTorch Lightning MCP

We're excited to have you contribute! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

- **Clear title and description** of what went wrong
- **Steps to reproduce** the issue
- **Expected vs. actual behavior**
- **Environment details** (Python version, PyTorch version, OS)
- **Relevant logs or error messages**

### Suggesting Features

We welcome feature suggestions! Please:

- Use a **clear and descriptive title**
- Provide a **detailed description** of the proposed feature
- Explain **why this feature would be useful**
- List **alternative approaches** you've considered
- Provide **examples** of how it would work

### Pull Requests

#### Prerequisites

- Python 3.10+ (3.11 recommended)
- `uv` package manager (see [DEVELOPMENT.md](DEVELOPMENT.md))
- Familiarity with Git and GitHub

#### Steps

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pytorch-lightning-mcp.git
   cd pytorch-lightning-mcp
   ```

3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Set up the development environment** (see [DEVELOPMENT.md](DEVELOPMENT.md)):
   ```bash
   uv sync --all-extras
   ```

5. **Make your changes** with clear, focused commits:
   ```bash
   git add .
   git commit -m "Add feature: clear description of change"
   ```

6. **Run tests locally**:
   ```bash
   pytest tests/ -v
   ```

7. **Check code quality**:
   ```bash
   ruff check src/ tests/
   mypy src/lightning_mcp --ignore-missing-imports
   ```

8. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

9. **Open a Pull Request** with:
   - Clear description of changes
   - Reference to related issues (e.g., "Fixes #123")
   - Screenshots or examples if applicable

#### PR Requirements

- All tests must pass (`pytest tests/`)
- Code must pass linting (`ruff check`)
- Type hints are encouraged
- Documentation should be updated if needed
- Commit messages should be clear and descriptive

## Development Workflow

See [DEVELOPMENT.md](DEVELOPMENT.md) for:

- Setting up your environment
- Running tests
- Building documentation
- Code style guidelines
- Pre-commit hooks

## Testing Guidelines

- Write tests for new features
- Maintain or improve code coverage
- Test edge cases and error conditions
- Use descriptive test names
- See `tests/` for examples

## Code Style

We use:

- **Ruff** for linting and formatting
- **mypy** for static type checking
- **Black-compatible** formatting (via Ruff)

Run `pre-commit` hooks to catch issues early:

```bash
pre-commit run --all-files
```

## Documentation

- Update [README.md](README.md) for user-facing changes
- Add docstrings to new functions/classes
- Use clear, concise language
- Include examples where helpful

## Questions?

- Check existing [issues](https://github.com/pratyushmishra/pytorch-lightning-mcp/issues)
- Review [DEVELOPMENT.md](DEVELOPMENT.md) for setup help
- Open a discussion for design questions

Thank you for contributing! 🎉

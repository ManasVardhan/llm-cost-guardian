# Contributing to LLM Cost Guardian

Thank you for your interest in contributing! Here is how you can help.

## Development Setup

```bash
git clone https://github.com/manasvardhan/llm-cost-guardian.git
cd llm-cost-guardian
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
ruff check .
ruff format .
```

## Adding a New Model

1. Open `src/llm_cost_guardian/models.py`
2. Add a `ModelPricing` entry with the correct per-1M-token pricing
3. Add a test if the model has unusual behavior

## Pull Requests

1. Fork the repo and create a branch from `main`
2. Add tests for any new functionality
3. Ensure all tests pass and linting is clean
4. Open a PR with a clear description of the change

## Reporting Issues

Open an issue on GitHub with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS

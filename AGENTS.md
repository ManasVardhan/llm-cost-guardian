# AGENTS.md - LLM Cost Guardian

## Overview
- Real-time cost monitoring and budget enforcement for LLM API calls. Wraps OpenAI and Anthropic clients with transparent tracking and automatic budget policies so runaway loops never produce surprise bills.
- For developers building LLM-powered applications who need cost visibility and spending controls.
- Core value: drop-in wrappers with zero code changes, thread-safe tracking, layered budget policies (hard caps, soft warnings, sliding windows), and multi-format export.

## Architecture

```
+----------------+     +--------------------------------------+     +---------------+
|                |     |        LLM Cost Guardian              |     |               |
|   Your Code   | --> |  +----------+    +---------------+    | --> |   LLM API     |
|                |     |  | Tracker  |    | BudgetManager |    |     | (OpenAI /     |
|                | <-- |  | (costs)  |    | (policies)    |    | <-- |  Anthropic /  |
|                |     |  +----------+    +---------------+    |     |  Google)      |
+----------------+     |  +----------+    +---------------+    |     +---------------+
                       |  | Exporters|    |     CLI       |    |
                       |  | (JSON/CSV|    |               |    |
                       |  | Promeths)|    |               |    |
                       |  +----------+    +---------------+    |
                       +--------------------------------------+
```

**Data flow:**
1. Your code calls `TrackedOpenAI` or `TrackedAnthropic` (or records manually via `CostTracker.record()`)
2. The wrapper intercepts the API call, passes it through to the real client
3. On response, token usage is extracted and recorded in `CostTracker`
4. `BudgetManager.enforce()` evaluates all policies (hard cap, soft warning, sliding window)
5. If BLOCK is triggered, `BudgetError` is raised before the next call
6. Exporters serialize tracked data to JSON, CSV, or Prometheus format

## Directory Structure

```
llm-cost-guardian/
  .github/workflows/ci.yml        -- CI: lint + test on Python 3.10-3.13
  src/llm_cost_guardian/
    __init__.py                    -- Public API re-exports, __version__
    __main__.py                    -- python -m llm_cost_guardian entry
    models.py                      -- ModelPricing dataclass, PRICING registry, get_pricing(), register_model()
    tracker.py                     -- CostTracker (thread-safe), UsageRecord dataclass
    budget.py                      -- BudgetManager, HardCapPolicy, SoftWarningPolicy, SlidingWindowPolicy, BudgetError
    wrappers.py                    -- TrackedOpenAI, TrackedAnthropic drop-in wrappers
    exporters.py                   -- to_json(), to_csv(), to_prometheus(), save_json(), save_csv()
    cli.py                         -- Click CLI: models, estimate, report, compare, top
  examples/
    basic_tracking.py              -- Minimal tracker usage
    budget_enforcement.py          -- Policy stacking example
    openai_wrapper.py              -- Drop-in wrapper demo
  tests/                           -- 204 tests across 12 test files
    test_budget.py                 -- Budget policy logic
    test_cli.py                    -- CLI commands
    test_cli_integration.py        -- CLI integration tests
    test_cli_top.py                -- `top` command tests
    test_compare.py                -- `compare` command tests
    test_edge_cases.py             -- Edge case coverage
    test_exporters.py              -- JSON/CSV/Prometheus export
    test_models.py                 -- Pricing registry and model lookup
    test_register_model.py         -- Custom model registration
    test_tracker.py                -- Core tracker logic
    test_tracker_filter.py         -- Filter method tests
    test_wrappers.py               -- OpenAI/Anthropic wrapper tests
    test_wrappers_extended.py      -- Extended wrapper coverage
  pyproject.toml                   -- Hatchling build, metadata, deps
  README.md                        -- Full docs with examples
  ROADMAP.md                       -- v0.2 plans
  CONTRIBUTING.md                  -- Contribution guidelines
  GETTING_STARTED.md               -- Quick start guide
  LICENSE                          -- MIT
```

## Core Concepts

- **ModelPricing**: Frozen dataclass holding per-model pricing (input/output cost per 1M tokens). Stored in a global `PRICING` dict.
- **Provider**: Enum with values `openai`, `anthropic`, `google`.
- **CostTracker**: Thread-safe accumulator. Records `UsageRecord` objects, computes running totals. Has `filter()` for querying records by model, time, cost, or predicate.
- **UsageRecord**: Dataclass for a single API call (model, tokens, cost, timestamp, metadata).
- **BudgetPolicy**: Base class. Subclasses: `HardCapPolicy` (block at limit), `SoftWarningPolicy` (warn but allow), `SlidingWindowPolicy` (rolling time window limit).
- **BudgetManager**: Evaluates a stack of policies. `check()` returns the most restrictive result. `enforce()` raises `BudgetError` on BLOCK.
- **Action**: Enum: ALLOW, WARN, BLOCK.
- **TrackedOpenAI / TrackedAnthropic**: Proxy objects that intercept `chat.completions.create()` / `messages.create()`, forward to the real client, then record usage.

## API Reference

### CostTracker
```python
class CostTracker:
    def __init__(self, on_record: CostCallback | None = None) -> None
    def record(self, model: str, input_tokens: int, output_tokens: int, *, cost: float | None = None, metadata: dict | None = None) -> UsageRecord
    @property total_cost -> float
    @property total_tokens -> int
    @property total_input_tokens -> int
    @property total_output_tokens -> int
    @property records -> list[UsageRecord]
    @property average_cost -> float
    @property last_record -> UsageRecord | None
    def cost_by_model() -> dict[str, float]
    def filter(*, model=None, since=None, until=None, min_cost=None, predicate=None) -> list[UsageRecord]
    def summary() -> dict[str, object]
    def reset() -> None
```

### BudgetManager
```python
class BudgetManager:
    def add(self, policy: BudgetPolicy) -> BudgetManager  # fluent
    def check(self, tracker: CostTracker) -> BudgetResult
    def enforce(self, tracker: CostTracker) -> BudgetResult  # raises BudgetError on BLOCK
```

### Models
```python
def get_pricing(model: str) -> ModelPricing  # exact match then prefix match
def register_model(name, provider, input_cost_per_1m, output_cost_per_1m, context_window=None) -> ModelPricing
def list_models(provider: Provider | None = None) -> list[ModelPricing]
```

### Exporters
```python
def to_json(tracker, indent=2) -> str
def to_csv(tracker) -> str
def to_prometheus(tracker, prefix="llm_cost_guardian") -> str
def save_json(tracker, path, indent=2) -> None
def save_csv(tracker, path) -> None
```

### Wrappers
```python
TrackedOpenAI(client, tracker, budget=None)   # wraps openai.OpenAI
TrackedAnthropic(client, tracker, budget=None) # wraps anthropic.Anthropic
```

## CLI Commands

```bash
# List all supported models and pricing
llm-cost-guardian models
llm-cost-guardian models --provider openai --json-output

# Estimate cost for a specific call
llm-cost-guardian estimate gpt-4o --input-tokens 10000 --output-tokens 5000

# View a saved JSON report
llm-cost-guardian report usage_report.json

# Compare two JSON reports side by side
llm-cost-guardian compare report_a.json report_b.json

# Show the most expensive calls from a report
llm-cost-guardian top report.json --limit 20 --json-output

# Version
llm-cost-guardian --version
```

## Configuration

- **YAML config** (documented in README, not yet implemented in code): `llm_cost_guardian.yml` for budget, export, and custom model pricing.
- **Custom models**: Use `register_model()` at runtime to add models not in the built-in registry.
- **No env vars required** for core functionality. OpenAI/Anthropic wrappers use their respective client configs.

## Testing

```bash
# Run all tests
pip install -e ".[dev]"
pytest -v

# Run specific test file
pytest tests/test_tracker.py -v
```

- **204 tests** across 12 test files
- Tests are pure unit tests, no API calls needed
- Located in `tests/` directory
- Uses pytest, no fixtures or conftest

## Dependencies

- **click>=8.0**: CLI framework
- **openai>=1.0** (optional): For TrackedOpenAI wrapper
- **anthropic>=0.20** (optional): For TrackedAnthropic wrapper
- **Python >=3.10**

## CI/CD

- **GitHub Actions** (`.github/workflows/ci.yml`)
- Matrix: Python 3.10, 3.11, 3.12, 3.13
- Steps: install, ruff lint + format check, pytest
- Triggers: push/PR to main

## Current Status

- **Version**: 0.1.1
- **Published on PyPI**: yes (`pip install llm-cost-guardian`)
- **What works**: Full cost tracking, budget enforcement, OpenAI/Anthropic wrappers, JSON/CSV/Prometheus export, CLI (models, estimate, report, compare, top), custom model registration, thread safety
- **Known limitations**: No async wrapper support yet. YAML config file loading not implemented in code (documented in README). No Google wrapper yet.
- **Roadmap (v0.2)**: Slack/Discord webhook alerts, per-user cost attribution, dashboard TUI, project/tag-based grouping

## Development Guide

```bash
git clone https://github.com/manasvardhan/llm-cost-guardian.git
cd llm-cost-guardian
pip install -e ".[dev]"
pytest
```

- **Build system**: Hatchling
- **Source layout**: `src/llm_cost_guardian/`
- **Adding a new model**: Add a `ModelPricing` call in `models.py` `_register()` section
- **Adding a new exporter**: Add function in `exporters.py`, re-export in `__init__.py`
- **Adding a new policy**: Subclass `BudgetPolicy` in `budget.py`, implement `evaluate()`
- **Code style**: Ruff (E, F, I, UP, B rules), line length 100, target Python 3.10

## Git Conventions

- **Branch**: main
- **Commits**: Imperative style ("Add feature X", "Fix bug Y")
- Never use em dashes in commit messages or docs

## Context

- **Author**: Manas Vardhan (ManasVardhan on GitHub)
- **Part of**: A suite of AI agent tooling
- **Related repos**: agent-sentry (crash reporting), agent-replay (trace debugging), llm-shelter (safety guardrails), promptdiff (prompt versioning), mcp-forge (MCP server scaffolding), bench-my-llm (benchmarking)
- **PyPI package**: `llm-cost-guardian`

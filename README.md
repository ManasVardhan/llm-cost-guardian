<p align="center">
  <img src="assets/hero.svg" alt="llm-cost-guardian" width="800">
</p>

# LLM Cost Guardian

**Real-time cost monitoring and budget enforcement for LLM API calls.**

[![PyPI](https://img.shields.io/pypi/v/llm-cost-guardian)](https://pypi.org/project/llm-cost-guardian/)
[![Python](https://img.shields.io/pypi/pyversions/llm-cost-guardian)](https://pypi.org/project/llm-cost-guardian/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/manasvardhan/llm-cost-guardian/actions/workflows/ci.yml/badge.svg)](https://github.com/manasvardhan/llm-cost-guardian/actions)

---

## Why?

LLM API costs can spiral out of control fast - a single runaway loop can burn through hundreds of dollars in minutes. LLM Cost Guardian wraps your existing clients with transparent tracking and automatic budget enforcement so you never get a surprise bill again.

## Features

- 📊 **Real-time cost tracking** - automatic per-call cost calculation from token usage
- 🛡️ **Budget enforcement** - hard caps, soft warnings, and sliding window policies
- 🔌 **Drop-in wrappers** - wrap OpenAI and Anthropic clients with one line of code
- 📈 **Prometheus export** - expose metrics for your monitoring stack
- 💾 **JSON & CSV export** - save usage reports for analysis
- 🖥️ **CLI tool** - estimate costs and view reports from the terminal
- 🧩 **Extensible** - add custom models, policies, and exporters
- 🔒 **Thread-safe** - safe for concurrent use in async applications

## Quick Start

```bash
pip install llm-cost-guardian
```

```python
from llm_cost_guardian import CostTracker, HardCapPolicy, BudgetManager

tracker = CostTracker()
budget = BudgetManager().add(HardCapPolicy(limit_usd=5.00))

# Track a call (or use the wrapper for automatic tracking)
tracker.record("gpt-4o", input_tokens=1500, output_tokens=800)
budget.enforce(tracker)  # raises BudgetError if over limit
print(f"Cost so far: ${tracker.total_cost:.4f}")
```

## Architecture

```
┌──────────────┐     ┌─────────────────────────────────────┐     ┌──────────────┐
│              │     │        LLM Cost Guardian             │     │              │
│  Your Code   │────>│  ┌───────────┐   ┌──────────────┐   │────>│   LLM API    │
│              │     │  │  Tracker   │   │   Budget     │   │     │  (OpenAI /   │
│              │<────│  │  (costs)   │   │  (policies)  │   │<────│  Anthropic / │
│              │     │  └───────────┘   └──────────────┘   │     │   Google)    │
└──────────────┘     │  ┌───────────┐   ┌──────────────┐   │     └──────────────┘
                     │  │ Exporters │   │     CLI      │   │
                     │  │ (JSON/CSV/│   │              │   │
                     │  │ Prometheus)│   │              │   │
                     │  └───────────┘   └──────────────┘   │
                     └─────────────────────────────────────┘
```

## Usage

### Basic Cost Tracking

```python
from llm_cost_guardian import CostTracker

tracker = CostTracker()

# Record API calls manually
tracker.record("gpt-4o", input_tokens=1500, output_tokens=800)
tracker.record("claude-3-5-haiku-20241022", input_tokens=2000, output_tokens=600)

print(f"Total: ${tracker.total_cost:.6f}")
print(f"Tokens: {tracker.total_tokens:,}")
print(tracker.cost_by_model())
```

### Drop-in Client Wrappers

Wrap your existing client - zero code changes needed:

```python
from openai import OpenAI
from llm_cost_guardian import CostTracker, TrackedOpenAI

tracker = CostTracker()
client = TrackedOpenAI(OpenAI(), tracker)

# Use exactly like the normal client - costs tracked automatically
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(f"This call cost: ${tracker.total_cost:.6f}")
```

Works the same way with Anthropic:

```python
from anthropic import Anthropic
from llm_cost_guardian import CostTracker, TrackedAnthropic

tracker = CostTracker()
client = TrackedAnthropic(Anthropic(), tracker)

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Budget Policies

Stack multiple policies for layered protection:

```python
from llm_cost_guardian import (
    BudgetManager,
    HardCapPolicy,
    SoftWarningPolicy,
    SlidingWindowPolicy,
    CostTracker,
    TrackedOpenAI,
)

tracker = CostTracker()
budget = BudgetManager(
    on_warn=lambda result: print(f"WARNING: {result.message}")
)
budget.add(SoftWarningPolicy(warning_usd=1.00))       # warn at $1
budget.add(HardCapPolicy(limit_usd=5.00))              # block at $5
budget.add(SlidingWindowPolicy(                         # $0.50/hour max
    limit_usd=0.50,
    window_seconds=3600,
))

# Attach to a client
client = TrackedOpenAI(OpenAI(), tracker, budget)
# Budget is enforced automatically before each API call
```

### Exporting Data

```python
from llm_cost_guardian import to_json, to_csv, to_prometheus, save_json

# JSON string
print(to_json(tracker))

# CSV string
print(to_csv(tracker))

# Prometheus metrics
print(to_prometheus(tracker))

# Save to file
save_json(tracker, "usage_report.json")
```

### CLI Usage

```bash
# List supported models and pricing
llm-cost-guardian models
llm-cost-guardian models --provider openai --json-output

# Estimate cost for a specific call
llm-cost-guardian estimate gpt-4o --input-tokens 10000 --output-tokens 5000

# View a saved report
llm-cost-guardian report usage_report.json
```

### Prometheus Export

Expose a `/metrics` endpoint for your monitoring stack:

```python
from flask import Flask, Response
from llm_cost_guardian import CostTracker, to_prometheus

app = Flask(__name__)
tracker = CostTracker()  # shared instance

@app.route("/metrics")
def metrics():
    return Response(to_prometheus(tracker), content_type="text/plain")
```

Output format:

```
# HELP llm_cost_guardian_total_cost_usd Total cost in USD
# TYPE llm_cost_guardian_total_cost_usd gauge
llm_cost_guardian_total_cost_usd 0.01234500
# HELP llm_cost_guardian_cost_by_model_usd Cost per model in USD
# TYPE llm_cost_guardian_cost_by_model_usd gauge
llm_cost_guardian_cost_by_model_usd{model="gpt-4o"} 0.00750000
```

## Configuration

LLM Cost Guardian supports YAML configuration files:

```yaml
# llm_cost_guardian.yml
budget:
  hard_cap_usd: 10.00
  soft_warning_usd: 5.00
  sliding_window:
    limit_usd: 2.00
    window_seconds: 3600

export:
  format: json
  path: ./reports/usage.json

# Override or add custom model pricing
models:
  my-fine-tuned-model:
    provider: openai
    input_cost_per_1m: 5.00
    output_cost_per_1m: 15.00
```

## Supported Models

| Model | Provider | Input / 1M tokens | Output / 1M tokens |
|-------|----------|-------------------|---------------------|
| `gpt-4o` | OpenAI | $2.50 | $10.00 |
| `gpt-4o-mini` | OpenAI | $0.15 | $0.60 |
| `gpt-4-turbo` | OpenAI | $10.00 | $30.00 |
| `gpt-4` | OpenAI | $30.00 | $60.00 |
| `gpt-3.5-turbo` | OpenAI | $0.50 | $1.50 |
| `o1` | OpenAI | $15.00 | $60.00 |
| `o1-mini` | OpenAI | $3.00 | $12.00 |
| `o3-mini` | OpenAI | $1.10 | $4.40 |
| `claude-opus-4-20250514` | Anthropic | $15.00 | $75.00 |
| `claude-sonnet-4-20250514` | Anthropic | $3.00 | $15.00 |
| `claude-3-5-sonnet-20241022` | Anthropic | $3.00 | $15.00 |
| `claude-3-5-haiku-20241022` | Anthropic | $0.80 | $4.00 |
| `claude-3-opus-20240229` | Anthropic | $15.00 | $75.00 |
| `claude-3-haiku-20240307` | Anthropic | $0.25 | $1.25 |
| `gemini-2.0-flash` | Google | $0.10 | $0.40 |
| `gemini-1.5-pro` | Google | $1.25 | $5.00 |
| `gemini-1.5-flash` | Google | $0.075 | $0.30 |

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

# Getting Started with llm-cost-guardian

A step-by-step guide to get up and running from scratch.

## Prerequisites

You need **Python 3.10 or newer** installed on your machine.

**Check if you have Python:**
```bash
python3 --version
```
If you see `Python 3.10.x` or higher, you're good. If not, download it from [python.org](https://www.python.org/downloads/).

## Step 1: Clone the repository

Open your terminal (Terminal on Mac, Command Prompt or PowerShell on Windows) and run:

```bash
git clone https://github.com/ManasVardhan/llm-cost-guardian.git
cd llm-cost-guardian
```

## Step 2: Create a virtual environment

A virtual environment keeps this project's dependencies separate from your system Python.

```bash
python3 -m venv venv
```

**Activate it:**

- **Mac/Linux:** `source venv/bin/activate`
- **Windows:** `venv\Scripts\activate`

You should see `(venv)` appear at the start of your terminal prompt.

## Step 3: Install the package

```bash
pip install -e ".[dev]"
```

This installs the package in "editable" mode (so changes to the code take effect immediately) along with all development dependencies (pytest, ruff, etc.).

## Step 4: Run the tests

```bash
pytest tests/ -v
```

You should see all tests passing with green checkmarks. This confirms everything is installed correctly.

## Step 5: Try it out

### 5a. Explore the CLI

```bash
llm-cost-guardian --help
```

List all supported models and their pricing:
```bash
llm-cost-guardian models
```

Check a summary (will be empty since you haven't tracked anything yet):
```bash
llm-cost-guardian summary
```

### 5b. Run the example scripts

```bash
python examples/basic_tracking.py
```

This simulates tracking costs for a few LLM calls and prints a summary.

### 5c. Use it in your own Python code

Create a file called `test_it.py`:

```python
from llm_cost_guardian import CostTracker

tracker = CostTracker()

# Simulate an API call
tracker.record(
    model="gpt-4o",
    input_tokens=500,
    output_tokens=150
)

tracker.record(
    model="claude-3.5-sonnet",
    input_tokens=1000,
    output_tokens=300
)

# See what you've spent
summary = tracker.summary()
print(f"Total cost: ${summary['total_cost']:.4f}")
print(f"Total calls: {summary['total_requests']}")
for model, cost in summary['cost_by_model'].items():
    print(f"  {model}: ${cost:.4f}")
```

Run it:
```bash
python test_it.py
```

## Step 6: Run the linter (optional)

To check code quality:
```bash
ruff check src/ tests/
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python3: command not found` | Install Python from [python.org](https://www.python.org/downloads/) |
| `pip: command not found` | Try `python3 -m pip` instead of `pip` |
| `No module named llm_cost_guardian` | Make sure you ran `pip install -e ".[dev]"` with the venv activated |
| Tests fail | Make sure you're on the latest `main` branch: `git pull origin main` |

## What's next?

- Read the full [README](README.md) for advanced features like budget policies, client wrappers, and Prometheus export
- Check `examples/` for more usage patterns
- Try integrating it with your own OpenAI/Anthropic projects

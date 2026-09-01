
# LLM Cost Guardian

> **New here?** Start with the [Getting Started Guide](GETTING_STARTED.md).

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
- 👤 **Per-user cost attribution** - see who is spending what across users or API keys
- 📅 **Daily cost breakdown** - per-day spend with `cost_by_day()` and a `daily` CLI bar chart
- 🔔 **Slack / Discord alerts** - webhook notifications when spend crosses thresholds
- 📟 **Terminal dashboard** - `dashboard` CLI with budget gauge, trends, and live `--watch` mode
- 🧾 **Persistent cost ledger** - append-only JSONL file so costs survive process restarts
- 📥 **Merge and dedupe** - combine per-service ledgers and reports into one view with `merge`
- 📈 **Cost anomaly detection** - `anomalies` flags days, models, or users whose spend spikes
- 🧮 **Token efficiency report** - `efficiency` shows output/input ratios and cost per 1K output tokens by model and tag
- 🪟 **Context window utilization** - `context` shows avg/p95/max input tokens against each model's window and flags calls near the limit
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
│              │     │        LLM Cost Guardian            │     │              │
│  Your Code   │────>│  ┌───────────┐   ┌──────────────┐   │────>│   LLM API    │
│              │     │  │  Tracker  │   │   Budget     │   │     │  (OpenAI /   │
│              │<────│  │  (costs)  │   │  (policies)  │   │<────│  Anthropic / │
│              │     │  └───────────┘   └──────────────┘   │     │   Google)    │
└──────────────┘     │  ┌───────────┐   ┌──────────────┐   │     └──────────────┘
                     │  │ Exporters │   │     CLI      │   │
                     │  │ (JSON/CSV/│   │              │   │
                     │  │Prometheus)│   │              │   │
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

### Tag-Based Cost Grouping

Attach tags (project, environment, feature) to any call and group costs by tag:

```python
from llm_cost_guardian import CostTracker

tracker = CostTracker()
tracker.record("gpt-4o", 1500, 800, tags=["prod", "chatbot"])
tracker.record("gpt-4o-mini", 2000, 600, tags=["dev"])
tracker.record("gpt-4o-mini", 100, 50)  # untagged

print(tracker.cost_by_tag())
# {'prod': 0.0155, 'chatbot': 0.0155, 'dev': 0.00066, '(untagged)': 4.5e-05}

# Filter records by tag (combines with model/since/until/min_cost)
prod_calls = tracker.filter(tag="prod")
```

Tags flow through every exporter: JSON records carry a `tags` list, CSV gets a
`tags` column (semicolon-separated), Prometheus emits a `cost_by_tag_usd` gauge,
and markdown reports include a "Cost by tag" table. A call with multiple tags
counts toward each of its tags.

### Per-User Cost Attribution

Attribute each call to a user, team member, or API key alias and see who is
consuming what:

```python
from llm_cost_guardian import CostTracker

tracker = CostTracker()
tracker.record("gpt-4o", 1500, 800, user="alice")
tracker.record("gpt-4o-mini", 2000, 600, user="bob")
tracker.record("gpt-4o-mini", 100, 50)  # no attribution

print(tracker.cost_by_user())
# {'alice': 0.01175, 'bob': 0.00066, '(unattributed)': 4.5e-05}

# Filter records by user (combines with model/since/until/min_cost/tag)
alice_calls = tracker.filter(user="alice")
```

Unlike tags, each call has at most one user, so per-user costs always sum to
the tracker total. Users flow through every exporter: JSON records carry a
`user` field, CSV gets a `user` column, Prometheus emits a `cost_by_user_usd`
gauge, and markdown reports include a "Cost by user" table.

### Daily Cost Breakdown

See how spend evolves day by day. In Python, `cost_by_day()` buckets records
into calendar days (local time by default, `utc=True` for UTC dates):

```python
print(tracker.cost_by_day())
# {'2026-07-24': 0.0312, '2026-07-25': 0.0158}
```

The `daily` CLI command renders the same view from a saved report, with an
ASCII bar chart scaled to the most expensive day:

```
$ llm-cost-guardian daily usage_report.json
=== Cost by Day (local) ===
Day            Calls     Tokens         Cost
---------------------------------------------------------------------
2026-07-23         4      9,300 $   0.014500  ###########
2026-07-24        11     24,800 $   0.031200  ########################
2026-07-25         6     12,100 $   0.015800  ############
---------------------------------------------------------------------
Total             21     46,200 $   0.061500
```

Use `--days 7` to keep only the most recent days, `--utc` for UTC bucketing,
and `--json-output` for machine-readable output. Records without a usable
timestamp are grouped under `(unknown)` at the end.

### Slack / Discord Webhook Alerts

Get pinged in Slack or Discord the moment spend crosses a threshold. Rules can
watch total cost or be scoped to a model, a tag, a user, or any combination.
Each rule fires at most once (until `reset()`), and webhook failures are
swallowed so alerting can never break your application:

```python
from llm_cost_guardian import CostAlerter, CostTracker, DiscordWebhook, SlackWebhook

alerter = CostAlerter(
    [
        SlackWebhook("https://hooks.slack.com/services/T000/B000/XXXX"),
        DiscordWebhook("https://discord.com/api/webhooks/123/abc"),
    ]
)
alerter.add_rule(10.0, label="daily-budget")           # total spend
alerter.add_rule(5.0, model="gpt-4o", label="gpt-4o")  # per-model
alerter.add_rule(2.0, tag="prod", user="alice")        # scoped combos

tracker = CostTracker()
alerter.attach(tracker)  # checks run automatically after every record()

tracker.record("gpt-4o", 500_000, 250_000)  # fires when a threshold crosses
```

You can also call `alerter.check(tracker)` manually, inspect
`alerter.fired_rules`, and `alerter.reset()` to re-arm (for example at the
start of each day). Uses only the standard library, no extra dependencies.

For CI or cron, the `alert` CLI command checks a saved report and notifies:

```bash
# Exit 0 under threshold, 2 when crossed, 1 on delivery failure
llm-cost-guardian alert usage_report.json -t 10 \
  --slack-webhook https://hooks.slack.com/services/T000/B000/XXXX

# Scope to a model or tag, preview the payload without sending
llm-cost-guardian alert usage_report.json -t 5 --model gpt-4o --dry-run
llm-cost-guardian alert usage_report.json -t 2 --tag prod --json-output
```

### Terminal Dashboard

Get the whole picture in one screen: totals, budget utilization, cost by
model, a daily trend chart, top tags and users, and the most expensive calls.
The dashboard renders with [rich](https://github.com/Textualize/rich), which
ships as an optional extra:

```bash
pip install "llm-cost-guardian[dashboard]"

# One-shot render
llm-cost-guardian dashboard usage_report.json --budget 50

# Live mode: re-reads the report every 5 seconds until Ctrl+C
llm-cost-guardian dashboard usage_report.json --budget 50 --watch 5
```

The budget gauge turns yellow at 80% utilization and red when you are over
budget. Use `--top N` to size the breakdown sections, `--utc` to bucket the
daily trend by UTC dates, and `--json-output` to get the computed dashboard
data as JSON (works without rich installed, handy for scripting).

Live mode pairs well with a tracker that periodically calls
`save_json(tracker, "usage_report.json")`: point `--watch` at the file and
watch spend evolve in real time. Partially written files are skipped, the
dashboard just keeps the last good frame.

The same data is available in Python via `build_dashboard_data(report_dict)`
and can be rendered anywhere rich renders with `render_dashboard(dash)`.

### Persistent Cost Ledger

Trackers live in memory, so costs vanish when the process exits. Attach a
ledger and every record is also appended to a JSONL file, one JSON object
per line, using the same schema as the JSON exporter:

```python
from llm_cost_guardian import CostTracker

tracker = CostTracker()
tracker.attach_ledger("costs.jsonl", replay=True)  # replay loads prior records
tracker.record("gpt-4o", 1000, 200)                # persisted automatically
```

`replay=True` loads existing ledger entries into the tracker on attach
(without rewriting them or firing `on_record`), so restarts pick up right
where they left off. Read a ledger from any process:

```python
from llm_cost_guardian import CostLedger

ledger = CostLedger("costs.jsonl")
tracker = ledger.to_tracker()                 # or records(since=..., until=...)
print(tracker.total_cost, tracker.cost_by_model())
```

The `ledger` CLI command summarizes a ledger file and converts it into a
standard JSON report, which makes every other command work on persisted data:

```bash
llm-cost-guardian ledger costs.jsonl                          # summary
llm-cost-guardian ledger costs.jsonl --since 2026-08-01       # date range (local)
llm-cost-guardian ledger costs.jsonl --to-report report.json  # convert
llm-cost-guardian dashboard report.json --budget 50           # then anything
```

Malformed lines (partial writes, manual edits) are skipped with a warning,
never a crash, and the valid records still load.

### Merging Ledgers and Reports

Teams usually end up with one cost file per service, host, or CI run. The
`merge` command combines any mix of JSONL ledgers and JSON reports into a
single deduplicated report:

```bash
llm-cost-guardian merge api.jsonl worker.jsonl batch-report.json -o combined.json
llm-cost-guardian daily combined.json      # then use any command on it
```

Each source's format is auto-detected. Records identical in every field
(model, tokens, cost, timestamp, tags, user, metadata) are collapsed to one
so overlapping exports do not double count spend; pass `--no-dedupe` to keep
them. Without `-o` the merged report is printed to stdout, and with
`--json-output` the summary is machine-readable for CI.

The same pipeline is available in Python:

```python
from llm_cost_guardian import merge_sources

result = merge_sources(["api.jsonl", "worker.jsonl", "batch-report.json"])
print(result.tracker.total_cost)      # merged CostTracker
print(result.duplicates_removed)      # overlap between sources
for src in result.sources:            # per-source stats
    print(src.path, src.format, src.records, src.skipped)
```

### Cost Anomaly Detection

Catch spend spikes before the invoice does. The `anomalies` command buckets a
report into calendar days and compares each day's spend to the trailing
average of recent active days across three dimensions: total spend, per
model, and per user:

```bash
llm-cost-guardian anomalies usage_report.json
llm-cost-guardian anomalies usage_report.json --window 7 -t 2.0 --min-spend 0.01
llm-cost-guardian anomalies usage_report.json --utc --json-output
```

A day is flagged when its spend is at least `--threshold` times the mean of
the active days (spend above zero) inside the trailing `--window`, and at
least `--min-spend` USD. Quiet days are skipped in the baseline so bursty
every-other-day usage does not false alarm, and spend whose entire baseline
window was quiet is reported as `new` (for example, a model nobody used
before suddenly costing money). `--min-history` days of history are required
before anything is flagged, so short reports do not false alarm.

Exit codes make it CI and cron friendly: 0 when clean, 2 when at least one
anomaly is found, 1 on invalid input.

```
=== Cost Anomalies (local) ===
Analyzed 412 record(s) across 14 day(s); window=7d, threshold=2x, min spend $0.01

Day          Dimension  Key                                     Spend     Baseline    Ratio
------------------------------------------------------------------------------------------
2026-08-14   total      (total)                          $   9.804300 $   2.113471     4.6x
2026-08-14   model      gpt-4o                           $   8.113200 $   1.204119     6.7x
2026-08-14   user       batch-runner                     $   7.921000 $   0.884314     9.0x

ALERT: 3 anomalies found.
```

The same detection is available in Python via `analyze_anomalies(report_dict)`,
returning an `AnomalyReport` with typed `Anomaly` entries and `to_dict()` for
serialization.

### Token Efficiency Report

Spot prompts and models that burn input tokens without producing much output.
The `efficiency` command sums calls, input and output tokens, and cost for the
whole report and for each model and tag, then derives the output-to-input
token ratio and the cost per 1K output tokens:

```bash
llm-cost-guardian efficiency usage_report.json
llm-cost-guardian efficiency usage_report.json --json-output
```

A ratio below `1.00` means a bucket consumes more input than it produces (long
prompts, short answers), and a high cost per 1K output tokens means output is
expensive to generate. Rows are sorted by descending cost so the biggest spend
surfaces first, and records missing a model or with bad token or cost fields
are skipped and counted.

```
=== Token Efficiency ===
Analyzed 412 record(s); ratio is output/input tokens, cost per 1K output tokens.

Key                                Calls          Input         Output    Ratio       $/1K out
--------------------------------------------------------------------------------------------
(overall)                            412        820,000        240,000     0.29       $12.5000

By model:
gpt-4o                               260        610,000        150,000     0.25       $16.2000
gpt-4o-mini                          152        210,000         90,000     0.43        $2.1000

By tag:
summarization                        180        540,000         60,000     0.11       $28.0000
chat                                 232        280,000        180,000     0.64        $3.9000
```

The same analysis is available in Python via `analyze_efficiency(report_dict)`,
returning an `EfficiencyReport` with typed `EfficiencyStat` rows (which also
expose `cost_per_1k_input`) and `to_dict()` for serialization.

### Context Window Utilization

See how close your calls actually run to each model's context window. The
`context` command shows, per model, the average, p95, and max input tokens
against the model's context window, plus how many calls run at or above the
near-limit fraction (default 80% of the window):

```bash
llm-cost-guardian context usage_report.json
llm-cost-guardian context usage_report.json --near-limit 0.9 --json-output

# Custom or fine-tuned models: supply the window yourself
llm-cost-guardian context usage_report.json --window my-finetune=32000

# CI gate: exit 2 when any model's p95 utilization crosses the threshold
llm-cost-guardian context usage_report.json --fail-near-limit
```

```
=== Context Window Utilization ===
Analyzed 412 record(s); near-limit threshold 80% of the window.

Model                              Calls     Avg in     P95 in     Max in      Window  P95 util  Near
----------------------------------------------------------------------------------------------------
gpt-4                                 40      6,100      7,900      8,100       8,192     96.4%     9
gpt-4o                               260      2,300      5,100      9,800     128,000      4.0%     0
my-finetune                          112     18,000     29,500     31,900           ?         -     -

No known context window for: my-finetune. Use --window MODEL=TOKENS to supply one.

Near limit (80% p95 utilization): gpt-4
```

High p95 utilization means truncation risk and no headroom for retrieval or
history growth; very low utilization on an expensive long-context model means
you may be over-provisioned for the prompts you actually send. Window sizes
come from the built-in model registry (prefix matches cover versioned names
like `gpt-4o-2024-08-06`), and `--window MODEL=TOKENS` overrides or extends
it. The same analysis is available in Python via `analyze_context(report_dict,
windows=..., near_limit=...)`, returning a `ContextReport` with typed
`ContextStat` rows and `to_dict()` for serialization.

### CLI Usage

```bash
# List supported models and pricing
llm-cost-guardian models
llm-cost-guardian models --provider openai --json-output

# Estimate cost for a specific call
llm-cost-guardian estimate gpt-4o --input-tokens 10000 --output-tokens 5000

# View a saved report
llm-cost-guardian report usage_report.json

# Compare two reports side by side
llm-cost-guardian compare before.json after.json

# Most expensive calls in a report
llm-cost-guardian top usage_report.json --limit 5

# Cost distribution stats (mean, min/max, p50/p90/p99)
llm-cost-guardian stats usage_report.json

# Cost grouped by tag
llm-cost-guardian tags usage_report.json
llm-cost-guardian tags usage_report.json --json-output

# Cost attributed per user
llm-cost-guardian users usage_report.json
llm-cost-guardian users usage_report.json --json-output

# Cost per calendar day with a bar chart
llm-cost-guardian daily usage_report.json
llm-cost-guardian daily usage_report.json --utc --days 7 --json-output

# Project spend forward from the observed window
llm-cost-guardian forecast usage_report.json --days 30

# Check a report against a threshold and alert Slack/Discord
llm-cost-guardian alert usage_report.json -t 10 --slack-webhook https://hooks.slack.com/...

# Terminal dashboard (requires the [dashboard] extra)
llm-cost-guardian dashboard usage_report.json --budget 50 --watch 5

# Summarize a JSONL cost ledger, or convert it to a report
llm-cost-guardian ledger costs.jsonl --since 2026-08-01
llm-cost-guardian ledger costs.jsonl --to-report report.json

# Merge ledgers and reports into one deduplicated report
llm-cost-guardian merge api.jsonl worker.jsonl -o combined.json

# Flag spend spikes versus the trailing average (exit 2 when found)
llm-cost-guardian anomalies usage_report.json --window 7 -t 2.0

# Per-model context window utilization (exit 2 with --fail-near-limit)
llm-cost-guardian context usage_report.json --near-limit 0.8
```

Example `tags` output:

```
=== Cost by Tag ===
Tag                               Calls         Cost    Share
-------------------------------------------------------------
prod                                  2 $   0.031000    66.0%
chatbot                               1 $   0.015500    33.0%
dev                                   1 $   0.015500    33.0%
(untagged)                            1 $   0.000045     0.1%
-------------------------------------------------------------
Total                                 4 $   0.047000
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
| `gpt-4.1` | OpenAI | $2.00 | $8.00 |
| `gpt-4.1-mini` | OpenAI | $0.40 | $1.60 |
| `gpt-4.1-nano` | OpenAI | $0.10 | $0.40 |
| `gpt-4o` | OpenAI | $2.50 | $10.00 |
| `gpt-4o-mini` | OpenAI | $0.15 | $0.60 |
| `gpt-4-turbo` | OpenAI | $10.00 | $30.00 |
| `gpt-4` | OpenAI | $30.00 | $60.00 |
| `gpt-3.5-turbo` | OpenAI | $0.50 | $1.50 |
| `o1` | OpenAI | $15.00 | $60.00 |
| `o1-mini` | OpenAI | $3.00 | $12.00 |
| `o3` | OpenAI | $10.00 | $40.00 |
| `o3-mini` | OpenAI | $1.10 | $4.40 |
| `o4-mini` | OpenAI | $1.10 | $4.40 |
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

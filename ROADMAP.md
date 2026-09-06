# Roadmap - llm-cost-guardian

## Shipped in v0.1

### 📁 CSV / JSON / Prometheus Export
Export cost reports in CSV, JSON, and Prometheus text exposition formats for integration with billing systems, spreadsheets, monitoring stacks, or custom analytics pipelines.

### 🏷️ Project and Tag-Based Grouping
Group API calls by project, environment, or custom tags to get granular cost visibility across different workloads. Shipped as `tracker.record(..., tags=[...])`, `cost_by_tag()`, `filter(tag=...)`, a `tags` CLI command, and tag support in every exporter (JSON, CSV, Prometheus, markdown).

### 👤 Per-User Cost Attribution
Track and attribute API costs to individual users or API keys so team leads can see who is consuming what. Shipped as `tracker.record(..., user=...)`, `cost_by_user()`, `filter(user=...)`, a `users` CLI command, and user support in every exporter (JSON, CSV, Prometheus, markdown).

### 📅 Daily Cost Breakdown
See how spend evolves day by day. Shipped as `tracker.cost_by_day()` (local or UTC bucketing) and a `daily` CLI command with per-day calls, tokens, cost, an ASCII bar chart, `--days` limiting, `--utc`, and `--json-output`.

### 🔔 Slack / Discord Webhook Alerts
Send real-time cost alerts to Slack or Discord when spend exceeds configurable thresholds. Shipped as `CostAlerter` with `SlackWebhook` / `DiscordWebhook` senders (standard library only), threshold rules scoped to total, model, tag, or user, fire-once semantics with `reset()`, automatic checks via `alerter.attach(tracker)`, and an `alert` CLI command for CI and cron (exit 0 under threshold, 2 when crossed, 1 on delivery failure).

### 📟 Dashboard TUI
Terminal dashboard showing cost breakdowns, trend graphs, and budget utilization at a glance. Shipped as the `dashboard` CLI command (optional `[dashboard]` extra, built on `rich`): totals, a color-coded budget gauge (yellow at 80%, red when over), cost by model with share percentages, a daily trend bar chart, top tags and users, and the most expensive calls. `--watch N` keeps it live by re-reading the report file, `--json-output` emits the computed data without rich, and `build_dashboard_data()` / `render_dashboard()` expose the same pipeline in Python.

### 🧾 Persistent Cost Ledger
Append-only on-disk ledger so trackers persist across processes. Shipped as JSONL: `CostTracker.attach_ledger(path, replay=True)` appends every record durably and can replay prior entries on attach, `CostLedger` reads ledgers from any process (`records()` with since/until filters, `to_tracker()`, `skipped_lines` for corrupt-line visibility), and a `ledger` CLI command that summarizes a ledger, filters by `--since`/`--until` dates, and converts to a standard JSON report with `--to-report` so top, stats, daily, forecast, alert, and dashboard all work on persisted data.

### 📥 Ledger Import and Merge
Merge multiple ledgers or JSON reports into one so teams can combine per-service cost files into a single view. Shipped in v0.2.0 as the `merge` CLI command (`llm-cost-guardian merge a.jsonl b.json -o combined.json`) and `merge_sources()` / `load_records()` in Python: formats are auto-detected per source, records identical in every field are deduplicated by default (`--no-dedupe` to keep them), merged output is a standard JSON report usable by every other command, and per-source stats plus skipped-entry warnings make partial data visible.

### 📈 Cost Anomaly Detection
Flag days, models, or users whose spend spikes versus their trailing average, so unexpected cost jumps surface before the invoice does. Shipped in v0.3.0 as the `anomalies` CLI command and `analyze_anomalies()` in Python: report records are bucketed into calendar days (local or `--utc`), quiet days are zero-filled into the baseline, and each day's spend is compared to the mean of the trailing `--window` days (default 7) across total, per-model, and per-user dimensions. Days at or above `--threshold` times the baseline (default 2.0) and `--min-spend` USD are flagged, spend on a zero baseline is reported as new, `--min-history` suppresses false alarms on short reports, and exit codes (0 clean, 2 anomalies, 1 bad input) make it CI and cron friendly with `--json-output`.

### 🧮 Token Efficiency Report
Spot prompts and models that burn input tokens without producing output. Shipped in v0.4.0 as the `efficiency` CLI command and `analyze_efficiency()` in Python: for the whole report and for each model and tag it sums calls, input and output tokens, and cost, then derives the output-to-input token ratio and the cost per 1K output tokens (plus cost per 1K input tokens in the API). A ratio below 1.00 flags buckets that consume more input than they produce, and rows are sorted by descending cost so the biggest spend surfaces first. Records missing a model or with non-numeric or negative token or cost fields are skipped and counted, and `--json-output` makes it scriptable. Python API: `analyze_efficiency`, `EfficiencyReport`, `EfficiencyStat`.

### 🪟 Context Window Utilization
Per-model average and p95 input-token usage against each model's context window, so teams can see which calls run close to the limit and which models are over-provisioned for the prompts they actually receive. Shipped in v0.5.0 as the `context` CLI command and `analyze_context()` in Python: for each model it shows average, p95, and max input tokens against the window from the built-in registry (prefix matches cover versioned names), the p95 utilization percentage, and the number of calls at or above the `--near-limit` fraction (default 0.8). `--window MODEL=TOKENS` overrides or extends the registry for custom models, unknown models are listed with a hint instead of guessed, `--fail-near-limit` exits 2 for CI gating, and `--json-output` makes it scriptable. Python API: `analyze_context`, `ContextReport`, `ContextStat`, `resolve_window`.

### 🗃️ Cache-Aware Cost Tracking
Track prompt cache reads and writes separately from regular input tokens so reports reflect provider caching discounts, show real savings from caching, and flag calls that would benefit from enabling it. Shipped in v0.6.0: `tracker.record(..., cache_read_tokens=, cache_write_tokens=)` bills cache activity at each model's published cache prices (Anthropic read 0.1x and write 1.25x, OpenAI and Gemini discounted reads, input-rate fallback for models without cache pricing), the `TrackedOpenAI` and `TrackedAnthropic` wrappers extract cached token counts from responses automatically, and cache tokens flow through summaries, JSON, CSV, Prometheus, markdown, the ledger, and merge (pre-v0.6 files load unchanged). The `cache` CLI command and `analyze_cache()` in Python show per-model cache reads, writes, hit rate, spend, and savings versus paying the full input rate (write premiums count against savings), flag large-prompt models with zero cache usage as caching candidates via `--min-candidate-input`, and support `--json-output`. Python API: `analyze_cache`, `CacheReport`, `CacheStat`, plus cache price fields on `ModelPricing` and `register_model`.

---

## v0.7 (Planned)

### 💱 Configurable Pricing File
Load and pin model prices from a local JSON or YAML file so teams can track negotiated rates, new models, and price changes without upgrading the package, with a `prices` CLI command to view, diff, and validate the active price table.

---

Have ideas? Open an issue or start a discussion!

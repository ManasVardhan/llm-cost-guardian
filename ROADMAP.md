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

---

## v0.2 (Planned)

### 📥 Ledger Import and Merge
Merge multiple ledgers or JSON reports into one (`llm-cost-guardian merge a.jsonl b.jsonl -o combined.json`) with deduplication, so teams can combine per-service cost files into a single view.

---

Have ideas? Open an issue or start a discussion!

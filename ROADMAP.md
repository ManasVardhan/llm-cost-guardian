# Roadmap - llm-cost-guardian

## Shipped in v0.1

### 📁 CSV / JSON / Prometheus Export
Export cost reports in CSV, JSON, and Prometheus text exposition formats for integration with billing systems, spreadsheets, monitoring stacks, or custom analytics pipelines.

### 🏷️ Project and Tag-Based Grouping
Group API calls by project, environment, or custom tags to get granular cost visibility across different workloads. Shipped as `tracker.record(..., tags=[...])`, `cost_by_tag()`, `filter(tag=...)`, a `tags` CLI command, and tag support in every exporter (JSON, CSV, Prometheus, markdown).

### 👤 Per-User Cost Attribution
Track and attribute API costs to individual users or API keys so team leads can see who is consuming what. Shipped as `tracker.record(..., user=...)`, `cost_by_user()`, `filter(user=...)`, a `users` CLI command, and user support in every exporter (JSON, CSV, Prometheus, markdown).

### 🔔 Slack / Discord Webhook Alerts
Send real-time cost alerts to Slack or Discord when spend exceeds configurable thresholds. Shipped as `CostAlerter` with `SlackWebhook` / `DiscordWebhook` senders (standard library only), threshold rules scoped to total, model, tag, or user, fire-once semantics with `reset()`, automatic checks via `alerter.attach(tracker)`, and an `alert` CLI command for CI and cron (exit 0 under threshold, 2 when crossed, 1 on delivery failure).

---

## v0.2 (Planned)

### 📊 Dashboard TUI
Interactive terminal dashboard (built with `rich` / `textual`) showing live cost breakdowns, trend graphs, and budget utilization at a glance.

---

Have ideas? Open an issue or start a discussion!

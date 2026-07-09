"""Export tracked usage data to JSON, CSV, and Prometheus formats."""

from __future__ import annotations

import csv
import io
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tracker import CostTracker


def to_json(tracker: CostTracker, indent: int = 2) -> str:
    """Export all records as a JSON string."""
    records = [
        {
            "model": r.model,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cost_usd": round(r.cost, 8),
            "timestamp": r.timestamp,
            "metadata": r.metadata,
            "tags": list(r.tags),
        }
        for r in tracker.records
    ]
    payload = {
        "summary": tracker.summary(),
        "records": records,
    }
    return json.dumps(payload, indent=indent, default=str)


def to_csv(tracker: CostTracker) -> str:
    """Export all records as a CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["timestamp", "model", "input_tokens", "output_tokens", "cost_usd", "tags"])
    for r in tracker.records:
        writer.writerow(
            [
                r.timestamp,
                r.model,
                r.input_tokens,
                r.output_tokens,
                round(r.cost, 8),
                ";".join(r.tags),
            ]
        )
    return buf.getvalue()


def to_prometheus(tracker: CostTracker, prefix: str = "llm_cost_guardian") -> str:
    """Export metrics in Prometheus text exposition format."""
    lines: list[str] = []

    lines.append(f"# HELP {prefix}_total_cost_usd Total cost in USD")
    lines.append(f"# TYPE {prefix}_total_cost_usd gauge")
    lines.append(f"{prefix}_total_cost_usd {tracker.total_cost:.8f}")

    lines.append(f"# HELP {prefix}_total_requests Total number of API requests")
    lines.append(f"# TYPE {prefix}_total_requests counter")
    lines.append(f"{prefix}_total_requests {len(tracker.records)}")

    lines.append(f"# HELP {prefix}_total_input_tokens Total input tokens")
    lines.append(f"# TYPE {prefix}_total_input_tokens counter")
    lines.append(f"{prefix}_total_input_tokens {tracker.total_input_tokens}")

    lines.append(f"# HELP {prefix}_total_output_tokens Total output tokens")
    lines.append(f"# TYPE {prefix}_total_output_tokens counter")
    lines.append(f"{prefix}_total_output_tokens {tracker.total_output_tokens}")

    lines.append(f"# HELP {prefix}_cost_by_model_usd Cost per model in USD")
    lines.append(f"# TYPE {prefix}_cost_by_model_usd gauge")
    for model, cost in tracker.cost_by_model().items():
        lines.append(f'{prefix}_cost_by_model_usd{{model="{model}"}} {cost:.8f}')

    cost_by_tag = tracker.cost_by_tag()
    if cost_by_tag:
        lines.append(f"# HELP {prefix}_cost_by_tag_usd Cost per tag in USD")
        lines.append(f"# TYPE {prefix}_cost_by_tag_usd gauge")
        for tag, cost in cost_by_tag.items():
            lines.append(f'{prefix}_cost_by_tag_usd{{tag="{tag}"}} {cost:.8f}')

    lines.append("")
    return "\n".join(lines)


def to_markdown(tracker: CostTracker, title: str = "LLM Cost Report") -> str:
    """Export a tracker as a human-readable markdown report.

    Includes a summary header, a per-model breakdown table, and a recent calls
    table. Designed for pasting into pull requests, issues, or chat threads.
    """
    cost_by_model = tracker.cost_by_model()
    total_cost = tracker.total_cost
    total_requests = len(tracker.records)
    total_in = tracker.total_input_tokens
    total_out = tracker.total_output_tokens

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total cost (USD) | ${total_cost:.6f} |")
    lines.append(f"| Total requests | {total_requests:,} |")
    lines.append(f"| Input tokens | {total_in:,} |")
    lines.append(f"| Output tokens | {total_out:,} |")
    lines.append(f"| Total tokens | {total_in + total_out:,} |")
    if total_requests:
        lines.append(f"| Avg cost per request | ${total_cost / total_requests:.6f} |")
    lines.append("")

    if cost_by_model:
        lines.append("## Cost by model")
        lines.append("")
        lines.append("| Model | Cost (USD) | Share |")
        lines.append("|---|---|---|")
        for model, cost in sorted(cost_by_model.items(), key=lambda x: -x[1]):
            share = (cost / total_cost * 100) if total_cost else 0
            lines.append(f"| `{model}` | ${cost:.6f} | {share:.1f}% |")
        lines.append("")

    cost_by_tag = tracker.cost_by_tag()
    if cost_by_tag:
        lines.append("## Cost by tag")
        lines.append("")
        lines.append("| Tag | Cost (USD) | Share |")
        lines.append("|---|---|---|")
        for tag, cost in sorted(cost_by_tag.items(), key=lambda x: -x[1]):
            share = (cost / total_cost * 100) if total_cost else 0
            lines.append(f"| `{tag}` | ${cost:.6f} | {share:.1f}% |")
        lines.append("")

    records = tracker.records
    if records:
        lines.append(f"## Recent calls (showing up to {min(10, len(records))})")
        lines.append("")
        lines.append("| # | Model | Input | Output | Cost (USD) |")
        lines.append("|---|---|---|---|---|")
        # Most recent first
        for i, rec in enumerate(reversed(records[-10:]), 1):
            lines.append(
                f"| {i} | `{rec.model}` | {rec.input_tokens:,} | "
                f"{rec.output_tokens:,} | ${rec.cost:.6f} |"
            )
        lines.append("")

    return "\n".join(lines)


def save_json(tracker: CostTracker, path: str, indent: int = 2) -> None:
    """Write JSON export to a file."""
    with open(path, "w") as f:
        f.write(to_json(tracker, indent=indent))


def save_csv(tracker: CostTracker, path: str) -> None:
    """Write CSV export to a file."""
    with open(path, "w") as f:
        f.write(to_csv(tracker))


def save_markdown(tracker: CostTracker, path: str, title: str = "LLM Cost Report") -> None:
    """Write markdown export to a file."""
    with open(path, "w") as f:
        f.write(to_markdown(tracker, title=title))

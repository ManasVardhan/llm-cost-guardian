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
    writer.writerow(["timestamp", "model", "input_tokens", "output_tokens", "cost_usd"])
    for r in tracker.records:
        writer.writerow([r.timestamp, r.model, r.input_tokens, r.output_tokens, round(r.cost, 8)])
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

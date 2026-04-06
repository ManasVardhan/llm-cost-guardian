"""Command-line interface for llm-cost-guardian."""

from __future__ import annotations

import json
import math
import sys

import click

from .models import Provider, list_models


def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile (0-100) of *values* using linear interpolation.

    Returns 0.0 for an empty list. Mirrors numpy.percentile semantics for the
    'linear' method without requiring numpy as a runtime dependency.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (pct / 100) * (len(sorted_vals) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_vals[int(rank)])
    weight = rank - low
    return float(sorted_vals[low] * (1 - weight) + sorted_vals[high] * weight)


def _load_report(report_file: str) -> dict:
    try:
        with open(report_file) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        click.echo(f"Error: {report_file} is not valid JSON: {e}", err=True)
        sys.exit(1)

    if not isinstance(data, dict):
        click.echo(
            f"Error: Expected a JSON object in {report_file}, got {type(data).__name__}",
            err=True,
        )
        sys.exit(1)
    return data


@click.group()
@click.version_option(package_name="llm-cost-guardian")
def cli() -> None:
    """LLM Cost Guardian - Real-time cost monitoring for LLM APIs."""


@cli.command()
@click.option("--provider", type=click.Choice(["openai", "anthropic", "google"]), default=None)
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON")
def models(provider: str | None, as_json: bool) -> None:
    """List supported models and their pricing."""
    prov = Provider(provider) if provider else None
    model_list = list_models(prov)

    if as_json:
        data = [
            {
                "name": m.name,
                "provider": m.provider.value,
                "input_per_1m": m.input_cost_per_1m,
                "output_per_1m": m.output_cost_per_1m,
            }
            for m in model_list
        ]
        click.echo(json.dumps(data, indent=2))
        return

    click.echo(f"{'Model':<40} {'Provider':<12} {'Input/1M':>10} {'Output/1M':>10}")
    click.echo("-" * 74)
    for m in model_list:
        click.echo(
            f"{m.name:<40} {m.provider.value:<12} "
            f"${m.input_cost_per_1m:>8.3f} ${m.output_cost_per_1m:>8.3f}"
        )


@cli.command()
@click.argument("model")
@click.option("--input-tokens", "-i", type=int, required=True, help="Number of input tokens")
@click.option("--output-tokens", "-o", type=int, required=True, help="Number of output tokens")
def estimate(model: str, input_tokens: int, output_tokens: int) -> None:
    """Estimate cost for a given model and token count."""
    from .models import get_pricing

    if input_tokens < 0 or output_tokens < 0:
        click.echo(
            f"Error: Token counts must be non-negative, got "
            f"input_tokens={input_tokens}, output_tokens={output_tokens}",
            err=True,
        )
        sys.exit(1)

    try:
        pricing = get_pricing(model)
    except KeyError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    cost = pricing.calculate_cost(input_tokens, output_tokens)
    click.echo(f"Model:         {pricing.name}")
    click.echo(f"Input tokens:  {input_tokens:,}")
    click.echo(f"Output tokens: {output_tokens:,}")
    click.echo(f"Estimated cost: ${cost:.6f}")


@cli.command()
@click.argument("report_file", type=click.Path(exists=True))
def report(report_file: str) -> None:
    """Display a summary from a JSON report file."""
    data = _load_report(report_file)

    summary = data.get("summary", {})
    click.echo("=== LLM Cost Report ===")
    click.echo(f"Total cost:     ${summary.get('total_cost_usd', 0):.6f}")
    click.echo(f"Total requests: {summary.get('total_requests', 0)}")
    click.echo(f"Input tokens:   {summary.get('total_input_tokens', 0):,}")
    click.echo(f"Output tokens:  {summary.get('total_output_tokens', 0):,}")

    by_model = summary.get("cost_by_model", {})
    if by_model:
        click.echo("\nCost by model:")
        for model, cost in sorted(by_model.items()):
            click.echo(f"  {model:<35} ${cost:.6f}")


@cli.command()
@click.argument("report_a", type=click.Path(exists=True))
@click.argument("report_b", type=click.Path(exists=True))
def compare(report_a: str, report_b: str) -> None:
    """Compare two JSON report files side by side."""
    try:
        with open(report_a) as f:
            data_a = json.load(f)
        with open(report_b) as f:
            data_b = json.load(f)
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON: {e}", err=True)
        sys.exit(1)

    if not isinstance(data_a, dict) or not isinstance(data_b, dict):
        click.echo("Error: Both files must contain JSON objects.", err=True)
        sys.exit(1)

    sum_a = data_a.get("summary", {})
    sum_b = data_b.get("summary", {})

    cost_a = sum_a.get("total_cost_usd", 0)
    cost_b = sum_b.get("total_cost_usd", 0)
    cost_diff = cost_b - cost_a
    cost_pct = (cost_diff / cost_a * 100) if cost_a else 0

    req_a = sum_a.get("total_requests", 0)
    req_b = sum_b.get("total_requests", 0)

    tok_a = sum_a.get("total_input_tokens", 0) + sum_a.get("total_output_tokens", 0)
    tok_b = sum_b.get("total_input_tokens", 0) + sum_b.get("total_output_tokens", 0)

    click.echo("=== Cost Comparison ===")
    click.echo(f"{'Metric':<25} {'Report A':>12} {'Report B':>12} {'Change':>12}")
    click.echo("-" * 63)
    click.echo(f"{'Total cost (USD)':<25} ${cost_a:>11.6f} ${cost_b:>11.6f} {cost_diff:>+11.6f}")
    click.echo(f"{'Requests':<25} {req_a:>12,} {req_b:>12,} {req_b - req_a:>+12,}")
    click.echo(f"{'Total tokens':<25} {tok_a:>12,} {tok_b:>12,} {tok_b - tok_a:>+12,}")
    if cost_a:
        click.echo(f"\nCost change: {cost_pct:+.1f}%")

    # Per-model breakdown
    models_a = sum_a.get("cost_by_model", {})
    models_b = sum_b.get("cost_by_model", {})
    all_models = sorted(set(list(models_a.keys()) + list(models_b.keys())))

    if all_models:
        click.echo(f"\n{'Model':<35} {'A cost':>10} {'B cost':>10} {'Change':>10}")
        click.echo("-" * 67)
        for model in all_models:
            ca = models_a.get(model, 0)
            cb = models_b.get(model, 0)
            click.echo(f"{model:<35} ${ca:>9.6f} ${cb:>9.6f} {cb - ca:>+9.6f}")


@cli.command()
@click.argument("report_file", type=click.Path(exists=True))
@click.option("--limit", "-n", type=int, default=10, help="Number of records to show (default 10).")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def top(report_file: str, limit: int, as_json: bool) -> None:
    """Show the most expensive API calls from a JSON report."""
    data = _load_report(report_file)

    records = data.get("records", [])
    if not records:
        click.echo("No records found in report.")
        return

    sorted_records = sorted(records, key=lambda r: r.get("cost_usd", 0), reverse=True)
    top_records = sorted_records[:limit]

    if as_json:
        click.echo(json.dumps(top_records, indent=2))
        return

    click.echo(f"=== Top {min(limit, len(records))} Most Expensive Calls ===")
    click.echo(f"{'#':<4} {'Model':<35} {'Input':>8} {'Output':>8} {'Cost':>12}")
    click.echo("-" * 69)
    for i, rec in enumerate(top_records, 1):
        model = rec.get("model", "unknown")
        inp = rec.get("input_tokens", 0)
        out = rec.get("output_tokens", 0)
        cost = rec.get("cost_usd", 0)
        click.echo(f"{i:<4} {model:<35} {inp:>8,} {out:>8,} ${cost:>11.6f}")

    total = sum(r.get("cost_usd", 0) for r in records)
    top_total = sum(r.get("cost_usd", 0) for r in top_records)
    if total > 0:
        pct = top_total / total * 100
        n_top = len(top_records)
        click.echo(
            f"\nTop {n_top} account for ${top_total:.6f}"
            f" of ${total:.6f} total ({pct:.1f}%)"
        )


@cli.command()
@click.argument("report_file", type=click.Path(exists=True))
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def stats(report_file: str, as_json: bool) -> None:
    """Show distribution stats (percentiles, min/max) for a JSON report."""
    data = _load_report(report_file)

    records = data.get("records", [])
    if not records:
        if as_json:
            click.echo(json.dumps({"records": 0}))
        else:
            click.echo("No records found in report.")
        return

    costs = [float(r.get("cost_usd", 0) or 0) for r in records]
    in_tokens = [int(r.get("input_tokens", 0) or 0) for r in records]
    out_tokens = [int(r.get("output_tokens", 0) or 0) for r in records]
    total_tokens = [i + o for i, o in zip(in_tokens, out_tokens, strict=True)]

    n = len(records)
    total_cost = sum(costs)
    mean_cost = total_cost / n if n else 0.0

    p50 = _percentile(costs, 50)
    p90 = _percentile(costs, 90)
    p99 = _percentile(costs, 99)
    cost_min = min(costs) if costs else 0.0
    cost_max = max(costs) if costs else 0.0

    total_tokens_f = [float(t) for t in total_tokens]
    tok_p50 = _percentile(total_tokens_f, 50)
    tok_p90 = _percentile(total_tokens_f, 90)
    tok_p99 = _percentile(total_tokens_f, 99)

    payload = {
        "records": n,
        "total_cost_usd": round(total_cost, 6),
        "cost_per_call": {
            "mean": round(mean_cost, 6),
            "min": round(cost_min, 6),
            "max": round(cost_max, 6),
            "p50": round(p50, 6),
            "p90": round(p90, 6),
            "p99": round(p99, 6),
        },
        "tokens_per_call": {
            "mean": round(sum(total_tokens) / n, 2) if n else 0,
            "p50": round(tok_p50, 2),
            "p90": round(tok_p90, 2),
            "p99": round(tok_p99, 2),
        },
    }

    if as_json:
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo("=== Cost Distribution ===")
    click.echo(f"Records:           {n:,}")
    click.echo(f"Total cost:        ${total_cost:.6f}")
    click.echo()
    click.echo("Cost per call (USD):")
    click.echo(f"  mean:            ${mean_cost:.6f}")
    click.echo(f"  min:             ${cost_min:.6f}")
    click.echo(f"  p50 (median):    ${p50:.6f}")
    click.echo(f"  p90:             ${p90:.6f}")
    click.echo(f"  p99:             ${p99:.6f}")
    click.echo(f"  max:             ${cost_max:.6f}")
    click.echo()
    click.echo("Tokens per call:")
    click.echo(f"  p50: {int(tok_p50):,}   p90: {int(tok_p90):,}   p99: {int(tok_p99):,}")


@cli.command()
@click.argument("report_file", type=click.Path(exists=True))
@click.option("--days", type=float, default=30.0, help="Forecast horizon in days (default 30).")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def forecast(report_file: str, days: float, as_json: bool) -> None:
    """Project total cost forward based on the report's observed time window."""
    data = _load_report(report_file)

    records = data.get("records", [])
    if not records:
        click.echo("No records found in report.", err=not as_json)
        if as_json:
            click.echo(json.dumps({"records": 0}))
        sys.exit(1 if not as_json else 0)

    timestamps = [
        float(r.get("timestamp", 0) or 0) for r in records if r.get("timestamp") is not None
    ]
    timestamps = [t for t in timestamps if t > 0]
    costs = [float(r.get("cost_usd", 0) or 0) for r in records]
    total_cost = sum(costs)

    if len(timestamps) < 2:
        click.echo(
            "Error: forecast requires at least 2 records with valid timestamps.",
            err=True,
        )
        sys.exit(1)

    span_seconds = max(timestamps) - min(timestamps)
    if span_seconds <= 0:
        click.echo(
            "Error: report timestamps span 0 seconds; cannot forecast.",
            err=True,
        )
        sys.exit(1)

    seconds_per_day = 86400.0
    span_days = span_seconds / seconds_per_day
    cost_per_day = total_cost / span_days
    projected = cost_per_day * days

    payload = {
        "observed": {
            "records": len(records),
            "total_cost_usd": round(total_cost, 6),
            "span_days": round(span_days, 4),
        },
        "rates": {
            "cost_per_day_usd": round(cost_per_day, 6),
            "cost_per_hour_usd": round(cost_per_day / 24, 6),
        },
        "forecast": {
            "horizon_days": days,
            "projected_cost_usd": round(projected, 6),
        },
    }

    if as_json:
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo("=== Cost Forecast ===")
    click.echo(f"Observed records:    {len(records):,}")
    click.echo(f"Observed cost:       ${total_cost:.6f}")
    click.echo(f"Observed window:     {span_days:.2f} days")
    click.echo()
    click.echo(f"Cost per day:        ${cost_per_day:.6f}")
    click.echo(f"Cost per hour:       ${cost_per_day / 24:.6f}")
    click.echo()
    click.echo(f"Projected over {days:g} days:  ${projected:.6f}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()

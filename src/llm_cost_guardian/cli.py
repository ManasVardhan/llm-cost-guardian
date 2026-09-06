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
        click.echo(f"\nTop {n_top} account for ${top_total:.6f} of ${total:.6f} total ({pct:.1f}%)")


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
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def tags(report_file: str, as_json: bool) -> None:
    """Show cost grouped by tag from a JSON report."""
    data = _load_report(report_file)

    records = data.get("records", [])
    if not records:
        click.echo("No records found in report.")
        return

    total_cost = 0.0
    cost_by_tag: dict[str, float] = {}
    calls_by_tag: dict[str, int] = {}
    any_tags = False
    for rec in records:
        cost = float(rec.get("cost_usd", 0) or 0)
        total_cost += cost
        rec_tags = rec.get("tags") or []
        if rec_tags:
            any_tags = True
        else:
            rec_tags = ["(untagged)"]
        for tag in rec_tags:
            cost_by_tag[tag] = cost_by_tag.get(tag, 0.0) + cost
            calls_by_tag[tag] = calls_by_tag.get(tag, 0) + 1

    if not any_tags:
        click.echo("No tags found in report. Record calls with tags=[...] to enable tag grouping.")
        return

    rows = sorted(cost_by_tag.items(), key=lambda x: -x[1])

    if as_json:
        payload = {
            "total_cost_usd": round(total_cost, 6),
            "tags": [
                {
                    "tag": tag,
                    "cost_usd": round(cost, 6),
                    "calls": calls_by_tag[tag],
                    "share_pct": round(cost / total_cost * 100, 2) if total_cost else 0,
                }
                for tag, cost in rows
            ],
        }
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo("=== Cost by Tag ===")
    click.echo(f"{'Tag':<30} {'Calls':>8} {'Cost':>12} {'Share':>8}")
    click.echo("-" * 61)
    for tag, cost in rows:
        share = (cost / total_cost * 100) if total_cost else 0
        click.echo(f"{tag:<30} {calls_by_tag[tag]:>8,} ${cost:>11.6f} {share:>7.1f}%")
    click.echo("-" * 61)
    click.echo(f"{'Total':<30} {len(records):>8,} ${total_cost:>11.6f}")
    click.echo(
        "\nNote: calls with multiple tags count toward each tag, so shares can sum past 100%."
    )


@cli.command()
@click.argument("report_file", type=click.Path(exists=True))
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def users(report_file: str, as_json: bool) -> None:
    """Show cost attributed per user from a JSON report."""
    data = _load_report(report_file)

    records = data.get("records", [])
    if not records:
        click.echo("No records found in report.")
        return

    total_cost = 0.0
    cost_by_user: dict[str, float] = {}
    calls_by_user: dict[str, int] = {}
    any_users = False
    for rec in records:
        cost = float(rec.get("cost_usd", 0) or 0)
        total_cost += cost
        user = rec.get("user") or None
        if user:
            any_users = True
        else:
            user = "(unattributed)"
        cost_by_user[user] = cost_by_user.get(user, 0.0) + cost
        calls_by_user[user] = calls_by_user.get(user, 0) + 1

    if not any_users:
        click.echo(
            "No users found in report. Record calls with user=... to enable user attribution."
        )
        return

    rows = sorted(cost_by_user.items(), key=lambda x: -x[1])

    if as_json:
        payload = {
            "total_cost_usd": round(total_cost, 6),
            "users": [
                {
                    "user": user,
                    "cost_usd": round(cost, 6),
                    "calls": calls_by_user[user],
                    "share_pct": round(cost / total_cost * 100, 2) if total_cost else 0,
                }
                for user, cost in rows
            ],
        }
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo("=== Cost by User ===")
    click.echo(f"{'User':<30} {'Calls':>8} {'Cost':>12} {'Share':>8}")
    click.echo("-" * 61)
    for user, cost in rows:
        share = (cost / total_cost * 100) if total_cost else 0
        click.echo(f"{user:<30} {calls_by_user[user]:>8,} ${cost:>11.6f} {share:>7.1f}%")
    click.echo("-" * 61)
    click.echo(f"{'Total':<30} {len(records):>8,} ${total_cost:>11.6f}")


@cli.command()
@click.argument("report_file", type=click.Path(exists=True))
@click.option("--days", "-n", type=int, default=None, help="Show only the most recent N days.")
@click.option("--utc", is_flag=True, help="Bucket days by UTC instead of local time.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def daily(report_file: str, days: int | None, utc: bool, as_json: bool) -> None:
    """Show cost per calendar day from a JSON report."""
    from datetime import datetime, timezone

    data = _load_report(report_file)

    records = data.get("records", [])
    if not records:
        click.echo("No records found in report.")
        return

    if days is not None and days < 1:
        click.echo(f"Error: --days must be at least 1, got {days}.", err=True)
        sys.exit(1)

    tz = timezone.utc if utc else None
    cost_by_day: dict[str, float] = {}
    calls_by_day: dict[str, int] = {}
    tokens_by_day: dict[str, int] = {}
    unknown_key = "(unknown)"
    for rec in records:
        cost = float(rec.get("cost_usd", 0) or 0)
        tokens = int(rec.get("input_tokens", 0) or 0) + int(rec.get("output_tokens", 0) or 0)
        raw_ts = rec.get("timestamp")
        try:
            ts = float(raw_ts)  # type: ignore[arg-type]
            day = datetime.fromtimestamp(ts, tz=tz).date().isoformat() if ts > 0 else unknown_key
        except (TypeError, ValueError, OSError, OverflowError):
            day = unknown_key
        cost_by_day[day] = cost_by_day.get(day, 0.0) + cost
        calls_by_day[day] = calls_by_day.get(day, 0) + 1
        tokens_by_day[day] = tokens_by_day.get(day, 0) + tokens

    has_unknown = unknown_key in cost_by_day
    day_keys = sorted(k for k in cost_by_day if k != unknown_key)
    if days is not None:
        day_keys = day_keys[-days:]
    if has_unknown:
        day_keys.append(unknown_key)

    total_cost = sum(cost_by_day[k] for k in day_keys)
    total_calls = sum(calls_by_day[k] for k in day_keys)
    total_tokens = sum(tokens_by_day[k] for k in day_keys)

    if as_json:
        payload = {
            "total_cost_usd": round(total_cost, 6),
            "timezone": "utc" if utc else "local",
            "days": [
                {
                    "day": key,
                    "calls": calls_by_day[key],
                    "tokens": tokens_by_day[key],
                    "cost_usd": round(cost_by_day[key], 6),
                    "share_pct": (
                        round(cost_by_day[key] / total_cost * 100, 2) if total_cost else 0
                    ),
                }
                for key in day_keys
            ],
        }
        click.echo(json.dumps(payload, indent=2))
        return

    max_cost = max((cost_by_day[k] for k in day_keys), default=0.0)
    bar_width = 24
    line_width = 45 + bar_width
    click.echo(f"=== Cost by Day ({'UTC' if utc else 'local'}) ===")
    click.echo(f"{'Day':<12} {'Calls':>7} {'Tokens':>10} {'Cost':>12}")
    click.echo("-" * line_width)
    for key in day_keys:
        cost = cost_by_day[key]
        bar = "#" * round(cost / max_cost * bar_width) if max_cost > 0 else ""
        click.echo(
            f"{key:<12} {calls_by_day[key]:>7,} {tokens_by_day[key]:>10,} ${cost:>11.6f}  {bar}"
        )
    click.echo("-" * line_width)
    click.echo(f"{'Total':<12} {total_calls:>7,} {total_tokens:>10,} ${total_cost:>11.6f}")


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


@cli.command()
@click.argument("report_file", type=click.Path(exists=True))
@click.option("--threshold", "-t", type=float, required=True, help="Alert threshold in USD.")
@click.option("--model", default=None, help="Only count cost for this model.")
@click.option("--tag", default=None, help="Only count cost for records carrying this tag.")
@click.option("--user", default=None, help="Only count cost attributed to this user.")
@click.option("--label", default=None, help="Display name used in the alert message.")
@click.option("--slack-webhook", default=None, help="Slack incoming webhook URL.")
@click.option("--discord-webhook", default=None, help="Discord webhook URL.")
@click.option("--dry-run", is_flag=True, help="Print webhook payloads instead of sending.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def alert(
    report_file: str,
    threshold: float,
    model: str | None,
    tag: str | None,
    user: str | None,
    label: str | None,
    slack_webhook: str | None,
    discord_webhook: str | None,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Check a JSON report against a cost threshold and send webhook alerts.

    Exit codes: 0 when under threshold, 2 when the threshold is crossed,
    1 on invalid input or webhook delivery failure. Designed for CI and
    cron jobs: run it against a saved report and let Slack or Discord know
    when spend crosses the line.
    """
    from .alerts import AlertEvent, AlertRule, DiscordWebhook, SlackWebhook, Webhook

    data = _load_report(report_file)
    records = data.get("records", [])

    try:
        rule = AlertRule(threshold, model=model, tag=tag, user=user, label=label)
    except (TypeError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    matched_cost = 0.0
    matched_calls = 0
    for rec in records:
        if model is not None and rec.get("model") != model:
            continue
        if tag is not None and tag not in (rec.get("tags") or []):
            continue
        if user is not None and rec.get("user") != user:
            continue
        matched_cost += float(rec.get("cost_usd", 0) or 0)
        matched_calls += 1

    triggered = matched_cost >= threshold
    event = AlertEvent(rule=rule, current_cost=matched_cost)

    webhooks: list[Webhook] = []
    try:
        if slack_webhook:
            webhooks.append(SlackWebhook(slack_webhook))
        if discord_webhook:
            webhooks.append(DiscordWebhook(discord_webhook))
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    delivery: list[dict[str, object]] = []
    failures = 0
    if triggered:
        for webhook in webhooks:
            kind = type(webhook).__name__
            if dry_run:
                payload = webhook.format_payload(event)
                delivery.append({"webhook": kind, "dry_run": True, "payload": payload})
            elif webhook.send(event):
                delivery.append({"webhook": kind, "sent": True})
            else:
                failures += 1
                delivery.append({"webhook": kind, "sent": False, "error": webhook.last_error})

    if as_json:
        payload_out = {
            "threshold_usd": threshold,
            "scope": rule.scope_description(),
            "matched_calls": matched_calls,
            "matched_cost_usd": round(matched_cost, 6),
            "triggered": triggered,
            "delivery": delivery,
        }
        click.echo(json.dumps(payload_out, indent=2))
    else:
        click.echo("=== Cost Alert Check ===")
        click.echo(f"Scope:         {rule.scope_description()}")
        click.echo(f"Matched calls: {matched_calls:,}")
        click.echo(f"Matched cost:  ${matched_cost:.6f}")
        click.echo(f"Threshold:     ${threshold:.2f}")
        if triggered:
            click.echo(f"\nALERT: {event.message}")
            for entry in delivery:
                if entry.get("dry_run"):
                    click.echo(f"[dry-run] {entry['webhook']}: {json.dumps(entry['payload'])}")
                elif entry.get("sent"):
                    click.echo(f"Sent alert via {entry['webhook']}.")
                else:
                    click.echo(f"Failed to send via {entry['webhook']}: {entry['error']}", err=True)
            if not webhooks:
                click.echo("No webhooks configured; nothing sent.")
        else:
            click.echo("\nOK: under threshold.")

    if failures:
        sys.exit(1)
    if triggered:
        sys.exit(2)


@cli.command()
@click.argument("report_file", type=click.Path(exists=True))
@click.option("--budget", type=float, default=None, help="Budget in USD for the utilization gauge.")
@click.option(
    "--watch",
    type=float,
    default=None,
    help="Re-read the report and refresh every N seconds until Ctrl+C.",
)
@click.option("--utc", is_flag=True, help="Bucket the daily trend by UTC instead of local time.")
@click.option("--top", "top_n", type=int, default=5, help="Rows per section (default 5).")
@click.option(
    "--json-output",
    "as_json",
    is_flag=True,
    help="Print the computed dashboard data as JSON (no rich required).",
)
def dashboard(
    report_file: str,
    budget: float | None,
    watch: float | None,
    utc: bool,
    top_n: int,
    as_json: bool,
) -> None:
    """Show a terminal dashboard for a JSON report.

    Renders totals, budget utilization, cost by model, a daily trend chart,
    top tags and users, and the most expensive calls. Pass --watch N to keep
    the dashboard live while the report file is being rewritten. Requires the
    "rich" extra: pip install "llm-cost-guardian[dashboard]".
    """
    from .dashboard import build_dashboard_data

    def build(data: dict) -> dict:
        try:
            return build_dashboard_data(data, budget=budget, utc=utc, top=top_n)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    data = _load_report(report_file)
    dash = build(data)

    if as_json:
        click.echo(json.dumps(dash, indent=2))
        return

    try:
        from rich.console import Console
    except ImportError:
        click.echo(
            "Error: the dashboard requires the 'rich' package.\n"
            'Install it with: pip install "llm-cost-guardian[dashboard]"',
            err=True,
        )
        sys.exit(1)

    from .dashboard import render_dashboard

    console = Console()
    if watch is None:
        console.print(render_dashboard(dash))
        return

    if watch <= 0:
        click.echo(f"Error: --watch must be positive, got {watch}.", err=True)
        sys.exit(1)

    import time

    from rich.live import Live

    title = f"LLM Cost Guardian (watching {report_file}, every {watch:g}s)"
    try:
        with Live(render_dashboard(dash, title=title), console=console, screen=False) as live:
            while True:
                time.sleep(watch)
                try:
                    with open(report_file) as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        dash = build(data)
                except (OSError, json.JSONDecodeError):
                    continue  # keep the last good frame while the file is mid-write
                live.update(render_dashboard(dash, title=title))
    except KeyboardInterrupt:
        pass


@cli.command()
@click.argument("ledger_file", type=click.Path(exists=True))
@click.option(
    "--since", default=None, help="Only include records on or after this date (YYYY-MM-DD)."
)
@click.option(
    "--until", default=None, help="Only include records on or before this date (YYYY-MM-DD)."
)
@click.option(
    "--to-report",
    "to_report",
    type=click.Path(),
    default=None,
    help="Write a standard JSON report usable by every other command.",
)
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def ledger(
    ledger_file: str,
    since: str | None,
    until: str | None,
    to_report: str | None,
    as_json: bool,
) -> None:
    """Summarize a JSONL cost ledger written by CostTracker.attach_ledger.

    Reads the append-only ledger, optionally filters by date range (local
    time), and prints a summary. Use --to-report to convert the ledger into
    a JSON report so top, stats, daily, forecast, alert, and dashboard all
    work on persisted data.
    """
    from datetime import datetime, timedelta

    from .exporters import save_json, to_json
    from .ledger import CostLedger

    def _parse_day(value: str, name: str, *, end_of_day: bool) -> float:
        try:
            day = datetime.fromisoformat(value)
        except ValueError:
            click.echo(f"Error: --{name} must be a date like 2026-08-09, got {value!r}.", err=True)
            sys.exit(1)
        if end_of_day and day.time() == datetime.min.time():
            day = day + timedelta(days=1) - timedelta(microseconds=1)
        return day.timestamp()

    since_ts = _parse_day(since, "since", end_of_day=False) if since else None
    until_ts = _parse_day(until, "until", end_of_day=True) if until else None

    ledger_obj = CostLedger(ledger_file)
    tracker = ledger_obj.to_tracker(since=since_ts, until=until_ts)

    if ledger_obj.skipped_lines:
        click.echo(
            f"Warning: skipped {ledger_obj.skipped_lines} malformed line(s) in {ledger_file}.",
            err=True,
        )

    if to_report:
        save_json(tracker, to_report)
        click.echo(f"Wrote report with {len(tracker.records)} record(s) to {to_report}")
        return

    if as_json:
        click.echo(to_json(tracker))
        return

    records = tracker.records
    if not records:
        click.echo("No records found in ledger.")
        return

    summary = tracker.summary()
    first = min(r.timestamp for r in records)
    last = max(r.timestamp for r in records)
    fmt = "%Y-%m-%d %H:%M:%S"
    click.echo("=== Cost Ledger ===")
    click.echo(f"Ledger file:    {ledger_file}")
    click.echo(f"Records:        {len(records):,}")
    click.echo(f"First record:   {datetime.fromtimestamp(first).strftime(fmt)}")
    click.echo(f"Last record:    {datetime.fromtimestamp(last).strftime(fmt)}")
    click.echo(f"Total cost:     ${summary['total_cost_usd']:.6f}")
    click.echo(f"Input tokens:   {summary['total_input_tokens']:,}")
    click.echo(f"Output tokens:  {summary['total_output_tokens']:,}")

    by_model = summary.get("cost_by_model", {})
    if isinstance(by_model, dict) and by_model:
        click.echo("\nCost by model:")
        for model, cost in sorted(by_model.items(), key=lambda x: -x[1]):
            click.echo(f"  {model:<35} ${cost:.6f}")


@cli.command()
@click.argument("sources", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Write the merged JSON report to this file instead of stdout.",
)
@click.option(
    "--no-dedupe",
    is_flag=True,
    help="Keep records that appear in more than one source instead of collapsing them.",
)
@click.option(
    "--json-output",
    "as_json",
    is_flag=True,
    help="With --output, print the merge summary as JSON.",
)
def merge(
    sources: tuple[str, ...],
    output: str | None,
    no_dedupe: bool,
    as_json: bool,
) -> None:
    """Merge JSONL ledgers and JSON reports into one combined report.

    Accepts any mix of ledger files (written by CostTracker.attach_ledger)
    and JSON reports (written by save_json); each source's format is
    auto-detected. Records identical in every field are deduplicated by
    default so overlapping exports do not double count spend. The merged
    output is a standard JSON report usable by report, top, stats, daily,
    forecast, alert, and dashboard.
    """
    from .exporters import save_json, to_json
    from .merge import MergeError, merge_sources

    try:
        result = merge_sources(sources, dedupe=not no_dedupe)
    except MergeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    for src in result.sources:
        if src.skipped:
            click.echo(
                f"Warning: skipped {src.skipped} malformed entr"
                f"{'y' if src.skipped == 1 else 'ies'} in {src.path}.",
                err=True,
            )

    if output is None:
        click.echo(to_json(result.tracker))
        return

    save_json(result.tracker, output)

    if as_json:
        payload = {
            "sources": [
                {
                    "path": s.path,
                    "format": s.format,
                    "records": s.records,
                    "skipped": s.skipped,
                }
                for s in result.sources
            ],
            "duplicates_removed": result.duplicates_removed,
            "merged_records": result.total_records,
            "total_cost_usd": round(result.tracker.total_cost, 6),
            "output": output,
        }
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo("=== Merged Cost Report ===")
    click.echo("Sources:")
    for src in result.sources:
        click.echo(f"  {src.path:<40} {src.format:<7} {src.records:>7,} record(s)")
    click.echo(f"Duplicates removed: {result.duplicates_removed:,}")
    click.echo(f"Merged records:     {result.total_records:,}")
    click.echo(f"Total cost:         ${result.tracker.total_cost:.6f}")
    click.echo(f"\nWrote merged report to {output}")


@cli.command()
@click.argument("report_file", type=click.Path(exists=True))
@click.option(
    "--window", "-w", type=int, default=7, help="Trailing days for the baseline (default 7)."
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=2.0,
    help="Flag days at or above this multiple of the baseline (default 2.0).",
)
@click.option(
    "--min-spend",
    type=float,
    default=0.01,
    help="Ignore days below this many USD (default 0.01).",
)
@click.option(
    "--min-history",
    type=int,
    default=3,
    help="Days of history required before flagging (default 3).",
)
@click.option("--utc", is_flag=True, help="Bucket days by UTC instead of local time.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def anomalies(
    report_file: str,
    window: int,
    threshold: float,
    min_spend: float,
    min_history: int,
    utc: bool,
    as_json: bool,
) -> None:
    """Flag days, models, or users whose spend spikes versus their trailing average.

    Buckets the report into calendar days and compares each day's spend to
    the mean of the active days inside the trailing --window across three
    dimensions: total spend, per model, and per user. Spend whose entire
    baseline window was quiet is reported as new spend.

    Exit codes: 0 when no anomalies are found, 2 when at least one is,
    1 on invalid input. Designed for CI and cron: run it against a saved
    report and fail the job when spend spikes.
    """
    from .anomalies import analyze_anomalies

    data = _load_report(report_file)

    try:
        result = analyze_anomalies(
            data,
            window=window,
            threshold=threshold,
            min_spend=min_spend,
            min_history=min_history,
            utc=utc,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
        sys.exit(2 if result.has_anomalies else 0)

    click.echo(f"=== Cost Anomalies ({result.timezone}) ===")
    click.echo(
        f"Analyzed {result.records_analyzed:,} record(s) across "
        f"{result.days_analyzed} day(s); window={result.window}d, "
        f"threshold={result.threshold:g}x, min spend ${result.min_spend:g}"
    )
    if result.records_skipped:
        click.echo(
            f"Warning: skipped {result.records_skipped} record(s) without usable data.",
            err=True,
        )

    if not result.has_anomalies:
        click.echo("\nOK: no anomalies found.")
        return

    click.echo()
    click.echo(
        f"{'Day':<12} {'Dimension':<10} {'Key':<32} {'Spend':>12} {'Baseline':>12} {'Ratio':>8}"
    )
    click.echo("-" * 90)
    for a in result.anomalies:
        ratio = f"{a.ratio:.1f}x" if a.ratio is not None else "new"
        click.echo(
            f"{a.day:<12} {a.dimension:<10} {a.key:<32} "
            f"${a.spend_usd:>11.6f} ${a.baseline_usd:>11.6f} {ratio:>8}"
        )
    click.echo("-" * 90)
    plural = "y" if len(result.anomalies) == 1 else "ies"
    click.echo(f"\nALERT: {len(result.anomalies)} anomal{plural} found.")
    sys.exit(2)


def _fmt_ratio(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "-"


def _fmt_cost_per_1k(value: float | None) -> str:
    return f"${value:.4f}" if value is not None else "-"


def _echo_efficiency_rows(stats: list) -> None:
    for s in stats:
        click.echo(
            f"{s.key:<32.32} {s.calls:>7,} {s.input_tokens:>14,} "
            f"{s.output_tokens:>14,} {_fmt_ratio(s.output_input_ratio):>8} "
            f"{_fmt_cost_per_1k(s.cost_per_1k_output):>14}"
        )


@cli.command()
@click.argument("report_file", type=click.Path(exists=True))
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def efficiency(report_file: str, as_json: bool) -> None:
    """Report per-model and per-tag token efficiency for a saved report.

    For the whole report and for each model and tag, shows total calls,
    input and output tokens, the output-to-input token ratio, and the cost
    per 1K output tokens. A ratio below 1.00 means a bucket consumes more
    input than it produces; a high cost per 1K output tokens means output
    is expensive to generate. Use it to spot models and prompts that burn
    tokens without producing much output.

    Exit codes: 0 on success, 1 on invalid input.
    """
    from .efficiency import analyze_efficiency

    data = _load_report(report_file)
    result = analyze_efficiency(data)

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    click.echo("=== Token Efficiency ===")
    click.echo(
        f"Analyzed {result.records_analyzed:,} record(s); "
        f"ratio is output/input tokens, cost per 1K output tokens."
    )
    if result.records_skipped:
        click.echo(
            f"Warning: skipped {result.records_skipped} record(s) without usable data.",
            err=True,
        )

    if result.records_analyzed == 0:
        click.echo("\nNo usable records found.")
        return

    header = (
        f"{'Key':<32} {'Calls':>7} {'Input':>14} "
        f"{'Output':>14} {'Ratio':>8} {'$/1K out':>14}"
    )
    click.echo()
    click.echo(header)
    click.echo("-" * 92)
    _echo_efficiency_rows([result.overall])
    if result.by_model:
        click.echo("\nBy model:")
        _echo_efficiency_rows(result.by_model)
    if result.by_tag:
        click.echo("\nBy tag:")
        _echo_efficiency_rows(result.by_tag)


def _parse_window_overrides(overrides: tuple[str, ...]) -> dict[str, int]:
    windows: dict[str, int] = {}
    for spec in overrides:
        name, sep, raw = spec.partition("=")
        name = name.strip()
        try:
            size = int(raw.strip())
        except ValueError:
            size = 0
        if not sep or not name or size <= 0:
            click.echo(
                f"Error: invalid --window {spec!r}, expected MODEL=TOKENS "
                f"with a positive token count (e.g. my-model=32000).",
                err=True,
            )
            sys.exit(1)
        windows[name] = size
    return windows


@cli.command("context")
@click.argument("report_file", type=click.Path(exists=True))
@click.option(
    "--near-limit",
    type=float,
    default=0.8,
    show_default=True,
    help="Fraction of the window at which a model counts as near the limit.",
)
@click.option(
    "--window",
    "windows",
    multiple=True,
    metavar="MODEL=TOKENS",
    help="Override or add a context window size (repeatable).",
)
@click.option(
    "--fail-near-limit",
    is_flag=True,
    help="Exit 2 when any model's p95 utilization crosses --near-limit.",
)
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def context(
    report_file: str,
    near_limit: float,
    windows: tuple[str, ...],
    fail_near_limit: bool,
    as_json: bool,
) -> None:
    """Report per-model context window utilization for a saved report.

    For each model, shows average, p95, and max input tokens against the
    model's context window, plus how many calls run at or above the
    near-limit fraction. Use it to spot calls at truncation risk and
    models that are over-provisioned for the prompts they receive.
    Window sizes come from the built-in model registry; use --window
    MODEL=TOKENS to override or cover custom models.

    Exit codes: 0 on success, 1 on invalid input, 2 when
    --fail-near-limit is set and a model is near its limit.
    """
    from .context_window import analyze_context

    data = _load_report(report_file)
    try:
        result = analyze_context(
            data, windows=_parse_window_overrides(windows), near_limit=near_limit
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    near = result.models_near_limit

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        click.echo("=== Context Window Utilization ===")
        click.echo(
            f"Analyzed {result.records_analyzed:,} record(s); "
            f"near-limit threshold {near_limit * 100:.0f}% of the window."
        )
        if result.records_skipped:
            click.echo(
                f"Warning: skipped {result.records_skipped} record(s) without usable data.",
                err=True,
            )

        if result.records_analyzed == 0:
            click.echo("\nNo usable records found.")
            return

        header = (
            f"{'Model':<32} {'Calls':>7} {'Avg in':>10} {'P95 in':>10} "
            f"{'Max in':>10} {'Window':>11} {'P95 util':>9} {'Near':>5}"
        )
        click.echo()
        click.echo(header)
        click.echo("-" * 100)
        for s in result.by_model:
            window = f"{s.context_window:,}" if s.context_window is not None else "?"
            util = s.p95_utilization
            util_str = f"{util * 100:.1f}%" if util is not None else "-"
            near_str = str(s.calls_near_limit) if s.calls_near_limit is not None else "-"
            click.echo(
                f"{s.model:<32.32} {s.calls:>7,} {s.avg_input_tokens:>10,.0f} "
                f"{s.p95_input_tokens:>10,.0f} {s.max_input_tokens:>10,} "
                f"{window:>11} {util_str:>9} {near_str:>5}"
            )

        unknown = result.models_without_window
        if unknown:
            names = ", ".join(s.model for s in unknown)
            click.echo(
                f"\nNo known context window for: {names}. "
                f"Use --window MODEL=TOKENS to supply one."
            )
        if near:
            names = ", ".join(s.model for s in near)
            click.echo(
                f"\nNear limit ({near_limit * 100:.0f}% p95 utilization): {names}"
            )

    if fail_near_limit and near:
        sys.exit(2)


@cli.command("cache")
@click.argument("report_file", type=click.Path(exists=True))
@click.option(
    "--min-candidate-input",
    type=float,
    default=1024.0,
    show_default=True,
    help="Average input tokens per call at which an uncached model becomes a caching candidate.",
)
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
def cache(report_file: str, min_candidate_input: float, as_json: bool) -> None:
    """Report prompt cache usage and savings for a saved report.

    For each model, shows regular input tokens versus cache read and cache
    write tokens, the cache hit rate, actual spend, and the savings caching
    produced versus paying the full input rate (using each model's published
    cache prices; Anthropic-style write premiums count against savings).
    Models that send large prompts with no cache usage are flagged as
    candidates that would likely benefit from enabling caching.

    Reports written before v0.6 have no cache fields and are treated as
    fully uncached. Exit codes: 0 on success, 1 on invalid input.
    """
    from .cache import CacheStat, analyze_cache

    data = _load_report(report_file)
    try:
        result = analyze_cache(data, min_candidate_input=min_candidate_input)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    click.echo("=== Prompt Cache Usage ===")
    click.echo(
        f"Analyzed {result.records_analyzed:,} record(s); "
        f"hit rate is cache reads over cache reads plus regular input."
    )
    if result.records_skipped:
        click.echo(
            f"Warning: skipped {result.records_skipped} record(s) without usable data.",
            err=True,
        )

    if result.records_analyzed == 0:
        click.echo("\nNo usable records found.")
        return

    def _fmt_savings(stat: CacheStat) -> str:
        if stat.savings is None:
            return "?"
        return f"${stat.savings:,.4f}"

    header = (
        f"{'Model':<32} {'Calls':>7} {'Input':>12} {'Cache rd':>12} "
        f"{'Cache wr':>12} {'Hit rate':>9} {'Cost':>12} {'Saved':>12}"
    )
    click.echo()
    click.echo(header)
    click.echo("-" * 115)
    rows = [result.overall] + result.by_model if len(result.by_model) > 1 else result.by_model
    for s in rows:
        click.echo(
            f"{s.key:<32.32} {s.calls:>7,} {s.input_tokens:>12,} "
            f"{s.cache_read_tokens:>12,} {s.cache_write_tokens:>12,} "
            f"{s.hit_rate * 100:>8.1f}% {'$' + format(s.cost, ',.4f'):>12} "
            f"{_fmt_savings(s):>12}"
        )

    unpriced = result.unpriced_models
    if unpriced:
        click.echo(
            f"\nNo pricing data for: {', '.join(unpriced)}. "
            f"Savings shown are a lower bound over priced models."
        )
    if result.candidates:
        click.echo(
            f"\nCaching candidates (no cache usage, avg input >= "
            f"{min_candidate_input:,.0f} tokens): {', '.join(result.candidates)}"
        )
    elif not result.overall.uses_cache:
        click.echo("\nNo cache usage recorded and no obvious candidates.")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()

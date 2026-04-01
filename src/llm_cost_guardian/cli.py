"""Command-line interface for llm-cost-guardian."""

from __future__ import annotations

import json
import sys

import click

from .models import Provider, list_models


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


def main() -> None:
    cli()


if __name__ == "__main__":
    main()

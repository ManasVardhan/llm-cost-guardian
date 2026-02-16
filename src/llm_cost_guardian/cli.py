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
    with open(report_file) as f:
        data = json.load(f)

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


def main() -> None:
    cli()


if __name__ == "__main__":
    main()

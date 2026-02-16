"""Pricing data for supported LLM models.

All prices are in USD per 1M tokens unless otherwise noted.
Last updated: 2025-01-15
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-token pricing for a single model."""

    name: str
    provider: Provider
    input_cost_per_1m: float
    output_cost_per_1m: float
    context_window: int | None = None

    @property
    def input_cost_per_token(self) -> float:
        return self.input_cost_per_1m / 1_000_000

    @property
    def output_cost_per_token(self) -> float:
        return self.output_cost_per_1m / 1_000_000

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost for a given number of tokens."""
        return input_tokens * self.input_cost_per_token + output_tokens * self.output_cost_per_token


# ---------------------------------------------------------------------------
# Pricing registry
# ---------------------------------------------------------------------------

PRICING: dict[str, ModelPricing] = {}


def _register(*models: ModelPricing) -> None:
    for m in models:
        PRICING[m.name] = m


# OpenAI models
_register(
    ModelPricing("gpt-4o", Provider.OPENAI, 2.50, 10.00, 128_000),
    ModelPricing("gpt-4o-mini", Provider.OPENAI, 0.15, 0.60, 128_000),
    ModelPricing("gpt-4-turbo", Provider.OPENAI, 10.00, 30.00, 128_000),
    ModelPricing("gpt-4", Provider.OPENAI, 30.00, 60.00, 8_192),
    ModelPricing("gpt-3.5-turbo", Provider.OPENAI, 0.50, 1.50, 16_385),
    ModelPricing("o1", Provider.OPENAI, 15.00, 60.00, 200_000),
    ModelPricing("o1-mini", Provider.OPENAI, 3.00, 12.00, 128_000),
    ModelPricing("o3-mini", Provider.OPENAI, 1.10, 4.40, 200_000),
)

# Anthropic models
_register(
    ModelPricing("claude-opus-4-20250514", Provider.ANTHROPIC, 15.00, 75.00, 200_000),
    ModelPricing("claude-sonnet-4-20250514", Provider.ANTHROPIC, 3.00, 15.00, 200_000),
    ModelPricing("claude-3-5-sonnet-20241022", Provider.ANTHROPIC, 3.00, 15.00, 200_000),
    ModelPricing("claude-3-5-haiku-20241022", Provider.ANTHROPIC, 0.80, 4.00, 200_000),
    ModelPricing("claude-3-opus-20240229", Provider.ANTHROPIC, 15.00, 75.00, 200_000),
    ModelPricing("claude-3-haiku-20240307", Provider.ANTHROPIC, 0.25, 1.25, 200_000),
)

# Google models
_register(
    ModelPricing("gemini-2.0-flash", Provider.GOOGLE, 0.10, 0.40, 1_000_000),
    ModelPricing("gemini-1.5-pro", Provider.GOOGLE, 1.25, 5.00, 2_000_000),
    ModelPricing("gemini-1.5-flash", Provider.GOOGLE, 0.075, 0.30, 1_000_000),
)


def get_pricing(model: str) -> ModelPricing:
    """Look up pricing for a model name. Tries exact match then prefix match."""
    if model in PRICING:
        return PRICING[model]
    # Prefix match for versioned names like "gpt-4o-2024-08-06"
    for key, pricing in PRICING.items():
        if model.startswith(key):
            return pricing
    raise KeyError(f"Unknown model: {model!r}. Register it or use a known model name.")


def list_models(provider: Provider | None = None) -> list[ModelPricing]:
    """List all known models, optionally filtered by provider."""
    models = list(PRICING.values())
    if provider is not None:
        models = [m for m in models if m.provider == provider]
    return sorted(models, key=lambda m: m.name)

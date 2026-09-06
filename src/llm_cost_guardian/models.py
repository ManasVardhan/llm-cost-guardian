"""Pricing data for supported LLM models.

All prices are in USD per 1M tokens unless otherwise noted.
Last updated: 2026-03-07
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
    """Per-token pricing for a single model.

    ``cache_read_cost_per_1m`` and ``cache_write_cost_per_1m`` are the prices
    for prompt cache hits and cache writes. When a provider does not publish a
    separate price (or the model predates caching), they default to None and
    cached tokens are billed at the regular input rate.
    """

    name: str
    provider: Provider
    input_cost_per_1m: float
    output_cost_per_1m: float
    context_window: int | None = None
    cache_read_cost_per_1m: float | None = None
    cache_write_cost_per_1m: float | None = None

    @property
    def input_cost_per_token(self) -> float:
        return self.input_cost_per_1m / 1_000_000

    @property
    def output_cost_per_token(self) -> float:
        return self.output_cost_per_1m / 1_000_000

    @property
    def effective_cache_read_cost_per_1m(self) -> float:
        """Cache read price, falling back to the input price when unpublished."""
        if self.cache_read_cost_per_1m is None:
            return self.input_cost_per_1m
        return self.cache_read_cost_per_1m

    @property
    def effective_cache_write_cost_per_1m(self) -> float:
        """Cache write price, falling back to the input price when unpublished."""
        if self.cache_write_cost_per_1m is None:
            return self.input_cost_per_1m
        return self.cache_write_cost_per_1m

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Calculate total cost for a given number of tokens.

        ``input_tokens`` are regular (uncached) input tokens. Cache reads and
        writes are billed at the model's cache prices when known, otherwise at
        the regular input rate.
        """
        cost = input_tokens * self.input_cost_per_token + output_tokens * self.output_cost_per_token
        if cache_read_tokens:
            cost += cache_read_tokens * self.effective_cache_read_cost_per_1m / 1_000_000
        if cache_write_tokens:
            cost += cache_write_tokens * self.effective_cache_write_cost_per_1m / 1_000_000
        return cost


# ---------------------------------------------------------------------------
# Pricing registry
# ---------------------------------------------------------------------------

PRICING: dict[str, ModelPricing] = {}


def _register(*models: ModelPricing) -> None:
    for m in models:
        PRICING[m.name] = m


# OpenAI models
# Cached input pricing per OpenAI's prompt caching (reads discounted, writes
# are billed as regular input, so cache_write is left as None).
_register(
    ModelPricing("gpt-4o", Provider.OPENAI, 2.50, 10.00, 128_000, cache_read_cost_per_1m=1.25),
    ModelPricing(
        "gpt-4o-mini", Provider.OPENAI, 0.15, 0.60, 128_000, cache_read_cost_per_1m=0.075
    ),
    ModelPricing("gpt-4.1", Provider.OPENAI, 2.00, 8.00, 1_047_576, cache_read_cost_per_1m=0.50),
    ModelPricing(
        "gpt-4.1-mini", Provider.OPENAI, 0.40, 1.60, 1_047_576, cache_read_cost_per_1m=0.10
    ),
    ModelPricing(
        "gpt-4.1-nano", Provider.OPENAI, 0.10, 0.40, 1_047_576, cache_read_cost_per_1m=0.025
    ),
    ModelPricing("gpt-4-turbo", Provider.OPENAI, 10.00, 30.00, 128_000),
    ModelPricing("gpt-4", Provider.OPENAI, 30.00, 60.00, 8_192),
    ModelPricing("gpt-3.5-turbo", Provider.OPENAI, 0.50, 1.50, 16_385),
    ModelPricing("o1", Provider.OPENAI, 15.00, 60.00, 200_000, cache_read_cost_per_1m=7.50),
    ModelPricing("o1-mini", Provider.OPENAI, 3.00, 12.00, 128_000, cache_read_cost_per_1m=1.50),
    ModelPricing("o3", Provider.OPENAI, 10.00, 40.00, 200_000, cache_read_cost_per_1m=2.50),
    ModelPricing("o3-mini", Provider.OPENAI, 1.10, 4.40, 200_000, cache_read_cost_per_1m=0.55),
    ModelPricing("o4-mini", Provider.OPENAI, 1.10, 4.40, 200_000, cache_read_cost_per_1m=0.275),
)

# Anthropic models
# Prompt caching: 5-minute cache writes cost 1.25x the input rate, cache
# reads cost 0.1x the input rate.
_register(
    ModelPricing(
        "claude-opus-4-20250514",
        Provider.ANTHROPIC,
        15.00,
        75.00,
        200_000,
        cache_read_cost_per_1m=1.50,
        cache_write_cost_per_1m=18.75,
    ),
    ModelPricing(
        "claude-sonnet-4-20250514",
        Provider.ANTHROPIC,
        3.00,
        15.00,
        200_000,
        cache_read_cost_per_1m=0.30,
        cache_write_cost_per_1m=3.75,
    ),
    ModelPricing(
        "claude-3-5-sonnet-20241022",
        Provider.ANTHROPIC,
        3.00,
        15.00,
        200_000,
        cache_read_cost_per_1m=0.30,
        cache_write_cost_per_1m=3.75,
    ),
    ModelPricing(
        "claude-3-5-haiku-20241022",
        Provider.ANTHROPIC,
        0.80,
        4.00,
        200_000,
        cache_read_cost_per_1m=0.08,
        cache_write_cost_per_1m=1.00,
    ),
    ModelPricing(
        "claude-3-opus-20240229",
        Provider.ANTHROPIC,
        15.00,
        75.00,
        200_000,
        cache_read_cost_per_1m=1.50,
        cache_write_cost_per_1m=18.75,
    ),
    ModelPricing(
        "claude-3-haiku-20240307",
        Provider.ANTHROPIC,
        0.25,
        1.25,
        200_000,
        cache_read_cost_per_1m=0.03,
        cache_write_cost_per_1m=0.30,
    ),
)

# Google models
# Implicit context caching bills cached tokens at 0.25x the input rate.
_register(
    ModelPricing(
        "gemini-2.0-flash", Provider.GOOGLE, 0.10, 0.40, 1_000_000, cache_read_cost_per_1m=0.025
    ),
    ModelPricing(
        "gemini-1.5-pro", Provider.GOOGLE, 1.25, 5.00, 2_000_000, cache_read_cost_per_1m=0.3125
    ),
    ModelPricing(
        "gemini-1.5-flash", Provider.GOOGLE, 0.075, 0.30, 1_000_000, cache_read_cost_per_1m=0.01875
    ),
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


def register_model(
    name: str,
    provider: str | Provider,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
    context_window: int | None = None,
    cache_read_cost_per_1m: float | None = None,
    cache_write_cost_per_1m: float | None = None,
) -> ModelPricing:
    """Register a custom model for cost tracking.

    Parameters
    ----------
    name : model identifier (e.g. "my-finetuned-llama")
    provider : provider name or Provider enum value
    input_cost_per_1m : cost in USD per 1M input tokens
    output_cost_per_1m : cost in USD per 1M output tokens
    context_window : optional max context length
    cache_read_cost_per_1m : optional cost in USD per 1M cache read tokens
    cache_write_cost_per_1m : optional cost in USD per 1M cache write tokens

    Returns
    -------
    The newly registered ModelPricing instance.

    Raises
    ------
    ValueError
        If costs are negative.

    Example::

        from llm_cost_guardian import register_model
        register_model("my-model", "openai", 1.00, 3.00)
    """
    if input_cost_per_1m < 0 or output_cost_per_1m < 0:
        raise ValueError(
            f"Costs must be non-negative, got input={input_cost_per_1m}, "
            f"output={output_cost_per_1m}"
        )
    for label, value in (
        ("cache_read_cost_per_1m", cache_read_cost_per_1m),
        ("cache_write_cost_per_1m", cache_write_cost_per_1m),
    ):
        if value is not None and value < 0:
            raise ValueError(f"Costs must be non-negative, got {label}={value}")
    if isinstance(provider, str):
        provider = Provider(provider)
    pricing = ModelPricing(
        name=name,
        provider=provider,
        input_cost_per_1m=input_cost_per_1m,
        output_cost_per_1m=output_cost_per_1m,
        context_window=context_window,
        cache_read_cost_per_1m=cache_read_cost_per_1m,
        cache_write_cost_per_1m=cache_write_cost_per_1m,
    )
    PRICING[name] = pricing
    return pricing


def list_models(provider: Provider | None = None) -> list[ModelPricing]:
    """List all known models, optionally filtered by provider."""
    models = list(PRICING.values())
    if provider is not None:
        models = [m for m in models if m.provider == provider]
    return sorted(models, key=lambda m: m.name)

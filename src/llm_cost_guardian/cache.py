"""Prompt cache usage analysis for LLM spend reports.

Shows, per model, how many tokens were served from the provider's prompt
cache versus billed at the full input rate, the savings caching produced
(using each model's published cache prices), and which models send large
prompts without any cache usage and would likely benefit from enabling
caching.

Cache prices come from the built-in pricing registry (including prefix
matches for versioned model names). Models without pricing data still get
usage totals; their savings are reported as unknown instead of guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import ModelPricing, get_pricing

DEFAULT_MIN_CANDIDATE_INPUT = 1024.0


def _resolve_pricing(model: str) -> ModelPricing | None:
    try:
        return get_pricing(model)
    except KeyError:
        return None


@dataclass(slots=True)
class CacheStat:
    """Prompt cache usage for one model (or the whole report)."""

    key: str
    calls: int
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost: float
    savings: float | None
    priced: bool

    @property
    def cached_tokens(self) -> int:
        return self.cache_read_tokens + self.cache_write_tokens

    @property
    def uses_cache(self) -> bool:
        return self.cached_tokens > 0

    @property
    def hit_rate(self) -> float:
        """Fraction of prompt tokens served from cache (reads over reads plus input)."""
        prompt_tokens = self.input_tokens + self.cache_read_tokens
        if prompt_tokens <= 0:
            return 0.0
        return self.cache_read_tokens / prompt_tokens

    @property
    def avg_input_tokens(self) -> float:
        if not self.calls:
            return 0.0
        return self.input_tokens / self.calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "hit_rate_pct": round(self.hit_rate * 100, 2),
            "cost_usd": round(self.cost, 6),
            "savings_usd": round(self.savings, 6) if self.savings is not None else None,
            "priced": self.priced,
        }


@dataclass(slots=True)
class CacheReport:
    """Result of a prompt cache usage analysis."""

    overall: CacheStat
    by_model: list[CacheStat]
    candidates: list[str]
    min_candidate_input: float
    records_analyzed: int
    records_skipped: int

    @property
    def models_using_cache(self) -> list[CacheStat]:
        return [s for s in self.by_model if s.uses_cache]

    @property
    def unpriced_models(self) -> list[str]:
        return [s.key for s in self.by_model if not s.priced]

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_analyzed": self.records_analyzed,
            "records_skipped": self.records_skipped,
            "min_candidate_input_tokens": self.min_candidate_input,
            "candidates": list(self.candidates),
            "unpriced_models": self.unpriced_models,
            "overall": self.overall.to_dict(),
            "by_model": [s.to_dict() for s in self.by_model],
        }


def _record_savings(
    pricing: ModelPricing, cache_read_tokens: int, cache_write_tokens: int
) -> float:
    """Savings versus billing every cached token at the regular input rate.

    Cache reads save (input rate - read rate) per token. Cache writes cost
    (write rate - input rate) more per token on providers that bill a write
    premium, so heavy writes with few reads can produce negative savings.
    """
    input_rate = pricing.input_cost_per_1m
    read_saving = cache_read_tokens * (input_rate - pricing.effective_cache_read_cost_per_1m)
    write_saving = cache_write_tokens * (input_rate - pricing.effective_cache_write_cost_per_1m)
    return (read_saving + write_saving) / 1_000_000


def analyze_cache(
    data: dict[str, Any],
    min_candidate_input: float = DEFAULT_MIN_CANDIDATE_INPUT,
) -> CacheReport:
    """Compute prompt cache usage and savings for a report dict.

    Parameters
    ----------
    data : a report dict as produced by ``to_json`` / ``save_json``, with a
        ``records`` list. Records written before v0.6 have no cache fields
        and are treated as fully uncached.
    min_candidate_input : average input tokens per call at or above which a
        model with zero cache usage is flagged as a caching candidate
        (providers require a minimum cacheable prefix, commonly 1024 tokens).

    Returns a :class:`CacheReport` with an overall row and one row per
    model, models using the cache first, sorted by descending savings.
    Records missing a model or with unusable token or cost fields are
    skipped and counted in ``records_skipped``.
    """
    if min_candidate_input < 0:
        raise ValueError(f"min_candidate_input must be non-negative, got {min_candidate_input}")

    records = data.get("records", []) or []
    buckets: dict[str, dict[str, float]] = {}
    analyzed = 0
    skipped = 0

    for rec in records:
        if not isinstance(rec, dict):
            skipped += 1
            continue
        model = rec.get("model")
        if not model:
            skipped += 1
            continue
        try:
            input_tokens = int(rec.get("input_tokens", 0) or 0)
            cache_read = int(rec.get("cache_read_tokens", 0) or 0)
            cache_write = int(rec.get("cache_write_tokens", 0) or 0)
            cost = float(rec.get("cost_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if input_tokens < 0 or cache_read < 0 or cache_write < 0 or cost < 0:
            skipped += 1
            continue
        analyzed += 1
        bucket = buckets.setdefault(
            str(model),
            {"calls": 0, "input": 0, "read": 0, "write": 0, "cost": 0.0},
        )
        bucket["calls"] += 1
        bucket["input"] += input_tokens
        bucket["read"] += cache_read
        bucket["write"] += cache_write
        bucket["cost"] += cost

    stats: list[CacheStat] = []
    candidates: list[str] = []
    total_savings = 0.0
    all_priced = True

    for model, bucket in buckets.items():
        pricing = _resolve_pricing(model)
        savings: float | None = None
        if pricing is not None:
            savings = _record_savings(pricing, int(bucket["read"]), int(bucket["write"]))
            total_savings += savings
        else:
            all_priced = False
        stat = CacheStat(
            key=model,
            calls=int(bucket["calls"]),
            input_tokens=int(bucket["input"]),
            cache_read_tokens=int(bucket["read"]),
            cache_write_tokens=int(bucket["write"]),
            cost=bucket["cost"],
            savings=savings,
            priced=pricing is not None,
        )
        stats.append(stat)
        if not stat.uses_cache and stat.avg_input_tokens >= min_candidate_input:
            candidates.append(model)

    def _sort_key(stat: CacheStat) -> tuple[int, float, str]:
        # Cache users first by descending savings, then the rest by cost.
        if stat.uses_cache:
            return (0, -(stat.savings or 0.0), stat.key)
        return (1, -stat.cost, stat.key)

    stats.sort(key=_sort_key)
    candidates.sort()

    overall = CacheStat(
        key="(all models)",
        calls=sum(s.calls for s in stats),
        input_tokens=sum(s.input_tokens for s in stats),
        cache_read_tokens=sum(s.cache_read_tokens for s in stats),
        cache_write_tokens=sum(s.cache_write_tokens for s in stats),
        cost=sum(s.cost for s in stats),
        # With unpriced models present this is a lower bound over the priced
        # ones; priced=False signals the caveat.
        savings=total_savings if stats else None,
        priced=all_priced if stats else False,
    )

    return CacheReport(
        overall=overall,
        by_model=stats,
        candidates=candidates,
        min_candidate_input=min_candidate_input,
        records_analyzed=analyzed,
        records_skipped=skipped,
    )

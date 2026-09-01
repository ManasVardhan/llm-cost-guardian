"""Context window utilization analysis for LLM spend reports.

Shows, per model, how much of the context window the recorded calls
actually use: average, p95, and max input tokens against the model's
context window. Teams use this to spot calls running close to the limit
(truncation risk) and models that are over-provisioned for the prompts
they actually receive.

Context windows come from the built-in pricing registry (including
prefix matches for versioned model names) and can be overridden or
supplied for unknown models via the ``windows`` mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .models import get_pricing

DEFAULT_NEAR_LIMIT = 0.8


def percentile(values: list[int] | list[float], pct: float) -> float:
    """Linear-interpolation percentile (0-100) of *values*.

    Returns 0.0 for an empty list. Mirrors numpy.percentile's 'linear'
    method without requiring numpy at runtime.
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


def resolve_window(model: str, windows: dict[str, int] | None = None) -> int | None:
    """Find the context window for *model*.

    Overrides in *windows* win (exact name match), then the pricing
    registry (exact then prefix match). Returns None when unknown or the
    known window is not a positive integer.
    """
    window: int | None
    if windows and model in windows:
        window = windows[model]
    else:
        try:
            window = get_pricing(model).context_window
        except KeyError:
            return None
    if isinstance(window, int) and window > 0:
        return window
    return None


@dataclass(slots=True)
class ContextStat:
    """Context window usage for one model."""

    model: str
    calls: int
    input_tokens: list[int] = field(repr=False)
    context_window: int | None = None
    near_limit: float = DEFAULT_NEAR_LIMIT

    @property
    def avg_input_tokens(self) -> float:
        if not self.input_tokens:
            return 0.0
        return sum(self.input_tokens) / len(self.input_tokens)

    @property
    def p95_input_tokens(self) -> float:
        return percentile(self.input_tokens, 95)

    @property
    def max_input_tokens(self) -> int:
        return max(self.input_tokens) if self.input_tokens else 0

    @property
    def avg_utilization(self) -> float | None:
        """Average input tokens over the context window, or None when unknown."""
        if self.context_window is None:
            return None
        return self.avg_input_tokens / self.context_window

    @property
    def p95_utilization(self) -> float | None:
        """p95 input tokens over the context window, or None when unknown."""
        if self.context_window is None:
            return None
        return self.p95_input_tokens / self.context_window

    @property
    def max_utilization(self) -> float | None:
        """Max input tokens over the context window, or None when unknown."""
        if self.context_window is None:
            return None
        return self.max_input_tokens / self.context_window

    @property
    def calls_near_limit(self) -> int | None:
        """Number of calls at or above the near-limit fraction, or None."""
        if self.context_window is None:
            return None
        cutoff = self.near_limit * self.context_window
        return sum(1 for t in self.input_tokens if t >= cutoff)

    @property
    def is_near_limit(self) -> bool:
        """True when the p95 utilization crosses the near-limit fraction."""
        util = self.p95_utilization
        return util is not None and util >= self.near_limit

    def to_dict(self) -> dict[str, Any]:
        def _pct(value: float | None) -> float | None:
            return round(value * 100, 2) if value is not None else None

        return {
            "model": self.model,
            "calls": self.calls,
            "context_window": self.context_window,
            "avg_input_tokens": round(self.avg_input_tokens, 1),
            "p95_input_tokens": round(self.p95_input_tokens, 1),
            "max_input_tokens": self.max_input_tokens,
            "avg_utilization_pct": _pct(self.avg_utilization),
            "p95_utilization_pct": _pct(self.p95_utilization),
            "max_utilization_pct": _pct(self.max_utilization),
            "calls_near_limit": self.calls_near_limit,
            "near_limit": self.is_near_limit,
        }


@dataclass(slots=True)
class ContextReport:
    """Result of a context window utilization analysis."""

    by_model: list[ContextStat]
    records_analyzed: int
    records_skipped: int
    near_limit: float

    @property
    def models_near_limit(self) -> list[ContextStat]:
        """Models whose p95 utilization crosses the near-limit fraction."""
        return [s for s in self.by_model if s.is_near_limit]

    @property
    def models_without_window(self) -> list[ContextStat]:
        """Models with no known context window."""
        return [s for s in self.by_model if s.context_window is None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_analyzed": self.records_analyzed,
            "records_skipped": self.records_skipped,
            "near_limit_threshold_pct": round(self.near_limit * 100, 2),
            "models_near_limit": [s.model for s in self.models_near_limit],
            "models_without_window": [s.model for s in self.models_without_window],
            "by_model": [s.to_dict() for s in self.by_model],
        }


def _sort_key(stat: ContextStat) -> tuple[int, float, str]:
    """Known windows first by descending p95 utilization, unknown last."""
    util = stat.p95_utilization
    if util is None:
        return (1, -stat.p95_input_tokens, stat.model)
    return (0, -util, stat.model)


def analyze_context(
    data: dict[str, Any],
    windows: dict[str, int] | None = None,
    near_limit: float = DEFAULT_NEAR_LIMIT,
) -> ContextReport:
    """Compute per-model context window utilization for a report dict.

    Parameters
    ----------
    data : a report dict as produced by ``to_json`` / ``save_json``, with
        a ``records`` list of ``{model, input_tokens, ...}`` entries.
    windows : optional mapping of model name to context window size that
        overrides or extends the built-in registry.
    near_limit : fraction of the window (0 < near_limit <= 1) at which a
        call or model counts as near the limit. Default 0.8.

    Returns a :class:`ContextReport` with one row per model, sorted by
    descending p95 utilization (models with unknown windows last).
    Records missing a model or a usable non-negative ``input_tokens``
    are skipped and counted in ``records_skipped``.
    """
    if not 0 < near_limit <= 1:
        raise ValueError(f"near_limit must be in (0, 1], got {near_limit}")

    records = data.get("records", []) or []
    tokens_by_model: dict[str, list[int]] = {}
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
        except (TypeError, ValueError):
            skipped += 1
            continue
        if input_tokens < 0:
            skipped += 1
            continue
        analyzed += 1
        tokens_by_model.setdefault(str(model), []).append(input_tokens)

    stats = [
        ContextStat(
            model=model,
            calls=len(tokens),
            input_tokens=tokens,
            context_window=resolve_window(model, windows),
            near_limit=near_limit,
        )
        for model, tokens in tokens_by_model.items()
    ]

    return ContextReport(
        by_model=sorted(stats, key=_sort_key),
        records_analyzed=analyzed,
        records_skipped=skipped,
        near_limit=near_limit,
    )

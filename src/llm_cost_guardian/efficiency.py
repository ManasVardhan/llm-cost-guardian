"""Token efficiency reporting for LLM spend reports.

Surfaces per-model and per-tag output/input token ratios and cost per 1K
output tokens, so teams can spot prompts and models that burn input tokens
without producing much output.

The report buckets records into three dimensions: an overall row, one row
per model, and one row per tag (a record with several tags counts toward
each). For every bucket it sums calls, input tokens, output tokens, and
cost, then derives the output-to-input ratio and the cost per 1K output
tokens. A ratio below 1.0 means a bucket consumes more input than it
produces; a high cost per 1K output tokens means output is expensive to
generate.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

DIMENSION_OVERALL = "overall"
DIMENSION_MODEL = "model"
DIMENSION_TAG = "tag"

OVERALL_KEY = "(overall)"


@dataclass(slots=True)
class EfficiencyStat:
    """Token efficiency for one dimension key."""

    dimension: str
    key: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float

    @property
    def output_input_ratio(self) -> float | None:
        """Output tokens per input token, or None when no input tokens."""
        if self.input_tokens <= 0:
            return None
        return self.output_tokens / self.input_tokens

    @property
    def cost_per_1k_output(self) -> float | None:
        """USD per 1000 output tokens, or None when no output tokens."""
        if self.output_tokens <= 0:
            return None
        return self.cost_usd / self.output_tokens * 1000

    @property
    def cost_per_1k_input(self) -> float | None:
        """USD per 1000 input tokens, or None when no input tokens."""
        if self.input_tokens <= 0:
            return None
        return self.cost_usd / self.input_tokens * 1000

    def to_dict(self) -> dict[str, Any]:
        ratio = self.output_input_ratio
        cost_out = self.cost_per_1k_output
        cost_in = self.cost_per_1k_input
        return {
            "dimension": self.dimension,
            "key": self.key,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "output_input_ratio": round(ratio, 4) if ratio is not None else None,
            "cost_per_1k_output": round(cost_out, 6) if cost_out is not None else None,
            "cost_per_1k_input": round(cost_in, 6) if cost_in is not None else None,
        }


@dataclass(slots=True)
class EfficiencyReport:
    """Result of running an efficiency analysis over a report."""

    overall: EfficiencyStat
    by_model: list[EfficiencyStat]
    by_tag: list[EfficiencyStat]
    records_analyzed: int
    records_skipped: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_analyzed": self.records_analyzed,
            "records_skipped": self.records_skipped,
            "overall": self.overall.to_dict(),
            "by_model": [s.to_dict() for s in self.by_model],
            "by_tag": [s.to_dict() for s in self.by_tag],
        }


class _Bucket:
    """Mutable running totals for one dimension key."""

    __slots__ = ("calls", "input_tokens", "output_tokens", "cost_usd")

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

    def add(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost

    def to_stat(self, dimension: str, key: str) -> EfficiencyStat:
        return EfficiencyStat(
            dimension=dimension,
            key=key,
            calls=self.calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=self.cost_usd,
        )


def _sort_stats(stats: Iterable[EfficiencyStat]) -> list[EfficiencyStat]:
    """Sort most expensive first, then by key for stable ties."""
    return sorted(stats, key=lambda s: (-s.cost_usd, s.key))


def analyze_efficiency(data: dict[str, Any]) -> EfficiencyReport:
    """Compute token efficiency for a JSON report dict.

    Parameters
    ----------
    data : a report dict as produced by ``to_json`` / ``save_json``, with a
        ``records`` list of ``{model, input_tokens, output_tokens,
        cost_usd, tags, ...}`` entries.

    Returns an :class:`EfficiencyReport` with an overall row plus per-model
    and per-tag rows (sorted by descending cost). Records missing a usable
    model or numeric token or cost fields are skipped and counted in
    ``records_skipped``.
    """
    records = data.get("records", []) or []

    overall = _Bucket()
    by_model: dict[str, _Bucket] = {}
    by_tag: dict[str, _Bucket] = {}
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
            output_tokens = int(rec.get("output_tokens", 0) or 0)
            cost = float(rec.get("cost_usd", 0) or 0)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if input_tokens < 0 or output_tokens < 0:
            skipped += 1
            continue

        analyzed += 1
        overall.add(input_tokens, output_tokens, cost)
        by_model.setdefault(str(model), _Bucket()).add(input_tokens, output_tokens, cost)

        tags = rec.get("tags") or []
        if isinstance(tags, (list, tuple)):
            for tag in tags:
                if tag:
                    by_tag.setdefault(str(tag), _Bucket()).add(
                        input_tokens, output_tokens, cost
                    )

    return EfficiencyReport(
        overall=overall.to_stat(DIMENSION_OVERALL, OVERALL_KEY),
        by_model=_sort_stats(b.to_stat(DIMENSION_MODEL, k) for k, b in by_model.items()),
        by_tag=_sort_stats(b.to_stat(DIMENSION_TAG, k) for k, b in by_tag.items()),
        records_analyzed=analyzed,
        records_skipped=skipped,
    )

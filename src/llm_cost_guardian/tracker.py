"""Core cost tracking engine."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .models import get_pricing


@dataclass(slots=True)
class UsageRecord:
    """A single API call's usage and cost."""

    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


CostCallback = Callable[[UsageRecord, float], None]


class CostTracker:
    """Thread-safe accumulator for LLM API costs.

    Parameters
    ----------
    on_record : optional callback invoked after every record with
                (record, cumulative_cost).
    """

    def __init__(self, on_record: CostCallback | None = None) -> None:
        self._lock = threading.Lock()
        self._records: list[UsageRecord] = []
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._on_record = on_record

    # -- Public API ----------------------------------------------------------

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cost: float | None = None,
        metadata: dict[str, str] | None = None,
    ) -> UsageRecord:
        """Record a single API call. If *cost* is None it is calculated from pricing data."""
        if cost is None:
            pricing = get_pricing(model)
            cost = pricing.calculate_cost(input_tokens, output_tokens)

        rec = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            metadata=metadata or {},
        )

        with self._lock:
            self._records.append(rec)
            self._total_cost += cost
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            cumulative = self._total_cost

        if self._on_record:
            self._on_record(rec, cumulative)

        return rec

    @property
    def total_cost(self) -> float:
        with self._lock:
            return self._total_cost

    @property
    def total_input_tokens(self) -> int:
        with self._lock:
            return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        with self._lock:
            return self._total_output_tokens

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self._total_input_tokens + self._total_output_tokens

    @property
    def records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

    def _cost_by_model_unlocked(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for r in self._records:
            result[r.model] = result.get(r.model, 0.0) + r.cost
        return result

    def cost_by_model(self) -> dict[str, float]:
        """Return a mapping of model name to cumulative cost."""
        with self._lock:
            return self._cost_by_model_unlocked()

    def reset(self) -> None:
        """Clear all tracked data."""
        with self._lock:
            self._records.clear()
            self._total_cost = 0.0
            self._total_input_tokens = 0
            self._total_output_tokens = 0

    def summary(self) -> dict[str, object]:
        """Return a summary dict suitable for logging or display."""
        with self._lock:
            return {
                "total_cost_usd": round(self._total_cost, 6),
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "total_requests": len(self._records),
                "cost_by_model": self._cost_by_model_unlocked(),
            }

"""Core cost tracking engine."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .models import get_pricing

if TYPE_CHECKING:
    from .ledger import CostLedger

UNTAGGED = "(untagged)"
UNATTRIBUTED = "(unattributed)"


def _normalize_user(user: str | None) -> str | None:
    """Normalize a user identifier: strip whitespace, treat empty as None."""
    if user is None:
        return None
    if not isinstance(user, str):
        raise TypeError(f"User must be a string, got {type(user).__name__}: {user!r}")
    cleaned = user.strip()
    return cleaned or None


def _normalize_tags(tags: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize a tag sequence: strip whitespace, drop empties, dedupe in order."""
    if not tags:
        return ()
    seen: dict[str, None] = {}
    for tag in tags:
        if not isinstance(tag, str):
            raise TypeError(f"Tags must be strings, got {type(tag).__name__}: {tag!r}")
        cleaned = tag.strip()
        if cleaned:
            seen[cleaned] = None
    return tuple(seen)


@dataclass(slots=True)
class UsageRecord:
    """A single API call's usage and cost.

    ``input_tokens`` counts regular (uncached) input tokens.
    ``cache_read_tokens`` and ``cache_write_tokens`` count prompt cache hits
    and cache writes, which most providers bill at different rates.
    """

    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, str] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    user: str | None = None
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def cached_tokens(self) -> int:
        """Cache read plus cache write tokens."""
        return self.cache_read_tokens + self.cache_write_tokens


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
        self._total_cache_read_tokens: int = 0
        self._total_cache_write_tokens: int = 0
        self._on_record = on_record
        self._ledger: CostLedger | None = None

    # -- Public API ----------------------------------------------------------

    @property
    def on_record(self) -> CostCallback | None:
        """The callback invoked after every record, or None."""
        return self._on_record

    @on_record.setter
    def on_record(self, callback: CostCallback | None) -> None:
        self._on_record = callback

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cost: float | None = None,
        metadata: dict[str, str] | None = None,
        tags: Sequence[str] | None = None,
        user: str | None = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> UsageRecord:
        """Record a single API call. If *cost* is None it is calculated from pricing data.

        *tags* is an optional sequence of labels (project, environment, feature)
        used to group costs. Whitespace is stripped, empty tags are dropped, and
        duplicates are removed while preserving order.

        *user* is an optional identifier (username, email, API key alias) that
        attributes the call's cost to a person or key. Whitespace is stripped
        and empty strings are treated as no attribution.

        *cache_read_tokens* and *cache_write_tokens* count prompt cache hits
        and writes. When *cost* is None they are billed at the model's cache
        prices (falling back to the input rate when the model has none).
        *input_tokens* should count only regular, uncached input tokens.
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError(
                f"Token counts must be non-negative, got input_tokens={input_tokens}, "
                f"output_tokens={output_tokens}"
            )
        if cache_read_tokens < 0 or cache_write_tokens < 0:
            raise ValueError(
                f"Cache token counts must be non-negative, got "
                f"cache_read_tokens={cache_read_tokens}, "
                f"cache_write_tokens={cache_write_tokens}"
            )
        if cost is None:
            pricing = get_pricing(model)
            cost = pricing.calculate_cost(
                input_tokens,
                output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            )

        rec = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            metadata=metadata or {},
            tags=_normalize_tags(tags),
            user=_normalize_user(user),
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )

        with self._lock:
            self._records.append(rec)
            self._total_cost += cost
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_cache_read_tokens += cache_read_tokens
            self._total_cache_write_tokens += cache_write_tokens
            cumulative = self._total_cost
            ledger = self._ledger

        if ledger is not None:
            ledger.append(rec)

        if self._on_record:
            self._on_record(rec, cumulative)

        return rec

    def add_record(self, record: UsageRecord) -> UsageRecord:
        """Add an existing UsageRecord as-is, preserving its timestamp.

        Used when replaying persisted records (for example from a ledger).
        Unlike ``record``, this never recalculates cost, never writes to an
        attached ledger, and does not invoke the ``on_record`` callback.
        """
        if record.input_tokens < 0 or record.output_tokens < 0:
            raise ValueError(
                f"Token counts must be non-negative, got "
                f"input_tokens={record.input_tokens}, "
                f"output_tokens={record.output_tokens}"
            )
        if record.cache_read_tokens < 0 or record.cache_write_tokens < 0:
            raise ValueError(
                f"Cache token counts must be non-negative, got "
                f"cache_read_tokens={record.cache_read_tokens}, "
                f"cache_write_tokens={record.cache_write_tokens}"
            )
        with self._lock:
            self._records.append(record)
            self._total_cost += record.cost
            self._total_input_tokens += record.input_tokens
            self._total_output_tokens += record.output_tokens
            self._total_cache_read_tokens += record.cache_read_tokens
            self._total_cache_write_tokens += record.cache_write_tokens
        return record

    def attach_ledger(
        self,
        path: str | os.PathLike[str],
        *,
        replay: bool = False,
    ) -> CostLedger:
        """Attach a persistent JSONL ledger at *path*.

        Every subsequent ``record`` call is appended to the ledger file, so
        cost data survives process restarts. Pass ``replay=True`` to first
        load any records already in the ledger into this tracker (without
        rewriting them and without firing the ``on_record`` callback).

        Returns the attached CostLedger.
        """
        from .ledger import CostLedger

        ledger = CostLedger(path)
        if replay:
            for record in ledger.records():
                self.add_record(record)
        with self._lock:
            self._ledger = ledger
        return ledger

    def detach_ledger(self) -> None:
        """Stop persisting records to the attached ledger, if any."""
        with self._lock:
            self._ledger = None

    @property
    def ledger(self) -> CostLedger | None:
        """The attached CostLedger, or None."""
        with self._lock:
            return self._ledger

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
    def total_cache_read_tokens(self) -> int:
        with self._lock:
            return self._total_cache_read_tokens

    @property
    def total_cache_write_tokens(self) -> int:
        with self._lock:
            return self._total_cache_write_tokens

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return (
                self._total_input_tokens
                + self._total_output_tokens
                + self._total_cache_read_tokens
                + self._total_cache_write_tokens
            )

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

    def _cost_by_tag_unlocked(self) -> dict[str, float]:
        if not any(r.tags for r in self._records):
            return {}
        result: dict[str, float] = {}
        for r in self._records:
            if r.tags:
                for tag in r.tags:
                    result[tag] = result.get(tag, 0.0) + r.cost
            else:
                result[UNTAGGED] = result.get(UNTAGGED, 0.0) + r.cost
        return result

    def cost_by_tag(self) -> dict[str, float]:
        """Return a mapping of tag to cumulative cost.

        Records without tags are grouped under ``"(untagged)"``. Returns an
        empty dict when no record has any tags. A record with multiple tags
        contributes its full cost to each of its tags, so the sum across tags
        can exceed ``total_cost``.
        """
        with self._lock:
            return self._cost_by_tag_unlocked()

    def _cost_by_user_unlocked(self) -> dict[str, float]:
        if not any(r.user for r in self._records):
            return {}
        result: dict[str, float] = {}
        for r in self._records:
            key = r.user if r.user else UNATTRIBUTED
            result[key] = result.get(key, 0.0) + r.cost
        return result

    def cost_by_user(self) -> dict[str, float]:
        """Return a mapping of user to cumulative cost.

        Records without a user are grouped under ``"(unattributed)"``. Returns
        an empty dict when no record has a user. Unlike tags, each record has
        at most one user, so the sum across users always equals ``total_cost``.
        """
        with self._lock:
            return self._cost_by_user_unlocked()

    def _cost_by_day_unlocked(self, *, utc: bool) -> dict[str, float]:
        tz = timezone.utc if utc else None
        result: dict[str, float] = {}
        for r in self._records:
            day = datetime.fromtimestamp(r.timestamp, tz=tz).date().isoformat()
            result[day] = result.get(day, 0.0) + r.cost
        return dict(sorted(result.items()))

    def cost_by_day(self, *, utc: bool = False) -> dict[str, float]:
        """Return a mapping of calendar day (YYYY-MM-DD) to cumulative cost.

        Days are keyed in the local timezone by default; pass ``utc=True``
        to bucket by UTC dates instead. Keys are sorted chronologically.
        Returns an empty dict when there are no records.
        """
        with self._lock:
            return self._cost_by_day_unlocked(utc=utc)

    def reset(self) -> None:
        """Clear all tracked data."""
        with self._lock:
            self._records.clear()
            self._total_cost = 0.0
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._total_cache_read_tokens = 0
            self._total_cache_write_tokens = 0

    @property
    def average_cost(self) -> float:
        """Return the average cost per request, or 0.0 if no records."""
        with self._lock:
            if not self._records:
                return 0.0
            return self._total_cost / len(self._records)

    @property
    def last_record(self) -> UsageRecord | None:
        """Return the most recent usage record, or None if empty."""
        with self._lock:
            return self._records[-1] if self._records else None

    def filter(
        self,
        *,
        model: str | None = None,
        since: float | None = None,
        until: float | None = None,
        min_cost: float | None = None,
        tag: str | None = None,
        user: str | None = None,
        predicate: Callable[[UsageRecord], bool] | None = None,
    ) -> list[UsageRecord]:
        """Filter records by model name, time range, minimum cost, tag, user, or custom predicate.

        Parameters
        ----------
        model : only include records for this model (exact match)
        since : only include records with timestamp >= this value
        until : only include records with timestamp <= this value
        min_cost : only include records with cost >= this value
        tag : only include records carrying this tag (exact match)
        user : only include records attributed to this user (exact match)
        predicate : custom filter function applied to each record

        Returns
        -------
        A new list of matching UsageRecord objects.
        """
        with self._lock:
            results = list(self._records)
        if model is not None:
            results = [r for r in results if r.model == model]
        if since is not None:
            results = [r for r in results if r.timestamp >= since]
        if until is not None:
            results = [r for r in results if r.timestamp <= until]
        if min_cost is not None:
            results = [r for r in results if r.cost >= min_cost]
        if tag is not None:
            results = [r for r in results if tag in r.tags]
        if user is not None:
            results = [r for r in results if r.user == user]
        if predicate is not None:
            results = [r for r in results if predicate(r)]
        return results

    def summary(self) -> dict[str, object]:
        """Return a summary dict suitable for logging or display."""
        with self._lock:
            return {
                "total_cost_usd": round(self._total_cost, 6),
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "total_cache_read_tokens": self._total_cache_read_tokens,
                "total_cache_write_tokens": self._total_cache_write_tokens,
                "total_requests": len(self._records),
                "cost_by_model": self._cost_by_model_unlocked(),
                "cost_by_tag": self._cost_by_tag_unlocked(),
                "cost_by_user": self._cost_by_user_unlocked(),
            }

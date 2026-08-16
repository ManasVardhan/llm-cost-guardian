"""Merge multiple cost sources into a single tracker.

Teams often end up with one cost file per service, per host, or per CI run:
some are JSONL ledgers written by ``CostTracker.attach_ledger``, others are
JSON reports written by ``save_json``. This module combines any mix of them
into one deduplicated view::

    from llm_cost_guardian import merge_sources

    result = merge_sources(["api.jsonl", "worker.jsonl", "batch-report.json"])
    print(result.tracker.total_cost)
    print(result.duplicates_removed)

The same pipeline powers the ``llm-cost-guardian merge`` CLI command.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .ledger import _line_to_record, record_from_dict
from .tracker import CostTracker, UsageRecord


class MergeError(ValueError):
    """Raised when a merge source cannot be read or is not a recognized format."""


@dataclass(slots=True)
class SourceStats:
    """Per-source accounting for a merge."""

    path: str
    format: str  # "report" or "ledger"
    records: int
    skipped: int


@dataclass(slots=True)
class MergeResult:
    """Outcome of merging one or more cost sources."""

    tracker: CostTracker
    sources: list[SourceStats]
    duplicates_removed: int

    @property
    def total_records(self) -> int:
        """Number of records in the merged tracker."""
        return len(self.tracker.records)

    @property
    def total_skipped(self) -> int:
        """Total malformed entries skipped across all sources."""
        return sum(s.skipped for s in self.sources)


def load_records(path: str | os.PathLike[str]) -> tuple[str, list[UsageRecord], int]:
    """Load cost records from a JSON report or JSONL ledger, auto-detected.

    Returns ``(format, records, skipped)`` where *format* is ``"report"`` or
    ``"ledger"`` and *skipped* counts malformed entries that were ignored.

    Detection: if the whole file parses as a JSON object with a ``records``
    list it is treated as a report; a JSON object that is itself a single
    record is treated as a one-line ledger; anything else is read line by
    line as a JSONL ledger.
    """
    p = Path(path)
    try:
        text = p.read_text()
    except FileNotFoundError:
        raise MergeError(f"Source file not found: {p}") from None
    except OSError as e:
        raise MergeError(f"Could not read {p}: {e}") from None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict):
        raw_records = data.get("records")
        if isinstance(raw_records, list):
            records: list[UsageRecord] = []
            skipped = 0
            for item in raw_records:
                rec = record_from_dict(item)
                if rec is None:
                    skipped += 1
                else:
                    records.append(rec)
            return "report", records, skipped
        single = record_from_dict(data)
        if single is not None:
            return "ledger", [single], 0
        raise MergeError(
            f"{p} is a JSON object but is neither a cost report (no 'records' list) "
            f"nor a ledger record."
        )
    if data is not None:
        raise MergeError(
            f"{p} contains a JSON {type(data).__name__}; expected a report object "
            f"or a JSONL ledger."
        )

    records = []
    skipped = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rec = _line_to_record(stripped)
        if rec is None:
            skipped += 1
        else:
            records.append(rec)
    return "ledger", records, skipped


def _dedupe_key(record: UsageRecord) -> tuple[object, ...]:
    return (
        record.model,
        record.input_tokens,
        record.output_tokens,
        round(record.cost, 8),
        round(record.timestamp, 6),
        record.tags,
        record.user,
        tuple(sorted(record.metadata.items())),
    )


def merge_sources(
    paths: Sequence[str | os.PathLike[str]],
    *,
    dedupe: bool = True,
) -> MergeResult:
    """Merge JSONL ledgers and JSON reports into a single CostTracker.

    Records identical in every field (model, tokens, cost, timestamp, tags,
    user, metadata) are collapsed to one when *dedupe* is True, so merging
    overlapping exports does not double count spend. Merged records are
    sorted by timestamp.

    Raises MergeError when *paths* is empty or a source is unreadable or in
    an unrecognized format.
    """
    if not paths:
        raise MergeError("No sources given; pass at least one ledger or report file.")

    sources: list[SourceStats] = []
    combined: list[UsageRecord] = []
    for path in paths:
        fmt, records, skipped = load_records(path)
        sources.append(
            SourceStats(path=str(path), format=fmt, records=len(records), skipped=skipped)
        )
        combined.extend(records)

    duplicates = 0
    if dedupe:
        seen: set[tuple[object, ...]] = set()
        unique: list[UsageRecord] = []
        for record in combined:
            key = _dedupe_key(record)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            unique.append(record)
        combined = unique

    combined.sort(key=lambda r: r.timestamp)
    tracker = CostTracker()
    for record in combined:
        tracker.add_record(record)

    return MergeResult(tracker=tracker, sources=sources, duplicates_removed=duplicates)

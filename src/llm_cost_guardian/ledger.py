"""Persistent append-only cost ledger (JSONL).

A ledger lets cost data survive process restarts without manually saving and
merging JSON reports. Each line is one JSON object using the same record
schema as the JSON exporter, so ledgers are greppable, diffable, and easy to
ship to log pipelines.

Typical usage::

    tracker = CostTracker()
    tracker.attach_ledger("costs.jsonl", replay=True)
    tracker.record("gpt-4o", 1000, 200)   # appended to costs.jsonl

    # Later, in another process:
    ledger = CostLedger("costs.jsonl")
    tracker = ledger.to_tracker()
    print(tracker.total_cost)
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .tracker import CostTracker, UsageRecord

_REQUIRED_FIELDS = ("model", "input_tokens", "output_tokens", "cost_usd")


def _record_to_line(record: UsageRecord) -> str:
    """Serialize a UsageRecord to a single JSONL line (no trailing newline)."""
    return json.dumps(
        {
            "model": record.model,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "cost_usd": round(record.cost, 8),
            "timestamp": record.timestamp,
            "metadata": record.metadata,
            "tags": list(record.tags),
            "user": record.user,
        },
        default=str,
    )


def _line_to_record(line: str) -> UsageRecord | None:
    """Parse one JSONL line into a UsageRecord, or None if malformed."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    return record_from_dict(data)


def record_from_dict(data: object) -> UsageRecord | None:
    """Build a UsageRecord from a plain dict in the record schema, or None if malformed.

    Accepts the schema shared by ledger lines and JSON report entries:
    ``model``, ``input_tokens``, ``output_tokens``, ``cost_usd`` are required;
    ``timestamp``, ``metadata``, ``tags``, and ``user`` are optional.
    """
    if not isinstance(data, dict):
        return None
    for field in _REQUIRED_FIELDS:
        if field not in data:
            return None
    try:
        model = data["model"]
        if not isinstance(model, str) or not model:
            return None
        input_tokens = int(data["input_tokens"])
        output_tokens = int(data["output_tokens"])
        cost = float(data["cost_usd"])
        timestamp = float(data.get("timestamp") or 0.0)
    except (TypeError, ValueError):
        return None
    if input_tokens < 0 or output_tokens < 0 or timestamp < 0:
        return None

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    raw_tags = data.get("tags")
    tags: tuple[str, ...] = ()
    if isinstance(raw_tags, list):
        tags = tuple(t for t in raw_tags if isinstance(t, str) and t)
    user = data.get("user")
    if not isinstance(user, str) or not user.strip():
        user = None

    return UsageRecord(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        timestamp=timestamp,
        metadata={str(k): str(v) for k, v in metadata.items()},
        tags=tags,
        user=user,
    )


class CostLedger:
    """Append-only JSONL ledger for durable cost records.

    Parameters
    ----------
    path : file path for the ledger. Parent directories are created on first
           append. The file itself is created lazily.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._skipped_lines = 0

    @property
    def skipped_lines(self) -> int:
        """Number of malformed lines skipped during the most recent read."""
        return self._skipped_lines

    def append(self, record: UsageRecord) -> None:
        """Append a single record to the ledger file."""
        line = _record_to_line(record)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(line + "\n")

    def records(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
    ) -> list[UsageRecord]:
        """Read all records from the ledger, oldest first.

        Malformed lines are skipped (their count is available via
        ``skipped_lines``). Returns an empty list when the file does not
        exist yet.

        Parameters
        ----------
        since : only include records with timestamp >= this value
        until : only include records with timestamp <= this value
        """
        skipped = 0
        results: list[UsageRecord] = []
        try:
            with self._lock, open(self.path) as f:
                lines = f.readlines()
        except FileNotFoundError:
            self._skipped_lines = 0
            return []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            record = _line_to_record(stripped)
            if record is None:
                skipped += 1
                continue
            if since is not None and record.timestamp < since:
                continue
            if until is not None and record.timestamp > until:
                continue
            results.append(record)

        self._skipped_lines = skipped
        return results

    def to_tracker(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
    ) -> CostTracker:
        """Load ledger records into a fresh CostTracker.

        The returned tracker is not attached to this ledger, so recording on
        it will not write new lines. Use ``CostTracker.attach_ledger`` for a
        tracker that persists.
        """
        tracker = CostTracker()
        for record in self.records(since=since, until=until):
            tracker.add_record(record)
        return tracker

    def __len__(self) -> int:
        """Number of valid records currently in the ledger."""
        return len(self.records())

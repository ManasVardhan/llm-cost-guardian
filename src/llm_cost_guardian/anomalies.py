"""Cost anomaly detection for LLM spend reports.

Flags days, models, or users whose daily spend spikes versus their trailing
average, so unexpected cost jumps surface before the invoice does.

The detector buckets report records into calendar days (local time or UTC)
and compares each day's spend to the mean of the active days (spend above
zero) inside the trailing ``window``, so bursty every-other-day usage does
not false alarm. A day is anomalous when spend is at least ``threshold``
times its baseline and at least ``min_spend`` USD. Spend whose entire
baseline window was quiet is flagged as new spend (no ratio).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

DIMENSION_TOTAL = "total"
DIMENSION_MODEL = "model"
DIMENSION_USER = "user"

TOTAL_KEY = "(total)"


@dataclass(frozen=True, slots=True)
class Anomaly:
    """A single anomalous day for one dimension key."""

    dimension: str
    key: str
    day: str
    spend_usd: float
    baseline_usd: float
    ratio: float | None
    calls: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "key": self.key,
            "day": self.day,
            "spend_usd": round(self.spend_usd, 6),
            "baseline_usd": round(self.baseline_usd, 6),
            "ratio": round(self.ratio, 2) if self.ratio is not None else None,
            "calls": self.calls,
        }


@dataclass(slots=True)
class AnomalyReport:
    """Result of running anomaly detection over a report."""

    anomalies: list[Anomaly]
    days_analyzed: int
    records_analyzed: int
    records_skipped: int
    window: int
    threshold: float
    min_spend: float
    min_history: int
    timezone: str

    @property
    def has_anomalies(self) -> bool:
        return bool(self.anomalies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": {
                "window": self.window,
                "threshold": self.threshold,
                "min_spend": self.min_spend,
                "min_history": self.min_history,
                "timezone": self.timezone,
            },
            "days_analyzed": self.days_analyzed,
            "records_analyzed": self.records_analyzed,
            "records_skipped": self.records_skipped,
            "anomaly_count": len(self.anomalies),
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


def _record_day(record: dict[str, Any], tz: timezone | None) -> str | None:
    """Return the ISO calendar day for a record, or None if unusable."""
    raw_ts = record.get("timestamp")
    try:
        ts = float(raw_ts)  # type: ignore[arg-type]
        if ts <= 0:
            return None
        return datetime.fromtimestamp(ts, tz=tz).date().isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _detect_series(
    dimension: str,
    key: str,
    daily: dict[str, tuple[float, int]],
    all_days: list[str],
    *,
    window: int,
    threshold: float,
    min_spend: float,
    min_history: int,
) -> Iterable[Anomaly]:
    """Yield anomalies for one daily spend series.

    ``all_days`` is the full, sorted calendar range of the report. The
    baseline is the mean of the active days (spend above zero) inside the
    trailing window, so bursty every-other-day usage does not false alarm.
    A day whose entire window is quiet is reported as new spend.
    """
    spends = [daily.get(day, (0.0, 0))[0] for day in all_days]
    for i, day in enumerate(all_days):
        spend, calls = daily.get(day, (0.0, 0))
        if i < min_history or spend < min_spend:
            continue
        active = [s for s in spends[max(0, i - window) : i] if s > 0]
        if not active:
            yield Anomaly(dimension, key, day, spend, 0.0, None, calls)
            continue
        baseline = sum(active) / len(active)
        if spend >= threshold * baseline:
            yield Anomaly(dimension, key, day, spend, baseline, spend / baseline, calls)


def analyze_anomalies(
    data: dict[str, Any],
    *,
    window: int = 7,
    threshold: float = 2.0,
    min_spend: float = 0.01,
    min_history: int = 3,
    utc: bool = False,
) -> AnomalyReport:
    """Detect daily spend anomalies in a JSON report dict.

    Parameters
    ----------
    data : a report dict as produced by ``to_json`` / ``save_json``
    window : trailing days used for the baseline average (default 7)
    threshold : spend must be at least this multiple of the baseline
        to be flagged (default 2.0)
    min_spend : ignore days below this many USD, filtering noise on
        near-zero spend (default 0.01)
    min_history : calendar days of history required before a day can be
        flagged, avoiding false positives at the start of a report
        (default 3)
    utc : bucket days by UTC instead of local time

    Returns an :class:`AnomalyReport` covering three dimensions: total daily
    spend, per-model daily spend, and per-user daily spend.

    Raises
    ------
    ValueError
        If window or min_history is less than 1, or threshold or min_spend
        is not positive.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    if min_history < 1:
        raise ValueError(f"min_history must be at least 1, got {min_history}")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if min_spend <= 0:
        raise ValueError(f"min_spend must be positive, got {min_spend}")

    tz = timezone.utc if utc else None
    records = data.get("records", []) or []

    totals: dict[str, tuple[float, int]] = {}
    by_model: dict[str, dict[str, tuple[float, int]]] = {}
    by_user: dict[str, dict[str, tuple[float, int]]] = {}
    analyzed = 0
    skipped = 0

    def _add(bucket: dict[str, tuple[float, int]], day: str, cost: float) -> None:
        spend, calls = bucket.get(day, (0.0, 0))
        bucket[day] = (spend + cost, calls + 1)

    for rec in records:
        if not isinstance(rec, dict):
            skipped += 1
            continue
        day = _record_day(rec, tz)
        if day is None:
            skipped += 1
            continue
        try:
            cost = float(rec.get("cost_usd", 0) or 0)
        except (TypeError, ValueError):
            skipped += 1
            continue
        analyzed += 1
        _add(totals, day, cost)
        model = rec.get("model")
        if model:
            _add(by_model.setdefault(str(model), {}), day, cost)
        user = rec.get("user")
        if user:
            _add(by_user.setdefault(str(user), {}), day, cost)

    tz_label = "utc" if utc else "local"
    if not totals:
        return AnomalyReport(
            anomalies=[],
            days_analyzed=0,
            records_analyzed=analyzed,
            records_skipped=skipped,
            window=window,
            threshold=threshold,
            min_spend=min_spend,
            min_history=min_history,
            timezone=tz_label,
        )

    first = datetime.fromisoformat(min(totals)).date()
    last = datetime.fromisoformat(max(totals)).date()
    all_days = [
        (first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)
    ]

    anomalies: list[Anomaly] = []
    anomalies.extend(
        _detect_series(
            DIMENSION_TOTAL,
            TOTAL_KEY,
            totals,
            all_days,
            window=window,
            threshold=threshold,
            min_spend=min_spend,
            min_history=min_history,
        )
    )
    for model_name in sorted(by_model):
        anomalies.extend(
            _detect_series(
                DIMENSION_MODEL,
                model_name,
                by_model[model_name],
                all_days,
                window=window,
                threshold=threshold,
                min_spend=min_spend,
                min_history=min_history,
            )
        )
    for user_name in sorted(by_user):
        anomalies.extend(
            _detect_series(
                DIMENSION_USER,
                user_name,
                by_user[user_name],
                all_days,
                window=window,
                threshold=threshold,
                min_spend=min_spend,
                min_history=min_history,
            )
        )

    dimension_order = {DIMENSION_TOTAL: 0, DIMENSION_MODEL: 1, DIMENSION_USER: 2}
    anomalies.sort(key=lambda a: (a.day, dimension_order[a.dimension], -a.spend_usd, a.key))

    return AnomalyReport(
        anomalies=anomalies,
        days_analyzed=len(all_days),
        records_analyzed=analyzed,
        records_skipped=skipped,
        window=window,
        threshold=threshold,
        min_spend=min_spend,
        min_history=min_history,
        timezone=tz_label,
    )

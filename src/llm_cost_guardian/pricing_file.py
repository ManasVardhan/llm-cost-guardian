"""Load and pin model prices from a local JSON or YAML file.

Teams negotiate their own rates, providers add models, and prices change
between releases. This module lets you keep a price table in a file and load
it at runtime instead of waiting for a package upgrade:

    from llm_cost_guardian import apply_price_file
    apply_price_file("prices.json")   # registers every model in the file

The file is a JSON or YAML object with a "models" list (a bare list is also
accepted). Each entry needs a name, provider, input_cost_per_1m, and
output_cost_per_1m; context_window and the two cache prices are optional::

    {
      "models": [
        {
          "name": "acme-negotiated-gpt4",
          "provider": "openai",
          "input_cost_per_1m": 1.80,
          "output_cost_per_1m": 7.20,
          "context_window": 128000,
          "cache_read_cost_per_1m": 0.45
        }
      ]
    }

validate_price_file reports every problem without touching the registry, and
diff_price_file shows what a file would change versus the prices in effect
now, so a table can be reviewed before it is applied.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .models import PRICING, ModelPricing, Provider, register_model

REQUIRED_FIELDS = ("name", "provider", "input_cost_per_1m", "output_cost_per_1m")
OPTIONAL_FIELDS = (
    "context_window",
    "cache_read_cost_per_1m",
    "cache_write_cost_per_1m",
)
KNOWN_FIELDS = frozenset(REQUIRED_FIELDS + OPTIONAL_FIELDS)

# Numeric price fields compared by diff and shown by view.
PRICE_FIELDS = (
    "input_cost_per_1m",
    "output_cost_per_1m",
    "cache_read_cost_per_1m",
    "cache_write_cost_per_1m",
)


class PriceFileError(Exception):
    """Raised when a price file cannot be parsed or is invalid."""


@dataclass(frozen=True)
class PriceEntry:
    """One validated model price parsed from a file."""

    name: str
    provider: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    context_window: int | None = None
    cache_read_cost_per_1m: float | None = None
    cache_write_cost_per_1m: float | None = None

    def to_model_pricing(self) -> ModelPricing:
        return ModelPricing(
            name=self.name,
            provider=Provider(self.provider),
            input_cost_per_1m=self.input_cost_per_1m,
            output_cost_per_1m=self.output_cost_per_1m,
            context_window=self.context_window,
            cache_read_cost_per_1m=self.cache_read_cost_per_1m,
            cache_write_cost_per_1m=self.cache_write_cost_per_1m,
        )


def _read_raw(path: str) -> Any:
    if not os.path.exists(path):
        raise PriceFileError(f"Price file not found: {path}")
    with open(path) as f:
        text = f.read()

    is_yaml = path.lower().endswith((".yaml", ".yml"))
    if is_yaml:
        try:
            import yaml
        except ImportError as exc:
            raise PriceFileError(
                "Reading a YAML price file needs PyYAML. Install it with "
                "'pip install llm-cost-guardian[yaml]' or use a JSON file."
            ) from exc
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise PriceFileError(f"{path} is not valid YAML: {exc}") from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise PriceFileError(f"{path} is not valid JSON: {exc}") from exc


def _extract_entries(data: Any, path: str) -> list[Any]:
    if isinstance(data, dict):
        if "models" not in data:
            raise PriceFileError(
                f"{path} must have a 'models' list (or be a list of entries)."
            )
        models = data["models"]
    else:
        models = data
    if not isinstance(models, list):
        raise PriceFileError(f"{path}: 'models' must be a list, got {type(models).__name__}.")
    return models


def _validate_entry(raw: Any, index: int) -> tuple[PriceEntry | None, list[str]]:
    """Validate one raw entry. Returns (entry or None, errors)."""
    errors: list[str] = []
    where = f"entry {index}"
    if not isinstance(raw, dict):
        return None, [f"{where}: must be an object, got {type(raw).__name__}"]

    name = raw.get("name")
    if isinstance(name, str) and name:
        where = f"model {name!r}"

    unknown = set(raw) - KNOWN_FIELDS
    if unknown:
        errors.append(f"{where}: unknown field(s) {', '.join(sorted(unknown))}")

    for field in REQUIRED_FIELDS:
        if raw.get(field) is None:
            errors.append(f"{where}: missing required field '{field}'")

    if name is not None and (not isinstance(name, str) or not name):
        errors.append(f"{where}: 'name' must be a non-empty string")

    provider = raw.get("provider")
    if provider is not None:
        valid = {p.value for p in Provider}
        if provider not in valid:
            errors.append(
                f"{where}: unknown provider {provider!r} (expected one of "
                f"{', '.join(sorted(valid))})"
            )

    for field in ("input_cost_per_1m", "output_cost_per_1m",
                  "cache_read_cost_per_1m", "cache_write_cost_per_1m"):
        value = raw.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{where}: '{field}' must be a number, got {value!r}")
        elif value < 0:
            errors.append(f"{where}: '{field}' must be non-negative, got {value}")

    window = raw.get("context_window")
    if window is not None:
        if isinstance(window, bool) or not isinstance(window, int):
            errors.append(f"{where}: 'context_window' must be an integer, got {window!r}")
        elif window <= 0:
            errors.append(f"{where}: 'context_window' must be positive, got {window}")

    if errors:
        return None, errors

    return (
        PriceEntry(
            name=name,
            provider=provider,
            input_cost_per_1m=float(raw["input_cost_per_1m"]),
            output_cost_per_1m=float(raw["output_cost_per_1m"]),
            context_window=raw.get("context_window"),
            cache_read_cost_per_1m=raw.get("cache_read_cost_per_1m"),
            cache_write_cost_per_1m=raw.get("cache_write_cost_per_1m"),
        ),
        [],
    )


def load_price_file(path: str) -> list[PriceEntry]:
    """Parse and validate a price file, returning its entries.

    Raises PriceFileError with a message listing every problem when the file
    is missing, unparseable, or contains invalid entries. The pricing
    registry is not modified.
    """
    data = _read_raw(path)
    raw_entries = _extract_entries(data, path)

    entries: list[PriceEntry] = []
    errors: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_entries):
        entry, entry_errors = _validate_entry(raw, index)
        errors.extend(entry_errors)
        if entry is None:
            continue
        if entry.name in seen:
            errors.append(f"model {entry.name!r}: duplicate entry")
            continue
        seen.add(entry.name)
        entries.append(entry)

    if errors:
        joined = "\n  ".join(errors)
        raise PriceFileError(f"{path} has {len(errors)} problem(s):\n  {joined}")

    return entries


def validate_price_file(path: str) -> list[str]:
    """Return a list of problems with a price file, empty when it is valid.

    Unlike load_price_file this never raises for validation problems, so it
    suits a CLI validate command. It still surfaces parse and IO errors as a
    single-item list.
    """
    try:
        load_price_file(path)
    except PriceFileError as exc:
        message = str(exc)
        if "\n  " in message:
            return [line.strip() for line in message.split("\n  ")[1:]]
        return [message]
    return []


def apply_price_file(path: str, *, replace: bool = False) -> list[ModelPricing]:
    """Load a price file and register every model in it.

    Args:
        path: Path to the JSON or YAML price file.
        replace: When True, clear the built-in registry first so only the
            file's models are known (pin the table exactly). When False
            (default), the file's models are merged in, overriding built-ins
            with the same name and leaving the rest in place.

    Returns:
        The ModelPricing objects that were registered.

    Raises:
        PriceFileError: If the file is missing, unparseable, or invalid.
    """
    entries = load_price_file(path)
    if replace:
        PRICING.clear()
    registered: list[ModelPricing] = []
    for entry in entries:
        registered.append(
            register_model(
                name=entry.name,
                provider=entry.provider,
                input_cost_per_1m=entry.input_cost_per_1m,
                output_cost_per_1m=entry.output_cost_per_1m,
                context_window=entry.context_window,
                cache_read_cost_per_1m=entry.cache_read_cost_per_1m,
                cache_write_cost_per_1m=entry.cache_write_cost_per_1m,
            )
        )
    return registered


@dataclass(frozen=True)
class PriceChange:
    """A single field's old and new value for a changed model."""

    field: str
    old: float | None
    new: float | None


@dataclass(frozen=True)
class PriceDiffRow:
    """How one model in a file compares to the active registry."""

    name: str
    status: str  # "added", "changed", or "unchanged"
    changes: list[PriceChange]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "changes": [
                {"field": c.field, "old": c.old, "new": c.new} for c in self.changes
            ],
        }


@dataclass(frozen=True)
class PriceDiff:
    """The full comparison of a price file against the active registry."""

    rows: list[PriceDiffRow]

    @property
    def added(self) -> list[PriceDiffRow]:
        return [r for r in self.rows if r.status == "added"]

    @property
    def changed(self) -> list[PriceDiffRow]:
        return [r for r in self.rows if r.status == "changed"]

    @property
    def unchanged(self) -> list[PriceDiffRow]:
        return [r for r in self.rows if r.status == "unchanged"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": len(self.added),
            "changed": len(self.changed),
            "unchanged": len(self.unchanged),
            "rows": [r.to_dict() for r in self.rows],
        }


def diff_price_file(path: str) -> PriceDiff:
    """Compare a price file against the prices currently in effect.

    For every model in the file the diff reports whether it is new (added),
    differs from the registered price (changed, with per-field old and new
    values), or matches (unchanged). The registry is not modified.

    Raises:
        PriceFileError: If the file is missing, unparseable, or invalid.
    """
    entries = load_price_file(path)
    rows: list[PriceDiffRow] = []
    for entry in entries:
        current = PRICING.get(entry.name)
        if current is None:
            rows.append(PriceDiffRow(entry.name, "added", []))
            continue

        changes: list[PriceChange] = []
        for field in PRICE_FIELDS:
            old = getattr(current, field)
            new = getattr(entry, field)
            if old != new:
                changes.append(PriceChange(field, old, new))
        old_window = current.context_window
        new_window = entry.context_window
        if new_window is not None and new_window != old_window:
            changes.append(
                PriceChange("context_window", old_window, new_window)
            )

        status = "changed" if changes else "unchanged"
        rows.append(PriceDiffRow(entry.name, status, changes))
    return PriceDiff(rows=rows)

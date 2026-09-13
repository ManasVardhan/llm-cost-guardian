"""Tests for loading model prices from a local JSON or YAML file."""

from __future__ import annotations

import json

import pytest

from llm_cost_guardian import (
    PriceFileError,
    apply_price_file,
    diff_price_file,
    get_pricing,
    load_price_file,
    validate_price_file,
)
from llm_cost_guardian.models import PRICING


@pytest.fixture(autouse=True)
def restore_registry():
    """Snapshot the pricing registry and restore it after each test.

    apply_price_file(replace=True) clears the global registry, so every test
    that touches it must leave it exactly as it found it.
    """
    snapshot = dict(PRICING)
    try:
        yield
    finally:
        PRICING.clear()
        PRICING.update(snapshot)


def _write(tmp_path, name, data):
    path = tmp_path / name
    if name.endswith(".json"):
        path.write_text(json.dumps(data))
    else:
        path.write_text(data)
    return str(path)


VALID_ENTRY = {
    "name": "acme-gpt",
    "provider": "openai",
    "input_cost_per_1m": 1.80,
    "output_cost_per_1m": 7.20,
    "context_window": 128000,
    "cache_read_cost_per_1m": 0.45,
}


class TestLoadPriceFile:
    def test_loads_models_list(self, tmp_path):
        path = _write(tmp_path, "p.json", {"models": [VALID_ENTRY]})
        entries = load_price_file(path)
        assert len(entries) == 1
        assert entries[0].name == "acme-gpt"
        assert entries[0].input_cost_per_1m == 1.80
        assert entries[0].cache_read_cost_per_1m == 0.45

    def test_accepts_bare_list(self, tmp_path):
        path = _write(tmp_path, "p.json", [VALID_ENTRY])
        entries = load_price_file(path)
        assert len(entries) == 1

    def test_optional_fields_default_none(self, tmp_path):
        entry = {
            "name": "minimal",
            "provider": "google",
            "input_cost_per_1m": 1.0,
            "output_cost_per_1m": 2.0,
        }
        path = _write(tmp_path, "p.json", {"models": [entry]})
        loaded = load_price_file(path)[0]
        assert loaded.context_window is None
        assert loaded.cache_read_cost_per_1m is None
        assert loaded.cache_write_cost_per_1m is None

    def test_missing_file_raises(self):
        with pytest.raises(PriceFileError, match="not found"):
            load_price_file("/no/such/prices.json")

    def test_bad_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(PriceFileError, match="not valid JSON"):
            load_price_file(str(path))

    def test_dict_without_models_key_raises(self, tmp_path):
        path = _write(tmp_path, "p.json", {"prices": []})
        with pytest.raises(PriceFileError, match="must have a 'models' list"):
            load_price_file(path)

    def test_missing_required_field_raises(self, tmp_path):
        path = _write(tmp_path, "p.json", {"models": [{"name": "x", "provider": "openai"}]})
        with pytest.raises(PriceFileError, match="missing required field"):
            load_price_file(path)

    def test_unknown_provider_raises(self, tmp_path):
        entry = dict(VALID_ENTRY, provider="acme-cloud")
        path = _write(tmp_path, "p.json", {"models": [entry]})
        with pytest.raises(PriceFileError, match="unknown provider"):
            load_price_file(path)

    def test_negative_cost_raises(self, tmp_path):
        entry = dict(VALID_ENTRY, input_cost_per_1m=-1.0)
        path = _write(tmp_path, "p.json", {"models": [entry]})
        with pytest.raises(PriceFileError, match="non-negative"):
            load_price_file(path)

    def test_non_numeric_cost_raises(self, tmp_path):
        entry = dict(VALID_ENTRY, output_cost_per_1m="cheap")
        path = _write(tmp_path, "p.json", {"models": [entry]})
        with pytest.raises(PriceFileError, match="must be a number"):
            load_price_file(path)

    def test_bool_is_not_a_valid_cost(self, tmp_path):
        entry = dict(VALID_ENTRY, input_cost_per_1m=True)
        path = _write(tmp_path, "p.json", {"models": [entry]})
        with pytest.raises(PriceFileError, match="must be a number"):
            load_price_file(path)

    def test_unknown_field_raises(self, tmp_path):
        entry = dict(VALID_ENTRY, discount=0.1)
        path = _write(tmp_path, "p.json", {"models": [entry]})
        with pytest.raises(PriceFileError, match="unknown field"):
            load_price_file(path)

    def test_duplicate_name_raises(self, tmp_path):
        path = _write(tmp_path, "p.json", {"models": [VALID_ENTRY, VALID_ENTRY]})
        with pytest.raises(PriceFileError, match="duplicate entry"):
            load_price_file(path)

    def test_reports_multiple_problems(self, tmp_path):
        entry = {"name": "x", "provider": "nope", "input_cost_per_1m": -1}
        path = _write(tmp_path, "p.json", {"models": [entry]})
        with pytest.raises(PriceFileError) as exc:
            load_price_file(path)
        assert "problem(s)" in str(exc.value)


class TestValidatePriceFile:
    def test_valid_file_no_errors(self, tmp_path):
        path = _write(tmp_path, "p.json", {"models": [VALID_ENTRY]})
        assert validate_price_file(path) == []

    def test_invalid_file_lists_errors(self, tmp_path):
        entry = {"name": "x", "provider": "nope", "input_cost_per_1m": -1}
        path = _write(tmp_path, "p.json", {"models": [entry]})
        errors = validate_price_file(path)
        assert len(errors) >= 2
        assert any("provider" in e for e in errors)

    def test_missing_file_returns_error(self):
        errors = validate_price_file("/no/such.json")
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_does_not_touch_registry(self, tmp_path):
        before = dict(PRICING)
        path = _write(tmp_path, "p.json", {"models": [VALID_ENTRY]})
        validate_price_file(path)
        assert "acme-gpt" not in PRICING
        assert PRICING == before


class TestApplyPriceFile:
    def test_merge_registers_new_model(self, tmp_path):
        path = _write(tmp_path, "p.json", {"models": [VALID_ENTRY]})
        apply_price_file(path)
        pricing = get_pricing("acme-gpt")
        assert pricing.input_cost_per_1m == 1.80
        # Built-ins are still present under merge.
        assert "gpt-4o" in PRICING

    def test_merge_overrides_builtin(self, tmp_path):
        entry = {
            "name": "gpt-4o",
            "provider": "openai",
            "input_cost_per_1m": 0.99,
            "output_cost_per_1m": 3.99,
        }
        path = _write(tmp_path, "p.json", {"models": [entry]})
        apply_price_file(path)
        assert get_pricing("gpt-4o").input_cost_per_1m == 0.99

    def test_replace_pins_only_file_models(self, tmp_path):
        path = _write(tmp_path, "p.json", {"models": [VALID_ENTRY]})
        apply_price_file(path, replace=True)
        assert set(PRICING) == {"acme-gpt"}
        assert "gpt-4o" not in PRICING

    def test_returns_registered_models(self, tmp_path):
        path = _write(tmp_path, "p.json", {"models": [VALID_ENTRY]})
        registered = apply_price_file(path)
        assert [m.name for m in registered] == ["acme-gpt"]

    def test_invalid_file_registers_nothing(self, tmp_path):
        entry = dict(VALID_ENTRY, provider="nope")
        path = _write(tmp_path, "p.json", {"models": [entry]})
        with pytest.raises(PriceFileError):
            apply_price_file(path)
        assert "acme-gpt" not in PRICING


class TestDiffPriceFile:
    def test_added_model(self, tmp_path):
        path = _write(tmp_path, "p.json", {"models": [VALID_ENTRY]})
        diff = diff_price_file(path)
        assert len(diff.added) == 1
        assert diff.added[0].name == "acme-gpt"
        assert diff.changed == []

    def test_unchanged_model(self, tmp_path):
        current = PRICING["gpt-4o"]
        entry = {
            "name": "gpt-4o",
            "provider": "openai",
            "input_cost_per_1m": current.input_cost_per_1m,
            "output_cost_per_1m": current.output_cost_per_1m,
            "cache_read_cost_per_1m": current.cache_read_cost_per_1m,
        }
        path = _write(tmp_path, "p.json", {"models": [entry]})
        diff = diff_price_file(path)
        assert len(diff.unchanged) == 1
        assert diff.changed == []
        assert diff.added == []

    def test_changed_model_reports_fields(self, tmp_path):
        current = PRICING["gpt-4o"]
        entry = {
            "name": "gpt-4o",
            "provider": "openai",
            "input_cost_per_1m": current.input_cost_per_1m + 1.0,
            "output_cost_per_1m": current.output_cost_per_1m,
        }
        path = _write(tmp_path, "p.json", {"models": [entry]})
        diff = diff_price_file(path)
        assert len(diff.changed) == 1
        row = diff.changed[0]
        fields = {c.field for c in row.changes}
        assert "input_cost_per_1m" in fields
        change = next(c for c in row.changes if c.field == "input_cost_per_1m")
        assert change.old == current.input_cost_per_1m
        assert change.new == current.input_cost_per_1m + 1.0

    def test_diff_does_not_touch_registry(self, tmp_path):
        before = dict(PRICING)
        path = _write(tmp_path, "p.json", {"models": [VALID_ENTRY]})
        diff_price_file(path)
        assert PRICING == before

    def test_to_dict_shape(self, tmp_path):
        path = _write(tmp_path, "p.json", {"models": [VALID_ENTRY]})
        d = diff_price_file(path).to_dict()
        assert d["added"] == 1
        assert d["changed"] == 0
        assert isinstance(d["rows"], list)


class TestYamlSupport:
    def test_loads_yaml_when_available(self, tmp_path):
        pytest.importorskip("yaml")
        yaml_text = (
            "models:\n"
            "  - name: yaml-model\n"
            "    provider: anthropic\n"
            "    input_cost_per_1m: 2.5\n"
            "    output_cost_per_1m: 10.0\n"
        )
        path = _write(tmp_path, "prices.yaml", yaml_text)
        entries = load_price_file(path)
        assert entries[0].name == "yaml-model"
        assert entries[0].provider == "anthropic"

"""Tests for the `prices` CLI command group."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from llm_cost_guardian.cli import cli
from llm_cost_guardian.models import PRICING


@pytest.fixture(autouse=True)
def restore_registry():
    snapshot = dict(PRICING)
    try:
        yield
    finally:
        PRICING.clear()
        PRICING.update(snapshot)


def _price_file(tmp_path, models):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"models": models}))
    return str(path)


VALID = {
    "name": "acme-gpt",
    "provider": "openai",
    "input_cost_per_1m": 1.80,
    "output_cost_per_1m": 7.20,
}


class TestPricesView:
    def test_view_active_registry(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["prices", "view"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        assert "active registry" in result.output

    def test_view_provider_filter(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["prices", "view", "--provider", "google"])
        assert result.exit_code == 0
        assert "gemini" in result.output
        assert "gpt-4o" not in result.output

    def test_view_file(self, tmp_path):
        path = _price_file(tmp_path, [VALID])
        runner = CliRunner()
        result = runner.invoke(cli, ["prices", "view", "--file", path])
        assert result.exit_code == 0
        assert "acme-gpt" in result.output

    def test_view_json_output(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["prices", "view", "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert any(r["name"] == "gpt-4o" for r in data)

    def test_view_invalid_file_exits_1(self, tmp_path):
        path = _price_file(tmp_path, [{"name": "x", "provider": "nope"}])
        runner = CliRunner()
        result = runner.invoke(cli, ["prices", "view", "--file", path])
        assert result.exit_code == 1


class TestPricesValidate:
    def test_valid_exits_0(self, tmp_path):
        path = _price_file(tmp_path, [VALID])
        runner = CliRunner()
        result = runner.invoke(cli, ["prices", "validate", path])
        assert result.exit_code == 0
        assert "is valid" in result.output

    def test_invalid_exits_1(self, tmp_path):
        path = _price_file(tmp_path, [{"name": "x", "provider": "nope"}])
        runner = CliRunner()
        result = runner.invoke(cli, ["prices", "validate", path])
        assert result.exit_code == 1
        assert "invalid" in result.output


class TestPricesDiff:
    def test_added_exits_2(self, tmp_path):
        path = _price_file(tmp_path, [VALID])
        runner = CliRunner()
        result = runner.invoke(cli, ["prices", "diff", path])
        assert result.exit_code == 2
        assert "acme-gpt" in result.output
        assert "New models" in result.output

    def test_unchanged_exits_0(self, tmp_path):
        current = PRICING["gpt-4o"]
        entry = {
            "name": "gpt-4o",
            "provider": "openai",
            "input_cost_per_1m": current.input_cost_per_1m,
            "output_cost_per_1m": current.output_cost_per_1m,
            "cache_read_cost_per_1m": current.cache_read_cost_per_1m,
        }
        path = _price_file(tmp_path, [entry])
        runner = CliRunner()
        result = runner.invoke(cli, ["prices", "diff", path])
        assert result.exit_code == 0
        assert "No changes" in result.output

    def test_changed_exits_2(self, tmp_path):
        current = PRICING["gpt-4o"]
        entry = {
            "name": "gpt-4o",
            "provider": "openai",
            "input_cost_per_1m": current.input_cost_per_1m + 5.0,
            "output_cost_per_1m": current.output_cost_per_1m,
        }
        path = _price_file(tmp_path, [entry])
        runner = CliRunner()
        result = runner.invoke(cli, ["prices", "diff", path])
        assert result.exit_code == 2
        assert "Changed models" in result.output

    def test_diff_json_output(self, tmp_path):
        path = _price_file(tmp_path, [VALID])
        runner = CliRunner()
        result = runner.invoke(cli, ["prices", "diff", path, "--json-output"])
        assert result.exit_code == 2
        data = json.loads(result.output)
        assert data["added"] == 1

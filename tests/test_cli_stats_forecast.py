"""Tests for the new 'stats' and 'forecast' CLI commands."""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest
from click.testing import CliRunner

from llm_cost_guardian.cli import _percentile, cli


@pytest.fixture
def runner():
    return CliRunner()


def _write_report(records: list[dict], extra_summary: dict | None = None) -> str:
    payload = {
        "summary": {
            "total_cost_usd": sum(r["cost_usd"] for r in records),
            "total_requests": len(records),
            "total_input_tokens": sum(r.get("input_tokens", 0) for r in records),
            "total_output_tokens": sum(r.get("output_tokens", 0) for r in records),
            "cost_by_model": {},
        },
        "records": records,
    }
    if extra_summary:
        payload["summary"].update(extra_summary)

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


# _percentile helper ----------------------------------------------------------


def test_percentile_empty_list_returns_zero():
    assert _percentile([], 50) == 0.0


def test_percentile_single_value():
    assert _percentile([5.0], 50) == 5.0
    assert _percentile([5.0], 99) == 5.0


def test_percentile_median_odd_count():
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0


def test_percentile_median_even_count():
    # linear interpolation: rank = 0.5 * 3 = 1.5 -> halfway between 2 and 3
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_percentile_p100_is_max():
    assert _percentile([1.0, 2.0, 3.0, 9.0], 100) == 9.0


def test_percentile_p0_is_min():
    assert _percentile([1.0, 2.0, 3.0, 9.0], 0) == 1.0


# stats command ---------------------------------------------------------------


def test_stats_human_output(runner):
    records = [
        {"model": "gpt-4o", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.001},
        {"model": "gpt-4o", "input_tokens": 200, "output_tokens": 100, "cost_usd": 0.002},
        {"model": "gpt-4o", "input_tokens": 300, "output_tokens": 150, "cost_usd": 0.003},
        {"model": "gpt-4o", "input_tokens": 400, "output_tokens": 200, "cost_usd": 0.004},
        {"model": "gpt-4o", "input_tokens": 500, "output_tokens": 250, "cost_usd": 0.005},
    ]
    path = _write_report(records)
    try:
        result = runner.invoke(cli, ["stats", path])
        assert result.exit_code == 0
        assert "Cost Distribution" in result.output
        assert "p50 (median)" in result.output
        assert "p90" in result.output
        assert "p99" in result.output
        # Total cost = 0.015
        assert "$0.015000" in result.output
    finally:
        os.remove(path)


def test_stats_json_output(runner):
    records = [
        {"model": "gpt-4o", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.001},
        {"model": "gpt-4o", "input_tokens": 200, "output_tokens": 100, "cost_usd": 0.002},
        {"model": "gpt-4o", "input_tokens": 300, "output_tokens": 150, "cost_usd": 0.005},
    ]
    path = _write_report(records)
    try:
        result = runner.invoke(cli, ["stats", path, "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["records"] == 3
        assert data["total_cost_usd"] == pytest.approx(0.008)
        assert "cost_per_call" in data
        assert "p50" in data["cost_per_call"]
        assert "p90" in data["cost_per_call"]
        assert "p99" in data["cost_per_call"]
        assert "tokens_per_call" in data
    finally:
        os.remove(path)


def test_stats_empty_report(runner):
    path = _write_report([])
    try:
        result = runner.invoke(cli, ["stats", path])
        assert result.exit_code == 0
        assert "No records" in result.output
    finally:
        os.remove(path)


def test_stats_empty_report_json(runner):
    path = _write_report([])
    try:
        result = runner.invoke(cli, ["stats", path, "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["records"] == 0
    finally:
        os.remove(path)


def test_stats_invalid_json(runner):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        f.write("not json")
    try:
        result = runner.invoke(cli, ["stats", path])
        assert result.exit_code == 1
        assert "not valid JSON" in result.output
    finally:
        os.remove(path)


def test_stats_wrong_type(runner):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        json.dump([1, 2, 3], f)
    try:
        result = runner.invoke(cli, ["stats", path])
        assert result.exit_code == 1
        assert "JSON object" in result.output
    finally:
        os.remove(path)


def test_stats_handles_missing_token_fields(runner):
    records = [
        {"model": "x", "cost_usd": 0.001},
        {"model": "x", "cost_usd": 0.002},
    ]
    path = _write_report(records)
    try:
        result = runner.invoke(cli, ["stats", path, "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["records"] == 2
        assert data["tokens_per_call"]["mean"] == 0
    finally:
        os.remove(path)


# forecast command ------------------------------------------------------------


def test_forecast_human_output(runner):
    now = time.time()
    records = [
        {"model": "gpt-4o", "input_tokens": 100, "output_tokens": 50, "cost_usd": 1.0,
         "timestamp": now - 86400},  # 1 day ago
        {"model": "gpt-4o", "input_tokens": 100, "output_tokens": 50, "cost_usd": 1.0,
         "timestamp": now},  # now
    ]
    path = _write_report(records)
    try:
        result = runner.invoke(cli, ["forecast", path, "--days", "30"])
        assert result.exit_code == 0
        assert "Cost Forecast" in result.output
        assert "Cost per day" in result.output
        assert "Projected over 30 days" in result.output
    finally:
        os.remove(path)


def test_forecast_json_output(runner):
    now = time.time()
    records = [
        {"model": "x", "cost_usd": 0.5, "timestamp": now - 86400},
        {"model": "x", "cost_usd": 0.5, "timestamp": now},
    ]
    path = _write_report(records)
    try:
        result = runner.invoke(cli, ["forecast", path, "--days", "10", "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["observed"]["records"] == 2
        assert data["observed"]["total_cost_usd"] == pytest.approx(1.0)
        # 1 USD over 1 day -> 1 USD/day -> 10 USD over 10 days
        assert data["rates"]["cost_per_day_usd"] == pytest.approx(1.0, rel=0.01)
        assert data["forecast"]["projected_cost_usd"] == pytest.approx(10.0, rel=0.01)
        assert data["forecast"]["horizon_days"] == 10
    finally:
        os.remove(path)


def test_forecast_default_30_day_horizon(runner):
    now = time.time()
    records = [
        {"model": "x", "cost_usd": 2.0, "timestamp": now - 86400 * 2},
        {"model": "x", "cost_usd": 2.0, "timestamp": now},
    ]
    path = _write_report(records)
    try:
        result = runner.invoke(cli, ["forecast", path, "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["forecast"]["horizon_days"] == 30.0
        # 4 USD over 2 days -> 2 USD/day -> 60 USD/month
        assert data["forecast"]["projected_cost_usd"] == pytest.approx(60.0, rel=0.01)
    finally:
        os.remove(path)


def test_forecast_no_records(runner):
    path = _write_report([])
    try:
        result = runner.invoke(cli, ["forecast", path])
        assert result.exit_code == 1
    finally:
        os.remove(path)


def test_forecast_only_one_record(runner):
    records = [{"model": "x", "cost_usd": 1.0, "timestamp": time.time()}]
    path = _write_report(records)
    try:
        result = runner.invoke(cli, ["forecast", path])
        assert result.exit_code == 1
        assert "at least 2 records" in result.output
    finally:
        os.remove(path)


def test_forecast_zero_time_span(runner):
    now = time.time()
    records = [
        {"model": "x", "cost_usd": 1.0, "timestamp": now},
        {"model": "x", "cost_usd": 1.0, "timestamp": now},
    ]
    path = _write_report(records)
    try:
        result = runner.invoke(cli, ["forecast", path])
        assert result.exit_code == 1
        assert "0 seconds" in result.output
    finally:
        os.remove(path)


def test_forecast_invalid_json(runner):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        f.write("garbage")
    try:
        result = runner.invoke(cli, ["forecast", path])
        assert result.exit_code == 1
        assert "not valid JSON" in result.output
    finally:
        os.remove(path)


def test_forecast_records_without_timestamps(runner):
    records = [
        {"model": "x", "cost_usd": 1.0},
        {"model": "x", "cost_usd": 1.0},
    ]
    path = _write_report(records)
    try:
        result = runner.invoke(cli, ["forecast", path])
        assert result.exit_code == 1
        assert "at least 2 records" in result.output
    finally:
        os.remove(path)


# stats/forecast appear in --help --------------------------------------------


def test_help_lists_stats_and_forecast(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "stats" in result.output
    assert "forecast" in result.output

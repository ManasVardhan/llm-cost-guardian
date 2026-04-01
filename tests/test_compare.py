"""Tests for the CLI compare command."""

import json
import os
import tempfile

from click.testing import CliRunner

from llm_cost_guardian.cli import cli


def _write_report(data: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    return path


REPORT_A = {
    "summary": {
        "total_cost_usd": 0.05,
        "total_requests": 10,
        "total_input_tokens": 5000,
        "total_output_tokens": 2000,
        "cost_by_model": {
            "gpt-4o": 0.04,
            "gpt-4o-mini": 0.01,
        },
    },
    "records": [],
}

REPORT_B = {
    "summary": {
        "total_cost_usd": 0.12,
        "total_requests": 25,
        "total_input_tokens": 12000,
        "total_output_tokens": 5000,
        "cost_by_model": {
            "gpt-4o": 0.10,
            "claude-3-5-haiku-20241022": 0.02,
        },
    },
    "records": [],
}


class TestCompareCommand:
    def test_basic_compare(self) -> None:
        a = _write_report(REPORT_A)
        b = _write_report(REPORT_B)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["compare", a, b])
            assert result.exit_code == 0
            assert "Cost Comparison" in result.output
            assert "Total cost" in result.output
            assert "Requests" in result.output
        finally:
            os.unlink(a)
            os.unlink(b)

    def test_compare_shows_model_breakdown(self) -> None:
        a = _write_report(REPORT_A)
        b = _write_report(REPORT_B)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["compare", a, b])
            assert "gpt-4o" in result.output
            assert "gpt-4o-mini" in result.output
            assert "claude-3-5-haiku" in result.output
        finally:
            os.unlink(a)
            os.unlink(b)

    def test_compare_shows_percentage(self) -> None:
        a = _write_report(REPORT_A)
        b = _write_report(REPORT_B)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["compare", a, b])
            assert "Cost change:" in result.output
            assert "%" in result.output
        finally:
            os.unlink(a)
            os.unlink(b)

    def test_compare_identical_reports(self) -> None:
        a = _write_report(REPORT_A)
        b = _write_report(REPORT_A)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["compare", a, b])
            assert result.exit_code == 0
            assert "+0.000000" in result.output
        finally:
            os.unlink(a)
            os.unlink(b)

    def test_compare_empty_summaries(self) -> None:
        a = _write_report({"summary": {}})
        b = _write_report({"summary": {}})
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["compare", a, b])
            assert result.exit_code == 0
        finally:
            os.unlink(a)
            os.unlink(b)

    def test_compare_invalid_json(self) -> None:
        fd, bad_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write("not json{{{")
        good = _write_report(REPORT_A)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["compare", bad_path, good])
            assert result.exit_code != 0
            assert "Invalid JSON" in result.output
        finally:
            os.unlink(bad_path)
            os.unlink(good)

    def test_compare_non_object_json(self) -> None:
        fd, arr_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump([1, 2, 3], f)
        good = _write_report(REPORT_A)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["compare", arr_path, good])
            assert result.exit_code != 0
            assert "JSON objects" in result.output
        finally:
            os.unlink(arr_path)
            os.unlink(good)

    def test_compare_zero_cost_no_percentage(self) -> None:
        """When report A has zero cost, percentage line should not appear."""
        a = _write_report({"summary": {"total_cost_usd": 0}})
        b = _write_report(REPORT_B)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["compare", a, b])
            assert result.exit_code == 0
            assert "Cost change:" not in result.output
        finally:
            os.unlink(a)
            os.unlink(b)

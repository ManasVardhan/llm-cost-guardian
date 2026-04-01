"""Tests for the CLI top command."""

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


REPORT_WITH_RECORDS = {
    "summary": {
        "total_cost_usd": 1.35,
        "total_requests": 5,
        "total_input_tokens": 15000,
        "total_output_tokens": 7500,
        "cost_by_model": {"gpt-4o": 1.10, "gpt-4o-mini": 0.25},
    },
    "records": [
        {"model": "gpt-4o", "input_tokens": 5000, "output_tokens": 2500, "cost_usd": 0.50},
        {"model": "gpt-4o", "input_tokens": 3000, "output_tokens": 1500, "cost_usd": 0.30},
        {"model": "gpt-4o", "input_tokens": 2000, "output_tokens": 1000, "cost_usd": 0.20},
        {"model": "gpt-4o-mini", "input_tokens": 3000, "output_tokens": 1500, "cost_usd": 0.15},
        {"model": "gpt-4o-mini", "input_tokens": 2000, "output_tokens": 1000, "cost_usd": 0.10},
        {"model": "gpt-4o", "input_tokens": 500, "output_tokens": 250, "cost_usd": 0.10},
    ],
}


class TestTopCommand:
    def test_top_default(self) -> None:
        path = _write_report(REPORT_WITH_RECORDS)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["top", path])
            assert result.exit_code == 0
            assert "Most Expensive" in result.output
            assert "gpt-4o" in result.output
        finally:
            os.unlink(path)

    def test_top_with_limit(self) -> None:
        path = _write_report(REPORT_WITH_RECORDS)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["top", path, "-n", "3"])
            assert result.exit_code == 0
            assert "Top 3" in result.output
            # First entry should be the most expensive (0.50)
            lines = result.output.strip().split("\n")
            data_lines = [line for line in lines if line.startswith("1")]
            assert len(data_lines) == 1
            assert "0.500000" in data_lines[0]
        finally:
            os.unlink(path)

    def test_top_sorted_by_cost(self) -> None:
        path = _write_report(REPORT_WITH_RECORDS)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["top", path, "-n", "3"])
            assert result.exit_code == 0
            # Check costs are in descending order
            lines = result.output.strip().split("\n")
            costs = []
            for line in lines:
                if line and line[0].isdigit() and "$" in line:
                    cost_str = line.split("$")[-1].strip()
                    costs.append(float(cost_str))
            assert costs == sorted(costs, reverse=True)
        finally:
            os.unlink(path)

    def test_top_json_output(self) -> None:
        path = _write_report(REPORT_WITH_RECORDS)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["top", path, "--json-output"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert isinstance(data, list)
            assert data[0]["cost_usd"] >= data[1]["cost_usd"]
        finally:
            os.unlink(path)

    def test_top_no_records(self) -> None:
        report = {"summary": {}, "records": []}
        path = _write_report(report)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["top", path])
            assert result.exit_code == 0
            assert "No records found" in result.output
        finally:
            os.unlink(path)

    def test_top_no_records_key(self) -> None:
        report = {"summary": {}}
        path = _write_report(report)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["top", path])
            assert result.exit_code == 0
            assert "No records found" in result.output
        finally:
            os.unlink(path)

    def test_top_shows_percentage(self) -> None:
        path = _write_report(REPORT_WITH_RECORDS)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["top", path, "-n", "2"])
            assert result.exit_code == 0
            assert "%" in result.output
        finally:
            os.unlink(path)

    def test_top_invalid_json(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write("not json{{{")
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["top", path])
            assert result.exit_code != 0
            assert "not valid JSON" in result.output
        finally:
            os.unlink(path)

    def test_top_non_object_json(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump([1, 2, 3], f)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["top", path])
            assert result.exit_code != 0
            assert "JSON object" in result.output
        finally:
            os.unlink(path)

    def test_top_limit_exceeds_records(self) -> None:
        path = _write_report(REPORT_WITH_RECORDS)
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["top", path, "-n", "100"])
            assert result.exit_code == 0
            assert "Top 6" in result.output
        finally:
            os.unlink(path)

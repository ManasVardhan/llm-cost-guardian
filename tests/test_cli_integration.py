"""CLI integration tests: actually invoke the CLI and verify output."""

import json
import os
import subprocess
import sys
import tempfile

from llm_cost_guardian import CostTracker
from llm_cost_guardian.exporters import save_json

PYTHON = os.path.join(os.path.dirname(sys.executable), "python3.12")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable


def _run_cli(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run the CLI via subprocess."""
    return subprocess.run(
        [PYTHON, "-m", "llm_cost_guardian.cli", *args],
        capture_output=True,
        text=True,
        timeout=10,
    )


class TestCLIVersion:
    def test_version_flag(self):
        result = _run_cli("--version")
        assert result.returncode == 0
        assert "0.1.1" in result.stdout

    def test_help_flag(self):
        result = _run_cli("--help")
        assert result.returncode == 0
        assert "LLM Cost Guardian" in result.stdout
        assert "models" in result.stdout
        assert "estimate" in result.stdout
        assert "report" in result.stdout


class TestCLIModels:
    def test_models_table(self):
        result = _run_cli("models")
        assert result.returncode == 0
        assert "gpt-4o" in result.stdout
        assert "claude" in result.stdout
        assert "gemini" in result.stdout

    def test_models_json(self):
        result = _run_cli("models", "--json-output")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "name" in data[0]
        assert "provider" in data[0]
        assert "input_per_1m" in data[0]
        assert "output_per_1m" in data[0]

    def test_models_filter_openai(self):
        result = _run_cli("models", "--provider", "openai")
        assert result.returncode == 0
        assert "gpt-4o" in result.stdout
        # Should not have Anthropic models
        lines = result.stdout.strip().split("\n")
        data_lines = [
            ln for ln in lines
            if ln.strip() and not ln.startswith("-") and "Model" not in ln
        ]
        for line in data_lines:
            assert "openai" in line.lower()

    def test_models_filter_anthropic(self):
        result = _run_cli("models", "--provider", "anthropic")
        assert result.returncode == 0
        assert "claude" in result.stdout

    def test_models_filter_google(self):
        result = _run_cli("models", "--provider", "google")
        assert result.returncode == 0
        assert "gemini" in result.stdout


class TestCLIEstimate:
    def test_estimate_known_model(self):
        result = _run_cli("estimate", "gpt-4o", "-i", "1000", "-o", "500")
        assert result.returncode == 0
        assert "gpt-4o" in result.stdout
        assert "1,000" in result.stdout
        assert "500" in result.stdout
        assert "$" in result.stdout

    def test_estimate_unknown_model(self):
        result = _run_cli("estimate", "nonexistent-model", "-i", "100", "-o", "50")
        assert result.returncode != 0
        assert "Unknown model" in result.stderr

    def test_estimate_zero_tokens(self):
        result = _run_cli("estimate", "gpt-4o", "-i", "0", "-o", "0")
        assert result.returncode == 0
        assert "$0.000000" in result.stdout

    def test_estimate_negative_tokens(self):
        result = _run_cli("estimate", "gpt-4o", "-i", "-1", "-o", "0")
        assert result.returncode != 0
        assert "non-negative" in result.stderr

    def test_estimate_large_tokens(self):
        result = _run_cli("estimate", "gpt-4o", "-i", "1000000", "-o", "500000")
        assert result.returncode == 0
        assert "$" in result.stdout


class TestCLIReport:
    def test_report_valid_json(self):
        """Report should parse a valid JSON file."""
        tracker = CostTracker()
        tracker.record("gpt-4o", 1000, 500)
        tracker.record("gpt-4o", 2000, 1000)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_json(tracker, f.name)
            path = f.name

        try:
            result = _run_cli("report", path)
            assert result.returncode == 0
            assert "LLM Cost Report" in result.stdout
            assert "Total cost:" in result.stdout
            assert "Total requests: 2" in result.stdout
            assert "gpt-4o" in result.stdout
        finally:
            os.unlink(path)

    def test_report_nonexistent_file(self):
        result = _run_cli("report", "/tmp/nonexistent_file_12345.json")
        assert result.returncode != 0

    def test_report_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{broken json")
            path = f.name

        try:
            result = _run_cli("report", path)
            assert result.returncode != 0
            assert "not valid JSON" in result.stderr
        finally:
            os.unlink(path)

    def test_report_wrong_type(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([1, 2, 3], f)
            path = f.name

        try:
            result = _run_cli("report", path)
            assert result.returncode != 0
            assert "Expected a JSON object" in result.stderr
        finally:
            os.unlink(path)

    def test_report_empty_object(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            path = f.name

        try:
            result = _run_cli("report", path)
            assert result.returncode == 0
            assert "LLM Cost Report" in result.stdout
        finally:
            os.unlink(path)

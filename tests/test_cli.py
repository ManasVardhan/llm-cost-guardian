"""Tests for the CLI."""

import json
import os
import tempfile

from click.testing import CliRunner

from llm_cost_guardian.cli import cli


class TestModelsCommand:
    def test_models_lists_all(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        assert "claude-sonnet-4-20250514" in result.output
        assert "gemini-2.0-flash" in result.output

    def test_models_filter_by_provider(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["models", "--provider", "openai"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        # Should not include Anthropic or Google models
        assert "claude" not in result.output
        assert "gemini" not in result.output

    def test_models_json_output(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["models", "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) > 0
        first = data[0]
        assert "name" in first
        assert "provider" in first
        assert "input_per_1m" in first
        assert "output_per_1m" in first

    def test_models_json_with_provider_filter(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["models", "--provider", "anthropic", "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        for m in data:
            assert m["provider"] == "anthropic"


class TestEstimateCommand:
    def test_estimate_known_model(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["estimate", "gpt-4o", "-i", "10000", "-o", "5000"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        assert "$0.075000" in result.output
        assert "10,000" in result.output
        assert "5,000" in result.output

    def test_estimate_unknown_model(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["estimate", "nonexistent-model", "-i", "100", "-o", "100"])
        assert result.exit_code == 1
        assert "Unknown model" in result.output

    def test_estimate_zero_tokens(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["estimate", "gpt-4o", "-i", "0", "-o", "0"])
        assert result.exit_code == 0
        assert "$0.000000" in result.output

    def test_estimate_large_tokens(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["estimate", "gpt-4", "-i", "1000000", "-o", "500000"])
        assert result.exit_code == 0
        # gpt-4: $30/1M in, $60/1M out = $30 + $30 = $60
        assert "$60.000000" in result.output

    def test_estimate_negative_input_tokens(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["estimate", "gpt-4o", "-i", "-100", "-o", "50"])
        assert result.exit_code == 1
        assert "non-negative" in result.output

    def test_estimate_negative_output_tokens(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["estimate", "gpt-4o", "-i", "100", "-o", "-50"])
        assert result.exit_code == 1
        assert "non-negative" in result.output

    def test_estimate_prefix_matched_model(self):
        """Versioned model names should resolve via prefix matching."""
        runner = CliRunner()
        result = runner.invoke(cli, ["estimate", "gpt-4o-2024-08-06", "-i", "1000", "-o", "500"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output


class TestReportCommand:
    def test_report_valid_file(self):
        runner = CliRunner()
        report_data = {
            "summary": {
                "total_cost_usd": 1.234567,
                "total_requests": 42,
                "total_input_tokens": 100000,
                "total_output_tokens": 50000,
                "cost_by_model": {
                    "gpt-4o": 0.75,
                    "claude-sonnet-4-20250514": 0.48,
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report_data, f)
            f.flush()
            result = runner.invoke(cli, ["report", f.name])
        os.unlink(f.name)

        assert result.exit_code == 0
        assert "$1.234567" in result.output
        assert "42" in result.output
        assert "100,000" in result.output
        assert "gpt-4o" in result.output

    def test_report_empty_summary(self):
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"summary": {}}, f)
            f.flush()
            result = runner.invoke(cli, ["report", f.name])
        os.unlink(f.name)

        assert result.exit_code == 0
        assert "$0.000000" in result.output

    def test_report_nonexistent_file(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["report", "/tmp/does_not_exist_12345.json"])
        assert result.exit_code != 0

    def test_report_malformed_json(self):
        """Malformed JSON should produce a clear error, not a traceback."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("this is not json {{{")
            f.flush()
            result = runner.invoke(cli, ["report", f.name])
        os.unlink(f.name)

        assert result.exit_code == 1
        assert "not valid JSON" in result.output

    def test_report_json_array_instead_of_object(self):
        """A JSON array instead of an object should produce a clear error."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([1, 2, 3], f)
            f.flush()
            result = runner.invoke(cli, ["report", f.name])
        os.unlink(f.name)

        assert result.exit_code == 1
        assert "Expected a JSON object" in result.output

    def test_report_with_cost_by_model(self):
        """Verify that cost_by_model is rendered correctly."""
        runner = CliRunner()
        report_data = {
            "summary": {
                "total_cost_usd": 0.5,
                "total_requests": 2,
                "total_input_tokens": 2000,
                "total_output_tokens": 1000,
                "cost_by_model": {"gpt-4o": 0.3, "o3-mini": 0.2},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report_data, f)
            f.flush()
            result = runner.invoke(cli, ["report", f.name])
        os.unlink(f.name)

        assert result.exit_code == 0
        assert "Cost by model" in result.output
        assert "gpt-4o" in result.output
        assert "o3-mini" in result.output


class TestVersionFlag:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.1" in result.output


class TestHelp:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "LLM Cost Guardian" in result.output
        assert "estimate" in result.output
        assert "models" in result.output
        assert "report" in result.output

"""Tests for context window utilization analysis."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from llm_cost_guardian import (
    ContextReport,
    ContextStat,
    analyze_context,
    resolve_window,
)
from llm_cost_guardian.cli import cli
from llm_cost_guardian.context_window import percentile


def _record(model: str = "gpt-4o", input_tokens: int = 1000) -> dict:
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": 100,
        "cost_usd": 0.01,
    }


def _report(records: list[dict]) -> dict:
    return {"summary": {}, "records": records}


class TestPercentile:
    def test_empty_is_zero(self):
        assert percentile([], 95) == 0.0

    def test_single_value(self):
        assert percentile([42], 95) == 42.0

    def test_interpolates(self):
        assert percentile([0, 100], 50) == 50.0

    def test_p95_of_uniform(self):
        values = list(range(1, 101))
        assert percentile(values, 95) == pytest.approx(95.05)


class TestResolveWindow:
    def test_known_model(self):
        assert resolve_window("gpt-4o") == 128_000

    def test_prefix_match(self):
        assert resolve_window("gpt-4o-2024-08-06") == 128_000

    def test_unknown_model(self):
        assert resolve_window("my-custom-model") is None

    def test_override_wins(self):
        assert resolve_window("gpt-4o", {"gpt-4o": 64_000}) == 64_000

    def test_override_covers_unknown(self):
        assert resolve_window("my-custom-model", {"my-custom-model": 32_000}) == 32_000


class TestContextStat:
    def _stat(self, tokens: list[int], window: int | None = 1000) -> ContextStat:
        return ContextStat(
            model="m",
            calls=len(tokens),
            input_tokens=tokens,
            context_window=window,
            near_limit=0.8,
        )

    def test_basic_metrics(self):
        stat = self._stat([100, 300, 200])
        assert stat.avg_input_tokens == 200.0
        assert stat.max_input_tokens == 300
        assert stat.avg_utilization == pytest.approx(0.2)
        assert stat.max_utilization == pytest.approx(0.3)

    def test_unknown_window_gives_none(self):
        stat = self._stat([100], window=None)
        assert stat.avg_utilization is None
        assert stat.p95_utilization is None
        assert stat.max_utilization is None
        assert stat.calls_near_limit is None
        assert stat.is_near_limit is False

    def test_calls_near_limit_counts_cutoff_inclusive(self):
        stat = self._stat([799, 800, 900])
        assert stat.calls_near_limit == 2

    def test_is_near_limit_from_p95(self):
        assert self._stat([900, 900, 900]).is_near_limit is True
        assert self._stat([100, 100, 100]).is_near_limit is False

    def test_to_dict_percentages(self):
        d = self._stat([500]).to_dict()
        assert d["avg_utilization_pct"] == 50.0
        assert d["p95_utilization_pct"] == 50.0
        assert d["context_window"] == 1000
        assert d["near_limit"] is False

    def test_to_dict_none_for_unknown_window(self):
        d = self._stat([500], window=None).to_dict()
        assert d["avg_utilization_pct"] is None
        assert d["calls_near_limit"] is None


class TestAnalyzeContext:
    def test_groups_by_model(self):
        report = _report(
            [
                _record("gpt-4o", 1000),
                _record("gpt-4o", 3000),
                _record("gpt-4", 2000),
            ]
        )
        result = analyze_context(report)
        assert isinstance(result, ContextReport)
        assert result.records_analyzed == 3
        by_model = {s.model: s for s in result.by_model}
        assert by_model["gpt-4o"].calls == 2
        assert by_model["gpt-4o"].avg_input_tokens == 2000.0
        assert by_model["gpt-4"].context_window == 8_192

    def test_sorted_by_p95_utilization_unknown_last(self):
        report = _report(
            [
                _record("gpt-4", 7000),  # window 8192 -> ~85%
                _record("gpt-4o", 1000),  # window 128000 -> tiny
                _record("mystery-model", 999_999),
            ]
        )
        result = analyze_context(report)
        assert [s.model for s in result.by_model] == ["gpt-4", "gpt-4o", "mystery-model"]

    def test_models_near_limit(self):
        report = _report([_record("gpt-4", 8000), _record("gpt-4o", 100)])
        result = analyze_context(report)
        assert [s.model for s in result.models_near_limit] == ["gpt-4"]
        assert [s.model for s in result.models_without_window] == []

    def test_windows_override(self):
        report = _report([_record("my-model", 900)])
        result = analyze_context(report, windows={"my-model": 1000})
        stat = result.by_model[0]
        assert stat.context_window == 1000
        assert stat.is_near_limit is True

    def test_skips_bad_records(self):
        report = _report(
            [
                _record("gpt-4o", 100),
                {"model": "", "input_tokens": 5},
                {"model": "x", "input_tokens": "bad"},
                {"model": "y", "input_tokens": -5},
                "not-a-dict",
            ]
        )
        result = analyze_context(report)
        assert result.records_analyzed == 1
        assert result.records_skipped == 4

    def test_empty_report(self):
        result = analyze_context(_report([]))
        assert result.records_analyzed == 0
        assert result.by_model == []

    def test_bad_near_limit_raises(self):
        with pytest.raises(ValueError):
            analyze_context(_report([]), near_limit=0)
        with pytest.raises(ValueError):
            analyze_context(_report([]), near_limit=1.5)

    def test_to_dict_shape(self):
        report = _report([_record("gpt-4", 8000)])
        d = analyze_context(report).to_dict()
        assert d["records_analyzed"] == 1
        assert d["near_limit_threshold_pct"] == 80.0
        assert d["models_near_limit"] == ["gpt-4"]
        assert d["by_model"][0]["model"] == "gpt-4"


class TestContextCLI:
    def _write_report(self, tmp_path, records):
        path = tmp_path / "report.json"
        path.write_text(json.dumps(_report(records)))
        return str(path)

    def test_text_output(self, tmp_path):
        report = self._write_report(
            tmp_path, [_record("gpt-4", 7900), _record("gpt-4o", 500)]
        )
        result = CliRunner().invoke(cli, ["context", report])
        assert result.exit_code == 0
        assert "Context Window Utilization" in result.output
        assert "gpt-4" in result.output
        assert "Near limit" in result.output

    def test_unknown_model_hint(self, tmp_path):
        report = self._write_report(tmp_path, [_record("mystery-model", 500)])
        result = CliRunner().invoke(cli, ["context", report])
        assert result.exit_code == 0
        assert "No known context window" in result.output
        assert "mystery-model" in result.output

    def test_window_override(self, tmp_path):
        report = self._write_report(tmp_path, [_record("mystery-model", 900)])
        result = CliRunner().invoke(
            cli, ["context", report, "--window", "mystery-model=1000"]
        )
        assert result.exit_code == 0
        assert "No known context window" not in result.output
        assert "Near limit" in result.output

    def test_bad_window_spec_errors(self, tmp_path):
        report = self._write_report(tmp_path, [_record()])
        for spec in ["nope", "m=", "m=-5", "=100"]:
            result = CliRunner().invoke(cli, ["context", report, "--window", spec])
            assert result.exit_code == 1, spec
            assert "invalid --window" in result.output

    def test_json_output(self, tmp_path):
        report = self._write_report(tmp_path, [_record("gpt-4o", 500)])
        result = CliRunner().invoke(cli, ["context", report, "--json-output"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["records_analyzed"] == 1
        assert payload["by_model"][0]["context_window"] == 128_000

    def test_fail_near_limit_exit_2(self, tmp_path):
        report = self._write_report(tmp_path, [_record("gpt-4", 8100)])
        result = CliRunner().invoke(cli, ["context", report, "--fail-near-limit"])
        assert result.exit_code == 2

    def test_fail_near_limit_clean_exit_0(self, tmp_path):
        report = self._write_report(tmp_path, [_record("gpt-4o", 100)])
        result = CliRunner().invoke(cli, ["context", report, "--fail-near-limit"])
        assert result.exit_code == 0

    def test_custom_near_limit(self, tmp_path):
        report = self._write_report(tmp_path, [_record("gpt-4", 5000)])  # ~61%
        ok = CliRunner().invoke(cli, ["context", report, "--fail-near-limit"])
        assert ok.exit_code == 0
        strict = CliRunner().invoke(
            cli, ["context", report, "--near-limit", "0.5", "--fail-near-limit"]
        )
        assert strict.exit_code == 2

    def test_bad_near_limit_errors(self, tmp_path):
        report = self._write_report(tmp_path, [_record()])
        result = CliRunner().invoke(cli, ["context", report, "--near-limit", "2"])
        assert result.exit_code == 1
        assert "near_limit" in result.output

    def test_no_records_message(self, tmp_path):
        report = self._write_report(tmp_path, [])
        result = CliRunner().invoke(cli, ["context", report])
        assert result.exit_code == 0
        assert "No usable records" in result.output

    def test_missing_file_errors(self):
        result = CliRunner().invoke(cli, ["context", "/no/such/file.json"])
        assert result.exit_code != 0

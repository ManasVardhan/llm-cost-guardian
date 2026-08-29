"""Tests for token efficiency reporting."""

from __future__ import annotations

import json

from click.testing import CliRunner

from llm_cost_guardian import (
    EfficiencyReport,
    EfficiencyStat,
    analyze_efficiency,
)
from llm_cost_guardian.cli import cli


def _record(
    model: str = "gpt-4o",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cost: float = 0.01,
    tags: list[str] | None = None,
) -> dict:
    rec = {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
    }
    if tags is not None:
        rec["tags"] = tags
    return rec


def _report(records: list[dict]) -> dict:
    return {"summary": {}, "records": records}


class TestEfficiencyStat:
    def test_ratio_and_cost_per_1k(self):
        stat = EfficiencyStat(
            dimension="overall",
            key="(overall)",
            calls=1,
            input_tokens=200,
            output_tokens=100,
            cost_usd=0.02,
        )
        assert stat.output_input_ratio == 0.5
        assert stat.cost_per_1k_output == 0.2
        assert stat.cost_per_1k_input == 0.1

    def test_ratio_none_without_input(self):
        stat = EfficiencyStat("overall", "(overall)", 1, 0, 100, 0.02)
        assert stat.output_input_ratio is None
        assert stat.cost_per_1k_input is None
        assert stat.cost_per_1k_output == 0.2

    def test_cost_per_1k_output_none_without_output(self):
        stat = EfficiencyStat("overall", "(overall)", 1, 100, 0, 0.02)
        assert stat.cost_per_1k_output is None
        assert stat.output_input_ratio == 0.0

    def test_to_dict_rounds_and_keeps_nones(self):
        stat = EfficiencyStat("model", "gpt-4o", 3, 300, 0, 0.03)
        d = stat.to_dict()
        assert d["cost_per_1k_output"] is None
        assert d["output_input_ratio"] == 0.0
        assert d["cost_usd"] == 0.03


class TestAnalyzeEfficiency:
    def test_overall_aggregates_all_records(self):
        report = _report(
            [
                _record(input_tokens=100, output_tokens=50, cost=0.01),
                _record(input_tokens=300, output_tokens=150, cost=0.03),
            ]
        )
        result = analyze_efficiency(report)
        assert isinstance(result, EfficiencyReport)
        assert result.overall.calls == 2
        assert result.overall.input_tokens == 400
        assert result.overall.output_tokens == 200
        assert result.overall.output_input_ratio == 0.5
        assert result.records_analyzed == 2
        assert result.records_skipped == 0

    def test_by_model_split(self):
        report = _report(
            [
                _record(model="gpt-4o", cost=0.05),
                _record(model="gpt-4o-mini", cost=0.01),
                _record(model="gpt-4o", cost=0.05),
            ]
        )
        result = analyze_efficiency(report)
        keys = [s.key for s in result.by_model]
        assert keys == ["gpt-4o", "gpt-4o-mini"]  # sorted by descending cost
        assert result.by_model[0].calls == 2
        assert result.by_model[0].cost_usd == 0.10

    def test_by_tag_counts_each_tag(self):
        report = _report(
            [
                _record(tags=["prod", "chat"], cost=0.02),
                _record(tags=["prod"], cost=0.03),
                _record(tags=[], cost=0.01),
            ]
        )
        result = analyze_efficiency(report)
        by_tag = {s.key: s for s in result.by_tag}
        assert by_tag["prod"].calls == 2
        assert round(by_tag["prod"].cost_usd, 6) == 0.05
        assert by_tag["chat"].calls == 1

    def test_empty_report(self):
        result = analyze_efficiency(_report([]))
        assert result.records_analyzed == 0
        assert result.overall.calls == 0
        assert result.by_model == []
        assert result.by_tag == []

    def test_skips_bad_records(self):
        report = _report(
            [
                _record(cost=0.01),
                {"model": "", "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.1},
                {"model": "x", "input_tokens": "bad", "output_tokens": 1, "cost_usd": 0.1},
                {"model": "y", "input_tokens": -5, "output_tokens": 1, "cost_usd": 0.1},
                "not-a-dict",
            ]
        )
        result = analyze_efficiency(report)
        assert result.records_analyzed == 1
        assert result.records_skipped == 4

    def test_missing_records_key(self):
        result = analyze_efficiency({"summary": {}})
        assert result.records_analyzed == 0


class TestEfficiencyCLI:
    def _write_report(self, tmp_path, records):
        path = tmp_path / "report.json"
        path.write_text(json.dumps(_report(records)))
        return str(path)

    def test_text_output(self, tmp_path):
        report = self._write_report(
            tmp_path,
            [
                _record(model="gpt-4o", tags=["prod"], cost=0.05),
                _record(model="gpt-4o-mini", cost=0.01),
            ],
        )
        result = CliRunner().invoke(cli, ["efficiency", report])
        assert result.exit_code == 0
        assert "Token Efficiency" in result.output
        assert "gpt-4o" in result.output
        assert "By model:" in result.output
        assert "By tag:" in result.output

    def test_json_output(self, tmp_path):
        report = self._write_report(tmp_path, [_record(cost=0.02)])
        result = CliRunner().invoke(cli, ["efficiency", report, "--json-output"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["records_analyzed"] == 1
        assert payload["overall"]["output_input_ratio"] == 0.5
        assert "by_model" in payload

    def test_no_records_message(self, tmp_path):
        report = self._write_report(tmp_path, [])
        result = CliRunner().invoke(cli, ["efficiency", report])
        assert result.exit_code == 0
        assert "No usable records" in result.output

    def test_missing_file_errors(self):
        result = CliRunner().invoke(cli, ["efficiency", "/no/such/file.json"])
        assert result.exit_code != 0

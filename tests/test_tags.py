"""Tests for tag-based cost grouping: tracker, exporters, and CLI."""

import json
import tempfile

import pytest
from click.testing import CliRunner

from llm_cost_guardian import (
    UNTAGGED,
    CostTracker,
    to_csv,
    to_json,
    to_markdown,
    to_prometheus,
)
from llm_cost_guardian.cli import cli
from llm_cost_guardian.tracker import _normalize_tags


class TestNormalizeTags:
    def test_none_returns_empty(self):
        assert _normalize_tags(None) == ()

    def test_empty_list_returns_empty(self):
        assert _normalize_tags([]) == ()

    def test_basic(self):
        assert _normalize_tags(["prod", "chatbot"]) == ("prod", "chatbot")

    def test_strips_whitespace(self):
        assert _normalize_tags(["  prod  ", "\tchatbot\n"]) == ("prod", "chatbot")

    def test_drops_empty_strings(self):
        assert _normalize_tags(["prod", "", "   "]) == ("prod",)

    def test_dedupes_preserving_order(self):
        assert _normalize_tags(["b", "a", "b", "a"]) == ("b", "a")

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError, match="Tags must be strings"):
            _normalize_tags(["prod", 42])  # type: ignore[list-item]

    def test_accepts_tuple(self):
        assert _normalize_tags(("x", "y")) == ("x", "y")


class TestRecordTags:
    def test_default_is_empty_tuple(self):
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", 100, 50)
        assert rec.tags == ()

    def test_tags_stored_on_record(self):
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", 100, 50, tags=["prod", "search"])
        assert rec.tags == ("prod", "search")

    def test_tags_normalized_on_record(self):
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", 100, 50, tags=[" prod ", "prod", ""])
        assert rec.tags == ("prod",)

    def test_bad_tag_type_raises(self):
        tracker = CostTracker()
        with pytest.raises(TypeError):
            tracker.record("gpt-4o", 100, 50, tags=[None])  # type: ignore[list-item]
        assert tracker.records == []


class TestCostByTag:
    def test_empty_tracker(self):
        assert CostTracker().cost_by_tag() == {}

    def test_no_tags_anywhere_returns_empty(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=0.5)
        assert tracker.cost_by_tag() == {}

    def test_single_tag(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=0.5, tags=["prod"])
        assert tracker.cost_by_tag() == {"prod": 0.5}

    def test_accumulates_across_records(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=0.5, tags=["prod"])
        tracker.record("gpt-4o-mini", 100, 50, cost=0.25, tags=["prod"])
        assert tracker.cost_by_tag() == {"prod": 0.75}

    def test_multi_tag_record_counts_in_each(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=0.4, tags=["prod", "search"])
        by_tag = tracker.cost_by_tag()
        assert by_tag["prod"] == 0.4
        assert by_tag["search"] == 0.4

    def test_untagged_bucket_when_mixed(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=0.5, tags=["prod"])
        tracker.record("gpt-4o", 100, 50, cost=0.3)
        by_tag = tracker.cost_by_tag()
        assert by_tag["prod"] == 0.5
        assert by_tag[UNTAGGED] == 0.3

    def test_summary_includes_cost_by_tag(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=0.5, tags=["dev"])
        summary = tracker.summary()
        assert summary["cost_by_tag"] == {"dev": 0.5}

    def test_summary_cost_by_tag_empty_without_tags(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=0.5)
        assert tracker.summary()["cost_by_tag"] == {}


class TestFilterByTag:
    def test_filter_tag(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=0.5, tags=["prod"])
        tracker.record("gpt-4o", 100, 50, cost=0.3, tags=["dev"])
        tracker.record("gpt-4o", 100, 50, cost=0.1)
        results = tracker.filter(tag="prod")
        assert len(results) == 1
        assert results[0].cost == 0.5

    def test_filter_tag_no_match(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=0.5, tags=["prod"])
        assert tracker.filter(tag="staging") == []

    def test_filter_tag_combined_with_model(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=0.5, tags=["prod"])
        tracker.record("gpt-4o-mini", 100, 50, cost=0.2, tags=["prod"])
        results = tracker.filter(model="gpt-4o", tag="prod")
        assert len(results) == 1
        assert results[0].model == "gpt-4o"


class TestExportersWithTags:
    def _tracker(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=0.5, tags=["prod", "search"])
        tracker.record("gpt-4o-mini", 100, 50, cost=0.2)
        return tracker

    def test_json_includes_tags(self):
        data = json.loads(to_json(self._tracker()))
        assert data["records"][0]["tags"] == ["prod", "search"]
        assert data["records"][1]["tags"] == []
        assert data["summary"]["cost_by_tag"]["prod"] == 0.5

    def test_csv_includes_tags_column(self):
        output = to_csv(self._tracker())
        lines = output.strip().splitlines()
        assert lines[0].split(",")[-2] == "tags"
        assert "prod;search" in lines[1]

    def test_csv_untagged_record_has_empty_cell(self):
        output = to_csv(self._tracker())
        lines = output.strip().split("\r\n" if "\r\n" in output else "\n")
        assert lines[2].rstrip().endswith(",")

    def test_prometheus_includes_tag_gauge(self):
        output = to_prometheus(self._tracker())
        assert 'llm_cost_guardian_cost_by_tag_usd{tag="prod"}' in output
        assert 'tag="(untagged)"' in output

    def test_prometheus_omits_tag_gauge_without_tags(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=0.5)
        assert "cost_by_tag" not in to_prometheus(tracker)

    def test_markdown_includes_tag_section(self):
        md = to_markdown(self._tracker())
        assert "## Cost by tag" in md
        assert "`prod`" in md
        assert "`(untagged)`" in md

    def test_markdown_omits_tag_section_without_tags(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=0.5)
        assert "## Cost by tag" not in to_markdown(tracker)


class TestTagsCli:
    def _write_report(self, records):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"summary": {}, "records": records}, f)
        f.close()
        return f.name

    def test_tags_table_output(self):
        path = self._write_report(
            [
                {"model": "gpt-4o", "cost_usd": 0.5, "tags": ["prod"]},
                {"model": "gpt-4o", "cost_usd": 0.3, "tags": ["dev"]},
                {"model": "gpt-4o", "cost_usd": 0.2},
            ]
        )
        result = CliRunner().invoke(cli, ["tags", path])
        assert result.exit_code == 0
        assert "prod" in result.output
        assert "dev" in result.output
        assert "(untagged)" in result.output
        assert "Total" in result.output

    def test_tags_sorted_by_cost_descending(self):
        path = self._write_report(
            [
                {"model": "a", "cost_usd": 0.1, "tags": ["cheap"]},
                {"model": "b", "cost_usd": 0.9, "tags": ["pricey"]},
            ]
        )
        result = CliRunner().invoke(cli, ["tags", path])
        assert result.output.index("pricey") < result.output.index("cheap")

    def test_tags_json_output(self):
        path = self._write_report(
            [
                {"model": "gpt-4o", "cost_usd": 0.5, "tags": ["prod", "search"]},
                {"model": "gpt-4o", "cost_usd": 0.5, "tags": ["prod"]},
            ]
        )
        result = CliRunner().invoke(cli, ["tags", path, "--json-output"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["total_cost_usd"] == 1.0
        by_tag = {t["tag"]: t for t in payload["tags"]}
        assert by_tag["prod"]["cost_usd"] == 1.0
        assert by_tag["prod"]["calls"] == 2
        assert by_tag["search"]["share_pct"] == 50.0

    def test_tags_no_records(self):
        path = self._write_report([])
        result = CliRunner().invoke(cli, ["tags", path])
        assert result.exit_code == 0
        assert "No records" in result.output

    def test_tags_no_tags_in_report(self):
        path = self._write_report([{"model": "gpt-4o", "cost_usd": 0.5}])
        result = CliRunner().invoke(cli, ["tags", path])
        assert result.exit_code == 0
        assert "No tags found" in result.output

    def test_tags_invalid_json(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        f.write("not json")
        f.close()
        result = CliRunner().invoke(cli, ["tags", f.name])
        assert result.exit_code == 1

    def test_tags_missing_file(self):
        result = CliRunner().invoke(cli, ["tags", "/nonexistent/report.json"])
        assert result.exit_code == 2


class TestTagsRoundtrip:
    def test_export_then_cli(self, tmp_path):
        tracker = CostTracker()
        tracker.record("gpt-4o", 1000, 500, tags=["prod", "chatbot"])
        tracker.record("gpt-4o-mini", 2000, 1000, tags=["dev"])
        tracker.record("gpt-4o-mini", 100, 50)
        path = tmp_path / "report.json"
        path.write_text(to_json(tracker))

        result = CliRunner().invoke(cli, ["tags", str(path), "--json-output"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        tag_names = {t["tag"] for t in payload["tags"]}
        assert tag_names == {"prod", "chatbot", "dev", "(untagged)"}

"""Tests for per-user cost attribution."""

from __future__ import annotations

import csv
import io
import json

import pytest
from click.testing import CliRunner

from llm_cost_guardian import UNATTRIBUTED, CostTracker, to_csv, to_json, to_markdown, to_prometheus
from llm_cost_guardian.cli import cli
from llm_cost_guardian.tracker import _normalize_user


class TestNormalizeUser:
    def test_none_passthrough(self):
        assert _normalize_user(None) is None

    def test_strips_whitespace(self):
        assert _normalize_user("  alice  ") == "alice"

    def test_empty_becomes_none(self):
        assert _normalize_user("") is None
        assert _normalize_user("   ") is None

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            _normalize_user(42)  # type: ignore[arg-type]


class TestRecordWithUser:
    def test_default_user_is_none(self):
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", 100, 50)
        assert rec.user is None

    def test_user_stored_on_record(self):
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", 100, 50, user="alice")
        assert rec.user == "alice"

    def test_user_normalized(self):
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", 100, 50, user="  bob ")
        assert rec.user == "bob"

    def test_empty_user_becomes_none(self):
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", 100, 50, user="   ")
        assert rec.user is None

    def test_non_string_user_raises(self):
        tracker = CostTracker()
        with pytest.raises(TypeError):
            tracker.record("gpt-4o", 100, 50, user=123)  # type: ignore[arg-type]

    def test_user_and_tags_together(self):
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", 100, 50, user="alice", tags=["prod"])
        assert rec.user == "alice"
        assert rec.tags == ("prod",)


class TestCostByUser:
    def test_empty_tracker(self):
        assert CostTracker().cost_by_user() == {}

    def test_no_users_returns_empty(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50)
        assert tracker.cost_by_user() == {}

    def test_costs_grouped_per_user(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=1.0, user="alice")
        tracker.record("gpt-4o", 100, 50, cost=2.0, user="bob")
        tracker.record("gpt-4o", 100, 50, cost=3.0, user="alice")
        by_user = tracker.cost_by_user()
        assert by_user["alice"] == pytest.approx(4.0)
        assert by_user["bob"] == pytest.approx(2.0)

    def test_unattributed_bucket(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=1.0, user="alice")
        tracker.record("gpt-4o", 100, 50, cost=2.5)
        by_user = tracker.cost_by_user()
        assert by_user[UNATTRIBUTED] == pytest.approx(2.5)

    def test_sums_to_total_cost(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=1.0, user="alice")
        tracker.record("gpt-4o", 100, 50, cost=2.0, user="bob")
        tracker.record("gpt-4o", 100, 50, cost=0.5)
        assert sum(tracker.cost_by_user().values()) == pytest.approx(tracker.total_cost)

    def test_summary_includes_cost_by_user(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=1.0, user="alice")
        summary = tracker.summary()
        assert summary["cost_by_user"] == {"alice": pytest.approx(1.0)}

    def test_summary_empty_when_no_users(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=1.0)
        assert tracker.summary()["cost_by_user"] == {}


class TestFilterByUser:
    def test_filter_user(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, user="alice")
        tracker.record("gpt-4o", 100, 50, user="bob")
        tracker.record("gpt-4o", 100, 50)
        results = tracker.filter(user="alice")
        assert len(results) == 1
        assert results[0].user == "alice"

    def test_filter_user_no_match(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, user="alice")
        assert tracker.filter(user="carol") == []

    def test_filter_user_combined_with_model(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, user="alice")
        tracker.record("gpt-4o-mini", 100, 50, user="alice")
        results = tracker.filter(model="gpt-4o", user="alice")
        assert len(results) == 1
        assert results[0].model == "gpt-4o"


class TestExportersWithUser:
    def _tracker(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=1.0, user="alice")
        tracker.record("gpt-4o", 200, 80, cost=2.0, user="bob")
        tracker.record("gpt-4o", 10, 5, cost=0.5)
        return tracker

    def test_json_records_carry_user(self):
        data = json.loads(to_json(self._tracker()))
        users = [r["user"] for r in data["records"]]
        assert users == ["alice", "bob", None]
        assert data["summary"]["cost_by_user"]["alice"] == pytest.approx(1.0)

    def test_csv_has_user_column(self):
        rows = list(csv.reader(io.StringIO(to_csv(self._tracker()))))
        assert rows[0][-1] == "user"
        assert rows[1][-1] == "alice"
        assert rows[2][-1] == "bob"
        assert rows[3][-1] == ""

    def test_prometheus_cost_by_user_gauge(self):
        text = to_prometheus(self._tracker())
        assert 'llm_cost_guardian_cost_by_user_usd{user="alice"} 1.00000000' in text
        assert 'llm_cost_guardian_cost_by_user_usd{user="(unattributed)"}' in text

    def test_prometheus_omits_gauge_without_users(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=1.0)
        assert "cost_by_user" not in to_prometheus(tracker)

    def test_markdown_cost_by_user_table(self):
        md = to_markdown(self._tracker())
        assert "## Cost by user" in md
        assert "| `alice` | $1.000000 |" in md
        assert "| `(unattributed)` | $0.500000 |" in md

    def test_markdown_omits_section_without_users(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=1.0)
        assert "Cost by user" not in to_markdown(tracker)


class TestUsersCLI:
    def _write_report(self, tmp_path, with_users=True):
        tracker = CostTracker()
        if with_users:
            tracker.record("gpt-4o", 100, 50, cost=3.0, user="alice")
            tracker.record("gpt-4o", 100, 50, cost=1.0, user="bob")
            tracker.record("gpt-4o", 100, 50, cost=0.5)
        else:
            tracker.record("gpt-4o", 100, 50, cost=1.0)
        path = tmp_path / "report.json"
        path.write_text(to_json(tracker))
        return str(path)

    def test_table_output(self, tmp_path):
        result = CliRunner().invoke(cli, ["users", self._write_report(tmp_path)])
        assert result.exit_code == 0
        assert "Cost by User" in result.output
        assert "alice" in result.output
        assert "bob" in result.output
        assert "(unattributed)" in result.output

    def test_sorted_by_cost_desc(self, tmp_path):
        result = CliRunner().invoke(cli, ["users", self._write_report(tmp_path)])
        assert result.output.index("alice") < result.output.index("bob")

    def test_json_output(self, tmp_path):
        result = CliRunner().invoke(cli, ["users", self._write_report(tmp_path), "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total_cost_usd"] == pytest.approx(4.5)
        assert data["users"][0]["user"] == "alice"
        assert data["users"][0]["share_pct"] == pytest.approx(66.67)

    def test_no_users_message(self, tmp_path):
        result = CliRunner().invoke(cli, ["users", self._write_report(tmp_path, with_users=False)])
        assert result.exit_code == 0
        assert "No users found" in result.output

    def test_empty_records(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"summary": {}, "records": []}))
        result = CliRunner().invoke(cli, ["users", str(path)])
        assert result.exit_code == 0
        assert "No records found" in result.output

    def test_invalid_json_errors(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        result = CliRunner().invoke(cli, ["users", str(path)])
        assert result.exit_code == 1

    def test_missing_file_errors(self):
        result = CliRunner().invoke(cli, ["users", "/nonexistent/report.json"])
        assert result.exit_code != 0

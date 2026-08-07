"""Tests for the terminal dashboard: data builder, rendering, and CLI."""

from __future__ import annotations

import builtins
import json

import pytest
from click.testing import CliRunner

from llm_cost_guardian import CostTracker, to_json
from llm_cost_guardian.cli import cli
from llm_cost_guardian.dashboard import build_dashboard_data, render_dashboard

DAY_SECONDS = 86400.0
BASE_TS = 1_754_000_000.0  # fixed reference timestamp


def make_report(*, with_tags: bool = True, with_users: bool = True) -> dict:
    tracker = CostTracker()
    kw1 = {}
    kw2 = {}
    if with_tags:
        kw1["tags"] = ["prod", "chatbot"]
        kw2["tags"] = ["dev"]
    if with_users:
        kw1["user"] = "alice"
        kw2["user"] = "bob"
    tracker.record("gpt-4o", 1000, 500, **kw1)
    tracker.record("gpt-4o-mini", 2000, 1000, **kw2)
    tracker.record("claude-sonnet-4-20250514", 500, 250, **kw1)
    data = json.loads(to_json(tracker))
    for i, rec in enumerate(data["records"]):
        rec["timestamp"] = BASE_TS + i * DAY_SECONDS
    return data


class TestBuildDashboardData:
    def test_totals(self):
        data = make_report()
        dash = build_dashboard_data(data)
        totals = dash["totals"]
        assert totals["calls"] == 3
        assert totals["input_tokens"] == 3500
        assert totals["output_tokens"] == 1750
        assert totals["cost_usd"] == pytest.approx(
            sum(r["cost_usd"] for r in data["records"]), abs=1e-6
        )
        assert totals["avg_cost_per_call_usd"] == pytest.approx(totals["cost_usd"] / 3, abs=1e-6)

    def test_by_model_sorted_desc_with_share(self):
        dash = build_dashboard_data(make_report())
        by_model = dash["by_model"]
        assert len(by_model) == 3
        costs = [row["cost_usd"] for row in by_model]
        assert costs == sorted(costs, reverse=True)
        assert sum(row["share_pct"] for row in by_model) == pytest.approx(100.0, abs=0.1)

    def test_by_day_buckets(self):
        dash = build_dashboard_data(make_report(), utc=True)
        assert len(dash["by_day"]) == 3
        days = [row["day"] for row in dash["by_day"]]
        assert days == sorted(days)
        assert all(row["calls"] == 1 for row in dash["by_day"])

    def test_unknown_timestamp_goes_to_unknown_bucket(self):
        data = make_report()
        data["records"][0]["timestamp"] = None
        dash = build_dashboard_data(data)
        assert dash["by_day"][-1]["day"] == "(unknown)"

    def test_trend_days_limits_daily_rows(self):
        data = make_report()
        dash = build_dashboard_data(data, utc=True, trend_days=2)
        assert len(dash["by_day"]) == 2
        # keeps the most recent days
        all_days = sorted({row["day"] for row in build_dashboard_data(data, utc=True)["by_day"]})
        assert [r["day"] for r in dash["by_day"]] == all_days[-2:]

    def test_tags_and_users(self):
        dash = build_dashboard_data(make_report())
        tags = {row["tag"] for row in dash["by_tag"]}
        assert tags == {"prod", "chatbot", "dev"}
        users = {row["user"] for row in dash["by_user"]}
        assert users == {"alice", "bob"}

    def test_no_tags_or_users_yields_empty_sections(self):
        dash = build_dashboard_data(make_report(with_tags=False, with_users=False))
        assert dash["by_tag"] == []
        assert dash["by_user"] == []

    def test_top_limits_rows(self):
        dash = build_dashboard_data(make_report(), top=1)
        assert len(dash["by_model"]) == 1
        assert len(dash["top_calls"]) == 1
        assert len(dash["by_tag"]) == 1
        assert len(dash["by_user"]) == 1

    def test_top_calls_sorted_desc(self):
        dash = build_dashboard_data(make_report())
        costs = [row["cost_usd"] for row in dash["top_calls"]]
        assert costs == sorted(costs, reverse=True)

    def test_budget_ok_warn_over(self):
        data = make_report()
        total = build_dashboard_data(data)["totals"]["cost_usd"]

        ok = build_dashboard_data(data, budget=total * 10)["budget"]
        assert ok["status"] == "ok"
        assert ok["used_pct"] == pytest.approx(10.0, abs=0.1)

        warn = build_dashboard_data(data, budget=total / 0.9)["budget"]
        assert warn["status"] == "warn"

        over = build_dashboard_data(data, budget=total / 2)["budget"]
        assert over["status"] == "over"
        assert over["remaining_usd"] < 0

    def test_no_budget_is_none(self):
        assert build_dashboard_data(make_report())["budget"] is None

    def test_invalid_budget_raises(self):
        with pytest.raises(ValueError, match="budget must be positive"):
            build_dashboard_data(make_report(), budget=0)

    def test_invalid_top_raises(self):
        with pytest.raises(ValueError, match="top must be at least 1"):
            build_dashboard_data(make_report(), top=0)

    def test_invalid_trend_days_raises(self):
        with pytest.raises(ValueError, match="trend_days must be at least 1"):
            build_dashboard_data(make_report(), trend_days=0)

    def test_empty_report_falls_back_to_summary(self):
        data = {
            "summary": {
                "total_cost_usd": 1.5,
                "total_input_tokens": 10,
                "total_output_tokens": 20,
                "total_requests": 4,
            },
            "records": [],
        }
        dash = build_dashboard_data(data)
        assert dash["totals"]["cost_usd"] == pytest.approx(1.5)
        assert dash["totals"]["calls"] == 4
        assert dash["by_model"] == []
        assert dash["by_day"] == []

    def test_json_safe(self):
        dash = build_dashboard_data(make_report(), budget=5.0)
        json.dumps(dash)  # must not raise


class TestRenderDashboard:
    def _render_to_text(self, dash: dict) -> str:
        from rich.console import Console

        console = Console(record=True, width=100)
        console.print(render_dashboard(dash))
        return console.export_text()

    def test_render_contains_sections(self):
        text = self._render_to_text(build_dashboard_data(make_report(), budget=100.0))
        assert "Totals" in text
        assert "Budget" in text
        assert "Cost by Model" in text
        assert "Daily Trend" in text
        assert "Top Tags" in text
        assert "Top Users" in text
        assert "Most Expensive Calls" in text
        assert "gpt-4o" in text

    def test_render_over_budget_flag(self):
        data = make_report()
        total = build_dashboard_data(data)["totals"]["cost_usd"]
        text = self._render_to_text(build_dashboard_data(data, budget=total / 2))
        assert "OVER BUDGET" in text

    def test_render_minimal_report(self):
        dash = build_dashboard_data(make_report(with_tags=False, with_users=False))
        text = self._render_to_text(dash)
        assert "Totals" in text
        assert "Top Tags" not in text
        assert "Top Users" not in text


class TestDashboardCli:
    def _write_report(self, tmp_path, data=None) -> str:
        path = tmp_path / "report.json"
        path.write_text(json.dumps(data if data is not None else make_report()))
        return str(path)

    def test_dashboard_renders(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", self._write_report(tmp_path)])
        assert result.exit_code == 0
        assert "Totals" in result.output
        assert "Cost by Model" in result.output

    def test_dashboard_with_budget(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", self._write_report(tmp_path), "--budget", "100"])
        assert result.exit_code == 0
        assert "Budget" in result.output

    def test_dashboard_json_output(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["dashboard", self._write_report(tmp_path), "--json-output", "--budget", "100"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["totals"]["calls"] == 3
        assert payload["budget"]["limit_usd"] == 100

    def test_dashboard_invalid_budget(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", self._write_report(tmp_path), "--budget", "-1"])
        assert result.exit_code == 1
        assert "budget must be positive" in result.output

    def test_dashboard_invalid_watch(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", self._write_report(tmp_path), "--watch", "0"])
        assert result.exit_code == 1
        assert "--watch must be positive" in result.output

    def test_dashboard_missing_file(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", "nope.json"])
        assert result.exit_code != 0

    def test_dashboard_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", str(path)])
        assert result.exit_code == 1
        assert "not valid JSON" in result.output

    def test_dashboard_without_rich_shows_install_hint(self, tmp_path, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "rich.console" or name.startswith("rich.") or name == "rich":
                raise ImportError("No module named 'rich'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        runner = CliRunner()
        result = runner.invoke(cli, ["dashboard", self._write_report(tmp_path)])
        assert result.exit_code == 1
        assert "llm-cost-guardian[dashboard]" in result.output

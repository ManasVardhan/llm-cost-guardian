"""Tests for daily cost breakdown: cost_by_day API and the daily CLI command."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from click.testing import CliRunner

from llm_cost_guardian import CostTracker, to_json
from llm_cost_guardian.cli import cli


def _ts(year: int, month: int, day: int, hour: int = 12) -> float:
    """Epoch seconds for a UTC datetime."""
    return datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp()


def _make_tracker() -> CostTracker:
    """Three calls on Jan 5, two on Jan 6, one on Jan 8 (UTC noon, so the
    local date matches the UTC date in any timezone within UTC-12..UTC+11)."""
    tracker = CostTracker()
    for day, count in ((5, 3), (6, 2), (8, 1)):
        for i in range(count):
            rec = tracker.record("gpt-4o", 100, 50, cost=0.01)
            rec.timestamp = _ts(2026, 1, day, hour=12) + i * 60
    return tracker


class TestCostByDay:
    def test_empty_tracker(self):
        assert CostTracker().cost_by_day() == {}

    def test_groups_by_utc_day(self):
        tracker = _make_tracker()
        by_day = tracker.cost_by_day(utc=True)
        assert by_day == {
            "2026-01-05": 0.03,
            "2026-01-06": 0.02,
            "2026-01-08": 0.01,
        }

    def test_keys_sorted_chronologically(self):
        tracker = CostTracker()
        for day in (20, 3, 11):
            rec = tracker.record("gpt-4o", 10, 5, cost=0.001)
            rec.timestamp = _ts(2026, 2, day)
        assert list(tracker.cost_by_day(utc=True)) == [
            "2026-02-03",
            "2026-02-11",
            "2026-02-20",
        ]

    def test_sums_to_total_cost(self):
        tracker = _make_tracker()
        assert sum(tracker.cost_by_day(utc=True).values()) == round(tracker.total_cost, 10)

    def test_local_bucketing_near_midnight_utc(self):
        """A record just after midnight UTC lands on the previous local day
        in timezones behind UTC, so local and UTC buckets may differ."""
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", 10, 5, cost=0.001)
        rec.timestamp = _ts(2026, 3, 10, hour=0) + 60  # 00:01 UTC
        utc_days = list(tracker.cost_by_day(utc=True))
        local_days = list(tracker.cost_by_day())
        assert utc_days == ["2026-03-10"]
        local_day = datetime.fromtimestamp(rec.timestamp).date().isoformat()
        assert local_days == [local_day]


class TestDailyCli:
    def _report_path(self, tmp_path, tracker: CostTracker) -> str:
        path = tmp_path / "report.json"
        path.write_text(to_json(tracker))
        return str(path)

    def test_table_output(self, tmp_path):
        runner = CliRunner()
        path = self._report_path(tmp_path, _make_tracker())
        result = runner.invoke(cli, ["daily", path, "--utc"])
        assert result.exit_code == 0
        assert "Cost by Day (UTC)" in result.output
        assert "2026-01-05" in result.output
        assert "2026-01-06" in result.output
        assert "2026-01-08" in result.output
        assert "Total" in result.output
        # Biggest day gets the longest bar
        lines = {line.split()[0]: line for line in result.output.splitlines() if "#" in line}
        assert lines["2026-01-05"].count("#") > lines["2026-01-08"].count("#")

    def test_local_is_default_header(self, tmp_path):
        runner = CliRunner()
        path = self._report_path(tmp_path, _make_tracker())
        result = runner.invoke(cli, ["daily", path])
        assert result.exit_code == 0
        assert "Cost by Day (local)" in result.output

    def test_days_limit_keeps_most_recent(self, tmp_path):
        runner = CliRunner()
        path = self._report_path(tmp_path, _make_tracker())
        result = runner.invoke(cli, ["daily", path, "--utc", "--days", "2"])
        assert result.exit_code == 0
        assert "2026-01-05" not in result.output
        assert "2026-01-06" in result.output
        assert "2026-01-08" in result.output

    def test_days_must_be_positive(self, tmp_path):
        runner = CliRunner()
        path = self._report_path(tmp_path, _make_tracker())
        result = runner.invoke(cli, ["daily", path, "--days", "0"])
        assert result.exit_code == 1
        assert "must be at least 1" in result.output

    def test_json_output(self, tmp_path):
        runner = CliRunner()
        path = self._report_path(tmp_path, _make_tracker())
        result = runner.invoke(cli, ["daily", path, "--utc", "--json-output"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["timezone"] == "utc"
        assert payload["total_cost_usd"] == 0.06
        days = {d["day"]: d for d in payload["days"]}
        assert days["2026-01-05"]["calls"] == 3
        assert days["2026-01-05"]["tokens"] == 450
        assert days["2026-01-05"]["cost_usd"] == 0.03
        assert days["2026-01-05"]["share_pct"] == 50.0

    def test_json_share_sums_to_100(self, tmp_path):
        runner = CliRunner()
        path = self._report_path(tmp_path, _make_tracker())
        result = runner.invoke(cli, ["daily", path, "--utc", "--json-output"])
        payload = json.loads(result.output)
        assert round(sum(d["share_pct"] for d in payload["days"]), 1) == 100.0

    def test_unknown_timestamp_bucket(self, tmp_path):
        runner = CliRunner()
        report = {
            "summary": {},
            "records": [
                {
                    "model": "gpt-4o",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cost_usd": 0.01,
                    "timestamp": _ts(2026, 1, 5),
                },
                {
                    "model": "gpt-4o",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cost_usd": 0.02,
                    "timestamp": None,
                },
                {
                    "model": "gpt-4o",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cost_usd": 0.03,
                    "timestamp": "not-a-number",
                },
            ],
        }
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report))
        result = runner.invoke(cli, ["daily", str(path), "--utc", "--json-output"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        days = {d["day"]: d for d in payload["days"]}
        assert days["(unknown)"]["calls"] == 2
        assert days["(unknown)"]["cost_usd"] == 0.05
        # Unknown bucket is listed last
        assert payload["days"][-1]["day"] == "(unknown)"

    def test_empty_records(self, tmp_path):
        runner = CliRunner()
        path = tmp_path / "report.json"
        path.write_text(json.dumps({"summary": {}, "records": []}))
        result = runner.invoke(cli, ["daily", str(path)])
        assert result.exit_code == 0
        assert "No records found" in result.output

    def test_invalid_json_report(self, tmp_path):
        runner = CliRunner()
        path = tmp_path / "report.json"
        path.write_text("{not json")
        result = runner.invoke(cli, ["daily", str(path)])
        assert result.exit_code == 1

    def test_missing_file(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["daily", "does-not-exist.json"])
        assert result.exit_code != 0

    def test_zero_cost_records_no_bars(self, tmp_path):
        runner = CliRunner()
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", 0, 0, cost=0.0)
        rec.timestamp = _ts(2026, 1, 5)
        path = self._report_path(tmp_path, tracker)
        result = runner.invoke(cli, ["daily", path, "--utc"])
        assert result.exit_code == 0
        assert "#" not in result.output

"""Tests for cost anomaly detection."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

from llm_cost_guardian import Anomaly, AnomalyReport, analyze_anomalies
from llm_cost_guardian.cli import cli

UTC = timezone.utc


def _ts(day: str, hour: int = 12) -> float:
    """UTC timestamp for an ISO day at the given hour."""
    return datetime.fromisoformat(day).replace(hour=hour, tzinfo=UTC).timestamp()


def _record(
    day: str,
    cost: float,
    model: str = "gpt-4o",
    user: str | None = None,
    hour: int = 12,
) -> dict:
    rec = {
        "model": model,
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": cost,
        "timestamp": _ts(day, hour),
    }
    if user is not None:
        rec["user"] = user
    return rec


def _report(records: list[dict]) -> dict:
    return {"summary": {}, "records": records}


def _days(start: str, costs: list[float], **kwargs) -> list[dict]:
    """One record per day starting at ISO date start with the given costs."""
    first = datetime.fromisoformat(start).date()
    return [
        _record((first + timedelta(days=i)).isoformat(), cost, **kwargs)
        for i, cost in enumerate(costs)
    ]


class TestAnalyzeAnomalies:
    def test_flat_spend_has_no_anomalies(self):
        report = analyze_anomalies(
            _report(_days("2026-08-01", [1.0] * 10)), utc=True
        )
        assert report.anomalies == []
        assert not report.has_anomalies
        assert report.days_analyzed == 10
        assert report.records_analyzed == 10

    def test_total_spike_detected(self):
        records = _days("2026-08-01", [1.0, 1.0, 1.0, 1.0, 5.0])
        report = analyze_anomalies(_report(records), utc=True)
        total = [a for a in report.anomalies if a.dimension == "total"]
        assert len(total) == 1
        anomaly = total[0]
        assert anomaly.day == "2026-08-05"
        assert anomaly.key == "(total)"
        assert anomaly.spend_usd == pytest.approx(5.0)
        assert anomaly.baseline_usd == pytest.approx(1.0)
        assert anomaly.ratio == pytest.approx(5.0)
        assert anomaly.calls == 1

    def test_model_dimension_spike(self):
        # Steady gpt-4o spend, plus a claude spike on the last day.
        records = _days("2026-08-01", [1.0] * 8, model="gpt-4o")
        records += _days("2026-08-01", [0.1] * 7 + [2.0], model="claude-3-5-haiku-20241022")
        report = analyze_anomalies(_report(records), utc=True)
        model_hits = [a for a in report.anomalies if a.dimension == "model"]
        assert [a.key for a in model_hits] == ["claude-3-5-haiku-20241022"]
        assert model_hits[0].day == "2026-08-08"
        assert model_hits[0].ratio == pytest.approx(20.0)

    def test_user_dimension_spike(self):
        records = _days("2026-08-01", [0.5] * 8, user="alice")
        records += _days("2026-08-01", [0.5] * 7 + [4.0], user="bob")
        report = analyze_anomalies(_report(records), utc=True)
        user_hits = [a for a in report.anomalies if a.dimension == "user"]
        assert [a.key for a in user_hits] == ["bob"]
        assert user_hits[0].ratio == pytest.approx(8.0)

    def test_min_spend_filters_tiny_spikes(self):
        # 10x spike but only fractions of a cent.
        records = _days("2026-08-01", [0.0001] * 4 + [0.001])
        report = analyze_anomalies(_report(records), utc=True)
        assert report.anomalies == []
        # Lowering min_spend surfaces it.
        report = analyze_anomalies(_report(records), min_spend=0.0005, utc=True)
        assert any(a.dimension == "total" for a in report.anomalies)

    def test_new_spend_with_zero_baseline(self):
        # Nothing for days, then spend appears: flagged with ratio None.
        records = _days("2026-08-01", [0.0, 0.0, 0.0, 0.0, 3.0])
        report = analyze_anomalies(_report(records), utc=True)
        total = [a for a in report.anomalies if a.dimension == "total"]
        assert len(total) == 1
        assert total[0].ratio is None
        assert total[0].baseline_usd == 0.0
        assert total[0].spend_usd == pytest.approx(3.0)

    def test_bursty_every_other_day_usage_not_flagged(self):
        # Same $1 spend every other day: quiet days are skipped in the
        # baseline, so steady bursty usage does not false alarm.
        costs = [1.0, 0.0] * 5
        report = analyze_anomalies(_report(_days("2026-08-01", costs)), utc=True)
        assert report.anomalies == []

    def test_spike_after_gap_uses_active_day_baseline(self):
        # $1 on day 1, quiet week, then $3: baseline is the active day mean
        # (1.0), so day 8 is a 3x spike, while a return to $1 stays quiet.
        records = [_record("2026-08-01", 1.0), _record("2026-08-08", 3.0)]
        report = analyze_anomalies(_report(records), utc=True)
        total = [a for a in report.anomalies if a.dimension == "total"]
        assert len(total) == 1
        assert total[0].day == "2026-08-08"
        assert total[0].baseline_usd == pytest.approx(1.0)
        assert total[0].ratio == pytest.approx(3.0)

        calm = [_record("2026-08-01", 1.0), _record("2026-08-08", 1.0)]
        report = analyze_anomalies(_report(calm), utc=True)
        assert report.anomalies == []

    def test_window_limits_baseline(self):
        # Expensive early days roll out of a 3-day window, so a return to
        # 4.0 counts as a spike against the recent cheap days.
        costs = [4.0, 4.0, 4.0, 0.5, 0.5, 0.5, 4.0]
        report = analyze_anomalies(_report(_days("2026-08-01", costs)), window=3, utc=True)
        total = [a for a in report.anomalies if a.dimension == "total"]
        assert [a.day for a in total] == ["2026-08-07"]
        assert total[0].baseline_usd == pytest.approx(0.5)
        # With a wide window the early expensive days keep the baseline high.
        report = analyze_anomalies(_report(_days("2026-08-01", costs)), window=7, utc=True)
        assert [a for a in report.anomalies if a.dimension == "total"] == []

    def test_min_history_prevents_early_flagging(self):
        # Spike on day 2 is inside the min_history warmup.
        records = _days("2026-08-01", [1.0, 50.0, 1.0, 1.0])
        report = analyze_anomalies(_report(records), utc=True)
        assert report.anomalies == []
        report = analyze_anomalies(_report(records), min_history=1, utc=True)
        assert any(a.day == "2026-08-02" for a in report.anomalies)

    def test_invalid_records_skipped(self):
        records = _days("2026-08-01", [1.0] * 5)
        records += [
            {"model": "gpt-4o", "cost_usd": 1.0},  # no timestamp
            {"model": "gpt-4o", "cost_usd": 1.0, "timestamp": "not-a-number"},
            {"model": "gpt-4o", "cost_usd": 1.0, "timestamp": -5},
            "not a dict",
        ]
        report = analyze_anomalies(_report(records), utc=True)
        assert report.records_analyzed == 5
        assert report.records_skipped == 4

    def test_empty_report(self):
        report = analyze_anomalies(_report([]))
        assert report.anomalies == []
        assert report.days_analyzed == 0
        assert report.records_analyzed == 0

    def test_missing_records_key(self):
        report = analyze_anomalies({"summary": {}})
        assert report.anomalies == []

    def test_param_validation(self):
        data = _report(_days("2026-08-01", [1.0] * 5))
        with pytest.raises(ValueError, match="window"):
            analyze_anomalies(data, window=0)
        with pytest.raises(ValueError, match="threshold"):
            analyze_anomalies(data, threshold=0)
        with pytest.raises(ValueError, match="min_spend"):
            analyze_anomalies(data, min_spend=0)
        with pytest.raises(ValueError, match="min_history"):
            analyze_anomalies(data, min_history=0)

    def test_anomalies_sorted_by_day_then_dimension(self):
        records = _days("2026-08-01", [1.0] * 4 + [6.0, 1.0, 1.0, 8.0], user="alice")
        report = analyze_anomalies(_report(records), utc=True)
        keys = [(a.day, a.dimension) for a in report.anomalies]
        assert keys == sorted(
            keys, key=lambda k: (k[0], {"total": 0, "model": 1, "user": 2}[k[1]])
        )

    def test_to_dict_is_json_serializable(self):
        records = _days("2026-08-01", [1.0] * 4 + [5.0], user="alice")
        report = analyze_anomalies(_report(records), utc=True)
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["anomaly_count"] == len(report.anomalies)
        assert payload["params"]["window"] == 7
        assert payload["params"]["timezone"] == "utc"
        for entry in payload["anomalies"]:
            assert set(entry) == {
                "dimension",
                "key",
                "day",
                "spend_usd",
                "baseline_usd",
                "ratio",
                "calls",
            }

    def test_dataclass_exports(self):
        anomaly = Anomaly("total", "(total)", "2026-08-05", 5.0, 1.0, 5.0, 3)
        assert anomaly.to_dict()["ratio"] == 5.0
        report = AnomalyReport(
            anomalies=[anomaly],
            days_analyzed=5,
            records_analyzed=5,
            records_skipped=0,
            window=7,
            threshold=2.0,
            min_spend=0.01,
            min_history=3,
            timezone="utc",
        )
        assert report.has_anomalies


class TestAnomaliesCli:
    def _write_report(self, tmp_path, records: list[dict]) -> str:
        path = tmp_path / "report.json"
        path.write_text(json.dumps(_report(records)))
        return str(path)

    def test_no_anomalies_exit_zero(self, tmp_path):
        path = self._write_report(tmp_path, _days("2026-08-01", [1.0] * 6))
        result = CliRunner().invoke(cli, ["anomalies", path, "--utc"])
        assert result.exit_code == 0
        assert "no anomalies" in result.output

    def test_spike_exits_two_and_renders_table(self, tmp_path):
        path = self._write_report(tmp_path, _days("2026-08-01", [1.0] * 4 + [5.0]))
        result = CliRunner().invoke(cli, ["anomalies", path, "--utc"])
        assert result.exit_code == 2
        assert "2026-08-05" in result.output
        assert "(total)" in result.output
        assert "5.0x" in result.output
        assert "ALERT" in result.output

    def test_new_spend_renders_new_marker(self, tmp_path):
        path = self._write_report(tmp_path, _days("2026-08-01", [0.0] * 4 + [3.0]))
        result = CliRunner().invoke(cli, ["anomalies", path, "--utc"])
        assert result.exit_code == 2
        assert "new" in result.output

    def test_json_output(self, tmp_path):
        path = self._write_report(tmp_path, _days("2026-08-01", [1.0] * 4 + [5.0]))
        result = CliRunner().invoke(cli, ["anomalies", path, "--utc", "--json-output"])
        assert result.exit_code == 2
        payload = json.loads(result.output)
        assert payload["anomaly_count"] >= 1
        assert payload["anomalies"][0]["day"] == "2026-08-05"

    def test_json_output_clean_exit_zero(self, tmp_path):
        path = self._write_report(tmp_path, _days("2026-08-01", [1.0] * 6))
        result = CliRunner().invoke(cli, ["anomalies", path, "--utc", "--json-output"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["anomaly_count"] == 0

    def test_invalid_window_exits_one(self, tmp_path):
        path = self._write_report(tmp_path, _days("2026-08-01", [1.0] * 6))
        result = CliRunner().invoke(cli, ["anomalies", path, "--window", "0"])
        assert result.exit_code == 1
        assert "window" in result.output

    def test_threshold_option(self, tmp_path):
        # 1.5x jump is quiet at the default 2.0 threshold, loud at 1.2.
        path = self._write_report(tmp_path, _days("2026-08-01", [1.0] * 4 + [1.5]))
        result = CliRunner().invoke(cli, ["anomalies", path, "--utc"])
        assert result.exit_code == 0
        result = CliRunner().invoke(cli, ["anomalies", path, "--utc", "-t", "1.2"])
        assert result.exit_code == 2

    def test_skipped_records_warning(self, tmp_path):
        records = _days("2026-08-01", [1.0] * 6)
        records.append({"model": "gpt-4o", "cost_usd": 1.0})
        path = self._write_report(tmp_path, records)
        result = CliRunner().invoke(cli, ["anomalies", path, "--utc"])
        assert "skipped 1 record" in result.output

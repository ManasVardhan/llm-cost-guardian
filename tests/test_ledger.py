"""Tests for the persistent JSONL cost ledger."""

import json

import pytest
from click.testing import CliRunner

from llm_cost_guardian import CostLedger, CostTracker, UsageRecord
from llm_cost_guardian.cli import cli


def make_record(model="gpt-4o", cost=0.01, timestamp=1000.0, **kwargs):
    return UsageRecord(
        model=model,
        input_tokens=kwargs.pop("input_tokens", 100),
        output_tokens=kwargs.pop("output_tokens", 50),
        cost=cost,
        timestamp=timestamp,
        **kwargs,
    )


class TestCostLedger:
    def test_append_and_read_round_trip(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        ledger = CostLedger(path)
        rec = make_record(
            tags=("prod", "chat"),
            user="alice",
            metadata={"request_id": "r1"},
        )
        ledger.append(rec)

        loaded = ledger.records()
        assert len(loaded) == 1
        got = loaded[0]
        assert got.model == "gpt-4o"
        assert got.input_tokens == 100
        assert got.output_tokens == 50
        assert got.cost == pytest.approx(0.01)
        assert got.timestamp == pytest.approx(1000.0)
        assert got.tags == ("prod", "chat")
        assert got.user == "alice"
        assert got.metadata == {"request_id": "r1"}

    def test_read_missing_file_returns_empty(self, tmp_path):
        ledger = CostLedger(tmp_path / "nope.jsonl")
        assert ledger.records() == []
        assert ledger.skipped_lines == 0

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "costs.jsonl"
        ledger = CostLedger(path)
        ledger.append(make_record())
        assert path.exists()

    def test_malformed_lines_skipped(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        ledger = CostLedger(path)
        ledger.append(make_record())
        with open(path, "a") as f:
            f.write("not json\n")
            f.write('{"model": "gpt-4o"}\n')  # missing required fields
            f.write('[1, 2, 3]\n')  # not an object
            f.write("\n")  # blank lines are fine, not counted
        ledger.append(make_record(model="gpt-4"))

        loaded = ledger.records()
        assert [r.model for r in loaded] == ["gpt-4o", "gpt-4"]
        assert ledger.skipped_lines == 3

    def test_negative_tokens_rejected_on_read(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        with open(path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "model": "gpt-4o",
                        "input_tokens": -5,
                        "output_tokens": 1,
                        "cost_usd": 0.01,
                    }
                )
                + "\n"
            )
        ledger = CostLedger(path)
        assert ledger.records() == []
        assert ledger.skipped_lines == 1

    def test_since_until_filters(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        ledger = CostLedger(path)
        for ts in (100.0, 200.0, 300.0):
            ledger.append(make_record(timestamp=ts))

        assert len(ledger.records(since=150.0)) == 2
        assert len(ledger.records(until=250.0)) == 2
        assert len(ledger.records(since=150.0, until=250.0)) == 1

    def test_to_tracker_totals(self, tmp_path):
        ledger = CostLedger(tmp_path / "costs.jsonl")
        ledger.append(make_record(cost=0.01))
        ledger.append(make_record(model="gpt-4", cost=0.05))

        tracker = ledger.to_tracker()
        assert tracker.total_cost == pytest.approx(0.06)
        assert tracker.total_input_tokens == 200
        assert len(tracker.records) == 2
        assert set(tracker.cost_by_model()) == {"gpt-4o", "gpt-4"}

    def test_to_tracker_is_detached(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        ledger = CostLedger(path)
        ledger.append(make_record())
        tracker = ledger.to_tracker()
        tracker.record("gpt-4o", 10, 5, cost=0.001)
        # Recording on the detached tracker must not grow the file
        assert len(ledger.records()) == 1

    def test_len(self, tmp_path):
        ledger = CostLedger(tmp_path / "costs.jsonl")
        assert len(ledger) == 0
        ledger.append(make_record())
        assert len(ledger) == 1


class TestTrackerLedgerIntegration:
    def test_add_record_preserves_timestamp(self):
        tracker = CostTracker()
        rec = make_record(timestamp=123.0)
        tracker.add_record(rec)
        assert tracker.records[0].timestamp == pytest.approx(123.0)
        assert tracker.total_cost == pytest.approx(0.01)

    def test_add_record_rejects_negative_tokens(self):
        tracker = CostTracker()
        with pytest.raises(ValueError):
            tracker.add_record(make_record(input_tokens=-1))

    def test_attach_ledger_persists_records(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        tracker = CostTracker()
        tracker.attach_ledger(path)
        tracker.record("gpt-4o", 100, 50, cost=0.01, tags=["prod"], user="bob")

        reloaded = CostLedger(path).to_tracker()
        assert reloaded.total_cost == pytest.approx(0.01)
        assert reloaded.records[0].user == "bob"
        assert reloaded.records[0].tags == ("prod",)

    def test_attach_ledger_replay(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        first = CostTracker()
        first.attach_ledger(path)
        first.record("gpt-4o", 100, 50, cost=0.01)

        second = CostTracker()
        second.attach_ledger(path, replay=True)
        assert second.total_cost == pytest.approx(0.01)
        # Replay must not duplicate lines in the file
        assert len(CostLedger(path).records()) == 1

        second.record("gpt-4", 10, 5, cost=0.002)
        assert len(CostLedger(path).records()) == 2

    def test_replay_does_not_fire_callback(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        seed = CostTracker()
        seed.attach_ledger(path)
        seed.record("gpt-4o", 100, 50, cost=0.01)

        calls = []
        tracker = CostTracker(on_record=lambda rec, total: calls.append(rec))
        tracker.attach_ledger(path, replay=True)
        assert calls == []
        tracker.record("gpt-4o", 1, 1, cost=0.001)
        assert len(calls) == 1

    def test_detach_ledger_stops_writes(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        tracker = CostTracker()
        tracker.attach_ledger(path)
        tracker.record("gpt-4o", 100, 50, cost=0.01)
        tracker.detach_ledger()
        tracker.record("gpt-4o", 100, 50, cost=0.01)
        assert len(CostLedger(path).records()) == 1
        assert tracker.ledger is None

    def test_ledger_property(self, tmp_path):
        tracker = CostTracker()
        assert tracker.ledger is None
        ledger = tracker.attach_ledger(tmp_path / "costs.jsonl")
        assert tracker.ledger is ledger


class TestLedgerCLI:
    def _write_ledger(self, path, records):
        ledger = CostLedger(path)
        for rec in records:
            ledger.append(rec)
        return ledger

    def test_summary_output(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        self._write_ledger(path, [
            make_record(cost=0.01, timestamp=1700000000.0),
            make_record(model="gpt-4", cost=0.05, timestamp=1700086400.0),
        ])
        result = CliRunner().invoke(cli, ["ledger", str(path)])
        assert result.exit_code == 0
        assert "=== Cost Ledger ===" in result.output
        assert "Records:        2" in result.output
        assert "$0.060000" in result.output
        assert "gpt-4" in result.output

    def test_json_output(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        self._write_ledger(path, [make_record(cost=0.01)])
        result = CliRunner().invoke(cli, ["ledger", str(path), "--json-output"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["summary"]["total_cost_usd"] == 0.01
        assert len(payload["records"]) == 1

    def test_to_report_roundtrip(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        report = tmp_path / "report.json"
        self._write_ledger(path, [make_record(cost=0.01)])
        result = CliRunner().invoke(
            cli, ["ledger", str(path), "--to-report", str(report)]
        )
        assert result.exit_code == 0
        assert report.exists()
        # The generated report works with other commands
        result = CliRunner().invoke(cli, ["report", str(report)])
        assert result.exit_code == 0
        assert "$0.010000" in result.output

    def test_date_filtering(self, tmp_path):
        from datetime import datetime

        path = tmp_path / "costs.jsonl"
        day1 = datetime(2026, 8, 1, 12, 0).timestamp()
        day2 = datetime(2026, 8, 5, 12, 0).timestamp()
        self._write_ledger(path, [
            make_record(cost=0.01, timestamp=day1),
            make_record(cost=0.05, timestamp=day2),
        ])
        result = CliRunner().invoke(
            cli, ["ledger", str(path), "--since", "2026-08-03"]
        )
        assert result.exit_code == 0
        assert "Records:        1" in result.output
        assert "$0.050000" in result.output

        result = CliRunner().invoke(
            cli, ["ledger", str(path), "--until", "2026-08-03"]
        )
        assert result.exit_code == 0
        assert "$0.010000" in result.output

    def test_invalid_date_errors(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        self._write_ledger(path, [make_record()])
        result = CliRunner().invoke(
            cli, ["ledger", str(path), "--since", "not-a-date"]
        )
        assert result.exit_code == 1
        assert "must be a date" in result.output

    def test_malformed_lines_warn(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        self._write_ledger(path, [make_record()])
        with open(path, "a") as f:
            f.write("garbage\n")
        result = CliRunner().invoke(cli, ["ledger", str(path)])
        assert result.exit_code == 0
        assert "skipped 1 malformed line(s)" in result.output

    def test_empty_ledger(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        path.touch()
        result = CliRunner().invoke(cli, ["ledger", str(path)])
        assert result.exit_code == 0
        assert "No records found in ledger." in result.output

    def test_missing_file_errors(self, tmp_path):
        result = CliRunner().invoke(cli, ["ledger", str(tmp_path / "nope.jsonl")])
        assert result.exit_code != 0

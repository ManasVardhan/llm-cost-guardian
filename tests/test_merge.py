"""Tests for merging ledgers and reports."""

import json

import pytest
from click.testing import CliRunner

from llm_cost_guardian import (
    CostLedger,
    CostTracker,
    MergeError,
    UsageRecord,
    load_records,
    merge_sources,
    save_json,
)
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


def write_ledger(path, records):
    ledger = CostLedger(path)
    for rec in records:
        ledger.append(rec)
    return path


def write_report(path, records):
    tracker = CostTracker()
    for rec in records:
        tracker.add_record(rec)
    save_json(tracker, path)
    return path


class TestLoadRecords:
    def test_detects_ledger(self, tmp_path):
        path = write_ledger(tmp_path / "a.jsonl", [make_record(), make_record(model="gpt-4")])
        fmt, records, skipped = load_records(path)
        assert fmt == "ledger"
        assert [r.model for r in records] == ["gpt-4o", "gpt-4"]
        assert skipped == 0

    def test_detects_report(self, tmp_path):
        path = write_report(tmp_path / "a.json", [make_record(user="alice", tags=("prod",))])
        fmt, records, skipped = load_records(path)
        assert fmt == "report"
        assert len(records) == 1
        assert records[0].user == "alice"
        assert records[0].tags == ("prod",)
        assert skipped == 0

    def test_single_line_ledger_detected_as_ledger(self, tmp_path):
        path = write_ledger(tmp_path / "one.jsonl", [make_record()])
        fmt, records, skipped = load_records(path)
        assert fmt == "ledger"
        assert len(records) == 1
        assert skipped == 0

    def test_empty_file_is_empty_ledger(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        fmt, records, skipped = load_records(path)
        assert fmt == "ledger"
        assert records == []
        assert skipped == 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(MergeError, match="not found"):
            load_records(tmp_path / "nope.jsonl")

    def test_json_array_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(MergeError, match="list"):
            load_records(path)

    def test_json_object_without_records_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text('{"hello": "world"}')
        with pytest.raises(MergeError, match="neither"):
            load_records(path)

    def test_malformed_ledger_lines_counted(self, tmp_path):
        path = write_ledger(tmp_path / "a.jsonl", [make_record(), make_record()])
        with open(path, "a") as f:
            f.write("not json\n")
            f.write('{"model": "gpt-4o"}\n')
        fmt, records, skipped = load_records(path)
        assert fmt == "ledger"
        assert len(records) == 2
        assert skipped == 2

    def test_malformed_report_records_counted(self, tmp_path):
        path = tmp_path / "a.json"
        report = {
            "summary": {},
            "records": [
                {
                    "model": "gpt-4o",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cost_usd": 0.01,
                    "timestamp": 1000.0,
                },
                {"model": "gpt-4o"},
                "not a dict",
            ],
        }
        path.write_text(json.dumps(report))
        fmt, records, skipped = load_records(path)
        assert fmt == "report"
        assert len(records) == 1
        assert skipped == 2


class TestMergeSources:
    def test_merges_two_ledgers(self, tmp_path):
        a = write_ledger(tmp_path / "a.jsonl", [make_record(timestamp=2000.0, cost=0.02)])
        b = write_ledger(tmp_path / "b.jsonl", [make_record(timestamp=1000.0, cost=0.01)])
        result = merge_sources([a, b])
        assert result.total_records == 2
        assert result.duplicates_removed == 0
        assert result.tracker.total_cost == pytest.approx(0.03)
        # Sorted by timestamp regardless of source order.
        timestamps = [r.timestamp for r in result.tracker.records]
        assert timestamps == sorted(timestamps)

    def test_merges_ledger_and_report(self, tmp_path):
        a = write_ledger(tmp_path / "a.jsonl", [make_record(model="gpt-4o")])
        b = write_report(tmp_path / "b.json", [make_record(model="claude-3-5-sonnet-20241022")])
        result = merge_sources([a, b])
        assert result.total_records == 2
        assert {s.format for s in result.sources} == {"ledger", "report"}

    def test_dedupes_identical_records(self, tmp_path):
        shared = make_record(user="alice", tags=("prod",), metadata={"k": "v"})
        a = write_ledger(tmp_path / "a.jsonl", [shared, make_record(model="gpt-4")])
        b = write_ledger(tmp_path / "b.jsonl", [shared])
        result = merge_sources([a, b])
        assert result.total_records == 2
        assert result.duplicates_removed == 1
        assert result.tracker.total_cost == pytest.approx(0.02)

    def test_no_dedupe_keeps_duplicates(self, tmp_path):
        shared = make_record()
        a = write_ledger(tmp_path / "a.jsonl", [shared])
        b = write_ledger(tmp_path / "b.jsonl", [shared])
        result = merge_sources([a, b], dedupe=False)
        assert result.total_records == 2
        assert result.duplicates_removed == 0

    def test_near_duplicates_not_collapsed(self, tmp_path):
        a = write_ledger(tmp_path / "a.jsonl", [make_record(user="alice")])
        b = write_ledger(tmp_path / "b.jsonl", [make_record(user="bob")])
        result = merge_sources([a, b])
        assert result.total_records == 2
        assert result.duplicates_removed == 0

    def test_source_stats(self, tmp_path):
        a = write_ledger(tmp_path / "a.jsonl", [make_record(), make_record(model="gpt-4")])
        with open(a, "a") as f:
            f.write("garbage\n")
        result = merge_sources([a])
        assert result.sources[0].records == 2
        assert result.sources[0].skipped == 1
        assert result.total_skipped == 1

    def test_empty_paths_raises(self):
        with pytest.raises(MergeError, match="at least one"):
            merge_sources([])

    def test_missing_source_raises(self, tmp_path):
        a = write_ledger(tmp_path / "a.jsonl", [make_record()])
        with pytest.raises(MergeError, match="not found"):
            merge_sources([a, tmp_path / "nope.jsonl"])


class TestMergeCli:
    def test_merge_to_output_file(self, tmp_path):
        a = write_ledger(tmp_path / "a.jsonl", [make_record(cost=0.01)])
        b = write_report(tmp_path / "b.json", [make_record(model="gpt-4", cost=0.02)])
        out = tmp_path / "combined.json"
        runner = CliRunner()
        result = runner.invoke(cli, ["merge", str(a), str(b), "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert "Merged records:     2" in result.output
        assert "Duplicates removed: 0" in result.output

        data = json.loads(out.read_text())
        assert len(data["records"]) == 2
        assert data["summary"]["total_cost_usd"] == pytest.approx(0.03)

    def test_merge_stdout_is_report_json(self, tmp_path):
        a = write_ledger(tmp_path / "a.jsonl", [make_record()])
        runner = CliRunner()
        result = runner.invoke(cli, ["merge", str(a)])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["summary"]["total_requests"] == 1

    def test_merge_output_usable_by_report_command(self, tmp_path):
        a = write_ledger(tmp_path / "a.jsonl", [make_record()])
        b = write_ledger(tmp_path / "b.jsonl", [make_record(model="gpt-4")])
        out = tmp_path / "combined.json"
        runner = CliRunner()
        assert runner.invoke(cli, ["merge", str(a), str(b), "-o", str(out)]).exit_code == 0
        result = runner.invoke(cli, ["report", str(out)])
        assert result.exit_code == 0, result.output
        assert "Total requests: 2" in result.output

    def test_merge_dedupes_by_default(self, tmp_path):
        shared = make_record()
        a = write_ledger(tmp_path / "a.jsonl", [shared])
        b = write_ledger(tmp_path / "b.jsonl", [shared])
        out = tmp_path / "combined.json"
        runner = CliRunner()
        result = runner.invoke(cli, ["merge", str(a), str(b), "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert "Duplicates removed: 1" in result.output
        assert "Merged records:     1" in result.output

    def test_merge_no_dedupe_flag(self, tmp_path):
        shared = make_record()
        a = write_ledger(tmp_path / "a.jsonl", [shared])
        b = write_ledger(tmp_path / "b.jsonl", [shared])
        out = tmp_path / "combined.json"
        runner = CliRunner()
        result = runner.invoke(cli, ["merge", str(a), str(b), "--no-dedupe", "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert "Merged records:     2" in result.output

    def test_merge_json_summary(self, tmp_path):
        a = write_ledger(tmp_path / "a.jsonl", [make_record()])
        out = tmp_path / "combined.json"
        runner = CliRunner()
        result = runner.invoke(cli, ["merge", str(a), "-o", str(out), "--json-output"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["merged_records"] == 1
        assert payload["duplicates_removed"] == 0
        assert payload["sources"][0]["format"] == "ledger"

    def test_merge_warns_on_skipped_lines(self, tmp_path):
        a = write_ledger(tmp_path / "a.jsonl", [make_record()])
        with open(a, "a") as f:
            f.write("garbage\n")
        out = tmp_path / "combined.json"
        runner = CliRunner()
        result = runner.invoke(cli, ["merge", str(a), "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert "skipped 1 malformed entry" in result.output

    def test_merge_invalid_source_errors(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("[1, 2]")
        runner = CliRunner()
        result = runner.invoke(cli, ["merge", str(bad)])
        assert result.exit_code == 1
        assert "Error:" in result.output

    def test_merge_missing_file_errors(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["merge", str(tmp_path / "nope.jsonl")])
        assert result.exit_code == 2  # click Path(exists=True) usage error

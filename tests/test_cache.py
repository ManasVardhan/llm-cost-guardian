"""Tests for cache-aware cost tracking and the cache analysis command."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from llm_cost_guardian import (
    CostLedger,
    CostTracker,
    UsageRecord,
    analyze_cache,
    get_pricing,
    register_model,
    to_csv,
    to_json,
    to_markdown,
    to_prometheus,
)
from llm_cost_guardian.ledger import record_from_dict


def _report(tracker: CostTracker) -> dict:
    return json.loads(to_json(tracker))


class TestCachePricing:
    def test_cache_read_price_used(self):
        pricing = get_pricing("claude-sonnet-4-20250514")
        # 1M cache read tokens at $0.30 per 1M
        cost = pricing.calculate_cost(0, 0, cache_read_tokens=1_000_000)
        assert cost == pytest.approx(0.30)

    def test_cache_write_price_used(self):
        pricing = get_pricing("claude-sonnet-4-20250514")
        cost = pricing.calculate_cost(0, 0, cache_write_tokens=1_000_000)
        assert cost == pytest.approx(3.75)

    def test_openai_cache_write_falls_back_to_input_rate(self):
        pricing = get_pricing("gpt-4o")
        assert pricing.cache_write_cost_per_1m is None
        cost = pricing.calculate_cost(0, 0, cache_write_tokens=1_000_000)
        assert cost == pytest.approx(2.50)

    def test_model_without_cache_pricing_bills_input_rate(self):
        pricing = get_pricing("gpt-4")
        cost = pricing.calculate_cost(0, 0, cache_read_tokens=1_000_000)
        assert cost == pytest.approx(30.00)

    def test_mixed_token_types_sum(self):
        pricing = get_pricing("gpt-4o")
        cost = pricing.calculate_cost(
            1_000_000, 1_000_000, cache_read_tokens=1_000_000
        )
        assert cost == pytest.approx(2.50 + 10.00 + 1.25)

    def test_register_model_with_cache_prices(self):
        pricing = register_model(
            "cache-test-model", "openai", 1.00, 2.00,
            cache_read_cost_per_1m=0.10, cache_write_cost_per_1m=1.25,
        )
        assert pricing.effective_cache_read_cost_per_1m == 0.10
        assert pricing.effective_cache_write_cost_per_1m == 1.25

    def test_register_model_rejects_negative_cache_price(self):
        with pytest.raises(ValueError, match="non-negative"):
            register_model("bad-cache-model", "openai", 1.0, 2.0, cache_read_cost_per_1m=-1.0)


class TestTrackerCacheTokens:
    def test_record_with_cache_tokens_costs_less(self):
        cached = CostTracker()
        cached.record("claude-sonnet-4-20250514", 100, 500, cache_read_tokens=10_000)
        uncached = CostTracker()
        uncached.record("claude-sonnet-4-20250514", 10_100, 500)
        assert cached.total_cost < uncached.total_cost

    def test_totals_include_cache_tokens(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cache_read_tokens=900, cache_write_tokens=10)
        assert tracker.total_cache_read_tokens == 900
        assert tracker.total_cache_write_tokens == 10
        assert tracker.total_tokens == 100 + 50 + 900 + 10
        rec = tracker.last_record
        assert rec is not None
        assert rec.cached_tokens == 910
        assert rec.total_tokens == 1060

    def test_negative_cache_tokens_rejected(self):
        tracker = CostTracker()
        with pytest.raises(ValueError, match="non-negative"):
            tracker.record("gpt-4o", 100, 50, cache_read_tokens=-1)

    def test_summary_includes_cache_totals(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cache_read_tokens=200)
        summary = tracker.summary()
        assert summary["total_cache_read_tokens"] == 200
        assert summary["total_cache_write_tokens"] == 0

    def test_reset_clears_cache_totals(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cache_read_tokens=200, cache_write_tokens=100)
        tracker.reset()
        assert tracker.total_cache_read_tokens == 0
        assert tracker.total_cache_write_tokens == 0

    def test_explicit_cost_not_recalculated(self):
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", 100, 50, cost=1.23, cache_read_tokens=500)
        assert rec.cost == 1.23


class TestCacheSerialization:
    def test_json_round_trip(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cache_read_tokens=900, cache_write_tokens=25)
        data = _report(tracker)
        rec = data["records"][0]
        assert rec["cache_read_tokens"] == 900
        assert rec["cache_write_tokens"] == 25
        assert data["summary"]["total_cache_read_tokens"] == 900

    def test_csv_includes_cache_columns(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cache_read_tokens=900)
        output = to_csv(tracker)
        header, row = output.strip().splitlines()
        assert "cache_read_tokens" in header
        assert "cache_write_tokens" in header
        assert ",900," in row

    def test_prometheus_emits_cache_counters_only_when_present(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50)
        assert "cache_read" not in to_prometheus(tracker)
        tracker.record("gpt-4o", 100, 50, cache_read_tokens=10)
        output = to_prometheus(tracker)
        assert "llm_cost_guardian_total_cache_read_tokens 10" in output
        assert "llm_cost_guardian_total_cache_write_tokens 0" in output

    def test_markdown_shows_cache_rows_when_present(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50)
        assert "Cache read" not in to_markdown(tracker)
        tracker.record("gpt-4o", 100, 50, cache_read_tokens=10)
        md = to_markdown(tracker)
        assert "| Cache read tokens | 10 |" in md

    def test_ledger_round_trip(self, tmp_path):
        path = tmp_path / "costs.jsonl"
        tracker = CostTracker()
        tracker.attach_ledger(path)
        tracker.record("gpt-4o", 100, 50, cache_read_tokens=900, cache_write_tokens=25)

        loaded = CostLedger(path).records()
        assert len(loaded) == 1
        assert loaded[0].cache_read_tokens == 900
        assert loaded[0].cache_write_tokens == 25

    def test_old_ledger_line_defaults_to_zero_cache(self):
        rec = record_from_dict(
            {"model": "gpt-4o", "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01}
        )
        assert rec is not None
        assert rec.cache_read_tokens == 0
        assert rec.cache_write_tokens == 0

    def test_negative_cache_tokens_line_rejected(self):
        rec = record_from_dict(
            {
                "model": "gpt-4o",
                "input_tokens": 10,
                "output_tokens": 5,
                "cost_usd": 0.01,
                "cache_read_tokens": -3,
            }
        )
        assert rec is None


class TestAnalyzeCache:
    def _data(self) -> dict:
        tracker = CostTracker()
        for _ in range(4):
            tracker.record(
                "claude-sonnet-4-20250514", 200, 400,
                cache_read_tokens=20_000, tags=["prod"],
            )
        tracker.record("claude-sonnet-4-20250514", 200, 400, cache_write_tokens=20_000)
        for _ in range(3):
            tracker.record("gpt-4o", 5_000, 300)  # big prompts, no cache
        tracker.record("gpt-4o-mini", 100, 50)  # small prompts, no cache
        return _report(tracker)

    def test_by_model_stats(self):
        report = analyze_cache(self._data())
        assert report.records_analyzed == 9
        by_key = {s.key: s for s in report.by_model}
        sonnet = by_key["claude-sonnet-4-20250514"]
        assert sonnet.cache_read_tokens == 80_000
        assert sonnet.cache_write_tokens == 20_000
        assert sonnet.uses_cache
        # reads 80k at (3.00 - 0.30) minus writes 20k at (3.75 - 3.00) premium
        expected = (80_000 * 2.70 - 20_000 * 0.75) / 1_000_000
        assert sonnet.savings == pytest.approx(expected)
        assert sonnet.hit_rate == pytest.approx(80_000 / (80_000 + 1_000))

    def test_candidates_flagged_by_avg_input(self):
        report = analyze_cache(self._data())
        assert report.candidates == ["gpt-4o"]

    def test_min_candidate_input_zero_flags_all_uncached(self):
        report = analyze_cache(self._data(), min_candidate_input=0)
        assert set(report.candidates) == {"gpt-4o", "gpt-4o-mini"}

    def test_unpriced_model_has_unknown_savings(self):
        data = {
            "records": [
                {
                    "model": "totally-unknown-model-xyz",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.5,
                    "cache_read_tokens": 1_000,
                }
            ]
        }
        report = analyze_cache(data)
        assert report.by_model[0].savings is None
        assert report.unpriced_models == ["totally-unknown-model-xyz"]
        assert report.overall.priced is False

    def test_old_report_without_cache_fields(self):
        data = {
            "records": [
                {"model": "gpt-4o", "input_tokens": 5000, "output_tokens": 100, "cost_usd": 0.02}
            ]
        }
        report = analyze_cache(data)
        assert report.overall.cached_tokens == 0
        assert report.candidates == ["gpt-4o"]

    def test_malformed_records_skipped(self):
        data = {
            "records": [
                "nonsense",
                {"input_tokens": 5},
                {"model": "gpt-4o", "input_tokens": "x", "output_tokens": 1, "cost_usd": 0},
                {"model": "gpt-4o", "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01},
            ]
        }
        report = analyze_cache(data)
        assert report.records_analyzed == 1
        assert report.records_skipped == 3

    def test_empty_report(self):
        report = analyze_cache({"records": []})
        assert report.records_analyzed == 0
        assert report.overall.savings is None

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            analyze_cache({"records": []}, min_candidate_input=-1)

    def test_to_dict_shape(self):
        result = analyze_cache(self._data()).to_dict()
        assert set(result) == {
            "records_analyzed",
            "records_skipped",
            "min_candidate_input_tokens",
            "candidates",
            "unpriced_models",
            "overall",
            "by_model",
        }
        assert result["overall"]["key"] == "(all models)"


class TestCacheCLI:
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "llm_cost_guardian.cli", *args],
            capture_output=True,
            text=True,
        )

    def _write_report(self, tmp_path) -> str:
        tracker = CostTracker()
        tracker.record("claude-sonnet-4-20250514", 200, 400, cache_read_tokens=20_000)
        tracker.record("gpt-4o", 5_000, 300)
        path = tmp_path / "report.json"
        path.write_text(to_json(tracker))
        return str(path)

    def test_text_output(self, tmp_path):
        result = self._run("cache", self._write_report(tmp_path))
        assert result.returncode == 0
        assert "Prompt Cache Usage" in result.stdout
        assert "claude-sonnet-4-20250514" in result.stdout
        assert "Caching candidates" in result.stdout
        assert "gpt-4o" in result.stdout

    def test_json_output(self, tmp_path):
        result = self._run("cache", self._write_report(tmp_path), "--json-output")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["candidates"] == ["gpt-4o"]
        assert data["records_analyzed"] == 2

    def test_no_cache_usage_message(self, tmp_path):
        tracker = CostTracker()
        tracker.record("gpt-4o", 10, 5)
        path = tmp_path / "r.json"
        path.write_text(to_json(tracker))
        result = self._run("cache", str(path))
        assert result.returncode == 0
        assert "No cache usage recorded" in result.stdout

    def test_invalid_json_report(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        result = self._run("cache", str(path))
        assert result.returncode == 1

    def test_negative_threshold_errors(self, tmp_path):
        result = self._run(
            "cache", self._write_report(tmp_path), "--min-candidate-input", "-5"
        )
        assert result.returncode == 1
        assert "non-negative" in result.stderr


class TestWrapperCacheExtraction:
    def test_openai_wrapper_splits_cached_tokens(self):
        from llm_cost_guardian import TrackedOpenAI

        class Details:
            cached_tokens = 800

        class Usage:
            prompt_tokens = 1000
            completion_tokens = 50
            prompt_tokens_details = Details()

        class Response:
            model = "gpt-4o"
            usage = Usage()

        class Completions:
            def create(self, **kwargs):
                return Response()

        class Chat:
            completions = Completions()

        class Client:
            chat = Chat()

        tracker = CostTracker()
        TrackedOpenAI(Client(), tracker).chat.completions.create(model="gpt-4o", messages=[])
        rec = tracker.last_record
        assert rec is not None
        assert rec.input_tokens == 200
        assert rec.cache_read_tokens == 800
        expected = (200 * 2.50 + 800 * 1.25 + 50 * 10.00) / 1_000_000
        assert rec.cost == pytest.approx(expected)

    def test_openai_wrapper_without_details(self):
        from llm_cost_guardian import TrackedOpenAI

        class Usage:
            prompt_tokens = 1000
            completion_tokens = 50
            prompt_tokens_details = None

        class Response:
            model = "gpt-4o"
            usage = Usage()

        class Completions:
            def create(self, **kwargs):
                return Response()

        class Chat:
            completions = Completions()

        class Client:
            chat = Chat()

        tracker = CostTracker()
        TrackedOpenAI(Client(), tracker).chat.completions.create(model="gpt-4o", messages=[])
        rec = tracker.last_record
        assert rec is not None
        assert rec.input_tokens == 1000
        assert rec.cache_read_tokens == 0

    def test_anthropic_wrapper_reads_cache_fields(self):
        from llm_cost_guardian import TrackedAnthropic

        class Usage:
            input_tokens = 100
            output_tokens = 200
            cache_read_input_tokens = 5_000
            cache_creation_input_tokens = 1_000

        class Response:
            model = "claude-sonnet-4-20250514"
            usage = Usage()

        class Messages:
            def create(self, **kwargs):
                return Response()

        class Client:
            messages = Messages()

        tracker = CostTracker()
        TrackedAnthropic(Client(), tracker).messages.create(
            model="claude-sonnet-4-20250514", messages=[]
        )
        rec = tracker.last_record
        assert rec is not None
        assert rec.input_tokens == 100
        assert rec.cache_read_tokens == 5_000
        assert rec.cache_write_tokens == 1_000
        expected = (100 * 3.00 + 5_000 * 0.30 + 1_000 * 3.75 + 200 * 15.00) / 1_000_000
        assert rec.cost == pytest.approx(expected)

    def test_anthropic_wrapper_without_cache_fields(self):
        from llm_cost_guardian import TrackedAnthropic

        class Usage:
            input_tokens = 100
            output_tokens = 200

        class Response:
            model = "claude-sonnet-4-20250514"
            usage = Usage()

        class Messages:
            def create(self, **kwargs):
                return Response()

        class Client:
            messages = Messages()

        tracker = CostTracker()
        TrackedAnthropic(Client(), tracker).messages.create(
            model="claude-sonnet-4-20250514", messages=[]
        )
        rec = tracker.last_record
        assert rec is not None
        assert rec.cache_read_tokens == 0
        assert rec.cache_write_tokens == 0


class TestAddRecordCacheTokens:
    def test_add_record_accumulates_cache_totals(self):
        tracker = CostTracker()
        tracker.add_record(
            UsageRecord(
                model="gpt-4o",
                input_tokens=10,
                output_tokens=5,
                cost=0.01,
                cache_read_tokens=100,
                cache_write_tokens=50,
            )
        )
        assert tracker.total_cache_read_tokens == 100
        assert tracker.total_cache_write_tokens == 50

    def test_add_record_rejects_negative_cache_tokens(self):
        tracker = CostTracker()
        with pytest.raises(ValueError, match="non-negative"):
            tracker.add_record(
                UsageRecord(
                    model="gpt-4o",
                    input_tokens=10,
                    output_tokens=5,
                    cost=0.01,
                    cache_read_tokens=-1,
                )
            )

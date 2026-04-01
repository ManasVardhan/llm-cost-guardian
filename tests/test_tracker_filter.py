"""Tests for CostTracker.filter(), average_cost, and last_record."""

import time

from llm_cost_guardian import CostTracker


class TestAverageCost:
    def test_average_cost_empty(self) -> None:
        tracker = CostTracker()
        assert tracker.average_cost == 0.0

    def test_average_cost_single(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
        assert tracker.average_cost == tracker.total_cost

    def test_average_cost_multiple(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=1000, output_tokens=500, cost=0.10)
        tracker.record("gpt-4o", input_tokens=2000, output_tokens=1000, cost=0.20)
        assert abs(tracker.average_cost - 0.15) < 1e-10


class TestLastRecord:
    def test_last_record_empty(self) -> None:
        tracker = CostTracker()
        assert tracker.last_record is None

    def test_last_record_single(self) -> None:
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        assert tracker.last_record is rec

    def test_last_record_multiple(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        rec2 = tracker.record("gpt-4o-mini", input_tokens=200, output_tokens=100)
        assert tracker.last_record is rec2
        assert tracker.last_record.model == "gpt-4o-mini"


class TestFilter:
    def test_filter_by_model(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        tracker.record("gpt-4o-mini", input_tokens=200, output_tokens=100)
        tracker.record("gpt-4o", input_tokens=300, output_tokens=150)
        results = tracker.filter(model="gpt-4o")
        assert len(results) == 2
        assert all(r.model == "gpt-4o" for r in results)

    def test_filter_by_model_no_match(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        results = tracker.filter(model="nonexistent")
        assert len(results) == 0

    def test_filter_by_min_cost(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50, cost=0.01)
        tracker.record("gpt-4o", input_tokens=1000, output_tokens=500, cost=0.10)
        tracker.record("gpt-4o", input_tokens=10000, output_tokens=5000, cost=1.00)
        results = tracker.filter(min_cost=0.05)
        assert len(results) == 2

    def test_filter_by_since(self) -> None:
        tracker = CostTracker()
        old_rec = tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        cutoff = time.time()
        new_rec = tracker.record("gpt-4o", input_tokens=200, output_tokens=100)
        results = tracker.filter(since=cutoff)
        assert len(results) == 1
        assert results[0] is new_rec

    def test_filter_by_until(self) -> None:
        tracker = CostTracker()
        old_rec = tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        cutoff = time.time()
        tracker.record("gpt-4o", input_tokens=200, output_tokens=100)
        results = tracker.filter(until=cutoff)
        assert len(results) == 1
        assert results[0] is old_rec

    def test_filter_by_predicate(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        tracker.record("gpt-4o", input_tokens=5000, output_tokens=2500)
        results = tracker.filter(predicate=lambda r: r.total_tokens > 1000)
        assert len(results) == 1
        assert results[0].total_tokens == 7500

    def test_filter_combined(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50, cost=0.01)
        tracker.record("gpt-4o", input_tokens=1000, output_tokens=500, cost=0.10)
        tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50, cost=0.001)
        results = tracker.filter(model="gpt-4o", min_cost=0.05)
        assert len(results) == 1
        assert results[0].cost == 0.10

    def test_filter_no_args_returns_all(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        tracker.record("gpt-4o-mini", input_tokens=200, output_tokens=100)
        results = tracker.filter()
        assert len(results) == 2

    def test_filter_empty_tracker(self) -> None:
        tracker = CostTracker()
        results = tracker.filter(model="gpt-4o")
        assert len(results) == 0

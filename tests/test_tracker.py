"""Tests for the cost tracker."""

import pytest

from llm_cost_guardian import CostTracker, get_pricing


class TestCostTracker:
    def test_record_calculates_cost(self):
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
        expected = get_pricing("gpt-4o").calculate_cost(1000, 500)
        assert rec.cost == pytest.approx(expected)
        assert tracker.total_cost == pytest.approx(expected)

    def test_record_with_explicit_cost(self):
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", input_tokens=100, output_tokens=50, cost=0.42)
        assert rec.cost == 0.42
        assert tracker.total_cost == 0.42

    def test_accumulates_multiple_records(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
        tracker.record("gpt-4o", input_tokens=2000, output_tokens=1000)
        assert len(tracker.records) == 2
        assert tracker.total_input_tokens == 3000
        assert tracker.total_output_tokens == 1500

    def test_cost_by_model(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 1000, 500)
        tracker.record("claude-3-5-haiku-20241022", 1000, 500)
        by_model = tracker.cost_by_model()
        assert "gpt-4o" in by_model
        assert "claude-3-5-haiku-20241022" in by_model

    def test_reset(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 1000, 500)
        tracker.reset()
        assert tracker.total_cost == 0.0
        assert tracker.records == []

    def test_callback_fired(self):
        results = []
        tracker = CostTracker(on_record=lambda r, c: results.append((r, c)))
        tracker.record("gpt-4o", 1000, 500)
        assert len(results) == 1

    def test_summary(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 1000, 500)
        s = tracker.summary()
        assert s["total_requests"] == 1
        assert s["total_input_tokens"] == 1000

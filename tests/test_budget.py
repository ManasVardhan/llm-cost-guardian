"""Tests for budget policies."""

import time

import pytest

from llm_cost_guardian import (
    Action,
    BudgetError,
    BudgetManager,
    BudgetResult,
    CostTracker,
    HardCapPolicy,
    SlidingWindowPolicy,
    SoftWarningPolicy,
)


class TestBudgetPolicies:
    def test_hard_cap_blocks(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100_000, 50_000)  # ~$0.75
        policy = HardCapPolicy(limit_usd=0.01)
        result = policy.evaluate(tracker)
        assert result.action == Action.BLOCK

    def test_hard_cap_allows(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 10, 5)
        policy = HardCapPolicy(limit_usd=100.0)
        result = policy.evaluate(tracker)
        assert result.action == Action.ALLOW

    def test_hard_cap_at_exact_limit(self):
        tracker = CostTracker()
        # Record with explicit cost to hit exact boundary
        tracker.record("gpt-4o", 0, 0, cost=5.00)
        policy = HardCapPolicy(limit_usd=5.00)
        result = policy.evaluate(tracker)
        assert result.action == Action.BLOCK

    def test_soft_warning(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100_000, 50_000)
        policy = SoftWarningPolicy(warning_usd=0.01)
        result = policy.evaluate(tracker)
        assert result.action == Action.WARN

    def test_soft_warning_below_threshold(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 10, 5)
        policy = SoftWarningPolicy(warning_usd=100.0)
        result = policy.evaluate(tracker)
        assert result.action == Action.ALLOW

    def test_budget_manager_enforce_raises(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100_000, 50_000)
        manager = BudgetManager()
        manager.add(HardCapPolicy(limit_usd=0.01))
        with pytest.raises(BudgetError):
            manager.enforce(tracker)


class TestSlidingWindowPolicy:
    def test_sliding_window_blocks_within_window(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100_000, 50_000)
        policy = SlidingWindowPolicy(limit_usd=0.01, window_seconds=3600)
        result = policy.evaluate(tracker)
        assert result.action == Action.BLOCK

    def test_sliding_window_allows_below_limit(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 10, 5)
        policy = SlidingWindowPolicy(limit_usd=100.0, window_seconds=3600)
        result = policy.evaluate(tracker)
        assert result.action == Action.ALLOW

    def test_sliding_window_custom_action(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100_000, 50_000)
        policy = SlidingWindowPolicy(
            limit_usd=0.01, window_seconds=3600, action_on_exceed=Action.WARN
        )
        result = policy.evaluate(tracker)
        assert result.action == Action.WARN

    def test_sliding_window_short_window(self):
        """Records outside the window should not count."""
        tracker = CostTracker()
        tracker.record("gpt-4o", 100_000, 50_000)
        # Manually backdate the record timestamp
        tracker.records[0].timestamp = time.time() - 7200  # 2 hours ago
        policy = SlidingWindowPolicy(limit_usd=0.01, window_seconds=3600)
        result = policy.evaluate(tracker)
        assert result.action == Action.ALLOW


class TestBudgetManager:
    def test_check_no_policies(self):
        manager = BudgetManager()
        tracker = CostTracker()
        result = manager.check(tracker)
        assert result.action == Action.ALLOW

    def test_check_returns_worst(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100_000, 50_000)
        manager = BudgetManager()
        manager.add(SoftWarningPolicy(warning_usd=0.01))
        result = manager.check(tracker)
        assert result.action == Action.WARN

    def test_block_takes_priority_over_warn(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100_000, 50_000)
        manager = BudgetManager()
        manager.add(SoftWarningPolicy(warning_usd=0.01))
        manager.add(HardCapPolicy(limit_usd=0.01))
        result = manager.check(tracker)
        assert result.action == Action.BLOCK

    def test_enforce_calls_on_warn(self):
        warnings = []
        tracker = CostTracker()
        tracker.record("gpt-4o", 100_000, 50_000)
        manager = BudgetManager(on_warn=lambda r: warnings.append(r))
        manager.add(SoftWarningPolicy(warning_usd=0.01))
        result = manager.enforce(tracker)
        assert result.action == Action.WARN
        assert len(warnings) == 1
        assert warnings[0].action == Action.WARN

    def test_enforce_allows_when_under_budget(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 10, 5)
        manager = BudgetManager()
        manager.add(HardCapPolicy(limit_usd=100.0))
        result = manager.enforce(tracker)
        assert result.action == Action.ALLOW

    def test_add_returns_self(self):
        """add() should return the manager for chaining."""
        manager = BudgetManager()
        result = manager.add(HardCapPolicy(limit_usd=1.0))
        assert result is manager

    def test_budget_error_has_result(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100_000, 50_000)
        manager = BudgetManager()
        manager.add(HardCapPolicy(limit_usd=0.01))
        with pytest.raises(BudgetError) as exc_info:
            manager.enforce(tracker)
        assert exc_info.value.result.action == Action.BLOCK
        assert exc_info.value.result.limit == 0.01


class TestBudgetResult:
    def test_budget_result_fields(self):
        result = BudgetResult(Action.WARN, "test message", 1.5, 2.0)
        assert result.action == Action.WARN
        assert result.message == "test message"
        assert result.current_cost == 1.5
        assert result.limit == 2.0

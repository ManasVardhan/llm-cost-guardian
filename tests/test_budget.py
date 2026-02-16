"""Tests for budget policies."""

import pytest

from llm_cost_guardian import (
    Action,
    BudgetError,
    BudgetManager,
    CostTracker,
    HardCapPolicy,
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

    def test_soft_warning(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100_000, 50_000)
        policy = SoftWarningPolicy(warning_usd=0.01)
        result = policy.evaluate(tracker)
        assert result.action == Action.WARN

    def test_budget_manager_enforce_raises(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100_000, 50_000)
        manager = BudgetManager()
        manager.add(HardCapPolicy(limit_usd=0.01))
        with pytest.raises(BudgetError):
            manager.enforce(tracker)

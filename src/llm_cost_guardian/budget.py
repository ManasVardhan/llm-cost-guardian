"""Budget policies for cost enforcement."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tracker import CostTracker


class Action(str, Enum):
    """What to do when a budget threshold is crossed."""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class BudgetResult:
    action: Action
    message: str
    current_cost: float
    limit: float


class BudgetPolicy:
    """Base class for budget policies."""

    def evaluate(self, tracker: CostTracker) -> BudgetResult:
        raise NotImplementedError


@dataclass
class HardCapPolicy(BudgetPolicy):
    """Block requests once total cost exceeds *limit_usd*."""

    limit_usd: float

    def evaluate(self, tracker: CostTracker) -> BudgetResult:
        cost = tracker.total_cost
        if cost >= self.limit_usd:
            return BudgetResult(
                Action.BLOCK,
                f"Hard cap reached: ${cost:.4f} >= ${self.limit_usd:.2f}",
                cost,
                self.limit_usd,
            )
        return BudgetResult(
            Action.ALLOW,
            f"Within budget: ${cost:.4f} / ${self.limit_usd:.2f}",
            cost,
            self.limit_usd,
        )


@dataclass
class SoftWarningPolicy(BudgetPolicy):
    """Warn (but allow) when cost exceeds *warning_usd*."""

    warning_usd: float

    def evaluate(self, tracker: CostTracker) -> BudgetResult:
        cost = tracker.total_cost
        if cost >= self.warning_usd:
            return BudgetResult(
                Action.WARN,
                f"Warning: ${cost:.4f} exceeds soft limit ${self.warning_usd:.2f}",
                cost,
                self.warning_usd,
            )
        return BudgetResult(
            Action.ALLOW,
            f"Within budget: ${cost:.4f} / ${self.warning_usd:.2f}",
            cost,
            self.warning_usd,
        )


@dataclass
class SlidingWindowPolicy(BudgetPolicy):
    """Enforce a cost limit over a rolling time window.

    Parameters
    ----------
    limit_usd : maximum spend in the window
    window_seconds : length of the sliding window (default 3600 = 1 hour)
    action : what to do when limit is exceeded (default BLOCK)
    """

    limit_usd: float
    window_seconds: float = 3600.0
    action_on_exceed: Action = Action.BLOCK

    def evaluate(self, tracker: CostTracker) -> BudgetResult:
        now = time.time()
        cutoff = now - self.window_seconds
        window_cost = sum(
            r.cost for r in tracker.records if r.timestamp >= cutoff
        )
        if window_cost >= self.limit_usd:
            return BudgetResult(
                self.action_on_exceed,
                f"Sliding window ({self.window_seconds}s) "
                f"cost ${window_cost:.4f} >= ${self.limit_usd:.2f}",
                window_cost,
                self.limit_usd,
            )
        return BudgetResult(
            Action.ALLOW,
            f"Window cost ${window_cost:.4f} / ${self.limit_usd:.2f}",
            window_cost,
            self.limit_usd,
        )


class BudgetError(Exception):
    """Raised when a BLOCK policy prevents an API call."""

    def __init__(self, result: BudgetResult) -> None:
        self.result = result
        super().__init__(result.message)


@dataclass
class BudgetManager:
    """Evaluates a stack of policies and enforces the strictest outcome."""

    policies: list[BudgetPolicy] = field(default_factory=list)
    on_warn: callable = None  # type: ignore[assignment]

    def add(self, policy: BudgetPolicy) -> BudgetManager:
        self.policies.append(policy)
        return self

    def check(self, tracker: CostTracker) -> BudgetResult:
        """Evaluate all policies. Returns the most restrictive result."""
        worst = BudgetResult(Action.ALLOW, "No policies configured", 0.0, 0.0)
        for policy in self.policies:
            result = policy.evaluate(tracker)
            if result.action == Action.BLOCK:
                return result
            if result.action == Action.WARN and worst.action == Action.ALLOW:
                worst = result
        return worst

    def enforce(self, tracker: CostTracker) -> BudgetResult:
        """Check policies and raise BudgetError if BLOCK is triggered."""
        result = self.check(tracker)
        if result.action == Action.WARN and self.on_warn:
            self.on_warn(result)
        if result.action == Action.BLOCK:
            raise BudgetError(result)
        return result

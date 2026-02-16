"""Budget enforcement example - demonstrates hard caps and warnings."""

from llm_cost_guardian import (
    BudgetError,
    BudgetManager,
    CostTracker,
    HardCapPolicy,
    SoftWarningPolicy,
)

tracker = CostTracker()
budget = BudgetManager(
    on_warn=lambda r: print(f"WARNING: {r.message}")
)
budget.add(SoftWarningPolicy(warning_usd=0.005))
budget.add(HardCapPolicy(limit_usd=0.01))

# Simulate calls until budget is hit
for i in range(20):
    try:
        budget.enforce(tracker)
        tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
        print(f"Call {i + 1}: ${tracker.total_cost:.6f}")
    except BudgetError as e:
        print(f"Call {i + 1} BLOCKED: {e}")
        break

print(f"\nFinal cost: ${tracker.total_cost:.6f}")

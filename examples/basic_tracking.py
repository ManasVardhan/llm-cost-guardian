"""Basic cost tracking example - no API keys needed."""

from llm_cost_guardian import CostTracker, to_json

tracker = CostTracker()

# Simulate some API calls by recording usage manually
tracker.record("gpt-4o", input_tokens=1500, output_tokens=800)
tracker.record("gpt-4o-mini", input_tokens=3000, output_tokens=1200)
tracker.record("claude-3-5-haiku-20241022", input_tokens=2000, output_tokens=600)

print(f"Total cost: ${tracker.total_cost:.6f}")
print(f"Total tokens: {tracker.total_tokens:,}")
print()

print("Cost by model:")
for model, cost in tracker.cost_by_model().items():
    print(f"  {model}: ${cost:.6f}")

print()
print("JSON export:")
print(to_json(tracker))

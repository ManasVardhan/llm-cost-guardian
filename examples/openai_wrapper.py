"""Example using the OpenAI wrapper (requires openai package and API key)."""

# from openai import OpenAI
# from llm_cost_guardian import CostTracker, TrackedOpenAI, BudgetManager, HardCapPolicy

# tracker = CostTracker()
# budget = BudgetManager().add(HardCapPolicy(limit_usd=1.00))
# client = TrackedOpenAI(OpenAI(), tracker, budget)
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "Hello!"}],
# )
#
# print(response.choices[0].message.content)
# print(f"Cost so far: ${tracker.total_cost:.6f}")

print("Uncomment the code above and set OPENAI_API_KEY to run this example.")
print()
print("How it works:")
print("  1. Wrap your OpenAI client: TrackedOpenAI(OpenAI(), tracker)")
print("  2. Use it exactly like the normal client")
print("  3. Costs are tracked automatically from response.usage")
print("  4. Budget policies can block calls before they happen")

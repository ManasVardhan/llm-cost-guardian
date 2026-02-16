"""Tests for client wrappers using mock objects."""

from types import SimpleNamespace

import pytest

from llm_cost_guardian import (
    BudgetError,
    BudgetManager,
    CostTracker,
    HardCapPolicy,
    TrackedAnthropic,
    TrackedOpenAI,
)


def _make_openai_response(model="gpt-4o", prompt_tokens=100, completion_tokens=50):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello!"))],
    )


def _make_anthropic_response(model="claude-3-5-haiku-20241022", input_tokens=100, output_tokens=50):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        content=[SimpleNamespace(text="Hello!")],
    )


class TestTrackedOpenAI:
    def test_tracks_usage(self):
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: _make_openai_response())
            )
        )
        tracker = CostTracker()
        client = TrackedOpenAI(mock_client, tracker)
        client.chat.completions.create(model="gpt-4o", messages=[])
        assert tracker.total_cost > 0
        assert tracker.total_input_tokens == 100

    def test_budget_blocks(self):
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: _make_openai_response())
            )
        )
        tracker = CostTracker()
        tracker.record("gpt-4o", 100_000, 100_000)  # burn budget
        budget = BudgetManager(policies=[HardCapPolicy(limit_usd=0.001)])
        client = TrackedOpenAI(mock_client, tracker, budget)
        with pytest.raises(BudgetError):
            client.chat.completions.create(model="gpt-4o", messages=[])


class TestTrackedAnthropic:
    def test_tracks_usage(self):
        mock_client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kw: _make_anthropic_response())
        )
        tracker = CostTracker()
        client = TrackedAnthropic(mock_client, tracker)
        client.messages.create(model="claude-3-5-haiku-20241022", messages=[], max_tokens=100)
        assert tracker.total_cost > 0
        assert tracker.total_input_tokens == 100

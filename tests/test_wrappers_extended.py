"""Extended wrapper tests: passthrough attributes, budget interactions, edge cases."""

from types import SimpleNamespace

import pytest

from llm_cost_guardian import (
    BudgetError,
    BudgetManager,
    CostTracker,
    HardCapPolicy,
    SlidingWindowPolicy,
    SoftWarningPolicy,
    TrackedAnthropic,
    TrackedOpenAI,
)


def _make_openai_response(model="gpt-4o", prompt_tokens=100, completion_tokens=50):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello!"))],
    )


def _make_anthropic_response(model="claude-sonnet-4-20250514", input_tokens=100, output_tokens=50):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        content=[SimpleNamespace(text="Hello!")],
    )


class TestTrackedOpenAIExtended:
    def test_passthrough_attributes(self):
        """Non-chat attributes should be proxied to the underlying client."""
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: _make_openai_response())
            ),
            api_key="test-key",
            base_url="https://api.openai.com",
        )
        tracker = CostTracker()
        client = TrackedOpenAI(mock_client, tracker)
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.openai.com"

    def test_tracks_correct_model_from_response(self):
        """Wrapper should use the model from the response, not the request."""
        def create(**kw):
            return _make_openai_response(model="gpt-4o-2024-08-06")

        mock_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        tracker = CostTracker()
        client = TrackedOpenAI(mock_client, tracker)
        client.chat.completions.create(model="gpt-4o", messages=[])
        records = tracker.records
        assert len(records) == 1
        assert records[0].model == "gpt-4o-2024-08-06"

    def test_no_usage_in_response(self):
        """If response has no usage, no record should be created."""
        response_no_usage = SimpleNamespace(
            model="gpt-4o",
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello!"))],
        )
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: response_no_usage)
            )
        )
        tracker = CostTracker()
        client = TrackedOpenAI(mock_client, tracker)
        client.chat.completions.create(model="gpt-4o", messages=[])
        assert tracker.total_cost == 0
        assert len(tracker.records) == 0

    def test_no_usage_attr_in_response(self):
        """If response doesn't have a usage attribute at all."""
        response_raw = SimpleNamespace(model="gpt-4o", choices=[])
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: response_raw)
            )
        )
        tracker = CostTracker()
        client = TrackedOpenAI(mock_client, tracker)
        client.chat.completions.create(model="gpt-4o", messages=[])
        assert tracker.total_cost == 0

    def test_multiple_calls_accumulate(self):
        """Multiple calls should accumulate tokens and cost."""
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: _make_openai_response())
            )
        )
        tracker = CostTracker()
        client = TrackedOpenAI(mock_client, tracker)

        for _ in range(5):
            client.chat.completions.create(model="gpt-4o", messages=[])

        assert len(tracker.records) == 5
        assert tracker.total_input_tokens == 500
        assert tracker.total_output_tokens == 250

    def test_budget_soft_warning(self):
        """Soft warning budget should allow the call but trigger the callback."""
        warnings = []
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: _make_openai_response())
            )
        )
        tracker = CostTracker()
        # Record enough to trigger warning
        tracker.record("gpt-4o", 100_000, 100_000)
        budget = BudgetManager(
            policies=[SoftWarningPolicy(warning_usd=0.001)],
            on_warn=lambda r: warnings.append(r),
        )
        client = TrackedOpenAI(mock_client, tracker, budget)
        # Should NOT raise, just warn
        response = client.chat.completions.create(model="gpt-4o", messages=[])
        assert response is not None
        assert len(warnings) == 1

    def test_exception_from_client_propagates(self):
        """If the underlying client raises, it should propagate."""
        def failing_create(**kw):
            raise ConnectionError("API unreachable")

        mock_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=failing_create))
        )
        tracker = CostTracker()
        client = TrackedOpenAI(mock_client, tracker)

        with pytest.raises(ConnectionError, match="API unreachable"):
            client.chat.completions.create(model="gpt-4o", messages=[])

        # No record should be created on error
        assert len(tracker.records) == 0

    def test_large_token_counts(self):
        """Should handle large token counts correctly."""
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: _make_openai_response(
                        prompt_tokens=1_000_000, completion_tokens=500_000
                    )
                )
            )
        )
        tracker = CostTracker()
        client = TrackedOpenAI(mock_client, tracker)
        client.chat.completions.create(model="gpt-4o", messages=[])
        assert tracker.total_input_tokens == 1_000_000
        assert tracker.total_output_tokens == 500_000
        assert tracker.total_cost > 0


class TestTrackedAnthropicExtended:
    def test_passthrough_attributes(self):
        """Non-messages attributes should be proxied to the underlying client."""
        mock_client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kw: _make_anthropic_response()),
            api_key="test-key",
            base_url="https://api.anthropic.com",
        )
        tracker = CostTracker()
        client = TrackedAnthropic(mock_client, tracker)
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.anthropic.com"

    def test_budget_blocks_anthropic(self):
        """Budget enforcement should work with Anthropic wrapper."""
        mock_client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kw: _make_anthropic_response())
        )
        tracker = CostTracker()
        tracker.record("claude-sonnet-4-20250514", 100_000, 100_000)
        budget = BudgetManager(policies=[HardCapPolicy(limit_usd=0.001)])
        client = TrackedAnthropic(mock_client, tracker, budget)
        with pytest.raises(BudgetError):
            client.messages.create(model="claude-sonnet-4-20250514", messages=[], max_tokens=100)

    def test_no_usage_in_response(self):
        """If response has no usage, no record should be created."""
        response_no_usage = SimpleNamespace(
            model="claude-sonnet-4-20250514",
            usage=None,
            content=[SimpleNamespace(text="Hello!")],
        )
        mock_client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kw: response_no_usage)
        )
        tracker = CostTracker()
        client = TrackedAnthropic(mock_client, tracker)
        client.messages.create(model="claude-sonnet-4-20250514", messages=[], max_tokens=100)
        assert tracker.total_cost == 0
        assert len(tracker.records) == 0

    def test_tracks_correct_model_from_response(self):
        """Wrapper should use the model from the response."""
        def create(**kw):
            return _make_anthropic_response(model="claude-sonnet-4-20250514")

        mock_client = SimpleNamespace(messages=SimpleNamespace(create=create))
        tracker = CostTracker()
        client = TrackedAnthropic(mock_client, tracker)
        client.messages.create(model="claude-sonnet-4-20250514", messages=[], max_tokens=100)
        records = tracker.records
        assert len(records) == 1
        assert records[0].model == "claude-sonnet-4-20250514"

    def test_multiple_calls_accumulate(self):
        """Multiple calls should accumulate."""
        mock_client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kw: _make_anthropic_response())
        )
        tracker = CostTracker()
        client = TrackedAnthropic(mock_client, tracker)

        for _ in range(3):
            client.messages.create(
                model="claude-sonnet-4-20250514", messages=[], max_tokens=100
            )

        assert len(tracker.records) == 3
        assert tracker.total_input_tokens == 300
        assert tracker.total_output_tokens == 150

    def test_exception_from_client_propagates(self):
        """If the underlying client raises, it should propagate."""
        def failing_create(**kw):
            raise ConnectionError("API unreachable")

        mock_client = SimpleNamespace(messages=SimpleNamespace(create=failing_create))
        tracker = CostTracker()
        client = TrackedAnthropic(mock_client, tracker)

        with pytest.raises(ConnectionError, match="API unreachable"):
            client.messages.create(
                model="claude-sonnet-4-20250514", messages=[], max_tokens=100
            )

        assert len(tracker.records) == 0

    def test_budget_soft_warning_anthropic(self):
        """Soft warning budget should allow the call for Anthropic."""
        warnings = []
        mock_client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kw: _make_anthropic_response())
        )
        tracker = CostTracker()
        tracker.record("claude-sonnet-4-20250514", 100_000, 100_000)
        budget = BudgetManager(
            policies=[SoftWarningPolicy(warning_usd=0.001)],
            on_warn=lambda r: warnings.append(r),
        )
        client = TrackedAnthropic(mock_client, tracker, budget)
        response = client.messages.create(
            model="claude-sonnet-4-20250514", messages=[], max_tokens=100
        )
        assert response is not None
        assert len(warnings) == 1


class TestBudgetIntegration:
    def test_sliding_window_blocks(self):
        """SlidingWindowPolicy should block when window cost exceeded."""
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: _make_openai_response())
            )
        )
        tracker = CostTracker()
        # Burn budget within the window
        tracker.record("gpt-4o", 500_000, 500_000)
        budget = BudgetManager(
            policies=[SlidingWindowPolicy(limit_usd=0.001, window_seconds=3600)]
        )
        client = TrackedOpenAI(mock_client, tracker, budget)
        with pytest.raises(BudgetError):
            client.chat.completions.create(model="gpt-4o", messages=[])

    def test_multiple_policies_strictest_wins(self):
        """When multiple policies are set, the strictest should apply."""
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: _make_openai_response())
            )
        )
        tracker = CostTracker()
        tracker.record("gpt-4o", 100_000, 100_000)  # spend some money
        budget = BudgetManager(
            policies=[
                SoftWarningPolicy(warning_usd=0.0001),  # Would warn
                HardCapPolicy(limit_usd=0.001),  # Would block
            ]
        )
        client = TrackedOpenAI(mock_client, tracker, budget)
        with pytest.raises(BudgetError):
            client.chat.completions.create(model="gpt-4o", messages=[])

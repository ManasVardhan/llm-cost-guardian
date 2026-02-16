"""Drop-in wrappers for OpenAI and Anthropic clients."""

from __future__ import annotations

from typing import Any

from .budget import BudgetManager
from .tracker import CostTracker


class TrackedOpenAI:
    """Wraps an ``openai.OpenAI`` client to automatically track costs.

    Usage::

        from openai import OpenAI
        from llm_cost_guardian import CostTracker, TrackedOpenAI

        tracker = CostTracker()
        client = TrackedOpenAI(OpenAI(), tracker)
        response = client.chat.completions.create(model="gpt-4o", messages=[...])
        print(tracker.total_cost)

    The wrapper intercepts ``chat.completions.create`` and records token usage
    from the response's ``usage`` field.
    """

    def __init__(
        self,
        client: Any,
        tracker: CostTracker,
        budget: BudgetManager | None = None,
    ) -> None:
        self._client = client
        self._tracker = tracker
        self._budget = budget
        self.chat = _OpenAIChatNamespace(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _OpenAIChatNamespace:
    def __init__(self, wrapper: TrackedOpenAI) -> None:
        self._wrapper = wrapper
        self.completions = _OpenAICompletions(wrapper)


class _OpenAICompletions:
    def __init__(self, wrapper: TrackedOpenAI) -> None:
        self._wrapper = wrapper

    def create(self, **kwargs: Any) -> Any:
        if self._wrapper._budget:
            self._wrapper._budget.enforce(self._wrapper._tracker)

        response = self._wrapper._client.chat.completions.create(**kwargs)

        if hasattr(response, "usage") and response.usage is not None:
            model = getattr(response, "model", kwargs.get("model", "unknown"))
            self._wrapper._tracker.record(
                model=model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

        return response


class TrackedAnthropic:
    """Wraps an ``anthropic.Anthropic`` client to automatically track costs.

    Usage::

        from anthropic import Anthropic
        from llm_cost_guardian import CostTracker, TrackedAnthropic

        tracker = CostTracker()
        client = TrackedAnthropic(Anthropic(), tracker)
        response = client.messages.create(model="claude-sonnet-4-20250514", ...)
        print(tracker.total_cost)
    """

    def __init__(
        self,
        client: Any,
        tracker: CostTracker,
        budget: BudgetManager | None = None,
    ) -> None:
        self._client = client
        self._tracker = tracker
        self._budget = budget
        self.messages = _AnthropicMessages(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _AnthropicMessages:
    def __init__(self, wrapper: TrackedAnthropic) -> None:
        self._wrapper = wrapper

    def create(self, **kwargs: Any) -> Any:
        if self._wrapper._budget:
            self._wrapper._budget.enforce(self._wrapper._tracker)

        response = self._wrapper._client.messages.create(**kwargs)

        if hasattr(response, "usage") and response.usage is not None:
            model = getattr(response, "model", kwargs.get("model", "unknown"))
            self._wrapper._tracker.record(
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        return response

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
            usage = response.usage
            # OpenAI reports cache hits in prompt_tokens_details.cached_tokens
            # and includes them inside prompt_tokens, so split them out.
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            input_tokens = max(usage.prompt_tokens - cached, 0)
            self._wrapper._tracker.record(
                model=model,
                input_tokens=input_tokens,
                output_tokens=usage.completion_tokens,
                cache_read_tokens=cached,
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
            usage = response.usage
            # Anthropic reports cache activity separately from input_tokens:
            # cache_read_input_tokens (hits) and cache_creation_input_tokens
            # (writes billed at a premium).
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
            self._wrapper._tracker.record(
                model=model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            )

        return response

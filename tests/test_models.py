"""Tests for model pricing data and lookups."""

import pytest

from llm_cost_guardian import ModelPricing, Provider, get_pricing, list_models


class TestModelPricing:
    def test_calculate_cost(self):
        model = ModelPricing("test", Provider.OPENAI, 2.50, 10.00)
        # 1M input tokens = $2.50, 1M output tokens = $10.00
        cost = model.calculate_cost(1_000_000, 1_000_000)
        assert cost == pytest.approx(12.50)

    def test_calculate_cost_zero_tokens(self):
        model = ModelPricing("test", Provider.OPENAI, 2.50, 10.00)
        assert model.calculate_cost(0, 0) == 0.0

    def test_input_cost_per_token(self):
        model = ModelPricing("test", Provider.OPENAI, 2.50, 10.00)
        assert model.input_cost_per_token == pytest.approx(2.50 / 1_000_000)

    def test_output_cost_per_token(self):
        model = ModelPricing("test", Provider.OPENAI, 2.50, 10.00)
        assert model.output_cost_per_token == pytest.approx(10.00 / 1_000_000)

    def test_context_window_optional(self):
        model = ModelPricing("test", Provider.OPENAI, 1.0, 2.0)
        assert model.context_window is None

    def test_context_window_set(self):
        model = ModelPricing("test", Provider.OPENAI, 1.0, 2.0, context_window=128_000)
        assert model.context_window == 128_000

    def test_frozen(self):
        model = ModelPricing("test", Provider.OPENAI, 1.0, 2.0)
        with pytest.raises(AttributeError):
            model.name = "changed"


class TestGetPricing:
    def test_exact_match(self):
        pricing = get_pricing("gpt-4o")
        assert pricing.name == "gpt-4o"
        assert pricing.provider == Provider.OPENAI

    def test_prefix_match(self):
        """Versioned model names should match via prefix."""
        pricing = get_pricing("gpt-4o-2024-08-06")
        assert pricing.name == "gpt-4o"

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_pricing("totally-fake-model")

    def test_all_providers_have_models(self):
        for provider in Provider:
            models = list_models(provider)
            assert len(models) > 0, f"No models for {provider.value}"


class TestListModels:
    def test_list_all(self):
        models = list_models()
        assert len(models) >= 22  # At least the models we know about

    def test_list_filtered(self):
        openai_models = list_models(Provider.OPENAI)
        for m in openai_models:
            assert m.provider == Provider.OPENAI

    def test_list_sorted_by_name(self):
        models = list_models()
        names = [m.name for m in models]
        assert names == sorted(names)

    def test_anthropic_models(self):
        models = list_models(Provider.ANTHROPIC)
        names = [m.name for m in models]
        assert "claude-sonnet-4-20250514" in names
        assert "claude-opus-4-20250514" in names

    def test_google_models(self):
        models = list_models(Provider.GOOGLE)
        names = [m.name for m in models]
        assert "gemini-2.0-flash" in names

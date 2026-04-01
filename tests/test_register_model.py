"""Tests for register_model() public API."""

import pytest

from llm_cost_guardian import CostTracker, get_pricing, list_models, register_model
from llm_cost_guardian.models import PRICING, Provider


class TestRegisterModel:
    """Test custom model registration."""

    def setup_method(self) -> None:
        # Remove any test models from previous runs
        for key in list(PRICING.keys()):
            if key.startswith("test-"):
                del PRICING[key]

    def test_register_basic(self) -> None:
        result = register_model("test-custom-v1", "openai", 5.0, 15.0)
        assert result.name == "test-custom-v1"
        assert result.provider == Provider.OPENAI
        assert result.input_cost_per_1m == 5.0
        assert result.output_cost_per_1m == 15.0

    def test_register_with_context_window(self) -> None:
        result = register_model(
            "test-custom-v2", "anthropic", 3.0, 12.0, context_window=32_000
        )
        assert result.context_window == 32_000

    def test_registered_model_in_get_pricing(self) -> None:
        register_model("test-lookup", "openai", 1.0, 2.0)
        pricing = get_pricing("test-lookup")
        assert pricing.name == "test-lookup"
        assert pricing.input_cost_per_1m == 1.0

    def test_registered_model_in_list(self) -> None:
        register_model("test-listed", "google", 0.5, 1.5)
        models = list_models()
        names = [m.name for m in models]
        assert "test-listed" in names

    def test_registered_model_in_list_filtered(self) -> None:
        register_model("test-google-model", "google", 0.1, 0.3)
        models = list_models(Provider.GOOGLE)
        names = [m.name for m in models]
        assert "test-google-model" in names

    def test_register_with_provider_enum(self) -> None:
        result = register_model("test-enum", Provider.ANTHROPIC, 2.0, 8.0)
        assert result.provider == Provider.ANTHROPIC

    def test_register_overwrites_existing(self) -> None:
        register_model("test-overwrite", "openai", 1.0, 2.0)
        register_model("test-overwrite", "openai", 5.0, 10.0)
        pricing = get_pricing("test-overwrite")
        assert pricing.input_cost_per_1m == 5.0

    def test_register_negative_cost_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            register_model("test-neg", "openai", -1.0, 2.0)

    def test_register_negative_output_cost_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            register_model("test-neg-out", "openai", 1.0, -2.0)

    def test_register_zero_cost_allowed(self) -> None:
        result = register_model("test-free", "openai", 0.0, 0.0)
        assert result.input_cost_per_1m == 0.0

    def test_register_and_track(self) -> None:
        """End-to-end: register custom model and track costs."""
        register_model("test-e2e", "openai", 10.0, 30.0)
        tracker = CostTracker()
        tracker.record("test-e2e", input_tokens=1000, output_tokens=500)
        expected = 1000 * 10.0 / 1_000_000 + 500 * 30.0 / 1_000_000
        assert abs(tracker.total_cost - expected) < 1e-10

    def test_register_invalid_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            register_model("test-bad-prov", "invalid_provider", 1.0, 2.0)

    def teardown_method(self) -> None:
        for key in list(PRICING.keys()):
            if key.startswith("test-"):
                del PRICING[key]

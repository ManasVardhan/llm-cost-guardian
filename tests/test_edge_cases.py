"""Edge case and integration tests for llm-cost-guardian."""

import json
import os
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest

from llm_cost_guardian import (
    Action,
    BudgetError,
    BudgetManager,
    BudgetResult,
    CostTracker,
    HardCapPolicy,
    ModelPricing,
    Provider,
    SlidingWindowPolicy,
    SoftWarningPolicy,
    TrackedAnthropic,
    TrackedOpenAI,
    UsageRecord,
    get_pricing,
    list_models,
    save_csv,
    save_json,
    to_csv,
    to_json,
    to_prometheus,
)

# ---------------------------------------------------------------------------
# UsageRecord edge cases
# ---------------------------------------------------------------------------


class TestUsageRecordEdgeCases:
    def test_total_tokens(self):
        rec = UsageRecord(model="gpt-4o", input_tokens=100, output_tokens=50, cost=0.01)
        assert rec.total_tokens == 150

    def test_zero_tokens(self):
        rec = UsageRecord(model="gpt-4o", input_tokens=0, output_tokens=0, cost=0.0)
        assert rec.total_tokens == 0
        assert rec.cost == 0.0

    def test_metadata_default_empty(self):
        rec = UsageRecord(model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.001)
        assert rec.metadata == {}

    def test_metadata_stored(self):
        rec = UsageRecord(
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
            cost=0.001,
            metadata={"session": "abc", "user": "test"},
        )
        assert rec.metadata["session"] == "abc"
        assert rec.metadata["user"] == "test"

    def test_timestamp_auto_set(self):
        before = time.time()
        rec = UsageRecord(model="gpt-4o", input_tokens=10, output_tokens=5, cost=0.001)
        after = time.time()
        assert before <= rec.timestamp <= after

    def test_large_token_counts(self):
        rec = UsageRecord(model="gpt-4o", input_tokens=1_000_000, output_tokens=500_000, cost=7.5)
        assert rec.total_tokens == 1_500_000


# ---------------------------------------------------------------------------
# CostTracker edge cases
# ---------------------------------------------------------------------------


class TestCostTrackerEdgeCases:
    def test_negative_input_tokens_raises(self):
        tracker = CostTracker()
        with pytest.raises(ValueError, match="non-negative"):
            tracker.record("gpt-4o", input_tokens=-1, output_tokens=0)

    def test_negative_output_tokens_raises(self):
        tracker = CostTracker()
        with pytest.raises(ValueError, match="non-negative"):
            tracker.record("gpt-4o", input_tokens=0, output_tokens=-5)

    def test_zero_tokens_allowed(self):
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", input_tokens=0, output_tokens=0)
        assert rec.cost == 0.0
        assert tracker.total_cost == 0.0

    def test_metadata_passed_through(self):
        tracker = CostTracker()
        rec = tracker.record("gpt-4o", 100, 50, metadata={"run_id": "r1"})
        assert rec.metadata["run_id"] == "r1"

    def test_total_tokens_property(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 1000, 500)
        tracker.record("gpt-4o", 2000, 1000)
        assert tracker.total_tokens == 4500

    def test_records_returns_copy(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50)
        records = tracker.records
        records.clear()
        assert len(tracker.records) == 1  # original not affected

    def test_unknown_model_raises(self):
        tracker = CostTracker()
        with pytest.raises(KeyError, match="Unknown model"):
            tracker.record("nonexistent-model-xyz", 100, 50)

    def test_explicit_cost_skips_pricing(self):
        tracker = CostTracker()
        # Even with unknown model, explicit cost works
        rec = tracker.record("totally-fake-model", 100, 50, cost=0.99)
        assert rec.cost == 0.99

    def test_callback_receives_cumulative_cost(self):
        cumulative_costs = []
        tracker = CostTracker(on_record=lambda r, c: cumulative_costs.append(c))
        tracker.record("gpt-4o", 1000, 500)
        tracker.record("gpt-4o", 1000, 500)
        assert len(cumulative_costs) == 2
        assert cumulative_costs[1] == pytest.approx(cumulative_costs[0] * 2)

    def test_concurrent_recording(self):
        """Verify thread-safety of the tracker."""
        tracker = CostTracker()
        n_threads = 10
        n_records = 50
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            for _ in range(n_records):
                tracker.record("gpt-4o", 100, 50)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(tracker.records) == n_threads * n_records
        assert tracker.total_input_tokens == n_threads * n_records * 100
        assert tracker.total_output_tokens == n_threads * n_records * 50

    def test_summary_after_reset(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 1000, 500)
        tracker.reset()
        s = tracker.summary()
        assert s["total_cost_usd"] == 0
        assert s["total_requests"] == 0
        assert s["cost_by_model"] == {}

    def test_cost_by_model_multiple_models(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 1000, 500)
        tracker.record("gpt-4o", 2000, 1000)
        tracker.record("o3-mini", 500, 200)
        by_model = tracker.cost_by_model()
        assert len(by_model) == 2
        assert "gpt-4o" in by_model
        assert "o3-mini" in by_model


# ---------------------------------------------------------------------------
# ModelPricing edge cases
# ---------------------------------------------------------------------------


class TestModelPricingEdgeCases:
    def test_zero_cost_model(self):
        m = ModelPricing("free-model", Provider.OPENAI, 0.0, 0.0)
        assert m.calculate_cost(1000, 500) == 0.0

    def test_very_small_tokens(self):
        m = get_pricing("gpt-4o")
        cost = m.calculate_cost(1, 1)
        assert cost > 0
        assert cost < 0.001

    def test_context_window_none_by_default(self):
        m = ModelPricing("test", Provider.OPENAI, 1.0, 2.0)
        assert m.context_window is None

    def test_prefix_match_versioned_model(self):
        pricing = get_pricing("gpt-4o-2024-08-06")
        assert pricing.name == "gpt-4o"

    def test_list_models_all(self):
        models = list_models()
        assert len(models) > 10
        # Verify sorted
        names = [m.name for m in models]
        assert names == sorted(names)

    def test_list_models_openai_only(self):
        models = list_models(Provider.OPENAI)
        assert all(m.provider == Provider.OPENAI for m in models)
        assert len(models) > 5

    def test_list_models_anthropic_only(self):
        models = list_models(Provider.ANTHROPIC)
        assert all(m.provider == Provider.ANTHROPIC for m in models)
        assert len(models) >= 4

    def test_list_models_google_only(self):
        models = list_models(Provider.GOOGLE)
        assert all(m.provider == Provider.GOOGLE for m in models)
        assert len(models) >= 3

    def test_pricing_frozen(self):
        m = get_pricing("gpt-4o")
        with pytest.raises(AttributeError):
            m.name = "something-else"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Budget policy edge cases
# ---------------------------------------------------------------------------


class TestBudgetEdgeCases:
    def test_hard_cap_at_exact_limit(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=5.00)
        policy = HardCapPolicy(limit_usd=5.00)
        result = policy.evaluate(tracker)
        assert result.action == Action.BLOCK

    def test_hard_cap_just_under(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=4.9999)
        policy = HardCapPolicy(limit_usd=5.00)
        result = policy.evaluate(tracker)
        assert result.action == Action.ALLOW

    def test_soft_warning_at_exact_limit(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=1.00)
        policy = SoftWarningPolicy(warning_usd=1.00)
        result = policy.evaluate(tracker)
        assert result.action == Action.WARN

    def test_sliding_window_custom_action(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=2.00)
        policy = SlidingWindowPolicy(
            limit_usd=1.00,
            window_seconds=3600,
            action_on_exceed=Action.WARN,
        )
        result = policy.evaluate(tracker)
        assert result.action == Action.WARN

    def test_sliding_window_old_records_excluded(self):
        tracker = CostTracker()
        # Record with a timestamp 2 hours ago
        old_rec = UsageRecord(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost=10.00,
            timestamp=time.time() - 7200,
        )
        with tracker._lock:
            tracker._records.append(old_rec)
            tracker._total_cost += old_rec.cost

        policy = SlidingWindowPolicy(limit_usd=5.00, window_seconds=3600)
        result = policy.evaluate(tracker)
        assert result.action == Action.ALLOW  # old record outside window

    def test_budget_manager_empty_policies(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50)
        bm = BudgetManager()
        result = bm.check(tracker)
        assert result.action == Action.ALLOW

    def test_budget_manager_chain_add(self):
        bm = BudgetManager()
        result = bm.add(HardCapPolicy(5.0)).add(SoftWarningPolicy(1.0))
        assert result is bm
        assert len(bm.policies) == 2

    def test_budget_error_has_result_and_message(self):
        result = BudgetResult(Action.BLOCK, "Over limit!", 10.0, 5.0)
        err = BudgetError(result)
        assert err.result.action == Action.BLOCK
        assert "Over limit!" in str(err)

    def test_enforce_calls_on_warn_callback(self):
        warnings = []
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=2.00)
        bm = BudgetManager(
            policies=[SoftWarningPolicy(warning_usd=1.00)],
            on_warn=lambda r: warnings.append(r),
        )
        bm.enforce(tracker)
        assert len(warnings) == 1
        assert warnings[0].action == Action.WARN

    def test_enforce_block_raises(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=10.00)
        bm = BudgetManager(policies=[HardCapPolicy(limit_usd=1.00)])
        with pytest.raises(BudgetError):
            bm.enforce(tracker)

    def test_block_takes_priority_over_warn(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, cost=10.00)
        bm = BudgetManager()
        bm.add(SoftWarningPolicy(warning_usd=1.00))
        bm.add(HardCapPolicy(limit_usd=5.00))
        result = bm.check(tracker)
        assert result.action == Action.BLOCK


# ---------------------------------------------------------------------------
# Exporter edge cases
# ---------------------------------------------------------------------------


class TestExporterEdgeCases:
    def test_json_empty_tracker(self):
        tracker = CostTracker()
        data = json.loads(to_json(tracker))
        assert data["records"] == []
        assert data["summary"]["total_requests"] == 0

    def test_csv_empty_tracker(self):
        tracker = CostTracker()
        output = to_csv(tracker)
        lines = output.strip().split("\n")
        assert len(lines) == 1  # header only

    def test_prometheus_empty_tracker(self):
        tracker = CostTracker()
        output = to_prometheus(tracker)
        assert "total_cost_usd 0.00000000" in output
        assert "total_requests 0" in output

    def test_prometheus_custom_prefix(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50)
        output = to_prometheus(tracker, prefix="myapp")
        assert "myapp_total_cost_usd" in output
        assert "myapp_total_requests" in output

    def test_save_json_creates_file(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 1000, 500)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_json(tracker, path)
            with open(path) as f:
                data = json.load(f)
            assert len(data["records"]) == 1
            assert data["summary"]["total_requests"] == 1
        finally:
            os.unlink(path)

    def test_save_csv_creates_file(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 1000, 500)
        tracker.record("o3-mini", 2000, 1000)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save_csv(tracker, path)
            with open(path) as f:
                lines = f.read().strip().split("\n")
            assert len(lines) == 3  # header + 2 records
        finally:
            os.unlink(path)

    def test_json_preserves_metadata(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50, metadata={"session": "s1"})
        data = json.loads(to_json(tracker))
        assert data["records"][0]["metadata"]["session"] == "s1"

    def test_json_custom_indent(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 100, 50)
        compact = to_json(tracker, indent=None)
        assert "\n" not in compact.strip() or len(compact) < len(to_json(tracker, indent=4))


# ---------------------------------------------------------------------------
# Wrapper edge cases
# ---------------------------------------------------------------------------


class TestWrapperEdgeCases:
    def test_openai_no_usage_in_response(self):
        """When response has no usage field, nothing is tracked."""
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(
                        model="gpt-4o", usage=None, choices=[]
                    )
                )
            )
        )
        tracker = CostTracker()
        client = TrackedOpenAI(mock_client, tracker)
        client.chat.completions.create(model="gpt-4o", messages=[])
        assert tracker.total_cost == 0.0
        assert len(tracker.records) == 0

    def test_openai_getattr_passthrough(self):
        """Attributes not overridden pass through to the real client."""
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: None)
            ),
            models=SimpleNamespace(list=lambda: ["gpt-4o"]),
        )
        tracker = CostTracker()
        client = TrackedOpenAI(mock_client, tracker)
        assert client.models.list() == ["gpt-4o"]

    def test_anthropic_no_usage_in_response(self):
        mock_client = SimpleNamespace(
            messages=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    model="claude-3-haiku-20240307", usage=None, content=[]
                )
            )
        )
        tracker = CostTracker()
        client = TrackedAnthropic(mock_client, tracker)
        client.messages.create(model="claude-3-haiku-20240307", messages=[], max_tokens=100)
        assert tracker.total_cost == 0.0

    def test_anthropic_getattr_passthrough(self):
        mock_client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kw: None),
            count_tokens=lambda text: len(text.split()),
        )
        tracker = CostTracker()
        client = TrackedAnthropic(mock_client, tracker)
        assert client.count_tokens("hello world test") == 3

    def test_openai_with_budget_allows(self):
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(
                        model="gpt-4o",
                        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                    )
                )
            )
        )
        tracker = CostTracker()
        budget = BudgetManager(policies=[HardCapPolicy(limit_usd=100.0)])
        client = TrackedOpenAI(mock_client, tracker, budget)
        client.chat.completions.create(model="gpt-4o", messages=[])
        assert tracker.total_cost > 0

    def test_anthropic_with_budget_blocks(self):
        mock_client = SimpleNamespace(
            messages=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    model="claude-3-haiku-20240307",
                    usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                )
            )
        )
        tracker = CostTracker()
        tracker.record("gpt-4o", 100000, 100000, cost=100.0)
        budget = BudgetManager(policies=[HardCapPolicy(limit_usd=0.001)])
        client = TrackedAnthropic(mock_client, tracker, budget)
        with pytest.raises(BudgetError):
            client.messages.create(model="claude-3-haiku-20240307", messages=[], max_tokens=10)

    def test_openai_model_from_response(self):
        """Model name comes from response, not kwargs."""
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(
                        model="gpt-4o-2024-08-06",
                        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
                    )
                )
            )
        )
        tracker = CostTracker()
        client = TrackedOpenAI(mock_client, tracker)
        client.chat.completions.create(model="gpt-4o", messages=[])
        assert tracker.records[0].model == "gpt-4o-2024-08-06"


# ---------------------------------------------------------------------------
# Integration: tracker + budget + exporter
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_workflow(self):
        """Track calls, enforce budget, export results."""
        tracker = CostTracker()
        warnings = []
        budget = BudgetManager(
            policies=[
                SoftWarningPolicy(warning_usd=0.001),
                HardCapPolicy(limit_usd=1.00),
            ],
            on_warn=lambda r: warnings.append(r),
        )

        tracker.record("gpt-4o", 1000, 500)
        budget.enforce(tracker)

        # Should have triggered a warning
        assert len(warnings) >= 1

        # Export to JSON
        data = json.loads(to_json(tracker))
        assert data["summary"]["total_requests"] == 1

        # Export to CSV
        csv_output = to_csv(tracker)
        assert "gpt-4o" in csv_output

        # Export to Prometheus
        prom = to_prometheus(tracker)
        assert "total_cost_usd" in prom

    def test_multi_model_report(self):
        tracker = CostTracker()
        models_used = ["gpt-4o", "gpt-4o-mini", "o3-mini", "claude-3-5-haiku-20241022"]
        for model in models_used:
            tracker.record(model, 500, 250)

        summary = tracker.summary()
        assert summary["total_requests"] == 4
        assert len(summary["cost_by_model"]) == 4

        # Verify total is sum of parts
        total_from_parts = sum(summary["cost_by_model"].values())
        assert summary["total_cost_usd"] == pytest.approx(round(total_from_parts, 6))

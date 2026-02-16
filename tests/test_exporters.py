"""Tests for exporters."""

import json

from llm_cost_guardian import CostTracker, to_csv, to_json, to_prometheus


class TestExporters:
    def setup_method(self):
        self.tracker = CostTracker()
        self.tracker.record("gpt-4o", 1000, 500)
        self.tracker.record("claude-3-5-haiku-20241022", 2000, 1000)

    def test_json_export(self):
        data = json.loads(to_json(self.tracker))
        assert len(data["records"]) == 2
        assert "summary" in data

    def test_csv_export(self):
        output = to_csv(self.tracker)
        lines = output.strip().split("\n")
        assert len(lines) == 3  # header + 2 records

    def test_prometheus_export(self):
        output = to_prometheus(self.tracker)
        assert "llm_cost_guardian_total_cost_usd" in output
        assert 'model="gpt-4o"' in output

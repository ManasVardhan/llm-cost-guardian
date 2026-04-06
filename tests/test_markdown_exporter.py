"""Tests for the markdown exporter."""

from __future__ import annotations

import os
import tempfile

import pytest

from llm_cost_guardian import CostTracker, save_markdown, to_markdown


@pytest.fixture
def populated_tracker():
    tracker = CostTracker()
    tracker.record("gpt-4o", input_tokens=1000, output_tokens=500, cost=0.0125)
    tracker.record("gpt-4o-mini", input_tokens=2000, output_tokens=1500, cost=0.00075)
    tracker.record("claude-3-5-sonnet", input_tokens=500, output_tokens=300, cost=0.006)
    tracker.record("gpt-4o", input_tokens=200, output_tokens=100, cost=0.0025)
    return tracker


def test_to_markdown_includes_title(populated_tracker):
    md = to_markdown(populated_tracker)
    assert md.startswith("# LLM Cost Report")


def test_to_markdown_custom_title(populated_tracker):
    md = to_markdown(populated_tracker, title="Sprint 17 LLM Spend")
    assert md.startswith("# Sprint 17 LLM Spend")


def test_to_markdown_summary_table(populated_tracker):
    md = to_markdown(populated_tracker)
    assert "## Summary" in md
    assert "| Total cost (USD)" in md
    assert "| Total requests" in md
    assert "| Avg cost per request" in md
    # Total cost = 0.0125 + 0.00075 + 0.006 + 0.0025 = 0.02175
    assert "$0.021750" in md


def test_to_markdown_cost_by_model_section(populated_tracker):
    md = to_markdown(populated_tracker)
    assert "## Cost by model" in md
    assert "`gpt-4o`" in md
    assert "`gpt-4o-mini`" in md
    assert "`claude-3-5-sonnet`" in md
    # Most expensive (gpt-4o = 0.015) should appear before claude-3-5-sonnet (0.006)
    gpt_idx = md.find("`gpt-4o` ")
    claude_idx = md.find("`claude-3-5-sonnet`")
    assert gpt_idx >= 0 and claude_idx >= 0
    assert gpt_idx < claude_idx


def test_to_markdown_cost_by_model_share_percentages(populated_tracker):
    md = to_markdown(populated_tracker)
    # Shares should sum visibly to ~100% across the displayed models.
    assert "%" in md


def test_to_markdown_recent_calls_section(populated_tracker):
    md = to_markdown(populated_tracker)
    assert "## Recent calls" in md
    # Most recent first (last record was gpt-4o 200/100)
    recent_idx = md.find("## Recent calls")
    section = md[recent_idx:]
    rows = [line for line in section.splitlines() if line.startswith("| ")]
    # header row + separator + at least 1 data row
    assert len(rows) >= 3


def test_to_markdown_empty_tracker():
    tracker = CostTracker()
    md = to_markdown(tracker)
    assert "## Summary" in md
    assert "## Cost by model" not in md
    assert "## Recent calls" not in md
    # Avg cost row not shown for zero requests
    assert "Avg cost per request" not in md


def test_to_markdown_recent_calls_caps_at_10():
    tracker = CostTracker()
    for _ in range(25):
        tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50, cost=0.0001)
    md = to_markdown(tracker)
    recent_idx = md.find("## Recent calls")
    section = md[recent_idx:]
    data_rows = [
        line
        for line in section.splitlines()
        if line.startswith("| ") and "---" not in line and "Model" not in line
    ]
    assert len(data_rows) == 10


def test_save_markdown_writes_file(populated_tracker):
    fd, path = tempfile.mkstemp(suffix=".md")
    os.close(fd)
    try:
        save_markdown(populated_tracker, path, title="Test Report")
        with open(path) as f:
            content = f.read()
        assert content.startswith("# Test Report")
        assert "## Summary" in content
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_to_markdown_handles_zero_cost_records():
    tracker = CostTracker()
    tracker.record("free-model", input_tokens=10, output_tokens=10, cost=0.0)
    md = to_markdown(tracker)
    assert "$0.000000" in md
    # Should not divide by zero in share calculation
    assert "## Cost by model" in md

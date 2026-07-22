"""Tests for Slack / Discord webhook cost alerts."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest
from click.testing import CliRunner

from llm_cost_guardian import (
    AlertEvent,
    AlertRule,
    CostAlerter,
    CostTracker,
    DiscordWebhook,
    SlackWebhook,
)
from llm_cost_guardian.cli import cli


class FakeWebhook:
    """Records sent events without touching the network."""

    def __init__(self) -> None:
        self.events: list[AlertEvent] = []
        self.last_error: str | None = None

    def send(self, event: AlertEvent) -> bool:
        self.events.append(event)
        return True


class ExplodingWebhook:
    """A badly behaved custom webhook that raises from send()."""

    last_error = None

    def send(self, event: AlertEvent) -> bool:
        raise RuntimeError("boom")


class TestAlertRule:
    def test_zero_threshold_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            AlertRule(0)

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            AlertRule(-5.0)

    def test_non_numeric_threshold_rejected(self):
        with pytest.raises(TypeError):
            AlertRule("10")  # type: ignore[arg-type]

    def test_bool_threshold_rejected(self):
        with pytest.raises(TypeError):
            AlertRule(True)  # type: ignore[arg-type]

    def test_scope_description_total(self):
        assert AlertRule(1.0).scope_description() == "total"

    def test_scope_description_combined(self):
        rule = AlertRule(1.0, model="gpt-4o", tag="prod", user="alice")
        assert rule.scope_description() == "model=gpt-4o, tag=prod, user=alice"

    def test_current_cost_total(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 1, 1, cost=2.0)
        tracker.record("gpt-4o-mini", 1, 1, cost=3.0)
        assert AlertRule(1.0).current_cost(tracker) == pytest.approx(5.0)

    def test_current_cost_scoped_to_model(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 1, 1, cost=2.0)
        tracker.record("gpt-4o-mini", 1, 1, cost=3.0)
        rule = AlertRule(1.0, model="gpt-4o")
        assert rule.current_cost(tracker) == pytest.approx(2.0)

    def test_current_cost_scoped_to_tag_and_user(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 1, 1, cost=2.0, tags=["prod"], user="alice")
        tracker.record("gpt-4o", 1, 1, cost=4.0, tags=["prod"], user="bob")
        tracker.record("gpt-4o", 1, 1, cost=8.0, tags=["dev"], user="alice")
        rule = AlertRule(1.0, tag="prod", user="alice")
        assert rule.current_cost(tracker) == pytest.approx(2.0)


class TestAlertEvent:
    def test_message_uses_label(self):
        event = AlertEvent(AlertRule(10.0, label="daily-budget"), 12.5)
        assert "daily-budget" in event.message
        assert "$12.5000" in event.message
        assert "$10.00" in event.message

    def test_message_without_label_uses_scope(self):
        event = AlertEvent(AlertRule(10.0, model="gpt-4o"), 11.0)
        assert "[model=gpt-4o]" in event.message


class TestWebhookValidation:
    def test_rejects_non_http_url(self):
        with pytest.raises(ValueError, match="http"):
            SlackWebhook("ftp://example.com/hook")

    def test_rejects_empty_url(self):
        with pytest.raises(ValueError):
            DiscordWebhook("")

    def test_accepts_https(self):
        assert SlackWebhook("https://hooks.slack.com/services/X").url


class TestPayloadFormats:
    def test_slack_payload(self):
        hook = SlackWebhook("https://hooks.slack.com/services/X")
        event = AlertEvent(AlertRule(10.0, label="budget"), 12.0)
        payload = hook.format_payload(event)
        assert set(payload) == {"text"}
        assert ":rotating_light:" in payload["text"]
        assert "budget" in payload["text"]

    def test_discord_payload_truncated(self):
        hook = DiscordWebhook("https://discord.com/api/webhooks/X")
        event = AlertEvent(AlertRule(10.0, label="x" * 3000), 12.0)
        payload = hook.format_payload(event)
        assert set(payload) == {"content"}
        assert len(payload["content"]) <= 2000


class _FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestWebhookSend:
    def test_send_success_posts_json(self, monkeypatch):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode())
            captured["content_type"] = request.get_header("Content-type")
            captured["timeout"] = timeout
            return _FakeResponse(b"ok")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        hook = SlackWebhook("https://hooks.slack.com/services/X", timeout=3.0)
        event = AlertEvent(AlertRule(1.0), 2.0)
        assert hook.send(event) is True
        assert hook.last_error is None
        assert captured["url"] == "https://hooks.slack.com/services/X"
        assert captured["content_type"] == "application/json"
        assert captured["timeout"] == 3.0
        assert ":rotating_light:" in captured["body"]["text"]

    def test_send_http_error_returns_false(self, monkeypatch):
        def fake_urlopen(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        hook = DiscordWebhook("https://discord.com/api/webhooks/X")
        assert hook.send(AlertEvent(AlertRule(1.0), 2.0)) is False
        assert "404" in hook.last_error

    def test_send_network_error_returns_false(self, monkeypatch):
        def fake_urlopen(request, timeout):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        hook = SlackWebhook("https://hooks.slack.com/services/X")
        assert hook.send(AlertEvent(AlertRule(1.0), 2.0)) is False
        assert "connection refused" in hook.last_error

    def test_send_non_2xx_status_returns_false(self, monkeypatch):
        response = _FakeResponse(b"rate limited")
        response.status = 429
        monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: response)
        hook = SlackWebhook("https://hooks.slack.com/services/X")
        assert hook.send(AlertEvent(AlertRule(1.0), 2.0)) is False
        assert hook.last_error == "HTTP 429"


class TestCostAlerter:
    def test_fires_when_threshold_crossed(self):
        hook = FakeWebhook()
        alerter = CostAlerter([hook])
        alerter.add_rule(1.0, label="budget")
        tracker = CostTracker()
        tracker.record("gpt-4o", 1, 1, cost=2.0)
        events = alerter.check(tracker)
        assert len(events) == 1
        assert events[0].current_cost == pytest.approx(2.0)
        assert hook.events == events

    def test_does_not_fire_under_threshold(self):
        alerter = CostAlerter([FakeWebhook()])
        alerter.add_rule(10.0)
        tracker = CostTracker()
        tracker.record("gpt-4o", 1, 1, cost=2.0)
        assert alerter.check(tracker) == []

    def test_fires_at_most_once(self):
        hook = FakeWebhook()
        alerter = CostAlerter([hook])
        alerter.add_rule(1.0)
        tracker = CostTracker()
        tracker.record("gpt-4o", 1, 1, cost=2.0)
        assert len(alerter.check(tracker)) == 1
        tracker.record("gpt-4o", 1, 1, cost=2.0)
        assert alerter.check(tracker) == []
        assert len(hook.events) == 1

    def test_reset_rearms_rules(self):
        alerter = CostAlerter([FakeWebhook()])
        alerter.add_rule(1.0)
        tracker = CostTracker()
        tracker.record("gpt-4o", 1, 1, cost=2.0)
        alerter.check(tracker)
        alerter.reset()
        assert len(alerter.check(tracker)) == 1

    def test_scoped_rule_ignores_other_models(self):
        alerter = CostAlerter([FakeWebhook()])
        alerter.add_rule(3.0, model="gpt-4o")
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", 1, 1, cost=100.0)
        assert alerter.check(tracker) == []
        tracker.record("gpt-4o", 1, 1, cost=3.5)
        assert len(alerter.check(tracker)) == 1

    def test_multiple_rules_fire_independently(self):
        alerter = CostAlerter([FakeWebhook()])
        alerter.add_rule(1.0, label="low")
        alerter.add_rule(100.0, label="high")
        tracker = CostTracker()
        tracker.record("gpt-4o", 1, 1, cost=2.0)
        events = alerter.check(tracker)
        assert [e.rule.label for e in events] == ["low"]
        assert [r.label for r in alerter.fired_rules] == ["low"]

    def test_exploding_webhook_is_swallowed(self):
        alerter = CostAlerter([ExplodingWebhook(), FakeWebhook()])
        alerter.add_rule(1.0)
        tracker = CostTracker()
        tracker.record("gpt-4o", 1, 1, cost=2.0)
        events = alerter.check(tracker)
        assert len(events) == 1

    def test_attach_fires_on_record(self):
        hook = FakeWebhook()
        alerter = CostAlerter([hook])
        alerter.add_rule(1.5)
        tracker = CostTracker()
        alerter.attach(tracker)
        tracker.record("gpt-4o", 1, 1, cost=1.0)
        assert hook.events == []
        tracker.record("gpt-4o", 1, 1, cost=1.0)
        assert len(hook.events) == 1

    def test_attach_chains_existing_callback(self):
        seen = []
        tracker = CostTracker(on_record=lambda rec, total: seen.append(total))
        alerter = CostAlerter([FakeWebhook()])
        alerter.add_rule(1.0)
        alerter.attach(tracker)
        tracker.record("gpt-4o", 1, 1, cost=2.0)
        assert seen == [pytest.approx(2.0)]

    def test_initial_rules_accepted(self):
        alerter = CostAlerter([FakeWebhook()], rules=[AlertRule(1.0)])
        tracker = CostTracker()
        tracker.record("gpt-4o", 1, 1, cost=2.0)
        assert len(alerter.check(tracker)) == 1


def _write_report(path, records):
    path.write_text(json.dumps({"summary": {}, "records": records}))
    return str(path)


SAMPLE_RECORDS = [
    {"model": "gpt-4o", "cost_usd": 3.0, "tags": ["prod"], "user": "alice"},
    {"model": "gpt-4o-mini", "cost_usd": 1.0, "tags": ["dev"], "user": "bob"},
    {"model": "gpt-4o", "cost_usd": 2.0, "tags": [], "user": None},
]


class TestAlertCommand:
    def test_under_threshold_exit_0(self, tmp_path):
        report = _write_report(tmp_path / "r.json", SAMPLE_RECORDS)
        result = CliRunner().invoke(cli, ["alert", report, "-t", "100"])
        assert result.exit_code == 0
        assert "OK: under threshold" in result.output

    def test_over_threshold_exit_2(self, tmp_path):
        report = _write_report(tmp_path / "r.json", SAMPLE_RECORDS)
        result = CliRunner().invoke(cli, ["alert", report, "-t", "5"])
        assert result.exit_code == 2
        assert "ALERT" in result.output
        assert "No webhooks configured" in result.output

    def test_scoped_by_tag(self, tmp_path):
        report = _write_report(tmp_path / "r.json", SAMPLE_RECORDS)
        result = CliRunner().invoke(cli, ["alert", report, "-t", "2", "--tag", "prod"])
        assert result.exit_code == 2
        assert "Matched calls: 1" in result.output

    def test_scoped_by_model_under(self, tmp_path):
        report = _write_report(tmp_path / "r.json", SAMPLE_RECORDS)
        result = CliRunner().invoke(cli, ["alert", report, "-t", "2", "--model", "gpt-4o-mini"])
        assert result.exit_code == 0

    def test_dry_run_prints_payloads(self, tmp_path):
        report = _write_report(tmp_path / "r.json", SAMPLE_RECORDS)
        result = CliRunner().invoke(
            cli,
            [
                "alert",
                report,
                "-t",
                "5",
                "--slack-webhook",
                "https://hooks.slack.com/services/X",
                "--discord-webhook",
                "https://discord.com/api/webhooks/X",
                "--dry-run",
            ],
        )
        assert result.exit_code == 2
        assert "[dry-run] SlackWebhook" in result.output
        assert "[dry-run] DiscordWebhook" in result.output
        assert ":rotating_light:" in result.output

    def test_json_output(self, tmp_path):
        report = _write_report(tmp_path / "r.json", SAMPLE_RECORDS)
        result = CliRunner().invoke(
            cli, ["alert", report, "-t", "5", "--json-output", "--label", "budget"]
        )
        assert result.exit_code == 2
        payload = json.loads(result.output)
        assert payload["triggered"] is True
        assert payload["matched_cost_usd"] == pytest.approx(6.0)
        assert payload["matched_calls"] == 3
        assert payload["scope"] == "total"

    def test_invalid_threshold_exit_1(self, tmp_path):
        report = _write_report(tmp_path / "r.json", SAMPLE_RECORDS)
        result = CliRunner().invoke(cli, ["alert", report, "-t", "0"])
        assert result.exit_code == 1
        assert "positive" in result.output

    def test_invalid_webhook_url_exit_1(self, tmp_path):
        report = _write_report(tmp_path / "r.json", SAMPLE_RECORDS)
        result = CliRunner().invoke(cli, ["alert", report, "-t", "5", "--slack-webhook", "notaurl"])
        assert result.exit_code == 1
        assert "http" in result.output

    def test_send_success_exit_2(self, tmp_path, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: _FakeResponse(b"ok"))
        report = _write_report(tmp_path / "r.json", SAMPLE_RECORDS)
        result = CliRunner().invoke(
            cli,
            ["alert", report, "-t", "5", "--slack-webhook", "https://hooks.slack.com/services/X"],
        )
        assert result.exit_code == 2
        assert "Sent alert via SlackWebhook" in result.output

    def test_send_failure_exit_1(self, tmp_path, monkeypatch):
        def fake_urlopen(request, timeout):
            raise urllib.error.URLError("no route to host")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        report = _write_report(tmp_path / "r.json", SAMPLE_RECORDS)
        result = CliRunner().invoke(
            cli,
            ["alert", report, "-t", "5", "--slack-webhook", "https://hooks.slack.com/services/X"],
        )
        assert result.exit_code == 1
        assert "Failed to send via SlackWebhook" in result.output

    def test_webhooks_not_called_under_threshold(self, tmp_path, monkeypatch):
        def fake_urlopen(request, timeout):  # pragma: no cover - must never run
            raise AssertionError("webhook should not be called under threshold")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        report = _write_report(tmp_path / "r.json", SAMPLE_RECORDS)
        result = CliRunner().invoke(
            cli,
            ["alert", report, "-t", "100", "--slack-webhook", "https://hooks.slack.com/services/X"],
        )
        assert result.exit_code == 0

    def test_empty_report_under_threshold(self, tmp_path):
        report = _write_report(tmp_path / "r.json", [])
        result = CliRunner().invoke(cli, ["alert", report, "-t", "1"])
        assert result.exit_code == 0
        assert "Matched calls: 0" in result.output

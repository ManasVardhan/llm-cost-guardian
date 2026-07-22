"""Slack and Discord webhook alerts for cost thresholds.

Send real-time notifications when tracked spend crosses configurable
thresholds. Thresholds can watch total cost or be scoped to a model, a tag,
a user, or any combination of the three. Uses only the standard library
(urllib), so no extra dependencies are required.

Example
-------
>>> from llm_cost_guardian import CostAlerter, CostTracker, SlackWebhook
>>> alerter = CostAlerter([SlackWebhook("https://hooks.slack.com/services/XXX")])
>>> alerter.add_rule(10.0, label="daily-budget")
>>> alerter.add_rule(5.0, model="gpt-4o", label="gpt4o-budget")
>>> tracker = CostTracker()
>>> alerter.attach(tracker)  # alerts fire automatically as calls are recorded
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tracker import CostTracker

_DISCORD_CONTENT_LIMIT = 2000


@dataclass(frozen=True)
class AlertRule:
    """A cost threshold to watch.

    Parameters
    ----------
    threshold_usd : fire when matched cost reaches this value (must be > 0)
    model : only count records for this model (exact match)
    tag : only count records carrying this tag (exact match)
    user : only count records attributed to this user (exact match)
    label : optional display name used in alert messages
    """

    threshold_usd: float
    model: str | None = None
    tag: str | None = None
    user: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.threshold_usd, (int, float)) or isinstance(self.threshold_usd, bool):
            raise TypeError(
                f"threshold_usd must be a number, got {type(self.threshold_usd).__name__}"
            )
        if self.threshold_usd <= 0:
            raise ValueError(f"threshold_usd must be positive, got {self.threshold_usd}")

    def scope_description(self) -> str:
        """Human-readable description of what this rule watches."""
        parts = []
        if self.model:
            parts.append(f"model={self.model}")
        if self.tag:
            parts.append(f"tag={self.tag}")
        if self.user:
            parts.append(f"user={self.user}")
        return ", ".join(parts) if parts else "total"

    def current_cost(self, tracker: CostTracker) -> float:
        """Return the cost currently matched by this rule's scope."""
        if self.model is None and self.tag is None and self.user is None:
            return tracker.total_cost
        matched = tracker.filter(model=self.model, tag=self.tag, user=self.user)
        return sum(r.cost for r in matched)


@dataclass(frozen=True)
class AlertEvent:
    """A fired alert: which rule crossed its threshold and at what cost."""

    rule: AlertRule
    current_cost: float

    @property
    def message(self) -> str:
        name = self.rule.label or self.rule.scope_description()
        return (
            f"LLM cost alert [{name}]: ${self.current_cost:.4f} spent on "
            f"{self.rule.scope_description()}, threshold ${self.rule.threshold_usd:.2f} crossed."
        )


class Webhook:
    """Base webhook sender. Subclasses define the JSON payload format."""

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        if not url or not url.startswith(("http://", "https://")):
            raise ValueError(f"Webhook URL must start with http:// or https://, got {url!r}")
        self.url = url
        self.timeout = timeout
        self.last_error: str | None = None

    def format_payload(self, event: AlertEvent) -> dict[str, object]:
        raise NotImplementedError

    def send(self, event: AlertEvent) -> bool:
        """POST the alert to the webhook URL. Returns True on success.

        Never raises: network and HTTP errors are captured in ``last_error``
        so a failing webhook can never break the calling application.
        """
        body = json.dumps(self.format_payload(event)).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = getattr(response, "status", 200)
                if 200 <= status < 300:
                    self.last_error = None
                    return True
                self.last_error = f"HTTP {status}"
                return False
        except urllib.error.HTTPError as e:
            self.last_error = f"HTTP {e.code}: {e.reason}"
            return False
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.last_error = str(e)
            return False


class SlackWebhook(Webhook):
    """Sends alerts to a Slack incoming webhook."""

    def format_payload(self, event: AlertEvent) -> dict[str, object]:
        return {"text": f":rotating_light: {event.message}"}


class DiscordWebhook(Webhook):
    """Sends alerts to a Discord webhook."""

    def format_payload(self, event: AlertEvent) -> dict[str, object]:
        content = f"\N{POLICE CARS REVOLVING LIGHT} {event.message}"
        return {"content": content[:_DISCORD_CONTENT_LIMIT]}


@dataclass
class CostAlerter:
    """Evaluates alert rules against a tracker and dispatches webhooks.

    Each rule fires at most once until :meth:`reset` is called, so repeated
    checks (for example one per recorded call) never spam a channel.

    Parameters
    ----------
    webhooks : webhook destinations to notify
    rules : initial alert rules (more can be added with :meth:`add_rule`)
    """

    webhooks: Sequence[Webhook] = ()
    rules: list[AlertRule] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._fired: set[AlertRule] = set()

    def add_rule(
        self,
        threshold_usd: float,
        *,
        model: str | None = None,
        tag: str | None = None,
        user: str | None = None,
        label: str | None = None,
    ) -> AlertRule:
        """Create, register, and return a new alert rule."""
        rule = AlertRule(threshold_usd, model=model, tag=tag, user=user, label=label)
        with self._lock:
            self.rules.append(rule)
        return rule

    def check(self, tracker: CostTracker) -> list[AlertEvent]:
        """Evaluate all rules and dispatch webhooks for newly crossed thresholds.

        Returns the list of alerts fired by this call. Webhook failures are
        swallowed (see ``Webhook.last_error``) so cost tracking never breaks.
        """
        fired: list[AlertEvent] = []
        with self._lock:
            pending = [r for r in self.rules if r not in self._fired]
        for rule in pending:
            cost = rule.current_cost(tracker)
            if cost >= rule.threshold_usd:
                with self._lock:
                    if rule in self._fired:
                        continue
                    self._fired.add(rule)
                event = AlertEvent(rule=rule, current_cost=cost)
                for webhook in self.webhooks:
                    try:
                        webhook.send(event)
                    except Exception:  # noqa: BLE001 - alerts must never break tracking
                        continue
                fired.append(event)
        return fired

    def reset(self) -> None:
        """Re-arm all rules so they can fire again."""
        with self._lock:
            self._fired.clear()

    @property
    def fired_rules(self) -> list[AlertRule]:
        """Rules that have already fired since the last reset."""
        with self._lock:
            return [r for r in self.rules if r in self._fired]

    def attach(self, tracker: CostTracker) -> None:
        """Automatically run :meth:`check` after every recorded call.

        Chains with any existing ``on_record`` callback rather than
        replacing it.
        """
        from .tracker import UsageRecord

        previous = tracker.on_record

        def _on_record(record: UsageRecord, cumulative: float) -> None:
            if previous is not None:
                previous(record, cumulative)
            self.check(tracker)

        tracker.on_record = _on_record

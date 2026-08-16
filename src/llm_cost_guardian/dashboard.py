"""Terminal dashboard for cost reports.

The heavy lifting lives in :func:`build_dashboard_data`, a pure function that
turns a JSON report (as produced by ``to_json`` / ``save_json``) into a plain
dict of display-ready numbers. Rendering with ``rich`` happens separately in
:func:`render_dashboard` so the data path stays dependency-free and testable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import RenderableType

UNKNOWN_DAY = "(unknown)"

BUDGET_WARN_PCT = 80.0


def _record_day(record: dict[str, Any], *, utc: bool) -> str:
    """Return the YYYY-MM-DD bucket for a record, or ``(unknown)``."""
    tz = timezone.utc if utc else None
    raw_ts = record.get("timestamp")
    try:
        ts = float(raw_ts)  # type: ignore[arg-type]
        if ts <= 0:
            return UNKNOWN_DAY
        return datetime.fromtimestamp(ts, tz=tz).date().isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return UNKNOWN_DAY


def build_dashboard_data(
    data: dict[str, Any],
    *,
    budget: float | None = None,
    utc: bool = False,
    top: int = 5,
    trend_days: int = 14,
) -> dict[str, Any]:
    """Compute all dashboard sections from a parsed JSON report.

    Parameters
    ----------
    data : parsed report dict with ``records`` (and optionally ``summary``)
    budget : optional budget in USD for the utilization gauge
    utc : bucket the daily trend by UTC dates instead of local time
    top : number of rows for the model, tag, user, and top-call sections
    trend_days : number of most recent days to keep in the daily trend

    Returns
    -------
    A JSON-safe dict with ``totals``, ``by_model``, ``by_day``, ``by_tag``,
    ``by_user``, ``top_calls``, and ``budget`` sections.
    """
    if top < 1:
        raise ValueError(f"top must be at least 1, got {top}")
    if trend_days < 1:
        raise ValueError(f"trend_days must be at least 1, got {trend_days}")

    records = data.get("records") or []
    summary = data.get("summary") or {}

    total_cost = 0.0
    total_input = 0
    total_output = 0
    cost_by_model: dict[str, float] = {}
    calls_by_model: dict[str, int] = {}
    cost_by_day: dict[str, float] = {}
    calls_by_day: dict[str, int] = {}
    cost_by_tag: dict[str, float] = {}
    calls_by_tag: dict[str, int] = {}
    cost_by_user: dict[str, float] = {}
    calls_by_user: dict[str, int] = {}
    any_tags = False
    any_users = False

    for rec in records:
        cost = float(rec.get("cost_usd", 0) or 0)
        total_cost += cost
        total_input += int(rec.get("input_tokens", 0) or 0)
        total_output += int(rec.get("output_tokens", 0) or 0)

        model = rec.get("model") or "(unknown)"
        cost_by_model[model] = cost_by_model.get(model, 0.0) + cost
        calls_by_model[model] = calls_by_model.get(model, 0) + 1

        day = _record_day(rec, utc=utc)
        cost_by_day[day] = cost_by_day.get(day, 0.0) + cost
        calls_by_day[day] = calls_by_day.get(day, 0) + 1

        rec_tags = rec.get("tags") or []
        if rec_tags:
            any_tags = True
        for tag in rec_tags:
            cost_by_tag[tag] = cost_by_tag.get(tag, 0.0) + cost
            calls_by_tag[tag] = calls_by_tag.get(tag, 0) + 1

        user = rec.get("user") or None
        if user:
            any_users = True
            cost_by_user[user] = cost_by_user.get(user, 0.0) + cost
            calls_by_user[user] = calls_by_user.get(user, 0) + 1

    if not records and summary:
        total_cost = float(summary.get("total_cost_usd", 0) or 0)
        total_input = int(summary.get("total_input_tokens", 0) or 0)
        total_output = int(summary.get("total_output_tokens", 0) or 0)

    n_calls = len(records) or int(summary.get("total_requests", 0) or 0)

    by_model = [
        {
            "model": model,
            "calls": calls_by_model[model],
            "cost_usd": round(cost, 6),
            "share_pct": round(cost / total_cost * 100, 2) if total_cost else 0.0,
        }
        for model, cost in sorted(cost_by_model.items(), key=lambda x: -x[1])[:top]
    ]

    day_keys = sorted(k for k in cost_by_day if k != UNKNOWN_DAY)[-trend_days:]
    if UNKNOWN_DAY in cost_by_day:
        day_keys.append(UNKNOWN_DAY)
    by_day = [
        {
            "day": key,
            "calls": calls_by_day[key],
            "cost_usd": round(cost_by_day[key], 6),
        }
        for key in day_keys
    ]

    by_tag = (
        [
            {
                "tag": tag,
                "calls": calls_by_tag[tag],
                "cost_usd": round(cost, 6),
            }
            for tag, cost in sorted(cost_by_tag.items(), key=lambda x: -x[1])[:top]
        ]
        if any_tags
        else []
    )

    by_user = (
        [
            {
                "user": user,
                "calls": calls_by_user[user],
                "cost_usd": round(cost, 6),
            }
            for user, cost in sorted(cost_by_user.items(), key=lambda x: -x[1])[:top]
        ]
        if any_users
        else []
    )

    top_calls = [
        {
            "model": rec.get("model") or "(unknown)",
            "input_tokens": int(rec.get("input_tokens", 0) or 0),
            "output_tokens": int(rec.get("output_tokens", 0) or 0),
            "cost_usd": round(float(rec.get("cost_usd", 0) or 0), 6),
        }
        for rec in sorted(records, key=lambda r: -float(r.get("cost_usd", 0) or 0))[:top]
    ]

    budget_info: dict[str, Any] | None = None
    if budget is not None:
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        pct = total_cost / budget * 100
        if pct >= 100:
            status = "over"
        elif pct >= BUDGET_WARN_PCT:
            status = "warn"
        else:
            status = "ok"
        budget_info = {
            "limit_usd": budget,
            "spent_usd": round(total_cost, 6),
            "remaining_usd": round(budget - total_cost, 6),
            "used_pct": round(pct, 2),
            "status": status,
        }

    return {
        "totals": {
            "cost_usd": round(total_cost, 6),
            "calls": n_calls,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "avg_cost_per_call_usd": round(total_cost / n_calls, 6) if n_calls else 0.0,
        },
        "timezone": "utc" if utc else "local",
        "by_model": by_model,
        "by_day": by_day,
        "by_tag": by_tag,
        "by_user": by_user,
        "top_calls": top_calls,
        "budget": budget_info,
    }


def render_dashboard(dash: dict[str, Any], *, title: str = "LLM Cost Guardian") -> RenderableType:
    """Render dashboard data (from :func:`build_dashboard_data`) with rich.

    Requires the ``rich`` package (``pip install "llm-cost-guardian[dashboard]"``).
    """
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    totals = dash["totals"]
    renderables: list[Any] = []

    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    header.add_row(
        Text(title, style="bold cyan"),
        Text(f"tz: {dash['timezone']}", style="dim"),
    )
    renderables.append(header)

    totals_table = Table(show_header=False, box=None, padding=(0, 2))
    totals_table.add_column(style="bold")
    totals_table.add_column(justify="right")
    totals_table.add_row("Total cost", f"${totals['cost_usd']:.6f}")
    totals_table.add_row("Calls", f"{totals['calls']:,}")
    totals_table.add_row(
        "Tokens",
        f"{totals['input_tokens']:,} in / {totals['output_tokens']:,} out",
    )
    totals_table.add_row("Avg cost/call", f"${totals['avg_cost_per_call_usd']:.6f}")
    renderables.append(Panel(totals_table, title="Totals", border_style="cyan"))

    budget = dash.get("budget")
    if budget:
        colors = {"ok": "green", "warn": "yellow", "over": "red"}
        color = colors[budget["status"]]
        bar_width = 30
        filled = min(bar_width, round(budget["used_pct"] / 100 * bar_width))
        bar = Text()
        bar.append("#" * filled, style=color)
        bar.append("-" * (bar_width - filled), style="dim")
        line = Text()
        line.append(f"${budget['spent_usd']:.4f} / ${budget['limit_usd']:.2f}  ")
        line.append(bar)
        line.append(f"  {budget['used_pct']:.1f}%", style=f"bold {color}")
        if budget["status"] == "over":
            line.append("  OVER BUDGET", style="bold red")
        elif budget["status"] == "warn":
            line.append("  approaching limit", style="yellow")
        renderables.append(Panel(line, title="Budget", border_style=color))

    if dash["by_model"]:
        model_table = Table(title=None, expand=True)
        model_table.add_column("Model", style="bold", overflow="fold")
        model_table.add_column("Calls", justify="right")
        model_table.add_column("Cost", justify="right")
        model_table.add_column("Share", justify="right")
        for row in dash["by_model"]:
            model_table.add_row(
                row["model"],
                f"{row['calls']:,}",
                f"${row['cost_usd']:.6f}",
                f"{row['share_pct']:.1f}%",
            )
        renderables.append(Panel(model_table, title="Cost by Model", border_style="magenta"))

    if dash["by_day"]:
        max_cost = max(row["cost_usd"] for row in dash["by_day"]) or 0.0
        day_table = Table(expand=True)
        day_table.add_column("Day")
        day_table.add_column("Calls", justify="right")
        day_table.add_column("Cost", justify="right")
        day_table.add_column("Trend", ratio=1)
        bar_width = 20
        for row in dash["by_day"]:
            trend_bar = "#" * round(row["cost_usd"] / max_cost * bar_width) if max_cost > 0 else ""
            day_table.add_row(
                row["day"],
                f"{row['calls']:,}",
                f"${row['cost_usd']:.6f}",
                Text(trend_bar, style="green"),
            )
        renderables.append(Panel(day_table, title="Daily Trend", border_style="green"))

    columns = Table.grid(expand=True, padding=(0, 1))
    columns.add_column(ratio=1)
    columns.add_column(ratio=1)
    side_panels: list[Any] = []
    if dash["by_tag"]:
        tag_table = Table(expand=True)
        tag_table.add_column("Tag", overflow="fold")
        tag_table.add_column("Calls", justify="right")
        tag_table.add_column("Cost", justify="right")
        for row in dash["by_tag"]:
            tag_table.add_row(row["tag"], f"{row['calls']:,}", f"${row['cost_usd']:.6f}")
        side_panels.append(Panel(tag_table, title="Top Tags", border_style="blue"))
    if dash["by_user"]:
        user_table = Table(expand=True)
        user_table.add_column("User", overflow="fold")
        user_table.add_column("Calls", justify="right")
        user_table.add_column("Cost", justify="right")
        for row in dash["by_user"]:
            user_table.add_row(row["user"], f"{row['calls']:,}", f"${row['cost_usd']:.6f}")
        side_panels.append(Panel(user_table, title="Top Users", border_style="blue"))
    if len(side_panels) == 2:
        columns.add_row(*side_panels)
        renderables.append(columns)
    elif side_panels:
        renderables.append(side_panels[0])

    if dash["top_calls"]:
        calls_table = Table(expand=True)
        calls_table.add_column("#", justify="right")
        calls_table.add_column("Model", overflow="fold")
        calls_table.add_column("Input", justify="right")
        calls_table.add_column("Output", justify="right")
        calls_table.add_column("Cost", justify="right")
        for i, row in enumerate(dash["top_calls"], 1):
            calls_table.add_row(
                str(i),
                row["model"],
                f"{row['input_tokens']:,}",
                f"{row['output_tokens']:,}",
                f"${row['cost_usd']:.6f}",
            )
        renderables.append(Panel(calls_table, title="Most Expensive Calls", border_style="red"))

    return Group(*renderables)

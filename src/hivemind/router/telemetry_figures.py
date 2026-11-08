# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Figures shown on the portal, join and bee pages come from the live telemetry payload or are shown as n/a.

A page that printed ``int(section.get("key", 147))`` showed 147 workunits whenever the payload lacked the key, as the
ruche's local payload always did (IP-146, IP-148). Here a missing or null figure renders as ``n/a``; nothing else is
ever printed in its place.
"""

from __future__ import annotations

from typing import Any

NOT_AVAILABLE = "n/a"


def shown_figure(value: Any, decimals: int | None = None) -> str:
    """Format a telemetry figure for a page: thousands separators, ``decimals`` places for floats, ``n/a`` when absent."""
    if value is None or isinstance(value, bool):
        return NOT_AVAILABLE
    if isinstance(value, str):
        return value if value else NOT_AVAILABLE
    try:
        number = float(value)
    except (TypeError, ValueError):
        return NOT_AVAILABLE
    if number != number or number in (float("inf"), float("-inf")):
        return NOT_AVAILABLE
    if decimals is None:
        return f"{int(number):,}" if number == int(number) else f"{number:,.2f}"
    return f"{number:,.{decimals}f}"

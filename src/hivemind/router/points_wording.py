"""Display wording: HiveMind runs on HiveMind Reflexion Points (HMT) only; no euro or dollar amount is ever shown.

`to_points_wording` rewrites user-facing text (pages and account JSON, never model output). Rules, in order:
1. A parenthesised currency note, e.g. "(0.00 EUR / 0.00 USD)", "(0,00 EUR / 0,00 USD)", "(0.00 EUR/USD)", is dropped:
   the surrounding sentence already names the points.
2. Phrases that put a currency where a unit belongs are reworded to the points ("at 0.00 EUR cost", "0.00 EUR / USD Charter");
   a "| 0.00 EUR Cloud Cost" segment is dropped (hosting is not free, and costs are not shown in currency).
3. Any other "0.00 EUR[/USD]" becomes "HiveMind Reflexion Points (HMT)".
4. Zero-dollar hosting labels ("$0/mo Idle", "$0.00/mo Free Tier") lose the dollar amount.
5. Point names are unified to Reflexion Points in English, French, German, Spanish and Chinese.
"""

from __future__ import annotations

import re

UNIT: str = "HiveMind Reflexion Points (HMT)"
_ZERO = r"0[.,]00"
_CURRENCY = rf"{_ZERO}\s*EUR(?:\s*/\s*(?:{_ZERO}\s*)?USD)?"

_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"\s*\((?:<code>)?{_CURRENCY}(?:</code>)?\)"), ""),
    (re.compile(rf"at {_CURRENCY} cost"), f"for {UNIT}"),
    (re.compile(rf"{_CURRENCY} non-monetary"), "non-monetary"),
    (re.compile(rf"\s*\|\s*{_CURRENCY} Cloud Cost"), ""),
    (re.compile(rf"{_CURRENCY} ([Cc])harter"), r"Reflexion Points \1harter"),
    (re.compile(_CURRENCY), UNIT),
    (re.compile(r"\s*\(\$0(?:\.00)?/mo\)"), ""),
    (re.compile(r"\(\$0(?:\.00)?/mo Idle\)"), "(scales to zero when idle)"),
    (re.compile(r"\$0(?:\.00)?/mo Idle"), "Scales to zero when idle"),
    (re.compile(r"\$0(?:\.00)?/mo Free Tier"), "Free Tier"),
    (re.compile(r"HiveMind Compute Points"), "HiveMind Reflexion Points"),
    (re.compile(r"\bPoints HiveMind\b"), "Points de Réflexion HiveMind"),
    (re.compile(r"\bHiveMind-Punkte\b"), "HiveMind-Reflexionspunkte"),
    (re.compile(r"\bPuntos HiveMind\b"), "Puntos de Reflexión HiveMind"),
    (re.compile("算力积分"), "反思积分"),
    (re.compile(r"\\u7b97\\u529b\\u79ef\\u5206"), r"\\u53cd\\u601d\\u79ef\\u5206"),
)

LEFTOVER_CURRENCY: re.Pattern[str] = re.compile(r"\bEUR\b|\bUSD\b|€|\$\d")


def to_points_wording(text: str) -> str:
    """Apply the rules above; idempotent."""
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


def currency_leftovers(text: str) -> list[str]:
    """Currency mentions a rewrite did not cover, with 40 characters of context each (empty when clean)."""
    return [text[max(0, m.start() - 40): m.end() + 40] for m in LEFTOVER_CURRENCY.finditer(text)]

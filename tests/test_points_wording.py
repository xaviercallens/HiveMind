"""The displayed economy is points only: every currency phrase seen on the live pages must be rewritten."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hivemind.router.points_wording import UNIT, currency_leftovers, to_points_wording

# Phrases observed on the live router, Ruche and Tribute pages on 2026-10-25.
OBSERVED: list[tuple[str, str]] = [
    ("Every workunit awards HiveMind Compute Points (0.00 EUR / 0.00 USD). Spend",
     "Every workunit awards HiveMind Reflexion Points. Spend"),
    ("Charter (0,00 EUR / 0,00 USD) • Partagez", "Charter • Partagez"),
    ("rapporte +250 Points HiveMind (0,00 EUR / 0,00 USD). Utilisez", "rapporte +250 Points de Réflexion HiveMind. Utilisez"),
    ("bringt +250 HiveMind-Punkte (0,00 EUR / 0,00 USD). L", "bringt +250 HiveMind-Reflexionspunkte. L"),
    ("otorga +250 Puntos HiveMind (0.00 EUR / 0.00 USD).", "otorga +250 Puntos de Reflexión HiveMind."),
    ("+250 算力积分 (0.00 EUR/USD)。", "+250 反思积分。"),
    ("+250 \\u7b97\\u529b\\u79ef\\u5206 (0.00 EUR/USD)", "+250 \\u53cd\\u601d\\u79ef\\u5206"),
    ("Reciprocity: <span>0.00 EUR</span>", f"Reciprocity: <span>{UNIT}</span>"),
    ("developers and researchers at 0.00 EUR cost.", f"developers and researchers for {UNIT}."),
    ("0.00 EUR / USD Charter</span>", "Reflexion Points Charter</span>"),
    ("Explain how HiveMind's 0.00 EUR non-monetary compute", "Explain how HiveMind's non-monetary compute"),
    ("Charter (<code>0.00 EUR / 0.00 USD</code>)</div>", "Charter</div>"),
    ("<strong>1,250 pts</strong> (0.00 EUR/USD)</span>", "<strong>1,250 pts</strong></span>"),
    ("hivemind-at-home-ruche ($0/mo Idle)</span>", "hivemind-at-home-ruche (scales to zero when idle)</span>"),
    ("Low-Cost Storage ($0.00/mo)</span>", "Low-Cost Storage</span>"),
    ("'kpi-val'>$0.00/mo Free Tier</div>", "'kpi-val'>Free Tier</div>"),
    ("kpi-sub\">$0.00/mo Idle &bull; 512Mi", "kpi-sub\">Scales to zero when idle &bull; 512Mi"),
    ("<strong style=\"color:#fbbf24;\">0.00 EUR / 0.00 USD</strong>", f"<strong style=\"color:#fbbf24;\">{UNIT}</strong>"),    ("24 PoUW Blocks Mined | 0.00 EUR Cloud Cost\",", "24 PoUW Blocks Mined\","),
    ("Sigma-MoE Peri-LN routing, 0.00 EUR/USD charter).", "Sigma-MoE Peri-LN routing, Reflexion Points charter)."),
]


@pytest.mark.parametrize("before, after", OBSERVED)
def test_each_observed_phrase_is_rewritten_exactly(before: str, after: str) -> None:
    assert currency_leftovers(before) or "$0" in before
    assert to_points_wording(before) == after
    assert currency_leftovers(after) == []


def test_text_without_currency_is_unchanged_and_balances_keep_hmt() -> None:
    text = "Balance: 98.4165 HMT | price 0.4941 HMT/1k tokens | 1,250 pts"
    assert to_points_wording(text) == text
    assert currency_leftovers(text) == []


def test_non_zero_amounts_are_reported_not_silently_rewritten() -> None:
    text = "a GPU hour costs 0.35 USD elsewhere"
    assert to_points_wording(text) == text
    assert len(currency_leftovers(text)) == 1


_FRAGMENTS = st.sampled_from([b for b, _ in OBSERVED] + ["plain text ", "HMT ", "<b>", "</b>", " (", ")", "0.00 ", "EUR", " / "])


@given(st.lists(_FRAGMENTS, max_size=12).map("".join))
def test_rewrite_is_idempotent(text: str) -> None:
    once = to_points_wording(text)
    assert to_points_wording(once) == once

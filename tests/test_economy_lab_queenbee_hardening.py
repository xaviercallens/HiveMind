"""Positive and negative controls for economy views, swarm lab, and queenbee hardening (Iteration 111)."""

from __future__ import annotations

import hashlib
import pytest

from hivemind.economy.international_launch_views import (
    render_foundation_paper_portal_html,
    render_gcp_planet_forecast_html,
)
from hivemind.lab.swarm_lab import (
    Scenario,
    merge_deltas,
    retention,
    simulate,
)
from hivemind.queenbee.dance import (
    DanceRecord,
    recruitment_weights,
    rotate_scouts,
    wilson_interval,
)
from hivemind.queenbee.knowledge_market import (
    KnowledgeMarket,
    KnowledgeShare,
    admit_share,
    settle_knowledge_bonus,
)


def test_international_launch_views_positive_and_negative_controls() -> None:
    html_planet = render_gcp_planet_forecast_html({"data_source": "LIVE_GCP_PRODUCTION"})
    assert "<!DOCTYPE html>" in html_planet

    html_paper = render_foundation_paper_portal_html({"five_wave_international_roadmap": []})
    assert "<!DOCTYPE html>" in html_paper

    with pytest.raises(ValueError, match="initial_forecast must be a dict or None"):
        render_gcp_planet_forecast_html("not-a-dict")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="payload must be a dict or None"):
        render_foundation_paper_portal_html("not-a-dict")  # type: ignore[arg-type]


def test_swarm_lab_positive_and_negative_controls() -> None:
    out = simulate(Scenario(M=2, K=2, rounds=2, H=2, dim=4))
    assert len(out["loss_curve"]) == 3
    merged = merge_deltas([([1.0, 2.0], 1.0)], "direct")
    assert merged == [1.0, 2.0]
    assert retention(2.0, 1.0, 0.5) == pytest.approx((2.0 - 1.0) / (2.0 - 0.5))

    with pytest.raises(ValueError, match="Scenario or dict"):
        simulate("not-a-scenario")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="items must be non-empty"):
        merge_deltas([], "direct")

    with pytest.raises(ValueError, match="mode must be 'direct' or 'rda'"):
        merge_deltas([([1.0], 1.0)], "invalid")

    with pytest.raises(ValueError, match="finite numeric values"):
        retention(True, 1.0, 0.5)  # type: ignore[arg-type]


def test_queenbee_dance_positive_and_negative_controls() -> None:
    rec = DanceRecord("exp-1", "exp-1", "cluster-a", 10, 8, 0.5, 1, "unit_tests", "signer-1").signed(b"secret")
    assert rec.verify(b"secret")
    lo, hi = wilson_interval(8, 10)
    assert 0.0 <= lo <= hi <= 1.0
    scouts = rotate_scouts(["s1", "s2", "s3"], epoch=1, k=2)
    assert len(scouts) == 2

    with pytest.raises(ValueError, match="integers, not bool"):
        DanceRecord("exp-1", "exp-1", "cluster-a", 10.5, 8, 0.5, 1, "unit_tests", "signer-1")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cost_per_1k must be finite"):
        DanceRecord("exp-1", "exp-1", "cluster-a", 10, 8, True, 1, "unit_tests", "signer-1")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="need 0 <= successes <= n"):
        wilson_interval(8.0, 10)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="z must be a strictly positive"):
        wilson_interval(8, 10, z=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epoch and k must be integers"):
        rotate_scouts(["s1", "s2"], epoch=True, k=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="now_epoch and horizon_epochs"):
        recruitment_weights([rec], now_epoch=True, horizon_epochs=5, cap_per_record=0.5, secret=b"secret")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cap_per_record must be finite"):
        recruitment_weights([rec], now_epoch=1, horizon_epochs=5, cap_per_record=True, secret=b"secret")  # type: ignore[arg-type]


def test_queenbee_knowledge_market_positive_and_negative_controls() -> None:
    txt = "def f(x): return x + 1"
    h = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    rcpt = "a" * 64
    share = KnowledgeShare("acc-1", "verified_code", "MIT", True, 2, h, rcpt)
    adm = admit_share(share, txt, scanner=lambda _: True)
    assert adm.admitted is True

    st = settle_knowledge_bonus(2, 100.0, 0.2, {"acc-1": 1})
    assert st.pool_hmt == pytest.approx(20.0)

    with pytest.raises(ValueError, match="epoch must be >= 0"):
        KnowledgeShare("acc-1", "verified_code", "MIT", True, 2.5, h, rcpt)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cap_per_identity_per_epoch must be >= 1"):
        admit_share(share, txt, cap_per_identity_per_epoch=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="epoch must be >= 0"):
        KnowledgeMarket(epoch=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="validator_fee_hmt must be finite"):
        settle_knowledge_bonus(2, True, 0.2, {"acc-1": 1})  # type: ignore[arg-type]

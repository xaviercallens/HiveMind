"""HM-S7-05: worker EF21 residual telemetry in /api/strategy-optimizer and the strategy HUD."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hivemind.game.steam_edition import _generate_cyberpunk_hud_html, create_hack_miner_fastapi_app
from hivemind.swarm.strategy_optimizer import StrategyOptimizerTelemetryEngine


def _entry(worker: str = "w-01", branch: str = "Code", **overrides) -> dict:
    entry = {
        "worker_id": worker,
        "branch_name": branch,
        "outer_step": 3,
        "residual_norm": 0.42,
        "raw_delta_norm": 1.5,
        "contraction_factor": 0.28,
        "clip_events": 0,
        "topk_ratio": 0.2,
        "stale_decay": 1.0,
    }
    entry.update(overrides)
    return entry


def test_posted_telemetry_is_served_with_its_basis() -> None:
    client = TestClient(create_hack_miner_fastapi_app(worker_token="test-worker-token-not-a-secret"), raise_server_exceptions=False)
    empty = client.get("/api/strategy-optimizer").json()["ef21_live_telemetry"]
    assert empty["workers"] == [] and "self-reported" in empty["basis"]

    assert client.post("/api/strategy-optimizer", headers={"Authorization": "Bearer test-worker-token-not-a-secret"}, json={"ef21": _entry()}).status_code == 200
    posted = client.post(
        "/api/strategy-optimizer", headers={"Authorization": "Bearer test-worker-token-not-a-secret"}, json={"ef21": [_entry("w-02", "STEM", contraction_factor=None)]}
    )
    assert posted.status_code == 200
    workers = client.get("/api/strategy-optimizer").json()["ef21_live_telemetry"]["workers"]
    assert [(w["worker_id"], w["branch_name"]) for w in workers] == [("w-01", "Code"), ("w-02", "STEM")]
    assert workers[0]["residual_norm"] == 0.42 and workers[1]["contraction_factor"] is None


@pytest.mark.parametrize(
    "entry",
    [
        _entry(residual_norm="nan"),
        _entry(raw_delta_norm=-1.0),
        _entry(contraction_factor="inf"),
        _entry(worker=""),
        _entry(branch="x" * 129),
        _entry(clip_events=-1),
        _entry(outer_step=1.5),
        _entry(topk_ratio=0.0),
        _entry(stale_decay=1.5),
        "not-an-object",
    ],
)
def test_invalid_telemetry_is_rejected_without_changing_state(entry) -> None:
    engine = StrategyOptimizerTelemetryEngine(seed=2026)
    engine.update({"ef21": _entry()})
    before = dict(engine.ef21_telemetry)
    with pytest.raises(ValueError):
        engine.update({"ef21": [_entry("w-09"), entry]})
    assert engine.ef21_telemetry == before


def test_one_entry_per_worker_branch_and_a_bounded_store() -> None:
    engine = StrategyOptimizerTelemetryEngine(seed=2026)
    engine.update({"ef21": [_entry(f"w-{i:03d}") for i in range(200)]})
    engine.update({"ef21": [_entry("w-000", residual_norm=0.1)]})
    engine.update({"ef21": [_entry(f"w-{i:03d}") for i in range(200, 257)]})
    assert len(engine.ef21_telemetry) == 256
    assert ("w-001", "Code") not in engine.ef21_telemetry
    assert engine.ef21_telemetry[("w-000", "Code")]["residual_norm"] == 0.1
    with pytest.raises(ValueError, match="at most 256"):
        engine.update({"ef21": [_entry(f"x-{i}") for i in range(257)]})


def test_hud_renders_the_ef21_section() -> None:
    html = _generate_cyberpunk_hud_html({})
    assert "[EF21 RESIDUALS] self-reported by workers" in html
    assert "s.ef21_live_telemetry" in html
    assert "no worker has posted EF21 telemetry yet" in html

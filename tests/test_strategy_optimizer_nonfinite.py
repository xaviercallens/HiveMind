"""POST /api/strategy-optimizer rejects non-finite input without corrupting the shared engine."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hivemind.game.steam_edition import create_hack_miner_fastapi_app
from hivemind.swarm.strategy_optimizer import StrategyOptimizerTelemetryEngine

DOMAIN = list(StrategyOptimizerTelemetryEngine(seed=2026).domains)[0]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_hack_miner_fastapi_app(worker_token="test-worker-token-not-a-secret"), raise_server_exceptions=False)


@pytest.mark.parametrize(
    "payload",
    [
        {"ring_rtt_ms": "nan"},
        {"ring_rtt_ms": "inf"},
        {"bbr_pacing_mbps": "inf"},
        {"packet_loss_rate": "nan"},
        {"domain_name": DOMAIN, "loss_before": "inf", "loss_after": 1.0},
        {"domain_name": DOMAIN, "loss_before": 2.0, "loss_after": "nan"},
        {"domain_name": DOMAIN, "loss_before": 2.0, "loss_after": 1.0, "gpu_seconds": "inf"},
        {"ring_rtt_ms": "fast"},
        {"domain_name": DOMAIN, "loss_before": 2.0},
    ],
)
def test_bad_observation_is_rejected_and_leaves_the_tab_working(client, payload) -> None:
    before = client.get("/api/strategy-optimizer")
    assert before.status_code == 200

    rejected = client.post("/api/strategy-optimizer", headers={"Authorization": "Bearer test-worker-token-not-a-secret"}, json=payload)
    assert rejected.status_code == 400

    after = client.get("/api/strategy-optimizer")
    assert after.status_code == 200
    for key in ("live_inputs", "si02_ucb1_curriculum_bandit", "si03_adaptive_diloco_cadence"):
        assert after.json()[key] == before.json()[key], key


def test_engine_state_is_unchanged_after_a_rejected_update() -> None:
    engine = StrategyOptimizerTelemetryEngine(seed=2026)
    link, revision = dict(engine.link_profile), engine.revision
    pulls = dict(engine.scheduler.pull_counts)
    for payload in (
        {"ring_rtt_ms": float("nan")},
        {"domain_name": DOMAIN, "loss_before": float("inf"), "loss_after": 1.0},
    ):
        with pytest.raises(ValueError, match="finite"):
            engine.update(payload)
    assert engine.link_profile == link and engine.revision == revision
    assert engine.scheduler.pull_counts == pulls
    engine.update({"ring_rtt_ms": 42.0})
    assert engine.link_profile["ring_rtt_ms"] == 42.0


def test_a_snapshot_failure_rolls_the_update_back(monkeypatch) -> None:
    engine = StrategyOptimizerTelemetryEngine(seed=2026)
    link, revision = dict(engine.link_profile), engine.revision

    def boom(force_refresh: bool = False):
        raise OverflowError("cannot convert float infinity to integer")

    monkeypatch.setattr(engine, "snapshot", boom)
    with pytest.raises(ValueError, match="observation rejected"):
        engine.update({"ring_rtt_ms": 1e300})
    assert engine.link_profile == link and engine.revision == revision

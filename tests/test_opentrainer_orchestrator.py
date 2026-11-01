# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit and integration tests for the Low-Cost VM Permanent OpenTrainer Orchestrator.

Verifies:
1. Low-cost GCP Spot VM governance compliance (`e2-small`, 0 GPUs, `requires_t4_gpu=False`,
   hardened `systemd` unit).
2. T4-offline collaborative QLoRA MoE training reduces held-out evaluation loss across
   outer DiLoCo rounds using a learner on shard `l1` and the CPU sentinel on shard `l0`.
3. Atomic checkpoint persistence and crash-resume (`current_round`, `global_sha256`,
   `velocity`, `rounds`).
4. Byzantine / tampered `task_result` envelopes are rejected fail-closed (`script_sha256`,
   `global_sha256`, `delta_l2` cap), while honest learners advance on deadline expiry.
5. FastAPI HTTP endpoints (`/healthz`, `/v1/opentrainer/status`, `/v1/opentrainer/offer`,
   `/v1/opentrainer/claim`, `/v1/opentrainer/result`, `/v1/opentrainer/tick`) end-to-end.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import e8_qlora_moe_round as qlora_round  # noqa: E402
from hivemind.swarm import federation_agents as fed  # noqa: E402
from hivemind.swarm.amcp_kafka_transport import HMACKeyring, sign_envelope  # noqa: E402
from hivemind.swarm.opentrainer_orchestrator import (  # noqa: E402
    PIN_FILES,
    LowCostVMProfile,
    OpenTrainerConfig,
    PermanentOpenTrainerOrchestrator,
    create_opentrainer_app,
    resolve_opentrainer_keyring,
)


def _compute_learner_result_envelope(
    offer_env: Mapping[str, Any],
    *,
    learner_id: str,
    learner_index: int,
    keyring: HMACKeyring,
    host: str = "rtx2070-test-host",
) -> dict[str, Any]:
    payload = dict(offer_env["data"]["payload"])
    params = dict(payload.get("params") or {})
    seed = int(params.get("seed", 1))
    n_learners = int(params.get("learners", 2))
    pin = fed.code_pin(PIN_FILES)
    base = qlora_round.build_base_state(seed)
    base_sha = fed.tensors_sha256(base)
    start_state = fed.check_offer(payload, my_pin=pin, my_base_sha256=base_sha)

    model = qlora_round.load_model(base, "none", "cpu", seed)
    evalset = qlora_round.make_eval(n_learners, seed)
    shard = qlora_round.make_shard(learner_index, int(params.get("shard_size", 32)), seed)
    out = qlora_round.learner_round(
        model,
        start_state,
        shard,
        evalset,
        steps=int(payload["inner_steps"]),
        lr=float(params.get("inner_lr", 1e-2)),
        batch=int(params.get("batch", 4)),
        seed=seed,
        rnd=int(payload["round"]),
        learner_id=learner_index,
        device="cpu",
        shuffle_labels=False,
    )
    res_ev = fed.build_result(
        run_id=str(payload["run_id"]),
        rnd=int(payload["round"]),
        learner_id=learner_id,
        host=host,
        tokens=int(out["tokens"]),
        delta=out["delta"],
        delta_l2=float(out["delta_l2"]),
        eval_loss_global=float(out["eval_loss_global"]),
        pin=pin,
        global_sha256=str(payload["global_sha256"]),
        vram_peak_mib=0.0,
    )
    return sign_envelope(res_ev, keyring, ttl_s=900.0)


def test_lowcost_vm_profile_and_systemd_governance() -> None:
    profile = LowCostVMProfile()
    gov = profile.validate_governance()
    assert gov["compliant"] is True
    assert all(gov["checks"].values())
    assert profile.requires_t4_gpu is False
    assert profile.accelerators_count == 0
    assert profile.provisioning_model == "SPOT"
    assert profile.machine_type == "e2-small"
    assert profile.estimated_monthly_cost_usd <= 5.0

    service_path = (
        REPO_ROOT / "deploy" / "opentrainer_vm" / "hivemind-opentrainer.service"
    )
    assert service_path.is_file()
    service_txt = service_path.read_text(encoding="utf-8")
    assert "User=hivemind" in service_txt
    assert "NoNewPrivileges=true" in service_txt
    assert "ProtectSystem=strict" in service_txt


def test_t4_offline_collaborative_training_reduces_eval_loss(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "opentrainer_state.pt"
    report_path = tmp_path / "opentrainer_report.json"
    keyring = resolve_opentrainer_keyring()
    cfg = OpenTrainerConfig(
        run_id="test-t4free-collab",
        learners=("l0", "l1"),
        quorum=2,
        rounds=3,
        seed=1,
        inner_steps=20,
        inner_lr=1e-2,
        batch_size=4,
        shard_size=32,
        outer_lr=0.5,
        outer_mu=0.5,
        cpu_sentinel_when_t4_offline=True,
        cpu_sentinel_learner_id="l0",
        cpu_sentinel_inner_steps=10,
        checkpoint_path=ckpt_path,
        report_path=report_path,
    )
    orch = PermanentOpenTrainerOrchestrator(cfg, keyring=keyring)
    initial_loss = orch.initial_eval_loss

    for r in range(3):
        offer_env = orch.get_signed_offer_envelope()
        payload = offer_env["data"]["payload"]
        assert payload["round"] == r

        claim_ev = fed.build_claim(
            run_id=cfg.run_id,
            rnd=r,
            learner_id="l1",
            host="rtx2070-test-host",
        )
        claim_res = orch.submit_claim_envelope(sign_envelope(claim_ev, keyring, ttl_s=900.0))
        assert claim_res["accepted"] is True

        res_env = _compute_learner_result_envelope(
            offer_env,
            learner_id="l1",
            learner_index=1,
            keyring=keyring,
        )
        sub_res = orch.submit_result_envelope(res_env)
        assert sub_res["accepted"] is True
        assert sub_res["round_advanced"] is False

        # Since quorum=2 and T4 (l0) is offline, trigger CPU sentinel on l0
        sentinel_res = orch.run_cpu_sentinel_step_if_needed()
        assert sentinel_res["ran_sentinel"] is True
        assert sentinel_res["round_advanced"] is True

    rep = orch.build_report()
    assert rep["completed"] is True
    assert rep["current_round"] == 3
    assert rep["t4_required"] is False
    assert rep["t4_participated_rounds"] == 0
    assert rep["t4_independent_rounds_completed"] == 3
    assert rep["current_eval_loss"] < initial_loss


def test_atomic_checkpoint_persistence_and_crash_resume(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "crash_resume_state.pt"
    keyring = resolve_opentrainer_keyring()
    cfg = OpenTrainerConfig(
        run_id="test-crash-resume",
        learners=("l0", "l1"),
        quorum=1,
        rounds=2,
        inner_steps=12,
        cpu_sentinel_when_t4_offline=True,
        checkpoint_path=ckpt_path,
    )
    orch1 = PermanentOpenTrainerOrchestrator(cfg, keyring=keyring)
    offer_env = orch1.get_signed_offer_envelope()

    # Submit both l0 (via CPU sentinel) and l1 so round 0 closes immediately
    orch1.run_cpu_sentinel_step_if_needed()
    res_env = _compute_learner_result_envelope(
        offer_env,
        learner_id="l1",
        learner_index=1,
        keyring=keyring,
    )
    out = orch1.submit_result_envelope(res_env)
    assert out["accepted"] is True
    assert out["round_advanced"] is True
    rep1 = orch1.build_report()
    sha_after_r0 = rep1["global_sha256"]
    loss_after_r0 = orch1.current_eval_loss

    # Re-instantiate from checkpoint
    orch2 = PermanentOpenTrainerOrchestrator(cfg, keyring=keyring)
    rep2 = orch2.build_report()
    assert orch2.current_round == 1
    assert rep2["global_sha256"] == sha_after_r0
    assert abs(orch2.current_eval_loss - loss_after_r0) < 1e-5


def test_byzantine_and_tampered_result_rejected_fail_closed(tmp_path: Path) -> None:
    keyring = resolve_opentrainer_keyring()
    cfg = OpenTrainerConfig(
        run_id="test-byzantine-guard",
        learners=("l0", "l1"),
        quorum=1,
        rounds=2,
        inner_steps=10,
        round_deadline_s=30.0,
        max_delta_l2=0.0001,
        cpu_sentinel_when_t4_offline=False,
        checkpoint_path=tmp_path / "byz_state.pt",
    )
    orch = PermanentOpenTrainerOrchestrator(cfg, keyring=keyring)
    offer_env = orch.get_signed_offer_envelope()
    payload = offer_env["data"]["payload"]

    # 1. Tamper with global_sha256 -> rejected fail-closed
    honest_env = _compute_learner_result_envelope(
        offer_env,
        learner_id="l1",
        learner_index=1,
        keyring=keyring,
    )
    honest_payload = honest_env["data"]["payload"]
    bad_sha_ev = fed.build_result(
        run_id=cfg.run_id,
        rnd=0,
        learner_id="l1",
        host="byzantine-node",
        tokens=int(honest_payload["tokens"]),
        delta=fed.decode_tensors(str(honest_payload["delta_b64"])),
        delta_l2=float(honest_payload["delta_l2"]),
        eval_loss_global=float(honest_payload["eval_loss_global"]),
        pin=str(payload["script_sha256"]),
        global_sha256="0" * 64,
    )
    r1 = orch.submit_result_envelope(sign_envelope(bad_sha_ev, keyring, ttl_s=900.0))
    assert r1["accepted"] is False
    assert "different global state" in str(r1["rejection_reason"])

    # 2. Honest delta exceeds tight max_delta_l2=0.0001 -> rejected fail-closed
    r2 = orch.submit_result_envelope(honest_env)
    assert r2["accepted"] is False
    assert "exceeds the bound" in str(r2["rejection_reason"])
    assert orch.current_round == 0

    # 3. Relax max_delta_l2 to normal bound and advance on deadline expiry with quorum=1
    orch.config.max_delta_l2 = 50.0
    assert orch.current_book is not None
    orch.current_book.max_delta_l2 = 50.0
    r3 = orch.submit_result_envelope(honest_env, now=time.time() + 60.0)
    assert r3["accepted"] is True
    assert r3["round_advanced"] is True
    assert orch.current_round == 1


def test_opentrainer_fastapi_http_endpoints_end_to_end(tmp_path: Path) -> None:
    keyring = resolve_opentrainer_keyring()
    cfg = OpenTrainerConfig(
        run_id="test-http-e2e",
        learners=("l0", "l1"),
        quorum=1,
        rounds=2,
        inner_steps=10,
        cpu_sentinel_when_t4_offline=True,
        checkpoint_path=tmp_path / "http_state.pt",
    )
    orch = PermanentOpenTrainerOrchestrator(cfg, keyring=keyring)
    app = create_opentrainer_app(orch)
    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert health.json()["t4_required"] is False

    offer_resp = client.get("/v1/opentrainer/offer")
    assert offer_resp.status_code == 200
    offer_env = offer_resp.json()
    payload = offer_env["data"]["payload"]

    claim_ev = fed.build_claim(
        run_id=cfg.run_id,
        rnd=int(payload["round"]),
        learner_id="l1",
        host="http-learner-l1",
    )
    claim_resp = client.post(
        "/v1/opentrainer/claim",
        json=sign_envelope(claim_ev, keyring, ttl_s=900.0),
    )
    assert claim_resp.status_code == 200
    assert claim_resp.json()["accepted"] is True

    res_env = _compute_learner_result_envelope(
        offer_env,
        learner_id="l1",
        learner_index=1,
        keyring=keyring,
        host="http-learner-l1",
    )
    res_resp = client.post("/v1/opentrainer/result", json=res_env)
    assert res_resp.status_code == 200
    assert res_resp.json()["accepted"] is True

    # Trigger /tick with CPU sentinel so both l0 (sentinel) and l1 complete round 0
    tick_resp = client.post("/v1/opentrainer/tick", json={"run_sentinel_if_t4_absent": True})
    assert tick_resp.status_code == 200
    assert tick_resp.json()["current_round"] == 1

    status_resp = client.get("/v1/opentrainer/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["current_round"] == 1
    assert status_resp.json()["t4_independent_rounds_completed"] == 1

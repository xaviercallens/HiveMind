# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Comprehensive tests for the Weekend Nov 14–15, 2026 CrowdGPT Sparse-MoE + Decoupled OpenDiLoCo Raid."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from fastapi.testclient import TestClient
import pytest
import torch

from hivemind.cli.campaign_calendar_quorum import (
    CROWDGPT_WEEKEND_NOV14_15_EVENTS,
    generate_crowdgpt_weekend_ics_calendar,
)
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdRuntimeMLP,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CROWDGPT_MOE_8_EXPERTS,
    CROWDGPT_MOE_PUBLIC_MODEL_ID,
    CROWDGPT_WEEKEND_CAMPAIGN_ID,
    CrowdFineGrainedMoEMLP,
    CrowdLoRAMoEMLP,
    CrowdOpenDiLoCoCoordinator,
    assign_hiveminder_moe_shards,
    build_opendiloco_launch_spec,
    build_weekend_nov14_15_raid_manifest,
    extract_moe_trainable_buckets,
    load_crowdgpt_moe_dataset_plan,
    load_moe_trainable_buckets,
    run_crowd_moe_opendiloco_inner_steps,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.decoupled_diloco import LearnerArrival
from hivemind.swarm.gcp_t4_crowd_diloco import forward_crowd_v1_checkpointed

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_small_crowd_model(seed: int = 20261114) -> CrowdV1WithEngram:
    torch.manual_seed(seed)
    cfg = CrowdV1RuntimeConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=128,
        engram_buckets=256,
    )
    model = CrowdV1WithEngram(cfg)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim >= 2:
                p.mul_(0.02)
    return model


def _load_script(name: str, rel_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fine_grained_moe_step0_exact_algebraic_equivalence() -> None:
    """`CrowdFineGrainedMoEMLP` (`8 x 16 = 128`) adds zero expert params and matches dense SwiGLU at step 0."""
    torch.manual_seed(1114)
    dense = CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=64, intermediate_size=128))
    x = torch.randn(3, 7, 64, dtype=torch.float32)
    with torch.no_grad():
        y_dense = dense(x)

    moe_exact = CrowdFineGrainedMoEMLP(dense, num_experts=8, top_k=2, routing_mode="exact_dense_sum")
    expert_param_count = sum(
        p.numel()
        for lst in (moe_exact.experts_w1, moe_exact.experts_w2, moe_exact.experts_w3)
        for mod in lst
        for p in mod.parameters()
    )
    dense_param_count = sum(p.numel() for p in dense.parameters())
    assert expert_param_count == dense_param_count

    with torch.no_grad():
        y_exact = moe_exact(x)
    assert torch.max(torch.abs(y_exact - y_dense)).item() < 1e-5

    # Under zero-initialized router (`sigmoid(0)=0.5`), `top_k == num_experts` also matches dense identically
    moe_sigma_all = CrowdFineGrainedMoEMLP(dense, num_experts=8, top_k=8, routing_mode="sigma_topk")
    with torch.no_grad():
        y_sigma_all = moe_sigma_all(x)
    assert torch.max(torch.abs(y_sigma_all - y_dense)).item() < 1e-5


def test_lora_domain_experts_step0_identity_and_8gb_fomoe_ghost_skip() -> None:
    """`CrowdLoRAMoEMLP` matches `CrowdRuntimeMLP` identically at step 0 and supports 8 GB 2-expert residency."""
    torch.manual_seed(1115)
    dense = CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=64, intermediate_size=128))
    x = torch.randn(2, 9, 64, dtype=torch.float32)
    with torch.no_grad():
        y_dense = dense(x)

    moe_8gb = CrowdLoRAMoEMLP(
        dense,
        num_experts=8,
        top_k=2,
        rank=4,
        alpha=8.0,
        routing_mode="sigma_topk",
        resident_expert_indices=(1, 2),
    )
    assert sorted(moe_8gb.experts.keys()) == ["1", "2"]

    y_step0 = moe_8gb(x)
    assert torch.max(torch.abs(y_step0 - y_dense)).item() == 0.0
    assert 0.0 <= moe_8gb.last_ghost_fraction <= 1.0

    loss = y_step0.pow(2).mean() + 0.01 * moe_8gb.last_aux_loss
    loss.backward()
    assert moe_8gb.router.weight.grad is not None
    assert torch.isfinite(moe_8gb.router.weight.grad).all()
    assert all(p.grad is None for p in moe_8gb.base_mlp.parameters())


@pytest.mark.parametrize("outer_mode", ["opendiloco_nesterov", "pa_cgad"])
def test_crowd_opendiloco_coordinator_rda_convergence_and_ratchet_rollback(
    tmp_path: Path,
    outer_mode: str,
) -> None:
    """Decoupled OpenDiLoCo + RDA decreases held-out loss across 2 non-IID learners and rolls back poisoned steps."""
    model = _build_small_crowd_model(seed=20261114)
    rep = upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=8,
        top_k=2,
        lora_rank=4,
        routing_mode="sigma_topk",
    )
    assert rep.num_layers_upcycled == 4
    assert rep.trainable_parameters > 0

    init_buckets = extract_moe_trainable_buckets(model, num_router_fragments=4)
    assert len([k for k in init_buckets if k.startswith("router_frag_")]) == 4
    assert len([k for k in init_buckets if k.startswith("expert_")]) == 8

    coord = CrowdOpenDiLoCoCoordinator(
        init_buckets,
        min_quorum_k=2,
        merge_mode="rda",
        outer_optimizer_mode=outer_mode,
        outer_lr=0.6,
        cgad_lr=2e-3,
        outer_momentum=0.9,
        work_dir=tmp_path / f"work_{outer_mode}",
    )
    assert coord.flat_state.flat_master_cpu.device.type == "cpu"

    pattern = torch.tensor([(i * 5 + 7) % 64 for i in range(16)], dtype=torch.long)
    stream_a = pattern.repeat(16)
    stream_b = torch.roll(pattern, shifts=2).repeat(16)
    held_out = pattern.repeat(4).reshape(1, 64)

    def _eval_fn(cand_buckets: dict[str, torch.Tensor]) -> float:
        load_moe_trainable_buckets(model, cand_buckets, num_router_fragments=4)
        model.eval()
        with torch.no_grad():
            _, l_t = forward_crowd_v1_checkpointed(
                model,
                held_out,
                labels=held_out,
                use_grad_checkpointing=False,
                token_chunk_size=64,
            )
            assert l_t is not None
            return float(l_t.item())

    snap = coord.export_buckets()
    load_moe_trainable_buckets(model, snap, num_router_fragments=4)
    out_a = run_crowd_moe_opendiloco_inner_steps(model, stream_a, steps=5, batch_size=2, seq_len=32, lr=3e-3)

    load_moe_trainable_buckets(model, snap, num_router_fragments=4)
    out_b = run_crowd_moe_opendiloco_inner_steps(model, stream_b, steps=5, batch_size=2, seq_len=32, lr=3e-3)

    subs = {
        k: [
            (LearnerArrival("gcp-t4-node1", 0.10, out_a["steps_completed"], out_a["tokens_processed"]), out_a["pseudo_grads"][k], 0.0),
            (LearnerArrival("rtx2070-node", 0.14, out_b["steps_completed"], out_b["tokens_processed"]), out_b["pseudo_grads"][k], 1.0),
        ]
        for k in snap
    }
    receipt = coord.step_decoupled_round(subs, eval_fn=_eval_fn)
    assert receipt.accepted_by_loss_ratchet is True
    assert receipt.eval_loss_after < receipt.eval_loss_before
    assert len(receipt.fragments_synced) == 12

    # Poisoned outer update must be rejected and rolled back by the monotone validation-loss ratchet
    sha_good = coord.master_sha256()
    poison_subs = {
        k: [
            (LearnerArrival("byz-0", 0.10, 5, 320), -50.0 * torch.ones_like(v), 0.0),
            (LearnerArrival("byz-1", 0.12, 5, 320), -50.0 * torch.ones_like(v), 0.0),
        ]
        for k, v in coord.export_buckets().items()
    }
    bad_receipt = coord.step_decoupled_round(poison_subs, eval_fn=_eval_fn)
    assert bad_receipt.accepted_by_loss_ratchet is False
    assert coord.master_sha256() == sha_good


def test_s21_dataset_plan_enforces_agpl_allowlist_and_blocks_nc_datasets() -> None:
    """S21 dataset plan covers all 8 experts (`sum=1.0`) and excludes non-commercial licenses (`cc-by-nc-*`)."""
    plan = load_crowdgpt_moe_dataset_plan()
    assert plan["campaign_id"] == CROWDGPT_WEEKEND_CAMPAIGN_ID
    assert plan["public_model_id"] == CROWDGPT_MOE_PUBLIC_MODEL_ID
    assert plan["num_experts"] == len(CROWDGPT_MOE_8_EXPERTS) == 8
    assert plan["total_mixture_weight"] == pytest.approx(1.0)
    assert "allenai/sciq" in plan["rejected_non_allowlisted_datasets"]
    assert "camel-ai/physics" in plan["rejected_non_allowlisted_datasets"]

    for exp in plan["experts"]:
        for src in exp["accepted_sources"]:
            assert "nc" not in str(src["license"]).lower()

    t4_shard = assign_hiveminder_moe_shards("socreateai-agora-hermes-node1", "trainer", 15360)
    assert t4_shard["min_free_vram_floor_mib"] == 4300
    assert len(t4_shard["resident_expert_indices"]) == 8

    rtx_prod = assign_hiveminder_moe_shards("xavgamsrv", "producer", 8192)
    assert rtx_prod["min_free_vram_floor_mib"] == 1536
    assert len(rtx_prod["resident_expert_indices"]) == 2

    vxtzq_pros = assign_hiveminder_moe_shards("vxtzq", "prosumer", 8192)
    assert len(vxtzq_pros["resident_expert_indices"]) == 4

    spec = build_opendiloco_launch_spec(role="prosumer", node_handle="vxtzq", vram_total_mib=8192)
    assert CROWDGPT_WEEKEND_CAMPAIGN_ID in spec["commands"]["hivemind_weekend_worker"]


def test_weekend_raid_gateway_endpoint_calendar_ics_and_worker_cli(tmp_path: Path) -> None:
    """Gateway `/v1/hivemind/opentrainer/crowdgpt-weekend-raid`, `.ics` export, and worker CLI all pass end-to-end."""
    ics_out = tmp_path / "weekend.ics"
    ics_txt = generate_crowdgpt_weekend_ics_calendar(output_path=ics_out)
    assert ics_out.exists()
    assert ics_txt.count("BEGIN:VEVENT") == len(CROWDGPT_WEEKEND_NOV14_15_EVENTS) == 6
    assert "20261114T" in ics_txt and "20261115T" in ics_txt

    manifest = build_weekend_nov14_15_raid_manifest()
    assert manifest["gcp_spot_t4_budget_governance"]["within_budget"] is True
    assert manifest["gcp_spot_t4_budget_governance"]["projected_total_monthly_usd"] <= 150.0

    store = RouterStore(tmp_path / "raid_gateway.db", b"s" * 32)
    app = create_router_app(store)
    client = TestClient(app)

    r_raid = client.get("/v1/hivemind/opentrainer/crowdgpt-weekend-raid?handle=vxtzq&role=prosumer&vram_mib=8192")
    assert r_raid.status_code == 200, r_raid.text
    raid_json = r_raid.json()
    assert raid_json["campaign_id"] == CROWDGPT_WEEKEND_CAMPAIGN_ID
    assert raid_json["personalized_assignment"]["handle"] == "vxtzq"

    reg = client.post("/v1/hivemind/register", json={"handle": "weekend-trainer-1", "role": "trainer"}).json()
    r_sub = client.post(
        "/v1/hivemind/opentrainer/submit-delta",
        headers={"Authorization": f"Bearer {reg['api_key']}"},
        json={
            "round_id": 1,
            "campaign_id": CROWDGPT_WEEKEND_CAMPAIGN_ID,
            "inner_steps": 8,
            "loss_before": 2.65,
            "loss_after": 2.31,
            "delta_norm": 0.042,
            "delta_sha256": "a" * 64,
            "vram_free_mib": 2048,
            "vram_total_mib": 8192,
        },
    )
    assert r_sub.status_code == 200, r_sub.text
    assert r_sub.json()["accepted"] is True

    worker_mod = _load_script("gcp_t4_crowd_diloco_worker_test", "scripts/gcp_t4_crowd_diloco_worker.py")
    rc_worker = worker_mod.main(
        [
            "--mode",
            "moe-opendiloco-round",
            "--small-arch",
            "--cpu-only",
            "--rounds",
            "1",
            "--inner-steps",
            "3",
            "--seq-len",
            "32",
            "--work-dir",
            str(tmp_path / "moe_cli_work"),
        ]
    )
    assert rc_worker == 0
    assert (tmp_path / "moe_cli_work" / "crowd_moe_opendiloco_round_1.safetensors").exists()

    pub_mod = _load_script("publish_crowdgpt_weekend_raid_kit_test", "scripts/publish_crowdgpt_weekend_raid_kit.py")
    assert pub_mod.main(["--dry-run", "--print-urls"]) == 0


def test_moe_inner_steps_stop_as_soon_as_the_yield_guard_says_so() -> None:
    """The weekend T4 round must honour the HOLD file, live traffic, Spot preemption and the VRAM floor at every step."""
    from types import SimpleNamespace

    model = _build_small_crowd_model(seed=20261115)
    upcycle_crowd_v1_to_sparse_moe(model, split_mode="lora_domain_experts", num_experts=8, top_k=2, lora_rank=4,
                                   routing_mode="sigma_topk")
    stream = torch.tensor([(i * 5 + 7) % 64 for i in range(16)], dtype=torch.long).repeat(16)
    calls = {"n": 0}

    class _Guard:
        def check(self):
            calls["n"] += 1
            return SimpleNamespace(should_yield=calls["n"] > 2, trigger="operator_hold_file")

    out = run_crowd_moe_opendiloco_inner_steps(model, stream, steps=5, batch_size=2, seq_len=32, lr=3e-3, yield_guard=_Guard())
    assert out["steps_completed"] == 2 and out["yielded"] == "operator_hold_file"
    free = run_crowd_moe_opendiloco_inner_steps(model, stream, steps=3, batch_size=2, seq_len=32, lr=3e-3)
    assert free["steps_completed"] == 3 and free["yielded"] is None


def test_weekend_t4_schedule_ends_after_the_weekend_and_the_daily_window_comes_back(tmp_path: Path) -> None:
    """The weekend Spot schedule must not fire again next November, and the daily window must be restorable (cost rule)."""
    import os
    import subprocess

    log = tmp_path / "gcloud.log"
    fake = tmp_path / "bin" / "gcloud"
    fake.parent.mkdir()
    # 'describe' of a missing policy fails; of the daily window policy succeeds; every call is logged
    fake.write_text(
        "#!/bin/bash\n"
        f'echo "$*" >> {log}\n'
        'if [[ "$*" == *"resource-policies describe hivemind-t4-weekend-nov14-15"* ]]; then exit 1; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake.parent}:{os.environ['PATH']}", "CONFIRM": "yes"}
    script = Path(__file__).resolve().parents[1] / "scripts" / "t4_training_window.sh"

    assert subprocess.run(["bash", str(script), "weekend-raid"], env=env, capture_output=True, text=True, timeout=30).returncode == 0
    create = next(line for line in log.read_text().splitlines() if "create instance-schedule hivemind-t4-weekend-nov14-15" in line)
    assert "--end-date 2026-11-15T23:00:00Z" in create and "--initiation-date 2026-11-13T23:00:00Z" in create

    log.write_text("")
    assert subprocess.run(["bash", str(script), "weekend-raid-end"], env=env, capture_output=True, text=True, timeout=30).returncode == 0
    calls = log.read_text().splitlines()
    assert any("remove-resource-policies" in c and "hivemind-t4-weekend-nov14-15" in c for c in calls)
    assert any("add-resource-policies" in c and "hivemind-t4-training-window" in c for c in calls)
    no_confirm = {k: v for k, v in env.items() if k != "CONFIRM"}
    assert subprocess.run(["bash", str(script), "weekend-raid-end"], env=no_confirm, capture_output=True, text=True, timeout=30).returncode == 2


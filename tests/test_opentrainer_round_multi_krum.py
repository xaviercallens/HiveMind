# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The OpenTrainer round close applies Multi-Krum before the outer step (it used to average every accepted delta)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from hivemind.swarm import federation_agents as fed  # noqa: E402
from hivemind.swarm.amcp_kafka_transport import sign_envelope  # noqa: E402
from hivemind.swarm.opentrainer_orchestrator import (  # noqa: E402
    PIN_FILES,
    OpenTrainerConfig,
    PermanentOpenTrainerOrchestrator,
    multi_krum_round_selection,
    resolve_opentrainer_keyring,
)


def _helpers():
    spec = importlib.util.spec_from_file_location("_ot_orch_tests", REPO_ROOT / "tests" / "test_opentrainer_orchestrator.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_selection_excludes_an_outlier_when_enough_deltas() -> None:
    torch.manual_seed(0)
    honest = [{"w": torch.randn(64) * 0.01 + 0.1} for _ in range(4)]
    bad = {"w": torch.full((64,), -3.0)}
    ids = ["a", "b", "c", "d", "evil"]
    out = multi_krum_round_selection(ids, honest + [bad], [100] * 5, f_byzantine=1)
    assert out["active"] is True and out["excluded"] == ["evil"]
    assert set(out["selected"]) == {"a", "b", "c", "d"} and set(out["scores"]) == set(ids)


def test_selection_below_the_minimum_keeps_all_and_says_why() -> None:
    out = multi_krum_round_selection(["a", "b"], [{"w": torch.zeros(4)}, {"w": torch.ones(4)}], [1, 1], f_byzantine=1)
    assert out["active"] is False and out["selected"] == ["a", "b"] and out["excluded"] == []
    assert "2 accepted deltas < 2f+3 = 5" in out["reason"]
    assert multi_krum_round_selection(["a"] * 0 + ["x"], [{"w": torch.zeros(1)}], [1], f_byzantine=0)["reason"] == "byzantine_f is 0"


def _submit_round(orch, cfg, keyring, honest_ids, poisoned_id=None, scale=-8.0):
    h = _helpers()
    offer_env = orch.get_signed_offer_envelope()
    payload = offer_env["data"]["payload"]
    last = None
    for idx, lid in enumerate(honest_ids):
        orch.submit_claim_envelope(sign_envelope(fed.build_claim(run_id=cfg.run_id, rnd=payload["round"], learner_id=lid,
                                                                 host=f"host-{lid}"), keyring, ttl_s=900.0))
        last = orch.submit_result_envelope(h._compute_learner_result_envelope(
            offer_env, learner_id=lid, learner_index=idx, keyring=keyring, host=f"host-{lid}"))
        assert last["accepted"] is True, last
    if poisoned_id is not None:
        honest_env = h._compute_learner_result_envelope(offer_env, learner_id=poisoned_id, learner_index=0, keyring=keyring,
                                                        host=f"host-{poisoned_id}")
        delta = fed.decode_tensors(honest_env["data"]["payload"]["delta_b64"])
        assert delta is not None, "federation_agents.decode_tensors is needed to plant a poisoned delta"
        poisoned = {k: v * scale for k, v in delta.items()}
        l2 = float(torch.sqrt(sum((v.float() ** 2).sum() for v in poisoned.values())))
        assert l2 < cfg.max_delta_l2, f"poisoned delta must stay under the L2 cap to test Multi-Krum ({l2:.2f})"
        orch.submit_claim_envelope(sign_envelope(fed.build_claim(run_id=cfg.run_id, rnd=payload["round"],
                                                                 learner_id=poisoned_id, host=f"host-{poisoned_id}"),
                                                 keyring, ttl_s=900.0))
        res_ev = fed.build_result(run_id=cfg.run_id, rnd=int(payload["round"]), learner_id=poisoned_id,
                                  host=f"host-{poisoned_id}", tokens=int(honest_env["data"]["payload"]["tokens"]),
                                  delta=poisoned, delta_l2=l2, eval_loss_global=1.0, pin=fed.code_pin(PIN_FILES),
                                  global_sha256=str(payload["global_sha256"]), vram_peak_mib=0.0)
        last = orch.submit_result_envelope(sign_envelope(res_ev, keyring, ttl_s=900.0))
        assert last["accepted"] is True, last
    return last


def test_round_close_excludes_a_poisoned_learner_under_the_l2_cap(tmp_path: Path) -> None:
    keyring = resolve_opentrainer_keyring()
    learners = ("l0", "l1", "l2", "l3", "evil")
    cfg = OpenTrainerConfig(run_id="test-krum", learners=learners, quorum=5, rounds=1, seed=1, inner_steps=5,
                            shard_size=16, cpu_sentinel_when_t4_offline=False, byzantine_f=1,
                            checkpoint_path=tmp_path / "s.pt", report_path=tmp_path / "r.json")
    orch = PermanentOpenTrainerOrchestrator(cfg, keyring=keyring)
    _submit_round(orch, cfg, keyring, ["l0", "l1", "l2", "l3"], poisoned_id="evil")
    entry = orch.build_report()["rounds"][0]
    assert entry["byzantine_filter"]["active"] is True
    assert "evil" in entry["byzantine_filter"]["excluded"] and "evil" not in entry["aggregated"]
    assert set(entry["accepted"]) == set(learners)


def test_small_round_is_recorded_unprotected_or_refused(tmp_path: Path) -> None:
    keyring = resolve_opentrainer_keyring()
    base = dict(learners=("l0", "l1"), quorum=2, rounds=1, seed=1, inner_steps=5, shard_size=16,
                cpu_sentinel_when_t4_offline=False)
    orch = PermanentOpenTrainerOrchestrator(OpenTrainerConfig(run_id="test-small", **base,
                                                              checkpoint_path=tmp_path / "a.pt"), keyring=keyring)
    _submit_round(orch, orch.config, keyring, ["l0", "l1"])
    entry = orch.build_report()["rounds"][0]
    assert entry["byzantine_filter"]["active"] is False and entry["aggregated"] == ["l0", "l1"]

    strict = PermanentOpenTrainerOrchestrator(OpenTrainerConfig(run_id="test-strict", **base, checkpoint_path=tmp_path / "b.pt",
                                                                refuse_below_multi_krum_minimum=True), keyring=keyring)
    last = _submit_round(strict, strict.config, keyring, ["l0", "l1"])
    assert last["round_advanced"] is False and strict.build_report()["rounds"] == []
    assert strict.current_round == 0

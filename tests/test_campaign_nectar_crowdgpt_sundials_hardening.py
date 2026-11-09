"""Negative-control and positive validation tests for campaign_orchestrator, swarm_nectar_cache,
crowdgpt_loss_check, and rusty_sundials_scientific_bee input hardening."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hivemind.swarm.campaign_orchestrator import (
    CampaignOrchestrator,
    CampaignRequest,
    DomainEvidence,
    DreamPolicy,
    _membership_key,
    _producers,
    _seed,
    _select_domains,
    check_governor_state,
    check_hold_file,
)
from hivemind.swarm.crowdgpt_loss_check import (
    LossCheckReceipt,
    _replace,
    _rope,
    _rotate_half,
    cpu_recompute_fn,
    make_receipt,
    verify_receipt,
)
from hivemind.swarm.rusty_sundials_scientific_bee import _rk4_van_der_pol_step
from hivemind.swarm.swarm_nectar_cache import (
    FileNectarPersistence,
    InMemoryClusterNectarHub,
    MultiNodeNectarSyncBackend,
    NectarEntry,
    SwarmNectarCache,
)


def test_campaign_orchestrator_hardening(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="data must be a Mapping"):
        CampaignRequest.from_dict([])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="campaign_id and participant_id must be non-empty strings"):
        _membership_key("", "p1")

    assert _membership_key("camp-1", "p1") == "camp-1|p1"

    with pytest.raises(ValueError, match="night_date, domain, and salt must be non-empty strings"):
        _seed("", "math", "salt")

    assert isinstance(_seed("2026-11-10", "math", "s1"), int)

    pol = DreamPolicy()
    with pytest.raises(TypeError, match="policy must be a DreamPolicy"):
        _select_domains([], "bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ledger must be a sequence of DomainEvidence"):
        _select_domains(["not-evidence"], pol)  # type: ignore[list-item]

    ev = DomainEvidence(domain="math", verified_accept=300, verified_reject=10, unverified=20, heldout_eval_items=350)
    considered, picked, qualified = _select_domains([ev], pol)
    assert len(picked) == 1 and qualified == ["math"]

    with pytest.raises(TypeError, match="policy must be a DreamPolicy"):
        _producers([], "bad", {})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="campaigns must be a Mapping"):
        _producers([], pol, [])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="hold_file must be a Path, string, or None"):
        check_hold_file(123)  # type: ignore[arg-type]
    assert check_hold_file(tmp_path / "no_hold") == ""

    with pytest.raises(ValueError, match="now_utc must be a finite number"):
        check_governor_state(tmp_path / "gov.json", float("nan"), 900.0)
    with pytest.raises(TypeError, match="path must be a Path, string, or None"):
        check_governor_state(123, 1000.0, 900.0)  # type: ignore[arg-type]

    orch = CampaignOrchestrator(
        state_path=tmp_path / "state.json",
        secret=b"0123456789abcdef0123456789abcdef",
        publish=lambda e: {"offset": 1},
    )
    with pytest.raises(TypeError, match="request must be a CampaignRequest"):
        orch.handle("bad", 1000.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="now_utc must be a finite number"):
        req = CampaignRequest.from_dict(
            {
                "participant_id": "alice",
                "participant_kind": "gpu_producer",
                "campaign_id": "camp-1",
                "issued_at_utc": 1000.0,
                "nonce": "nonce_12345",
            }
        )
        orch.handle(req, True)  # type: ignore[arg-type]


def test_swarm_nectar_cache_hardening(tmp_path: Path) -> None:
    entry = NectarEntry(
        nectar_id="nec-1",
        tool_name="lean4",
        query_hash="abcd" * 16,
        content={"proof": "rfl"},
        nectar_richness=0.95,
        discovered_by_bee="bee-1",
    )
    with pytest.raises(ValueError, match="credit_royalty must be a boolean"):
        entry.record_access(credit_royalty=1)  # type: ignore[arg-type]
    entry.record_access(credit_royalty=True)
    assert entry.access_count == 1

    with pytest.raises(TypeError, match="data must be a Mapping"):
        NectarEntry.from_dict("bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="path must be a non-empty string or Path"):
        FileNectarPersistence("   ")

    fp = FileNectarPersistence(tmp_path / "nectar.json")
    with pytest.raises(TypeError, match="entries must be a Mapping"):
        fp.save([])  # type: ignore[arg-type]

    hub = InMemoryClusterNectarHub()
    with pytest.raises(ValueError, match="key must be a non-empty string"):
        hub.get("")
    with pytest.raises(ValueError, match="key must be a non-empty string"):
        hub.delete("   ")

    backend = MultiNodeNectarSyncBackend(kv_client=hub, namespace="ns")
    with pytest.raises(ValueError, match="query_hash must be a non-empty string"):
        backend._key("  ")
    with pytest.raises(TypeError, match="entry must be a NectarEntry"):
        backend.put_entry("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="query_hash must be a non-empty string"):
        backend.get_entry("")
    with pytest.raises(ValueError, match="query_hash must be a non-empty string"):
        backend.delete_entry("")

    backend.put_entry(entry)
    fetched = backend.get_entry(entry.query_hash)
    assert fetched is not None and fetched.query_hash == entry.query_hash
    backend.delete_entry(entry.query_hash)
    assert backend.get_entry(entry.query_hash) is None

    cache = SwarmNectarCache()
    with pytest.raises(TypeError, match="entry must be a NectarEntry"):
        cache.build_kafka_food_source_event("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="entry must be a NectarEntry"):
        cache.build_kafka_telepathy_kv_event("bad")  # type: ignore[arg-type]


def test_crowdgpt_loss_check_hardening() -> None:
    with pytest.raises(ValueError, match="head_dim must be an even integer >= 2"):
        _rope(7, 4, torch.device("cpu"))
    with pytest.raises(ValueError, match="seq_len must be a positive integer"):
        _rope(8, 0, torch.device("cpu"))

    cos, sin = _rope(8, 4, torch.device("cpu"))
    assert cos.shape == (1, 1, 4, 8) and sin.shape == (1, 1, 4, 8)

    with pytest.raises(ValueError, match="x must be a Tensor with an even last dimension >= 2"):
        _rotate_half([1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="x must be a Tensor with an even last dimension >= 2"):
        _rotate_half(torch.zeros(3))

    rotated = _rotate_half(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(rotated, torch.tensor([-3.0, -4.0, 1.0, 2.0]))

    with pytest.raises(TypeError, match="r must be a LossCheckReceipt"):
        _replace("not-a-receipt", round_id="r2")  # type: ignore[arg-type]

    r = LossCheckReceipt(
        evaluator_id="eval-1",
        base_sha256="a" * 64,
        candidate_sha256="b" * 64,
        heldout_sha256="c" * 64,
        window=4,
        seed=7,
        starts=(0, 4),
        losses_before=(2.1, 2.2),
        losses_after=(2.0, 2.1),
        compute_dtype="float32",
        device="cpu",
    )
    r2 = _replace(r, round_id="round-9")
    assert r2.round_id == "round-9"

    heldout = np.arange(32, dtype=np.int64)
    base_t = torch.zeros(8)
    cand_t = torch.ones(8)

    with pytest.raises(TypeError, match="model_factory must be callable"):
        make_receipt(
            evaluator_id="e",
            key=b"secret",
            base=base_t,
            candidate=cand_t,
            heldout=heldout,
            window=4,
            n=2,
            seed=1,
            model_factory="not-callable",  # type: ignore[arg-type]
            device="cpu",
        )

    with pytest.raises(ValueError, match="batch must be an integer >= 1"):
        make_receipt(
            evaluator_id="e",
            key=b"secret",
            base=base_t,
            candidate=cand_t,
            heldout=heldout,
            window=4,
            n=2,
            seed=1,
            model_factory=lambda: torch.nn.Linear(2, 2),
            device="cpu",
            batch=0,
        )

    with pytest.raises(TypeError, match="receipt must be a LossCheckReceipt"):
        verify_receipt(
            "bad",  # type: ignore[arg-type]
            evaluator_keys={"eval-1": b"k"},
            server_secret=b"s",
            expected_base_sha256="a" * 64,
            expected_candidate_sha256="b" * 64,
            heldout=heldout,
            recompute=lambda w, s: [2.0] * len(s),
        )

    with pytest.raises(ValueError, match="min_windows must be an integer >= 1"):
        verify_receipt(
            r,
            evaluator_keys={"eval-1": b"k"},
            server_secret=b"s",
            expected_base_sha256="a" * 64,
            expected_candidate_sha256="b" * 64,
            heldout=heldout,
            recompute=lambda w, s: [2.0] * len(s),
            min_windows=0,
        )

    with pytest.raises(TypeError, match="model_factory must be callable"):
        cpu_recompute_fn("bad", base_t, cand_t, heldout, 4)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="base and candidate must be torch.Tensor instances"):
        cpu_recompute_fn(lambda: torch.nn.Linear(2, 2), [1.0], cand_t, heldout, 4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="window must be an integer >= 2"):
        cpu_recompute_fn(lambda: torch.nn.Linear(2, 2), base_t, cand_t, heldout, 1)


def test_rk4_van_der_pol_step_hardening() -> None:
    x1, v1 = _rk4_van_der_pol_step(1.0, 0.0, 1.0, 0.01)
    assert np.isfinite(x1) and np.isfinite(v1)

    with pytest.raises(ValueError, match="x, v, mu, and h must be finite numbers"):
        _rk4_van_der_pol_step(True, 0.0, 1.0, 0.01)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="x, v, mu, and h must be finite numbers"):
        _rk4_van_der_pol_step(float("nan"), 0.0, 1.0, 0.01)
    with pytest.raises(ValueError, match="h must be > 0 and mu must be >= 0"):
        _rk4_van_der_pol_step(1.0, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="h must be > 0 and mu must be >= 0"):
        _rk4_van_der_pol_step(1.0, 0.0, -0.1, 0.01)

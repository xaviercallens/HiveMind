"""Engram support in the CrowdGPT loss check: current Crowd-v1 adds a bigram hash table to the token embedding
(client.py `EngramMemory`, `ENGRAM_ACTIVE` on by default). A check that ignores it measures a different model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hivemind.swarm import crowdgpt_loss_check as lc  # noqa: E402

TINY = {"vocabSize": 97, "dim": 32, "nLayers": 2, "nHeads": 4, "nKvHeads": 2, "headDim": 8, "maxSeqLen": 64, "mlpHidden": 48,
        "weightTying": True}
TINY_ENG = {**TINY, "engramBuckets": 211}
KEY, SECRET = b"evaluator-key", b"backend-secret"


def trunk_and_engram(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    m = lc.CrowdV1(TINY_ENG)
    trunk = torch.cat([p.detach().reshape(-1) for p in m._trunk_parameters()])
    eng = torch.randn(TINY_ENG["engramBuckets"] * TINY["dim"]) * 0.5
    return trunk, eng


def client_engram_rows(idx: torch.Tensor, buckets: int) -> torch.Tensor:
    """Reference: the hashing in CrowdGPT client.py `EngramMemory.forward`."""
    px = torch.cat([torch.zeros_like(idx[:, :1]), idx[:, :-1]], dim=1)
    return (px * 1000003 + idx) % buckets


def test_constants_match_crowd_v1_config() -> None:
    assert lc.CROWD_V1_ENGRAM_SIZE == 350_000_640  # engramParams in Vxtzq/Crowd-v1 config.json
    assert lc.CROWD_V1_ENGRAM_CONFIG["engramBuckets"] == 227_865
    assert lc.CrowdV1.__init__.__defaults__[0] is lc.CROWD_V1_CONFIG  # default stays trunk-only (backwards compatible)


def test_engram_hash_matches_client_and_is_added_to_the_embedding() -> None:
    trunk, eng = trunk_and_engram(1)
    m = lc.CrowdV1(TINY_ENG)
    m.load_flat(trunk, engram=eng)
    idx = torch.randint(0, TINY["vocabSize"], (2, 9))
    rows = client_engram_rows(idx, TINY_ENG["engramBuckets"])
    assert torch.equal(m.engram(idx), m.engram.table.weight[rows])
    plain = lc.CrowdV1(TINY)
    plain.load_flat(trunk)
    with torch.inference_mode():
        assert not torch.allclose(m(idx), plain(idx))  # the engram changes the logits


def test_engram_free_config_still_loads_trunk_only_and_rejects_stray_engram() -> None:
    trunk, eng = trunk_and_engram(2)
    m = lc.CrowdV1(TINY)
    m.load_flat(trunk)
    assert m.n_flat() == trunk.numel()
    with pytest.raises(ValueError):
        m.load_flat(trunk, engram=eng)
    with pytest.raises(ValueError):
        lc.CrowdV1(TINY_ENG).load_flat(trunk)  # engram config without engram weights
    with pytest.raises(ValueError):
        lc.CrowdV1({**TINY, "engramBuckets": -1})


def test_concatenated_trunk_and_engram_equals_separate_loading() -> None:
    trunk, eng = trunk_and_engram(3)
    a, b = lc.CrowdV1(TINY_ENG), lc.CrowdV1(TINY_ENG)
    a.load_flat(trunk, engram=eng)
    b.load_flat(torch.cat((trunk, eng)))
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.equal(p, q)
    with pytest.raises(ValueError):
        lc.CrowdV1(TINY_ENG).load_flat(torch.cat((trunk, eng[:-1])))


def test_receipt_binds_the_engram_and_verifies_end_to_end() -> None:
    trunk, eng = trunk_and_engram(4)
    base = torch.cat((trunk, eng))
    cand = base.clone()
    cand[trunk.numel():] += 0.05  # a round that only moved the engram table
    assert lc.weights_sha256(base) != lc.weights_sha256(cand)
    held = np.random.default_rng(0).integers(0, TINY["vocabSize"], size=400, dtype=np.uint32)
    factory = lambda: lc.CrowdV1(TINY_ENG)  # noqa: E731
    r = lc.make_receipt(evaluator_id="t4", key=KEY, base=base, candidate=cand, heldout=held, window=17, n=8, seed=5,
                        model_factory=factory, device="cpu")
    assert r.losses_before != r.losses_after  # engram-only change is visible to the check
    v = lc.verify_receipt(r, evaluator_keys={"t4": KEY}, server_secret=SECRET, expected_base_sha256=lc.weights_sha256(base),
                          expected_candidate_sha256=lc.weights_sha256(cand), heldout=held,
                          recompute=lc.cpu_recompute_fn(factory, base, cand, held, 17), k=4, tol=1e-4,
                          max_regression_tolerance=10.0, min_windows=8)
    assert v.accepted, v.reason


def test_load_crowd_v1_with_engram_checks_size(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    save_file({"weights": torch.zeros(10, dtype=torch.bfloat16)}, str(tmp_path / "m.safetensors"))
    save_file({"weights": torch.zeros(7, dtype=torch.bfloat16)}, str(tmp_path / "e.safetensors"))
    with pytest.raises(ValueError):
        lc.load_crowd_v1_with_engram(tmp_path / "m.safetensors", tmp_path / "e.safetensors")

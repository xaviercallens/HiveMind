# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Crowd-v1 MoE add-on serving: manifest checks, load order, read-back, zero-delta equivalence, model IDs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from safetensors.torch import save_file
import torch

from hivemind.swarm.crowd_v1_moe_addon import (
    MANIFEST_SCHEMA,
    AddonIntegrityError,
    apply_moe_addon,
    build_crowd_v1_moe_model,
    load_addon_manifest,
    parse_addon_manifest,
    pinned_base_urls,
    sha256_file,
    verify_file_sha256,
)
from hivemind.swarm.crowdgpt_live_runtime import (
    ADVERTISED_CROWD_MODEL_IDS,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
    create_crowdgpt_producer_app,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import extract_moe_trainable_buckets, upcycle_crowd_v1_to_sparse_moe

REPO = Path(__file__).resolve().parent.parent
REPO_MANIFEST = REPO / "deploy/crowd_v1_moe/e50_r00_manifest.json"
N_LAYERS = 2


def _tiny_cfg() -> CrowdV1RuntimeConfig:
    return CrowdV1RuntimeConfig(vocab_size=97, hidden_size=32, num_hidden_layers=N_LAYERS, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=8, intermediate_size=48, max_position_embeddings=64,
                                engram_buckets=127, tie_word_embeddings=True)


def _split(**over) -> dict:
    s = {"split_mode": "lora_domain_experts", "num_experts": 8, "top_k": 2, "routing_mode": "sigma_topk",
         "lora_rank": 4, "lora_alpha": 16.0, "resident_expert_indices": list(range(8)), "num_router_fragments": 2,
         "num_layers_upcycled": N_LAYERS}
    s.update(over)
    return s


def _manifest_dict(addon_sha: str = "a" * 64, **over) -> dict:
    d = {"schema": MANIFEST_SCHEMA, "addon_id": "test-addon", "addon_file": "addon.safetensors", "addon_sha256": addon_sha,
         "base": {"hf_repo": "Vxtzq/Crowd-v1", "hf_revision": "b" * 40, "weights_sha256": "c" * 64,
                  "tokenizer_sha256": "d" * 64},
         "split": _split(), "advertised_model_ids": ["hivemind/crowd-v1-moe-test"]}
    d.update(over)
    return d


def _dense() -> CrowdV1WithEngram:
    torch.manual_seed(7)
    m = CrowdV1WithEngram(_tiny_cfg())
    m.eval()
    return m


def _write_addon(tmp_path: Path, dense: CrowdV1WithEngram, *, perturb: bool) -> Path:
    """Buckets of an upcycled copy of ``dense``: the zero-delta start, or a random perturbation of it."""
    m = copy.deepcopy(dense)
    sp = _split()
    upcycle_crowd_v1_to_sparse_moe(m, split_mode=sp["split_mode"], num_experts=8, top_k=2, lora_rank=sp["lora_rank"],
                                   lora_alpha=sp["lora_alpha"], routing_mode=sp["routing_mode"],
                                   resident_expert_indices=tuple(range(8)))
    b = extract_moe_trainable_buckets(m, num_router_fragments=sp["num_router_fragments"])
    if perturb:
        g = torch.Generator().manual_seed(11)
        b = {k: v + 0.05 * torch.randn(v.shape, generator=g) for k, v in b.items()}
    p = tmp_path / "addon.safetensors"
    save_file({k: v.contiguous() for k, v in b.items()}, str(p))
    return p


def _logits(m: torch.nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return m(torch.tensor([[3, 5, 7, 11, 13, 17]]))[0]


def test_repo_manifest_parses_and_pins_the_judged_base() -> None:
    m = load_addon_manifest(REPO_MANIFEST)
    assert m.base_hf_revision == "b63212a843d662dadbec35f8544b6e0dafa6e72b"
    assert m.base_weights_sha256.startswith("1f8214ca")
    assert m.addon_sha256.startswith("80dccfa8")
    assert m.split["num_layers_upcycled"] == 24 and m.split["lora_rank"] == 8
    assert not set(m.advertised_model_ids) & set(ADVERTISED_CROWD_MODEL_IDS)
    urls = pinned_base_urls(m)
    assert all("/resolve/b63212a843d662dadbec35f8544b6e0dafa6e72b/" in u for u in urls.values())
    assert json.loads(REPO_MANIFEST.read_text())["evidence"]["not_measured"]


@pytest.mark.parametrize(
    "over, msg",
    [
        ({"schema": "other"}, "schema"),
        ({"addon_sha256": "xyz"}, "addon_sha256"),
        ({"advertised_model_ids": ["hivemind/crowd-v1"]}, "base model"),
        ({"advertised_model_ids": []}, "advertised_model_ids"),
        ({"addon_file": "../evil.safetensors"}, "plain file name"),
        ({"split": _split(split_mode="fine_grained_width_split")}, "lora_domain_experts"),
        ({"split": {k: v for k, v in _split().items() if k != "lora_alpha"}}, "missing"),
    ],
)
def test_manifest_rejects_bad_fields(over: dict, msg: str) -> None:
    with pytest.raises(AddonIntegrityError, match=msg):
        parse_addon_manifest(_manifest_dict(**over))


def test_manifest_rejects_unpinned_revision() -> None:
    d = _manifest_dict()
    d["base"]["hf_revision"] = "main"
    with pytest.raises(AddonIntegrityError, match="hf_revision"):
        parse_addon_manifest(d)


def test_sha_mismatch_refuses(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"x")
    assert verify_file_sha256(p, sha256_file(p), "f") == sha256_file(p)
    with pytest.raises(AddonIntegrityError, match="does not match"):
        verify_file_sha256(p, "0" * 64, "f")


def test_addon_loads_reads_back_and_changes_the_output(tmp_path: Path) -> None:
    dense = _dense()
    addon = _write_addon(tmp_path, dense, perturb=True)
    man = parse_addon_manifest(_manifest_dict(addon_sha=sha256_file(addon)))
    model = copy.deepcopy(dense)
    rep = apply_moe_addon(model, man, addon)
    assert rep["num_layers_upcycled"] == N_LAYERS
    assert rep["readback_max_abs_diff"] <= 1e-6
    assert rep["buckets"] == sorted(["router_frag_0", "router_frag_1"] + [f"expert_{i}" for i in range(8)])
    assert not any(p.requires_grad for p in model.parameters())
    assert not torch.allclose(_logits(model), _logits(dense), atol=1e-4)


def test_zero_delta_addon_equals_dense(tmp_path: Path) -> None:
    dense = _dense()
    addon = _write_addon(tmp_path, dense, perturb=False)
    man = parse_addon_manifest(_manifest_dict(addon_sha=sha256_file(addon)))
    model = copy.deepcopy(dense)
    apply_moe_addon(model, man, addon)
    assert torch.allclose(_logits(model), _logits(dense), atol=1e-5)


def test_refuses_wrong_sha_layers_shapes_and_double_upcycle(tmp_path: Path) -> None:
    dense = _dense()
    addon = _write_addon(tmp_path, dense, perturb=True)
    sha = sha256_file(addon)
    with pytest.raises(AddonIntegrityError, match="does not match"):
        apply_moe_addon(copy.deepcopy(dense), parse_addon_manifest(_manifest_dict()), addon)
    with pytest.raises(AddonIntegrityError, match="layers"):
        apply_moe_addon(copy.deepcopy(dense), parse_addon_manifest(_manifest_dict(sha, split=_split(num_layers_upcycled=24))),
                        addon)
    with pytest.raises(AddonIntegrityError, match="shape"):
        apply_moe_addon(copy.deepcopy(dense), parse_addon_manifest(_manifest_dict(sha, split=_split(lora_rank=2))), addon)
    model = copy.deepcopy(dense)
    man = parse_addon_manifest(_manifest_dict(sha))
    apply_moe_addon(model, man, addon)
    with pytest.raises(TypeError, match="already"):
        apply_moe_addon(model, man, addon)


def test_all_zero_experts_refused(tmp_path: Path) -> None:
    dense = _dense()
    p = _write_addon(tmp_path, dense, perturb=False)
    from safetensors.torch import load_file

    b = {k: (torch.zeros_like(v) if k.startswith("expert_") else v) for k, v in load_file(str(p)).items()}
    save_file(b, str(p))
    man = parse_addon_manifest(_manifest_dict(addon_sha=sha256_file(p)))
    with pytest.raises(AddonIntegrityError, match="no-op"):
        apply_moe_addon(copy.deepcopy(dense), man, p)


def test_build_checks_base_sha_and_loads(tmp_path: Path) -> None:
    dense = _dense()
    base = tmp_path / "base.safetensors"
    sd = {k: v.contiguous() for k, v in dense.state_dict().items() if k in dict(dense.named_parameters()) and k != "head.weight"}
    save_file(sd, str(base))
    addon = _write_addon(tmp_path, dense, perturb=True)
    d = _manifest_dict(addon_sha=sha256_file(addon))
    d["base"]["weights_sha256"] = sha256_file(base)
    man = parse_addon_manifest(d)
    m_dense, info_d = build_crowd_v1_moe_model(man, base, None, cfg=_tiny_cfg())
    assert info_d["addon"] is None
    assert torch.allclose(_logits(m_dense), _logits(dense), atol=1e-6)
    m_moe, info = build_crowd_v1_moe_model(man, base, addon, cfg=_tiny_cfg())
    assert info["addon"]["addon_sha256"] == sha256_file(addon)
    d["base"]["weights_sha256"] = "0" * 64
    with pytest.raises(AddonIntegrityError, match="base weights"):
        build_crowd_v1_moe_model(parse_addon_manifest(d), base, addon, cfg=_tiny_cfg())


def test_strict_app_serves_only_its_own_ids(tmp_path: Path) -> None:
    dense = _dense()
    addon = _write_addon(tmp_path, dense, perturb=True)
    man = parse_addon_manifest(_manifest_dict(addon_sha=sha256_file(addon)))
    model = copy.deepcopy(dense)
    apply_moe_addon(model, man, addon)
    model.place_for_inference(device="cpu", dtype=torch.float32, offload_engram_to_cpu=False)
    app = create_crowdgpt_producer_app(
        model, encode_fn=lambda t: [1 + ord(c) % 90 for c in t[:16]] or [1], decode_fn=lambda ids: " ".join(map(str, ids)),
        revision_info={"hf_revision": "b" * 12, "addon_sha256": man.addon_sha256}, sync_hook=None,
        advertised_model_ids=man.advertised_model_ids, model_repo="Vxtzq/Crowd-v1@bbbb+test-addon", strict_model_ids=True)
    with TestClient(app) as c:
        h = c.get("/healthz").json()
        assert h["advertised_models"] == ["hivemind/crowd-v1-moe-test"]
        assert h["model_repo"] == "Vxtzq/Crowd-v1@bbbb+test-addon"
        assert [m["id"] for m in c.get("/v1/models").json()["data"]] == ["hivemind/crowd-v1-moe-test"]
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3})
        assert r.status_code == 200 and r.json()["model"] == "hivemind/crowd-v1-moe-test"
        r = c.post("/v1/chat/completions", json={"model": "hivemind/crowd-v1", "messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"


def test_default_app_keeps_base_ids() -> None:
    app = create_crowdgpt_producer_app(_dense(), encode_fn=lambda t: [1, 2], decode_fn=lambda ids: "x")
    with TestClient(app) as c:
        assert c.get("/healthz").json()["advertised_models"] == list(ADVERTISED_CROWD_MODEL_IDS)
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "max_tokens": 2})
        assert r.json()["model"] == "hivemind/crowd-v1"

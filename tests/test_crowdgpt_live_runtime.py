# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit and integration tests for `crowdgpt_live_runtime`, `crowdgpt_loss_check` Engram support, and shield routing."""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest
from safetensors.torch import save_file
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from hivemind.swarm import crowdgpt_loss_check as lc  # noqa: E402
from hivemind.swarm.crowdgpt_bridge import unpack_crowdgpt_delta_stream  # noqa: E402
from hivemind.swarm.crowdgpt_live_runtime import (  # noqa: E402
    ADVERTISED_CROWD_MODEL_IDS,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
    create_crowdgpt_producer_app,
    execute_multi_gpu_crowdgpt_round,
    load_crowd_v1_safetensors_into_model,
    select_safe_execution_device,
)


def _tiny_cfg(engram_buckets: int = 127) -> CrowdV1RuntimeConfig:
    return CrowdV1RuntimeConfig(
        vocab_size=97,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=48,
        max_position_embeddings=64,
        engram_buckets=engram_buckets,
        tie_word_embeddings=True,
    )


def test_config_validation_and_hf_dict_parsing() -> None:
    cfg = CrowdV1RuntimeConfig.from_hf_dict(
        {
            "vocab_size": 151669,
            "hidden_size": 1536,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 96,
            "intermediate_size": 2560,
            "max_position_embeddings": 2048,
            "engram_buckets": 227865,
            "tie_word_embeddings": True,
        }
    )
    assert cfg.engram_buckets == 227_865
    assert cfg.vocab_size == 151_669

    with pytest.raises(ValueError, match="head_dim must be even"):
        CrowdV1RuntimeConfig(head_dim=7, hidden_size=28, num_attention_heads=4, num_key_value_heads=2)

    with pytest.raises(ValueError, match="divisible"):
        CrowdV1RuntimeConfig(num_attention_heads=5, num_key_value_heads=2)


def test_vram_safety_floors_for_t4_rtx2070_and_ampere() -> None:
    # 1. Tesla T4 (15,360 MiB): 5,000 MiB free - 1,350 MiB = 3,650 MiB < 4,300 MiB floor -> CPU fallback
    t4_tight = select_safe_execution_device(
        prefer_cuda=True,
        estimated_base_vram_mib=1350,
        cuda_available=True,
        vram_info_mib=(5000, 15360),
        compute_capability=(7, 5),
        gpu_name="NVIDIA Tesla T4",
    )
    assert t4_tight.device == "cpu"
    assert t4_tight.vram_safety_floor_mib == 4300
    assert "CPU_FALLBACK_VRAM_GUARD" in t4_tight.reason

    # 2. Tesla T4 with 6,500 MiB free -> 5,150 MiB remaining >= 4,300 MiB -> CUDA approved (fp16)
    t4_ok = select_safe_execution_device(
        prefer_cuda=True,
        estimated_base_vram_mib=1350,
        cuda_available=True,
        vram_info_mib=(6500, 15360),
        compute_capability=(7, 5),
        gpu_name="NVIDIA Tesla T4",
    )
    assert t4_ok.device == "cuda"
    assert t4_ok.dtype_name == "float16"
    assert t4_ok.cohort_key == "nvidia:sm75_pre_ampere:fp32:adamw_8bit"

    # 3. Local RTX 2070 (8,192 MiB) when warm Ollama leaves 2,041 MiB free -> CPU fallback protecting >= 1,536 MiB
    rtx_busy = select_safe_execution_device(
        prefer_cuda=True,
        estimated_base_vram_mib=1350,
        cuda_available=True,
        vram_info_mib=(2041, 8192),
        compute_capability=(7, 5),
        gpu_name="NVIDIA GeForce RTX 2070",
    )
    assert rtx_busy.device == "cpu"
    assert rtx_busy.vram_safety_floor_mib == 1536

    # 4. Local RTX 2070 with 3,400 MiB free -> 2,050 MiB remaining >= 1,536 MiB -> CUDA approved
    rtx_free = select_safe_execution_device(
        prefer_cuda=True,
        estimated_base_vram_mib=1350,
        cuda_available=True,
        vram_info_mib=(3400, 8192),
        compute_capability=(7, 5),
        gpu_name="NVIDIA GeForce RTX 2070",
    )
    assert rtx_free.device == "cuda"
    assert rtx_free.dtype_name == "float16"

    # 5. Ampere/Ada 24 GB GPU -> CUDA bfloat16 cohort
    ampere = select_safe_execution_device(
        prefer_cuda=True,
        estimated_base_vram_mib=1350,
        cuda_available=True,
        vram_info_mib=(20000, 24576),
        compute_capability=(8, 9),
        gpu_name="NVIDIA GeForce RTX 4090",
    )
    assert ampere.device == "cuda"
    assert ampere.dtype_name == "bfloat16"
    assert ampere.cohort_key == "nvidia:sm80_plus_ampere:bf16:adamw_8bit"


def test_kv_cache_and_loss_check_engram_equivalence(tmp_path: Path) -> None:
    torch.manual_seed(1234)
    cfg = _tiny_cfg(engram_buckets=127)
    runtime_model = CrowdV1WithEngram(cfg).train(False)  # inference mode, same as .eval()
    with torch.no_grad():
        for p in runtime_model.parameters():
            if p.ndim >= 2:
                p.mul_(0.02)

    # Save named safetensors matching Vxtzq/Crowd-v1/model.safetensors (without head.weight due to weight tying)
    sd = {k: v.contiguous() for k, v in runtime_model.state_dict().items() if k != "head.weight"}
    st_path = tmp_path / "model.safetensors"
    save_file(sd, str(st_path))

    reloaded = CrowdV1WithEngram(cfg).train(False)  # inference mode, same as .eval()
    sha = load_crowd_v1_safetensors_into_model(reloaded, st_path)
    assert len(sha) == 64

    seq = torch.tensor([[4, 19, 33, 7, 81, 12, 45, 9]], dtype=torch.long)
    with torch.no_grad():
        full_logits, _, _ = reloaded.forward(seq, use_cache=False)

        # Step-by-step KV-cached forward with bigram prev_token must match full sequence logits
        prefill = seq[:, :3]
        prefill_logits, _, kvs = reloaded.forward(prefill, use_cache=True)
        step_logits_list = [prefill_logits]
        for pos in range(3, seq.shape[1]):
            prev_tok = seq[:, pos - 1 : pos]
            cur_tok = seq[:, pos : pos + 1]
            sl, _, kvs = reloaded.forward(cur_tok, past_kvs=kvs, prev_token=prev_tok, use_cache=True)
            step_logits_list.append(sl)
        cached_logits = torch.cat(step_logits_list, dim=1)

    assert torch.allclose(full_logits, cached_logits, atol=1e-5)

    # Verify crowdgpt_loss_check.CrowdV1 with engramBuckets=127 produces the exact same logits!
    flat_trunk, flat_engram = lc.load_named_crowd_v1_safetensors(st_path, n_layers=2, weight_tying=True)
    assert flat_engram is not None
    lc_model = lc.CrowdV1(
        {
            "vocabSize": 97,
            "dim": 32,
            "nLayers": 2,
            "nHeads": 4,
            "nKvHeads": 2,
            "headDim": 8,
            "maxSeqLen": 64,
            "mlpHidden": 48,
            "weightTying": True,
            "engramBuckets": 127,
        }
    ).train(False)  # inference mode, same as .eval()
    lc_model.load_flat(flat_trunk, engram=flat_engram)
    with torch.no_grad():
        lc_logits = lc_model(seq)
    assert torch.allclose(full_logits, lc_logits, atol=1e-5)

    # Also verify held-out loss check receipt generation & spot-check verification with Engram enabled
    heldout = (np.arange(300, dtype=np.uint32) * 7 + 3) % 97
    cand_trunk = (flat_trunk.float() + 0.005 * torch.randn_like(flat_trunk.float())).to(torch.bfloat16)

    def _engram_factory() -> lc.CrowdV1:
        m = lc.CrowdV1(
            {
                "vocabSize": 97,
                "dim": 32,
                "nLayers": 2,
                "nHeads": 4,
                "nKvHeads": 2,
                "headDim": 8,
                "maxSeqLen": 64,
                "mlpHidden": 48,
                "weightTying": True,
                "engramBuckets": 127,
            }
        )
        assert flat_engram is not None
        m.engram.table.weight.data.copy_(flat_engram.view_as(m.engram.table.weight))
        return m

    receipt = lc.make_receipt(
        evaluator_id="rtx2070",
        key=b"eval-secret",
        base=flat_trunk.to(torch.bfloat16),
        candidate=cand_trunk,
        heldout=heldout,
        window=16,
        n=8,
        seed=42,
        model_factory=_engram_factory,
        device="cpu",
        round_id="step-348",
    )
    verdict = lc.verify_receipt(
        receipt,
        evaluator_keys={"rtx2070": b"eval-secret"},
        server_secret=b"hub-secret",
        expected_base_sha256=lc.weights_sha256(flat_trunk),
        expected_candidate_sha256=lc.weights_sha256(cand_trunk),
        heldout=heldout,
        recompute=lc.cpu_recompute_fn(_engram_factory, flat_trunk.to(torch.bfloat16), cand_trunk, heldout, 16),
        k=3,
        tol=1e-3,
        max_regression_tolerance=5.0,
        min_windows=8,
    )
    assert verdict.accepted, verdict.reason


def test_execute_multi_gpu_crowdgpt_round_packs_crosscheck_stream() -> None:
    torch.manual_seed(77)
    cfg = _tiny_cfg(engram_buckets=64)
    model = CrowdV1WithEngram(cfg)
    # Repeating pattern shard so inner AdamW steps measurably reduce next-token loss
    pattern = torch.tensor([3, 14, 15, 9, 26, 53, 58, 96] * 24, dtype=torch.long)

    result = execute_multi_gpu_crowdgpt_round(
        model,
        pattern,
        hiveminder_handle="xav_rtx2070",
        device_suffix="rtx2070",
        task_metadata={
            "taskId": "round-349-cohort-a",
            "globalStep": 348,
            "crossCheck": {
                "enabled": True,
                "targetSeed": 424242,
                "targetBatchSize": 2,
                "targetLocalSteps": 4,
                "targetAccumSteps": 1,
                "targetAutocast": "fp32",
            },
        },
        prefer_cuda=False,
        learning_rate=3e-3,
    )
    assert result["accepted"] is True
    assert result["loss_after"] < result["loss_before"]
    assert result["rms_guard"]["accepted"] is True
    assert result["ratchet_decision"]["accepted"] is True
    assert result["hardware_headers"]["X-Client-Device"] == "hm-xav_rtx2070-rtx2070"
    meta = result["submit_metadata"]
    assert meta["crossCheckUsed"] is True
    assert meta["actualSeed"] == 424242
    assert meta["actualSteps"] == 4
    assert meta["actualBatchSize"] == 2
    assert result["compressed_delta_bytes"] > 64


def test_producer_fastapi_app_and_shield_route_integration() -> None:
    import hivemind_producer_shield

    torch.manual_seed(99)
    cfg = _tiny_cfg(engram_buckets=32)
    model = CrowdV1WithEngram(cfg).train(False)  # inference mode, same as .eval()
    model.place_for_inference(device="cpu", dtype=torch.float32, offload_engram_to_cpu=True)

    sync_calls = [0]

    def _sync_hook() -> dict[str, object]:
        sync_calls[0] += 1
        return {"hf_revision": "59c016e94875", "global_step": 348 + sync_calls[0]}

    producer_app = create_crowdgpt_producer_app(
        model,
        encode_fn=lambda text: [max(1, ord(ch) % 90) for ch in text[:24]] or [1, 2, 3],
        decode_fn=lambda ids: " ".join(f"tok{i}" for i in ids),
        revision_info={"hf_revision": "59c016e94875", "global_step": 348},
        sync_hook=_sync_hook,
    )

    with TestClient(producer_app) as direct:
        h = direct.get("/healthz").json()
        assert h["ok"] is True
        assert "hivemind/crowd-v1" in h["advertised_models"]

        rev = direct.get("/v1/crowdgpt/revision").json()
        assert rev["global_step"] == 349

        models_resp = direct.get("/v1/models").json()
        listed_ids = {m["id"] for m in models_resp["data"]}
        assert set(ADVERTISED_CROWD_MODEL_IDS).issubset(listed_ids)

        chat_resp = direct.post(
            "/v1/chat/completions",
            json={
                "model": "hivemind/crowd-v1",
                "messages": [{"role": "user", "content": "Bonjour CrowdGPT"}],
                "max_tokens": 6,
                "temperature": 0.0,
            },
        )
        assert chat_resp.status_code == 200
        payload = chat_resp.json()
        assert payload["choices"][0]["message"]["content"].startswith("tok")
        assert payload["usage"]["completion_tokens"] == 6
        assert payload["crowdgpt_revision"]["hf_revision"] == "59c016e94875"

        stream_resp = direct.post(
            "/v1/chat/completions",
            json={
                "model": "hf.co/Vxtzq/Crowd-v1",
                "messages": [{"role": "user", "content": "Stream check"}],
                "max_tokens": 4,
                "temperature": 0.0,
                "stream": True,
            },
        )
        assert stream_resp.status_code == 200
        assert "data: [DONE]" in stream_resp.text

    # Verify hivemind_producer_shield routes `hf.co/Vxtzq/Crowd-v1` cleanly alongside Ollama upstream!
    crowd_transport = httpx.ASGITransport(app=producer_app)

    class _MuxTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.host == "crowd-local":
                return await crowd_transport.handle_async_request(request)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"object": "list", "data": [{"id": "qwen2.5-coder:3b", "object": "model"}]})
            return httpx.Response(404, json={"error": "not found"})

    token = "secret-endpoint-token-123456"
    shield_app = hivemind_producer_shield.create_shield_app(
        "http://ollama-local",
        token,
        allowed_models=["qwen2.5-coder:3b"],
        routes={"hf.co/Vxtzq/Crowd-v1": "http://crowd-local"},
        transport=_MuxTransport(),
    )
    with TestClient(shield_app) as shield:
        m_res = shield.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
        assert m_res.status_code == 200
        ids = {item["id"] for item in m_res.json()["data"]}
        assert {"qwen2.5-coder:3b", "hf.co/Vxtzq/Crowd-v1"}.issubset(ids)

        c_res = shield.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": "hf.co/Vxtzq/Crowd-v1",
                "messages": [{"role": "user", "content": "Test via shield"}],
                "max_tokens": 5,
                "temperature": 0.0,
            },
        )
        assert c_res.status_code == 200
        assert c_res.json()["choices"][0]["message"]["content"].startswith("tok")

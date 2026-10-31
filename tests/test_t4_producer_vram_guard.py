# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for PR #3:
- Tesla T4 producer VRAM headroom guard (`probe_model_sizes_mib`, `filter_models_by_vram_budget`)
- Accurate Qwen2.5-Coder-7B-Instruct resident process metadata in `t4_visibility.py`
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx

from hivemind.router.t4_visibility import build_t4_visibility_payload


def _load_producer_agent():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "hivemind_producer_agent.py"
    spec = importlib.util.spec_from_file_location("hivemind_producer_agent_vram_test", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_probe_model_sizes_and_filter_models_by_t4_vram_budget() -> None:
    agent = _load_producer_agent()

    tags_payload = {
        "models": [
            {"name": "qwen2.5-coder:1.5b", "size": 940 * 1024 * 1024},
            {"name": "qwen2.5-coder:7b", "size": 4466 * 1024 * 1024},
            {"name": "deepseek-r1:14b", "size": 8579 * 1024 * 1024},
            {"name": "gpt-oss:20b", "size": 13144 * 1024 * 1024},
        ]
    }
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=tags_payload))
    sizes = agent.probe_model_sizes_mib("http://127.0.0.1:11434", transport=transport)
    assert sizes == {
        "qwen2.5-coder:1.5b": 940,
        "qwen2.5-coder:7b": 4466,
        "deepseek-r1:14b": 8579,
        "gpt-oss:20b": 13144,
    }

    candidate_models = [
        {"id": "coder-lite", "local_name": "qwen2.5-coder:1.5b"},
        {"id": "coder", "local_name": "qwen2.5-coder:7b"},
        {"id": "reasoning", "local_name": "deepseek-r1:14b"},
        {"id": "moe-general", "local_name": "gpt-oss:20b"},
    ]

    # Tesla T4: 15,360 MiB total - 4,838 MiB pinned llama-server - 4,300 MiB required free headroom = 6,222 MiB budget
    kept, dropped = agent.filter_models_by_vram_budget(
        candidate_models,
        sizes,
        gpu_vram_mib=15360,
        pinned_vram_mib=4838,
        min_free_vram_mib=4300,
    )
    assert [m["id"] for m in kept] == ["coder-lite", "coder"]
    assert len(dropped) == 2
    assert any("deepseek-r1:14b" in d for d in dropped)
    assert any("gpt-oss:20b" in d for d in dropped)


def test_t4_visibility_reports_qwen25_coder_7b_for_bee_stars() -> None:
    payload = build_t4_visibility_payload(now_utc=1761868800.0)
    processes = payload["hardware"]["resident_processes"]
    assert any("Qwen2.5-Coder-7B-Instruct + NIGHT-MOE-1 Stars LoRA" in p["name"] for p in processes)
    assert not any("Qwen2.5-1.5B + NIGHT-MOE-1 Stars LoRA" in p["name"] for p in processes)

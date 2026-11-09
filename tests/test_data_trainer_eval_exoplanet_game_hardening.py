# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-path tests for hf_real_trainer, token_training_eval, exoplanet_hunter_cockpit, and exoplanet_hunter_engine."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hivemind.data.hf_real_trainer import (
    train_and_publish_astrophysics_domain_experts,
    train_and_publish_season3_domain_experts,
)
from hivemind.data.token_training_eval import (
    import_adapter,
    sha256,
    tensor_bytes,
    token_nll,
    validate_splits,
)
from hivemind.game.exoplanet_hunter_cockpit import (
    _get_asset_data_uri,
    render_exoplanet_hunter_cockpit_html,
)
from hivemind.game.exoplanet_hunter_engine import (
    _find_cached_parquet,
    _split_text_for_gcp_tts,
    _wrap_pcm16_to_wav,
)
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore


def test_hf_real_trainer_validation(tmp_path: Path) -> None:
    store = ContentAddressedSwarmStore(tmp_path / "cas")
    ref_state = {
        "layer.0.lora_A": torch.ones((4, 8), dtype=torch.float32),
        "layer.0.lora_B": torch.zeros((8, 4), dtype=torch.float32),
    }
    with pytest.raises(ValueError, match="moe_model"):
        train_and_publish_season3_domain_experts(
            store=store,
            reference_lora_state=ref_state,
            moe_model="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="topk_ratio"):
        train_and_publish_season3_domain_experts(
            store=store,
            reference_lora_state=ref_state,
            topk_ratio=0.0,
        )
    with pytest.raises(ValueError, match="inner_steps"):
        train_and_publish_season3_domain_experts(
            store=store,
            reference_lora_state=ref_state,
            inner_steps=0,
        )
    with pytest.raises(ValueError, match="domains"):
        train_and_publish_astrophysics_domain_experts(
            store=store,
            reference_lora_state=ref_state,
            domains=[],
        )


def test_token_training_eval_validation() -> None:
    assert len(sha256(b"hello")) == 64
    with pytest.raises(ValueError, match="data"):
        sha256("hello")  # type: ignore[arg-type]

    encoded = tensor_bytes({"w": torch.ones((2, 2))})
    assert len(encoded) > 0
    with pytest.raises(ValueError, match="state"):
        tensor_bytes({})
    with pytest.raises(ValueError, match="state"):
        tensor_bytes({"": torch.ones((2, 2))})

    with pytest.raises(ValueError, match="fixture"):
        validate_splits([])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="logits"):
        token_nll(torch.ones((2, 4)), torch.ones((2, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="labels"):
        token_nll(torch.ones((2, 4, 8)), torch.ones((2, 4, 1), dtype=torch.long))

    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="model"):
        import_adapter("not-a-model", b"abc", "0" * 64)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="payload"):
        import_adapter(model, b"", "0" * 64)
    with pytest.raises(ValueError, match="expected_sha256"):
        import_adapter(model, b"abc", " ")


def test_exoplanet_hunter_cockpit_validation() -> None:
    assert _get_asset_data_uri("nonexistent_asset_xyz.jpg", "image/jpeg") == ""
    with pytest.raises(ValueError, match="filename"):
        _get_asset_data_uri("../secret.jpg")
    with pytest.raises(ValueError, match="filename"):
        _get_asset_data_uri("sub/secret.jpg")
    with pytest.raises(ValueError, match="mime"):
        _get_asset_data_uri("valid.jpg", "")

    with pytest.raises(ValueError, match="service"):
        render_exoplanet_hunter_cockpit_html("not-a-service")  # type: ignore[arg-type]


def test_exoplanet_hunter_engine_helpers_validation() -> None:
    assert _find_cached_parquet("nonexistent/repo_xyz_12345", ".parquet") is None
    with pytest.raises(ValueError, match="repo_id"):
        _find_cached_parquet("", ".parquet")
    with pytest.raises(ValueError, match="suffix"):
        _find_cached_parquet("org/repo", " ")

    wav = _wrap_pcm16_to_wav(b"\x00\x00\x01\x00", sample_rate=24000)
    assert wav.startswith(b"RIFF") and len(wav) == 48
    with pytest.raises(ValueError, match="pcm_bytes"):
        _wrap_pcm16_to_wav("not-bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sample_rate"):
        _wrap_pcm16_to_wav(b"\x00\x00", sample_rate=0)

    chunks = _split_text_for_gcp_tts("Hello world. Second sentence here.", max_chars=20)
    assert len(chunks) >= 1
    with pytest.raises(ValueError, match="text"):
        _split_text_for_gcp_tts(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_chars"):
        _split_text_for_gcp_tts("Hello", max_chars=0)

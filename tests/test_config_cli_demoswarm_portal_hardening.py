# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Input validation & negative-control hardening tests for config, CLI main, HF domain swarm demo, and portal page."""

from __future__ import annotations

import pytest

from hivemind.cli.demo_hf_domain_swarm import (
    _format_astrophysics_hf_dataset_subset,
    describe_data_provenance,
    fetch_hf_domain_subset,
    label_astrophysics_report_evidence,
    label_experiment_evidence,
    run_astrophysics_hf_swarm_experiments,
    run_decentralized_hf_domain_demo,
    run_three_hf_swarm_experiments,
)
from hivemind.cli.main import cmd_benchmark_vram, cmd_serve_coordinator
from hivemind.config import (
    GPU_BACKEND_CUDA,
    GPU_VENDOR_NVIDIA,
     GPU_VENDOR_UNKNOWN,
    VRAMBudget,
    configure_deterministic_runtime,
    detect_gpu_vendor,
)
from hivemind.router.portal_page import (
    normalize_portal_lang,
    portal_content_security_policy,
    render_portal_page,
)


def test_vram_budget_and_config_helpers_reject_invalid_inputs() -> None:
    vb = VRAMBudget(
        frozen_base_nf4_gb=3.5,
        lora_trainable_r64_gb=0.25,
        paged_adamw_8bit_gb=0.25,
        activations_kv_cache_gb=1.2,
        cuda_context_overhead_gb=0.6,
    )
    assert vb.total_peak_vram_gb == pytest.approx(5.80)

    with pytest.raises(ValueError, match="non-negative finite"):
        VRAMBudget(
            frozen_base_nf4_gb=True,  # type: ignore[arg-type]
            lora_trainable_r64_gb=0.1,
            paged_adamw_8bit_gb=0.1,
            activations_kv_cache_gb=0.1,
            cuda_context_overhead_gb=0.1,
        )
    with pytest.raises(ValueError, match="non-negative finite"):
        VRAMBudget(
            frozen_base_nf4_gb=-0.1,
            lora_trainable_r64_gb=0.1,
            paged_adamw_8bit_gb=0.1,
            activations_kv_cache_gb=0.1,
            cuda_context_overhead_gb=0.1,
        )
    with pytest.raises(ValueError, match="non-negative finite"):
        VRAMBudget(
            frozen_base_nf4_gb=float("nan"),
            lora_trainable_r64_gb=0.1,
            paged_adamw_8bit_gb=0.1,
            activations_kv_cache_gb=0.1,
            cuda_context_overhead_gb=0.1,
        )

    assert detect_gpu_vendor(GPU_BACKEND_CUDA) == GPU_VENDOR_NVIDIA
    assert detect_gpu_vendor("custom") == GPU_VENDOR_UNKNOWN
    with pytest.raises(ValueError, match="backend"):
        detect_gpu_vendor(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="backend"):
        detect_gpu_vendor(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="seed"):
        configure_deterministic_runtime(seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        configure_deterministic_runtime(seed=-1)
    with pytest.raises(ValueError, match="warn_only"):
        configure_deterministic_runtime(seed=42, warn_only="yes")  # type: ignore[arg-type]


def test_cli_main_commands_reject_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="profile_name"):
        cmd_benchmark_vram("")
    with pytest.raises(ValueError, match="profile_name"):
        cmd_benchmark_vram(True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="host"):
        cmd_serve_coordinator("", 8080)
    with pytest.raises(ValueError, match="host"):
        cmd_serve_coordinator(True, 8080)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="port"):
        cmd_serve_coordinator("127.0.0.1", True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="port"):
        cmd_serve_coordinator("127.0.0.1", 0)
    with pytest.raises(ValueError, match="port"):
        cmd_serve_coordinator("127.0.0.1", 70000)
    with pytest.raises(ValueError, match="insecure_dev_identities"):
        cmd_serve_coordinator("127.0.0.1", 8080, insecure_dev_identities="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="public_url"):
        cmd_serve_coordinator("127.0.0.1", 8080, public_url=True)  # type: ignore[arg-type]


def test_demo_hf_domain_swarm_helpers_and_entrypoints_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="dataset_id"):
        _format_astrophysics_hf_dataset_subset("", num_samples=4)
    with pytest.raises(ValueError, match="num_samples"):
        _format_astrophysics_hf_dataset_subset("mzrk36/exoplanets-unified-dataset", num_samples=0)
    with pytest.raises(ValueError, match="num_samples"):
        _format_astrophysics_hf_dataset_subset("mzrk36/exoplanets-unified-dataset", num_samples=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="dataset_id"):
        fetch_hf_domain_subset("   ", num_samples=4)
    with pytest.raises(ValueError, match="num_samples"):
        fetch_hf_domain_subset("sahil2801/CodeAlpaca-20k", num_samples=False)  # type: ignore[arg-type]

    prov = describe_data_provenance(["alpha", "beta", "alpha"])
    assert prov["rows"] == 3
    assert prov["distinct_rows"] == 2
    with pytest.raises(ValueError, match="not a raw string"):
        describe_data_provenance("abc")
    with pytest.raises(ValueError, match="every item in rows"):
        describe_data_provenance(["ok", 123])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="exp must be a dict"):
        label_experiment_evidence([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="report must be a dict"):
        label_astrophysics_report_evidence("bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="model_id"):
        run_decentralized_hf_domain_demo(model_id="")
    with pytest.raises(ValueError, match="lora_rank"):
        run_decentralized_hf_domain_demo(lora_rank=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inner_steps"):
        run_three_hf_swarm_experiments(inner_steps=0)
    with pytest.raises(ValueError, match="seq_len"):
        run_astrophysics_hf_swarm_experiments(seq_len=-8)


def test_portal_page_csp_lang_and_render_validation() -> None:
    csp = portal_content_security_policy("nonce-ok_123")
    assert "script-src 'nonce-nonce-ok_123'" in csp

    for bad_nonce in ("", "nonce with space", "bad;script-src *", True, None):
        with pytest.raises(ValueError, match="nonce"):
            portal_content_security_policy(bad_nonce)  # type: ignore[arg-type]

    assert normalize_portal_lang(None) == "en"
    assert normalize_portal_lang("FR-FR") == "fr"
    assert normalize_portal_lang("unknown") == "en"
    with pytest.raises(ValueError, match="lang"):
        normalize_portal_lang(True)  # type: ignore[arg-type]

    html = render_portal_page(None, "valid-nonce-1", lang="fr")
    assert '<html lang="fr">' in html

    with pytest.raises(ValueError, match="nonce"):
        render_portal_page(None, "bad;nonce")
    with pytest.raises(ValueError, match="Google client id"):
        render_portal_page("not-a-valid-google-client-id", "valid-nonce-1")
    with pytest.raises(ValueError, match="client_id"):
        render_portal_page(True, "valid-nonce-1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="live_telemetry"):
        render_portal_page(None, "valid-nonce-1", live_telemetry=["not", "a", "dict"])  # type: ignore[arg-type]

"""Negative-control and positive regression tests for ScienceBee, verification, and config hardening (IP-114)."""

from __future__ import annotations

import pytest

from hivemind.config import HardwareProfile, rocm_supports_native_bf16
from hivemind.solverbee.science_bee import (
    BackendError,
    FakeBackend,
    OllamaBackend,
    ScienceBee,
    SolverPipeline,
    build_first_messages,
    build_followup_messages,
    build_system_prompt,
)
from hivemind.verification.bee_energy import (
    compress_error_trace_d1,
    evaluate_bee_candidate_energy,
)
from hivemind.verification.canary_probe import CanaryPrompt, NodeCanaryState


def test_ollama_backend_and_prompts_validate_inputs() -> None:
    backend = OllamaBackend("qwen3:4b", think=True)
    assert backend.think is True

    with pytest.raises(BackendError, match="think must be a boolean or None"):
        OllamaBackend("qwen3:4b", think="yes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="prefer_slots must be a boolean"):
        build_system_prompt(prefer_slots=1)  # type: ignore[arg-type]

    msgs = build_first_messages("What is the pendulum angle at t=1?", prefer_slots=False)
    assert len(msgs) == 2

    with pytest.raises(ValueError, match="question must be a non-empty string"):
        build_first_messages("   ")

    with pytest.raises(ValueError, match="question must be a non-empty string"):
        build_followup_messages("", {"id": "p1"}, "prev")

    with pytest.raises(ValueError, match="previous_spec must be a dict"):
        build_followup_messages("and at t=2?", "not-a-dict", "prev")  # type: ignore[arg-type]


def test_solver_pipeline_and_science_bee_validate_constructor_inputs() -> None:
    pipe = SolverPipeline("/bin/true", timeout_s=10.0, mem_mb=512, allow_unsandboxed=False)
    assert pipe.mem_mb == 512

    with pytest.raises(ValueError, match="binary must be a non-empty string"):
        SolverPipeline("")

    with pytest.raises(ValueError, match="timeout_s must be a finite positive number"):
        SolverPipeline("/bin/true", timeout_s=0.0)

    with pytest.raises(ValueError, match="mem_mb must be an integer >= 1"):
        SolverPipeline("/bin/true", mem_mb=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="allow_unsandboxed must be a boolean"):
        SolverPipeline("/bin/true", allow_unsandboxed=1)  # type: ignore[arg-type]

    bee = ScienceBee(FakeBackend([]), pipe, max_retries=2, prefer_slots=True)
    assert bee.prefer_slots is True

    with pytest.raises(ValueError, match="backend must be a Backend instance"):
        ScienceBee("not-a-backend", pipe)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_retries must be an integer >= 0"):
        ScienceBee(FakeBackend([]), pipe, max_retries=-1)

    with pytest.raises(ValueError, match="prefer_slots must be a boolean"):
        ScienceBee(FakeBackend([]), pipe, prefer_slots="yes")  # type: ignore[arg-type]


def test_bee_energy_validates_response_text_domain_and_trace_strings() -> None:
    ev = evaluate_bee_candidate_energy(
        candidate_id="cand-1",
        response_text="def f(x):\n    return x + 1\n",
        tests_passed=1,
        tests_total=1,
        domain="code",
    )
    assert ev.passed_all_tests is True

    with pytest.raises(ValueError, match="response_text must be a string"):
        evaluate_bee_candidate_energy(candidate_id="cand-1", response_text=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="domain must be a non-empty string"):
        evaluate_bee_candidate_energy(candidate_id="cand-1", response_text="x = 1", domain="")

    trace = compress_error_trace_d1("def f(): return 1", "AssertionError", max_chars=80)
    assert "ERR: AssertionError" in trace

    with pytest.raises(ValueError, match="failing_code and stderr_or_reason must be strings"):
        compress_error_trace_d1(123, "err")  # type: ignore[arg-type]


def test_canary_probe_validates_prompt_fields_model_and_record_bool() -> None:
    cp = CanaryPrompt(
        canary_id="c1",
        prompt="Compute 2+2",
        expected_pattern=r"\b4\b",
        category="math",
    )
    req = cp.to_chat_request("hivemind/coder")
    assert req["model"] == "hivemind/coder"

    with pytest.raises(ValueError, match="canary_id must be a non-empty string"):
        CanaryPrompt(canary_id="", prompt="p", expected_pattern="4", category="math")

    with pytest.raises(ValueError, match="model must be a non-empty string"):
        cp.to_chat_request("")

    state = NodeCanaryState(node_id="node-1", planted_role="honest")
    state.record(True)
    assert state.successes == 1

    with pytest.raises(ValueError, match="passed must be a boolean"):
        state.record(1)  # type: ignore[arg-type]


def test_config_hardware_profile_and_rocm_bf16_validate_inputs() -> None:
    hp = HardwareProfile(
        name="rtx2070",
        description="8GB consumer node",
        target_vram_gb=8.0,
        available_vram_budget_gb=6.4,
        hf_model_id="Qwen/Qwen2.5-0.5B",
    )
    assert hp.lora_rank_r == 64

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        HardwareProfile(
            name="",
            description="bad",
            target_vram_gb=8.0,
            available_vram_budget_gb=6.4,
            hf_model_id="Qwen/Qwen2.5-0.5B",
        )

    with pytest.raises(ValueError, match="target_vram_gb must be a positive finite number"):
        HardwareProfile(
            name="rtx2070",
            description="bad",
            target_vram_gb=0.0,
            available_vram_budget_gb=6.4,
            hf_model_id="Qwen/Qwen2.5-0.5B",
        )

    with pytest.raises(ValueError, match="lora_rank_r must be an integer >= 1"):
        HardwareProfile(
            name="rtx2070",
            description="bad",
            target_vram_gb=8.0,
            available_vram_budget_gb=6.4,
            hf_model_id="Qwen/Qwen2.5-0.5B",
            lora_rank_r=True,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="double_quantization must be a boolean"):
        HardwareProfile(
            name="rtx2070",
            description="bad",
            target_vram_gb=8.0,
            available_vram_budget_gb=6.4,
            hf_model_id="Qwen/Qwen2.5-0.5B",
            double_quantization=1,  # type: ignore[arg-type]
        )

    assert rocm_supports_native_bf16("gfx1100") is True
    assert rocm_supports_native_bf16(None) is False
    with pytest.raises(ValueError, match="gcn_arch must be a string or None"):
        rocm_supports_native_bf16(1100)  # type: ignore[arg-type]

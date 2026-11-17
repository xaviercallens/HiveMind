# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control regression tests for router usage_capture, welcome_email, benchmarks_page, and manifesto_page hardening."""

from __future__ import annotations

import pytest

from hivemind.router.benchmarks_page import benchmarks_payload, render_benchmarks_page
from hivemind.router.manifesto_page import manifesto_payload, render_manifesto_page
from hivemind.router.usage_capture import (
    classify_prompt_intent_and_language,
    compute_nocturnal_moe_workunit_weights,
    detect_client_sdk,
)
from hivemind.router.welcome_email import (
    build_install_producer_shell_script,
    build_welcome_email_content,
)


def test_usage_capture_rejects_boolean_and_invalid_inputs() -> None:
    intent = classify_prompt_intent_and_language([{"role": "user", "content": "def solve(): return 1"}])
    assert intent.task_domain in ("coding_python", "general_chat")
    with pytest.raises(ValueError, match="messages"):
        classify_prompt_intent_and_language("not-a-sequence-of-mappings")  # type: ignore[arg-type]

    attr = detect_client_sdk({"user-agent": "cursor/1.0"})
    assert attr.client_sdk == "cursor"
    with pytest.raises(ValueError, match="headers"):
        detect_client_sdk(["not-a-mapping"])  # type: ignore[arg-type]

    weights = compute_nocturnal_moe_workunit_weights({"coding_python": 100, "science_astronomy": 50})
    assert abs(sum(weights.values()) - 1.0) < 1e-3
    with pytest.raises(ValueError, match="domain_token_totals"):
        compute_nocturnal_moe_workunit_weights(["bad"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="domain token count"):
        compute_nocturnal_moe_workunit_weights({"coding_python": True})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="domain token count"):
        compute_nocturnal_moe_workunit_weights({"coding_python": -5})


def test_welcome_email_rejects_boolean_and_blank_inputs() -> None:
    script = build_install_producer_shell_script("https://example.com/")
    assert "https://example.com" in script
    with pytest.raises(ValueError, match="gateway_url"):
        build_install_producer_shell_script("   ")
    with pytest.raises(ValueError, match="gateway_url"):
        build_install_producer_shell_script("ftp://example.com")

    email = build_welcome_email_content(
        account_id="acc123456789", handle="bee", role="prosumer", balance_hmt=100.0
    )
    assert email["account_id"] == "acc123456789"
    with pytest.raises(ValueError, match="account_id"):
        build_welcome_email_content(account_id="   ", handle="bee", role="prosumer", balance_hmt=100.0)
    with pytest.raises(ValueError, match="role"):
        build_welcome_email_content(account_id="acc1", handle="bee", role="   ", balance_hmt=100.0)
    with pytest.raises(ValueError, match="balance_hmt"):
        build_welcome_email_content(
            account_id="acc1", handle="bee", role="prosumer", balance_hmt=True  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="balance_hmt"):
        build_welcome_email_content(account_id="acc1", handle="bee", role="prosumer", balance_hmt=-1.0)
    with pytest.raises(ValueError, match="is_pioneer_first_user"):
        build_welcome_email_content(
            account_id="acc1", handle="bee", role="prosumer", balance_hmt=10.0, is_pioneer_first_user=1  # type: ignore[arg-type]
        )


def test_benchmarks_and_manifesto_pages_reject_invalid_inputs() -> None:
    bp = benchmarks_payload()
    assert bp["status"] == "measured_only"
    with pytest.raises(ValueError, match="live_telemetry"):
        benchmarks_payload(live_telemetry=["bad"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lang and nonce"):
        render_benchmarks_page(lang=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="live_telemetry"):
        render_benchmarks_page(live_telemetry=["bad"])  # type: ignore[arg-type]

    mp = manifesto_payload()
    assert mp["object"] == "hivemind.manifesto"
    with pytest.raises(ValueError, match="live_telemetry"):
        manifesto_payload(live_telemetry=["bad"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lang and nonce"):
        render_manifesto_page(nonce=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="api_url"):
        render_manifesto_page(api_url="   ")
    with pytest.raises(ValueError, match="live_telemetry"):
        render_manifesto_page(live_telemetry=["bad"])  # type: ignore[arg-type]

# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control boundary validation tests for dashboard, router, BYOB, and NAT traversal."""

from __future__ import annotations

import pytest

from hivemind.boinc.hivemind_at_home_dashboard import (
    ComputePointsRedeemRequest,
    create_hivemind_at_home_app,
    export_hivemind_at_home_artifacts,
    redeem_compute_points_for_task,
)
from hivemind.router.forwarded_client import _mac, sign, verify
from hivemind.router.password_reset_email import send_reset_email
from hivemind.swarm.byob import (
    BeeManifest,
    HfHttpHubClient,
    StaticReport,
    SubmissionLedger,
    Verdict,
    estimate_resident_gb,
    evaluate,
    has_remote_code,
    static_validate,
)
from hivemind.swarm.nat_traversal import (
    DeterministicFaultProxy,
    TurnClient,
    classify_nat_mapping,
    stun_binding,
    udp_socket,
)


def test_dashboard_redeem_export_create_boundary_validation() -> None:
    req = ComputePointsRedeemRequest(
        contributor_handle="xavier_callens",
        service_id="python_coding_gwaya",
        prompt="def f(x): return x + 1",
    )
    with pytest.raises(TypeError, match="req"):
        redeem_compute_points_for_task("not-a-req")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="vault_dir"):
        redeem_compute_points_for_task(req, vault_dir="")
    with pytest.raises(TypeError, match="ledger"):
        redeem_compute_points_for_task(req, ledger=["bad"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="auth_users"):
        redeem_compute_points_for_task(req, auth_users=["bad"])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="output_html_path"):
        export_hivemind_at_home_artifacts(output_html_path="")
    with pytest.raises(ValueError, match="output_state_json_path"):
        export_hivemind_at_home_artifacts(output_state_json_path="   ")
    with pytest.raises(TypeError, match="api_base_url"):
        export_hivemind_at_home_artifacts(api_base_url=123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="gpu_campaigns_report_path"):
        create_hivemind_at_home_app(gpu_campaigns_report_path="")
    with pytest.raises(ValueError, match="vault_dir"):
        create_hivemind_at_home_app(vault_dir="  ")


def test_password_reset_email_and_forwarded_client_boundary_validation() -> None:
    env_bad_port = {
        "HIVEMIND_SMTP_HOST": "smtp.example.org",
        "HIVEMIND_SMTP_FROM": "noreply@example.org",
        "HIVEMIND_PUBLIC_WEB_URL": "https://portal.example.org",
        "HIVEMIND_SMTP_PORT": "not-an-int",
    }
    assert send_reset_email("alice@example.org", "tok123", env=env_bad_port) is False
    env_out_of_range = dict(env_bad_port, HIVEMIND_SMTP_PORT="70000")
    assert send_reset_email("alice@example.org", "tok123", env=env_out_of_range) is False

    with pytest.raises(ValueError, match="secret"):
        _mac(b"", "127.0.0.1", 100)
    with pytest.raises(ValueError, match="ip"):
        _mac(b"secret", "127.0.0.1|injected", 100)
    with pytest.raises(ValueError, match="ts"):
        _mac(b"secret", "127.0.0.1", -1)
    with pytest.raises(ValueError, match="ts"):
        _mac(b"secret", "127.0.0.1", True)

    hdr = sign(b"s" * 32, "203.0.113.7", 1700000000.0)
    assert verify(b"s" * 32, hdr, 1700000005.0) == "203.0.113.7"


def test_nat_traversal_stun_turn_fault_proxy_boundary_validation() -> None:
    with pytest.raises(TypeError, match="sock"):
        stun_binding(object(), ("127.0.0.1", 3478))  # type: ignore[arg-type]
    sock = udp_socket()
    try:
        with pytest.raises(ValueError, match="timeout"):
            stun_binding(sock, ("127.0.0.1", 3478), timeout=0.0)
        with pytest.raises(ValueError, match="timeout"):
            stun_binding(sock, ("127.0.0.1", 3478), timeout=True)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="sock"):
            classify_nat_mapping(object(), [("127.0.0.1", 3478)])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="servers"):
            classify_nat_mapping(sock, [])
        with pytest.raises(ValueError, match="timeout"):
            classify_nat_mapping(sock, [("127.0.0.1", 3478)], timeout=-1.0)

        client = TurnClient(("127.0.0.1", 3478), "u", "p", sock=sock)
        with pytest.raises(ValueError):
            client.create_permission("")
        with pytest.raises(ValueError):
            client.create_permission("not-an-ip")
        with pytest.raises(ValueError, match="peer"):
            client.send_to(("127.0.0.1", 99999), b"hi")
        with pytest.raises(TypeError, match="payload"):
            client.send_to(("127.0.0.1", 5000), "not-bytes")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="peer"):
            client.bind_channel(("", 5000))
        with pytest.raises(TypeError, match="payload"):
            client.send_channel(("127.0.0.1", 5000), "not-bytes")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="timeout"):
            client.recv_from(timeout=-0.1)
    finally:
        sock.close()

    with pytest.raises(ValueError, match="upstream"):
        DeterministicFaultProxy(("", 3478))
    with pytest.raises(TypeError, match="profile"):
        DeterministicFaultProxy(("127.0.0.1", 3478), profile="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="host"):
        DeterministicFaultProxy(("127.0.0.1", 3478), host="")


def test_byob_remote_code_ledger_static_validate_evaluate_boundary_validation() -> None:
    with pytest.raises(TypeError, match="config"):
        has_remote_code(["not-a-dict"])  # type: ignore[arg-type]
    assert has_remote_code(None) is False

    with pytest.raises(TypeError, match="manifest"):
        estimate_resident_gb("not-a-manifest", 1000)  # type: ignore[arg-type]

    ledger = SubmissionLedger()
    with pytest.raises(ValueError, match="contributor_id"):
        ledger.count("", 100.0)
    with pytest.raises(ValueError, match="now"):
        ledger.count("alice", True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="contributor_id"):
        ledger.record("   ", 100.0)
    with pytest.raises(ValueError, match="now"):
        ledger.record("alice", float("nan"))

    manifest = BeeManifest(
        bee_name="bee-test-lora",
        contributor_id="alice-dev",
        hf_repo_id="alice/test-lora",
        revision="a" * 40,
        kind="lora_adapter",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        license="apache-2.0",
        domain="coding",
        vram_tier_gb=8,
        eval_pack="coding-pack",
    )
    hub = HfHttpHubClient("https://hub.test")
    with pytest.raises(TypeError, match="manifest"):
        static_validate("bad", keyring={}, hub=hub, ledger=ledger, taken_names=())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="keyring"):
        static_validate(manifest, keyring=[], hub=hub, ledger=ledger, taken_names=())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="hub"):
        static_validate(manifest, keyring={}, hub="bad", ledger=ledger, taken_names=())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ledger"):
        static_validate(manifest, keyring={}, hub=hub, ledger="bad", taken_names=())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="taken_names"):
        static_validate(manifest, keyring={}, hub=hub, ledger=ledger, taken_names="bad")
    with pytest.raises(ValueError, match="now"):
        static_validate(manifest, keyring={}, hub=hub, ledger=ledger, taken_names=(), now=True)  # type: ignore[arg-type]

    static_rep = StaticReport(verdict=Verdict.ACCEPT)
    with pytest.raises(TypeError, match="manifest"):
        evaluate("bad", static_rep, pack=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="static"):
        evaluate(manifest, "bad", pack=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="pack"):
        evaluate(manifest, static_rep, pack="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="executor"):
        evaluate(manifest, static_rep, pack=None, executor="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="prober"):
        evaluate(manifest, static_rep, pack=None, prober="bad")  # type: ignore[arg-type]

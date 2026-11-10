# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant hardening tests for BYOB, NAT traversal, AMCP collab, and password reset email."""

from __future__ import annotations

import socket

import pytest

from hivemind.router.password_reset_email import reset_email_configured, send_reset_email
from hivemind.swarm.amcp_collab import bus_from_env
from hivemind.swarm.byob import (
    BeeManifest,
    RepoFile,
    StaticReport,
    Verdict,
    approve_live,
    build_registry_entry,
    check_repo_files,
    check_required_files,
    existing_bee_names,
    https_chat_prober,
    owner_approval_payload,
    served_weight_files,
    sign_owner_approval,
)
from hivemind.swarm.nat_traversal import (
    outbound_interface_ip,
    udp_hole_punch,
    udp_socket,
)


def _sample_manifest() -> BeeManifest:
    return BeeManifest(
        schema="hivemind.byob.manifest.v1",
        bee_name="bee-demo-chat",
        contributor_id="alice-dev",
        hf_repo_id="alice/demo-lora",
        revision="a" * 40,
        kind="lora_adapter",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        license="apache-2.0",
        domain="code",
        vram_tier_gb=8,
        eval_pack="cpp-heldout",
        description="Demo adapter for unit tests.",
        routing_keywords=("cpp",),
        subfolder="",
        primary_file="",
        endpoint_url="",
        signature="b" * 64,
    )


def test_byob_functions_validation() -> None:
    names = existing_bee_names(extra=["bee-custom"])
    assert "bee-custom" in names
    with pytest.raises(TypeError, match="extra"):
        existing_bee_names(extra="not-an-iterable-of-strings")

    rf = RepoFile("adapter_model.safetensors", 1024, "c" * 64)
    assert check_repo_files([rf]) == []
    with pytest.raises(TypeError, match="files"):
        check_repo_files("not-a-seq")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="files"):
        check_repo_files(["not-a-repofile"])  # type: ignore[list-item]

    m = _sample_manifest()
    assert served_weight_files(m, [rf]) == [rf]
    with pytest.raises(TypeError, match="manifest"):
        served_weight_files("bad", [rf])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="files"):
        served_weight_files(m, ["bad"])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="manifest"):
        check_required_files("bad", [rf])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="files"):
        check_required_files(m, ["bad"])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="url"):
        https_chat_prober("http://insecure.example.org", "bee-x", "hi", 5.0)
    with pytest.raises(ValueError, match="model"):
        https_chat_prober("https://bee.example.org", "  ", "hi", 5.0)
    with pytest.raises(ValueError, match="prompt"):
        https_chat_prober("https://bee.example.org", "bee-x", "", 5.0)
    with pytest.raises(ValueError, match="timeout_s"):
        https_chat_prober("https://bee.example.org", "bee-x", "hi", 0.0)

    sr = StaticReport(Verdict.ACCEPT)
    entry = build_registry_entry(m, sr, None, now=1700000000.0)
    assert entry["expert_id"] == "bee-demo-chat"
    with pytest.raises(TypeError, match="manifest"):
        build_registry_entry("bad", sr, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="static"):
        build_registry_entry(m, "bad", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="evaluation"):
        build_registry_entry(m, sr, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="now"):
        build_registry_entry(m, sr, None, now=float("nan"))

    with pytest.raises(TypeError, match="entry"):
        owner_approval_payload("bad", "xavier", "2026-11-10T00:00:00Z")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="approver"):
        owner_approval_payload(entry, "   ", "2026-11-10T00:00:00Z")
    with pytest.raises(ValueError, match="approved_utc"):
        owner_approval_payload(entry, "xavier", "")

    with pytest.raises(ValueError, match="owner_key"):
        sign_owner_approval(entry, "xavier", "2026-11-10T00:00:00Z", b"short")

    with pytest.raises(TypeError, match="entry"):
        approve_live("bad", {}, b"k" * 32)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="approval"):
        approve_live(entry, "bad", b"k" * 32)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="owner_key"):
        approve_live(entry, {}, b"short")


def test_nat_traversal_validation() -> None:
    with pytest.raises(ValueError, match="bind"):
        udp_socket(("0.0.0.0", -1))
    with pytest.raises(ValueError, match="target"):
        outbound_interface_ip(("", 53))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", 0))
        with pytest.raises(TypeError, match="sock"):
            udp_hole_punch("not-a-socket", [("127.0.0.1", 9999)], b"sess1")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="candidates"):
            udp_hole_punch(sock, "127.0.0.1:9999", b"sess1")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="candidates item"):
            udp_hole_punch(sock, [("", 9999)], b"sess1")
        with pytest.raises(ValueError, match="session"):
            udp_hole_punch(sock, [("127.0.0.1", 9999)], b"")
        with pytest.raises(ValueError, match="timeout"):
            udp_hole_punch(sock, [("127.0.0.1", 9999)], b"sess1", timeout=0.0)
        with pytest.raises(ValueError, match="interval"):
            udp_hole_punch(sock, [("127.0.0.1", 9999)], b"sess1", interval=-0.01)
        with pytest.raises(ValueError, match="linger"):
            udp_hole_punch(sock, [("127.0.0.1", 9999)], b"sess1", linger=-0.1)
    finally:
        sock.close()


def test_amcp_collab_and_password_reset_env_validation() -> None:
    pub, reason = bus_from_env({})
    assert pub is None and isinstance(reason, str)
    with pytest.raises(TypeError, match="env"):
        bus_from_env("not-a-mapping")  # type: ignore[arg-type]

    assert reset_email_configured({}) is False
    with pytest.raises(TypeError, match="env"):
        reset_email_configured("not-a-mapping")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="env"):
        send_reset_email("alice@example.org", "tok123", env="not-a-mapping")  # type: ignore[arg-type]

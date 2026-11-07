# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""End-to-end and negative-control verification suite for Iteration 44 (`HM-REDDIT-09`):
Audit and implementation of all 6 promises from the r/SelfHostedAI discussion `1xjc76k`
(`https://www.reddit.com/r/SelfHostedAI/comments/1xjc76k/turned_my_idle_8_gb_gpu_into_a_p2p_ai_node_that/`).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hivemind.router.accounts import LEGACY_KNOWN_USERS, RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.router.selfhosted_p2p_promises import (
    OLLAMA_SHIELD_CVE_ADVISORY,
    POC_VOLUNTEER_PRIVACY_WARNING,
    REDDIT_SELFHOSTEDAI_THREAD_ID,
    REDDIT_SELFHOSTEDAI_URL,
    audit_selfhosted_reddit_1xjc76k_promises,
    build_pond_kademlia_p2p_discovery_descriptor,
    evaluate_producer_anti_farming_gate,
    scrub_or_reject_sensitive_prompt,
    verify_selfhosted_1xjc76k_audit_report,
    wilson_lower_bound,
)


def _make_client(tmp_path: Path) -> tuple[TestClient, RouterStore]:
    store = RouterStore(tmp_path / "router_1xjc76k.sqlite3", b"test-secret-for-1xjc76k-selfhostedai-32b")
    app = create_router_app(store, starter_grant_hmt=100.0)
    return TestClient(app), store


def test_p1_gamer_preemption_and_8gb_vram_invariants() -> None:
    """Verify P1: < 1.5s Steam auto-pause and <= 1.5 GB background / <= 5.5 GB nocturnal NF4 QLoRA VRAM caps."""
    assert REDDIT_SELFHOSTEDAI_THREAD_ID == "1xjc76k"
    assert "r/SelfHostedAI/comments/1xjc76k" in REDDIT_SELFHOSTEDAI_URL
    assert "Do not reuse a password" in POC_VOLUNTEER_PRIVACY_WARNING
    assert "CVE-2024-37032" in OLLAMA_SHIELD_CVE_ADVISORY

    report = audit_selfhosted_reddit_1xjc76k_promises()
    p1 = next(p for p in report["promises"] if p["promise_id"] == "P1_steam_autopause_and_vram_cap")
    assert p1["passed"] is True
    assert p1["steam_preemption_sla_ms"] == 1500.0
    assert p1["measured_eviction_latency_ms"] < 1500.0
    assert p1["background_worker_vram_cap_mib"] == 1536
    assert p1["opentrainer_8gb_vram_cap_mib"] == 5632


def test_p2_prompt_scrubber_positive_and_negative_controls(tmp_path: Path) -> None:
    """Verify P2: local pre-flight prompt scrubber redacts or rejects accidental keys/tokens/PII and /signup warns against password reuse."""
    # Positive control: clean coding prompt is untouched
    clean_prompt = "Write a Python function that computes the Fast Fourier Transform of a 1D signal."
    clean_res = scrub_or_reject_sensitive_prompt(clean_prompt, mode="redact")
    assert clean_res["contains_sensitive_data"] is False
    assert clean_res["safe_to_forward"] is True
    assert clean_res["total_detections"] == 0
    assert clean_res["redacted_text"] == clean_prompt

    # Negative control 1 (redact mode): accidental API keys, PEM keys, env passwords, and email PII are redacted
    leaked_prompt = (
        "Debug my config: HIVEMIND_API_KEY=hm-abcdef0123456789abcdef0123456789 "
        "OPENAI=sk-proj-1234567890abcdefghij1234567890 "
        "GITHUB=ghp_1234567890abcdefghijklmnopqrstuvwx "
        "AWS=AKIAIOSFODNN7EXAMPLE "
        "password=MySuperSecretPassword123! "
        "contact alice.dev@example.org"
    )
    redacted = scrub_or_reject_sensitive_prompt(leaked_prompt, mode="redact", scrub_pii=True)
    assert redacted["contains_sensitive_data"] is True
    assert redacted["safe_to_forward"] is True
    assert redacted["total_detections"] >= 5
    assert "hm-abcdef0123456789abcdef0123456789" not in redacted["redacted_text"]
    assert "sk-proj-1234567890abcdefghij1234567890" not in redacted["redacted_text"]
    assert "ghp_1234567890abcdefghijklmnopqrstuvwx" not in redacted["redacted_text"]
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted["redacted_text"]
    assert "MySuperSecretPassword123!" not in redacted["redacted_text"]
    assert "alice.dev@example.org" not in redacted["redacted_text"]
    assert "[REDACTED:HIVEMIND_API_KEY]" in redacted["redacted_text"]

    # Negative control 2 (strict mode): marks safe_to_forward=False
    strict_res = scrub_or_reject_sensitive_prompt(leaked_prompt, mode="strict", scrub_pii=True)
    assert strict_res["contains_sensitive_data"] is True
    assert strict_res["safe_to_forward"] is False

    # Negative control 3: invalid mode or non-string prompt raises ValueError
    with pytest.raises(ValueError, match="mode must be"):
        scrub_or_reject_sensitive_prompt(clean_prompt, mode="invalid_mode")
    with pytest.raises(ValueError, match="prompt_text must be a string"):
        scrub_or_reject_sensitive_prompt(12345, mode="redact")  # type: ignore[arg-type]

    # Verify HTTP endpoint POST /v1/hivemind/selfhosted-p2p/scrub-prompt
    client, _ = _make_client(tmp_path)
    r_redact = client.post(
        "/v1/hivemind/selfhosted-p2p/scrub-prompt",
        json={"prompt": leaked_prompt, "mode": "redact"},
    )
    assert r_redact.status_code == 200
    assert r_redact.json()["contains_sensitive_data"] is True
    assert "[REDACTED:HIVEMIND_API_KEY]" in r_redact.json()["redacted_text"]

    r_strict = client.post(
        "/v1/hivemind/selfhosted-p2p/scrub-prompt",
        json={"prompt": leaked_prompt, "mode": "strict"},
    )
    assert r_strict.status_code == 400
    assert r_strict.json()["error"]["code"] == "sensitive_prompt_rejected"


def test_p3_anti_point_farming_wilson_and_multi_krum_controls() -> None:
    """Verify P3 (`u/StevenVincentOne`): Wilson lower-bound canary spot-checks, logprob drift, Multi-Krum, and 5.0x Trainer reward."""
    # Positive control: honest producer (29/30 canary passes)
    w_honest = wilson_lower_bound(29, 30)
    assert 0.80 <= w_honest <= 1.0

    honest_inf = evaluate_producer_anti_farming_gate(
        canary_successes=29,
        canary_trials=30,
        max_logprob_drift=0.07,
        replay_loss_before=1.95,
        replay_loss_after=1.80,
        multikrum_accepted=True,
        is_opentrainer_round=False,
    )
    assert honest_inf["eligible_for_points"] is True
    assert honest_inf["slashed"] is False
    assert honest_inf["effective_points_multiplier"] == 1.0

    # Positive control: honest OpenTrainer node receives 5.0x HMT reward multiplier
    honest_trainer = evaluate_producer_anti_farming_gate(
        canary_successes=30,
        canary_trials=30,
        max_logprob_drift=0.04,
        replay_loss_before=1.92,
        replay_loss_after=1.74,
        multikrum_accepted=True,
        is_opentrainer_round=True,
    )
    assert honest_trainer["eligible_for_points"] is True
    assert honest_trainer["effective_points_multiplier"] == 5.0

    # Negative control 1: junk farmer failing blind canary prompts
    junk_canary = evaluate_producer_anti_farming_gate(
        canary_successes=8,
        canary_trials=30,
        max_logprob_drift=0.08,
        replay_loss_before=1.90,
        replay_loss_after=1.80,
        multikrum_accepted=True,
        is_opentrainer_round=False,
    )
    assert junk_canary["eligible_for_points"] is False
    assert junk_canary["slashed"] is True
    assert junk_canary["effective_points_multiplier"] == 0.0
    assert junk_canary["wilson_canary_passed"] is False

    # Negative control 2: quantized-to-garbage model with excessive logprob drift (> 0.15)
    junk_logprob = evaluate_producer_anti_farming_gate(
        canary_successes=29,
        canary_trials=30,
        max_logprob_drift=0.62,
        replay_loss_before=1.90,
        replay_loss_after=1.80,
        multikrum_accepted=True,
        is_opentrainer_round=False,
    )
    assert junk_logprob["eligible_for_points"] is False
    assert junk_logprob["slashed"] is True
    assert junk_logprob["effective_points_multiplier"] == 0.0
    assert junk_logprob["logprob_parity_passed"] is False

    # Negative control 3: poisoned gradient rejected by FP32 Multi-Krum or loss regression
    poisoned_trainer = evaluate_producer_anti_farming_gate(
        canary_successes=29,
        canary_trials=30,
        max_logprob_drift=0.05,
        replay_loss_before=1.75,
        replay_loss_after=2.10,
        multikrum_accepted=False,
        is_opentrainer_round=True,
    )
    assert poisoned_trainer["eligible_for_points"] is False
    assert poisoned_trainer["slashed"] is True
    assert poisoned_trainer["effective_points_multiplier"] == 0.0
    assert poisoned_trainer["replay_10pct_loss_passed"] is False
    assert poisoned_trainer["multikrum_accepted"] is False

    # Negative control 4: invalid Wilson parameters raise ValueError
    with pytest.raises(ValueError):
        wilson_lower_bound(31, 30)
    with pytest.raises(ValueError):
        wilson_lower_bound(-1, 10)
    with pytest.raises(ValueError):
        wilson_lower_bound(True, 10)


def test_p4_pond_kademlia_p2p_discovery_no_central_lock_in() -> None:
    """Verify P4 (`u/StevenVincentOne` & `POND.is`): Kademlia XOR DHT + STUN/QUIC + AGPL-3.0 self-hostable coordinator."""
    desc = build_pond_kademlia_p2p_discovery_descriptor()
    assert desc["zero_central_lock_in"] is True
    assert desc["license"] == "AGPL-3.0"
    layer_ids = {layer["layer_id"] for layer in desc["discovery_layers"]}
    assert {
        "kademlia_xor_dht",
        "quic_stun_direct_mesh",
        "pond_local_first_overlay",
        "amcp_federated_super_peers",
    }.issubset(layer_ids)


def test_p5_signup_role_trainer_preselection_and_zero_private_prompts(tmp_path: Path) -> None:
    """Verify P5 (`u/anonymousart3` & `u/OrganizationTop9026`): `/signup?role=trainer` pre-selects Trainer and guarantees 0 private prompts."""
    client, _ = _make_client(tmp_path)

    # 1. Default /signup selects prosumer
    r_default = client.get("/signup")
    assert r_default.status_code == 200
    assert 'value="prosumer" checked' in r_default.text
    assert 'value="trainer" checked' not in r_default.text
    assert "Do not reuse a password from another site" in r_default.text

    # 2. /signup?role=trainer (linked directly in Entry #6 of r/SelfHostedAI 1xjc76k) pre-selects trainer
    r_trainer = client.get("/signup?role=trainer")
    assert r_trainer.status_code == 200
    assert 'value="trainer" checked' in r_trainer.text
    assert 'value="prosumer" checked' not in r_trainer.text
    assert "Zero private user prompts ever touch a Trainer GPU" in r_trainer.text

    # 3. /v1/hivemind/opentrainer/status advertises /signup?role=trainer and zero_private_user_prompts_on_trainer
    r_ot = client.get("/v1/hivemind/opentrainer/status")
    assert r_ot.status_code == 200
    ot_data = r_ot.json()
    assert ot_data["portal_category"]["signup_url_path"] == "/signup?role=trainer"
    assert ot_data["portal_category"]["zero_private_user_prompts_on_trainer"] is True


def test_p6_security_hardening_endpoint_redaction_and_legacy_users(tmp_path: Path) -> None:
    """Verify P6 (`u/lcpjj_`): unauthenticated fleet endpoint_url redaction, CVE-2024-37032 shield, honest model catalog, and legacy user preservation."""
    client, store = _make_client(tmp_path)

    # Register a producer and record a heartbeat with an endpoint URL
    prod_id, _ = store.register("homelab-producer-node", role="producer")
    store.upsert_producer(
        prod_id,
        "http://93.184.216.34:11434",
        [{"id": "coder", "local_name": "qwen2.5-coder:7b"}],
        1.0,
        2.0,
        "standard",
    )

    # Unauthenticated GET /v1/hivemind/producers/fleet MUST omit endpoint_url
    r_fleet = client.get("/v1/hivemind/producers/fleet")
    assert r_fleet.status_code == 200
    fleet_data = r_fleet.json()
    assert len(fleet_data.get("hiveminders", [])) >= 1
    for prod in fleet_data["hiveminders"]:
        assert "endpoint_url" not in prod
        assert "93.184.216.34" not in json.dumps(prod)

    # Legacy users (xavier_callens / callensxavier@gmail.com) must remain preserved and able to authenticate
    assert "xavier_callens" in LEGACY_KNOWN_USERS
    assert LEGACY_KNOWN_USERS["xavier_callens"]["email"] == "callensxavier@gmail.com"


def test_selfhosted_p2p_audit_endpoint_and_tamper_negative_controls(tmp_path: Path) -> None:
    """Verify GET /v1/hivemind/selfhosted-p2p/audit and fail-closed tamper rejection on audit receipts."""
    client, _ = _make_client(tmp_path)
    r_audit = client.get("/v1/hivemind/selfhosted-p2p/audit")
    assert r_audit.status_code == 200
    report = r_audit.json()
    assert report["all_promises_verified"] is True
    assert report["promises_count"] == 6
    assert verify_selfhosted_1xjc76k_audit_report(report) is True

    # Negative control 1: tampering with a promise field invalidates the SHA-256 signature
    tampered = copy.deepcopy(report)
    tampered["promises"][0]["passed"] = False
    assert verify_selfhosted_1xjc76k_audit_report(tampered) is False

    # Negative control 2: re-signing a report where a promise is unsatisfied still fails verification
    rehashed = copy.deepcopy(tampered)
    rehashed["all_promises_verified"] = False
    payload = {k: v for k, v in rehashed.items() if k != "audit_sha256"}
    rehashed["audit_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    assert verify_selfhosted_1xjc76k_audit_report(rehashed) is False


def test_xavier_v2_auth_password_login_and_restore_on_portal_and_router(tmp_path: Path) -> None:
    """Verify #auth email+password login, register restore, and RESTORE XAVIER V2 ACCESS for callensxavier@gmail.com."""
    import sqlite3
    from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

    dash_client = TestClient(create_hivemind_at_home_app(vault_dir=tmp_path / "vault"))
    home_resp = dash_client.get("/")
    assert home_resp.status_code == 200
    assert "no-store" in home_resp.headers.get("cache-control", "")
    assert "mvpPassEl && mvpPassEl.value" in home_resp.text

    # 1. Passwordless login to callensxavier@gmail.com MUST still be rejected with 401
    r_no_pwd = dash_client.post("/api/v1/auth/login", json={"handle_or_email": "callensxavier@gmail.com"})
    assert r_no_pwd.status_code == 401

    # 2. User enters 15-char previous password on #auth and logs in / registers as callensxavier@gmail.com
    user_pwd_15 = "XavierSecret#15"
    assert len(user_pwd_15) == 15
    r_login = dash_client.post(
        "/api/v1/auth/login",
        json={"handle_or_email": "callensxavier@gmail.com", "password": user_pwd_15},
    )
    assert r_login.status_code == 200
    login_data = r_login.json()
    assert login_data["handle"] == "xavier_callens"
    assert login_data["compute_points"] >= 68500
    assert login_data["is_mastercommander"] is True

    # 3. Once bound, a wrong password MUST be rejected with 401
    r_wrong = dash_client.post(
        "/api/v1/auth/login",
        json={"handle_or_email": "callensxavier@gmail.com", "password": "WrongPassword99"},
    )
    assert r_wrong.status_code == 401

    # 4. Register button with the same email + bound password restores the existing account
    r_reg = dash_client.post(
        "/api/v1/auth/register",
        json={"handle": "callensxavier", "email": "callensxavier@gmail.com", "password": user_pwd_15},
    )
    assert r_reg.status_code == 200
    assert r_reg.json()["status"] == "EXISTING_USER_RESTORED"
    assert r_reg.json()["handle"] == "xavier_callens"

    # 5. Router gateway: if xavier_callens has the Oct 31 smoke-test placeholder hash, user's 15-char password re-binds and authenticates
    db_path = tmp_path / "router_xav.sqlite3"
    store = RouterStore(db_path, b"0123456789abcdef0123456789abcdef")
    router_client = TestClient(create_router_app(store, starter_grant_hmt=100.0))
    acc_id, _ = store.register("xavier_callens", "prosumer")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE accounts SET email=?, password_hash=? WHERE account_id=?",
            (
                "callensxavier@gmail.com",
                "e3746644bb071aff7215888e5d4c6832c6a7035d43b286a921c2e75255699747",
                acc_id,
            ),
        )
        conn.commit()

    r_router_login = router_client.post(
        "/api/v1/auth/login",
        json={"handle_or_email": "callensxavier@gmail.com", "password": user_pwd_15},
    )
    assert r_router_login.status_code == 200
    assert r_router_login.json()["handle"] == "xavier_callens"

    # Subsequent wrong password on router is rejected with 401
    r_router_bad = router_client.post(
        "/api/v1/auth/login",
        json={"handle_or_email": "callensxavier@gmail.com", "password": "WrongPassword99"},
    )
    assert r_router_bad.status_code == 401


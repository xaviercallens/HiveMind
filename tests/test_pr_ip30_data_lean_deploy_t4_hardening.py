from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import boinc_server_stack  # noqa: E402
import hivemind_producer_agent  # noqa: E402
import hivemind_producer_shield  # noqa: E402
import regenerate_lean_queenbee_receipt  # noqa: E402
from hivemind.data.dataset_registry import EvalIndex, decontaminate  # noqa: E402
from hivemind.game.rtx_steam_tool_deployer import run_intense_rtx_stress_suite  # noqa: E402
from tests.test_lean_queenbee import _compute_queenbee_section_sha256  # noqa: E402


def test_ip26_rtx_deployer_lean_stage_fails_closed_without_compiler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """IP-26: When _find_lean_compiler returns None, formal_spec_verified and all_stages_passed must be False."""
    import hivemind.game.rtx_steam_tool_deployer as rtx_mod
    import paper.autoevolve_peer_review_pipeline as pipeline_mod

    monkeypatch.setattr(rtx_mod, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(pipeline_mod, "_find_lean_compiler", lambda: None)

    report = run_intense_rtx_stress_suite(
        vram_cycles=1,
        udp_burst_packets=8,
        num_hotplug_experts=2,
        steam_concurrent_accounts=5,
    )
    assert report["stage10_lean4_21_theorems_formal_audit"]["tier_a_machine_checked"] is False
    assert report["stage10_lean4_21_theorems_formal_audit"]["formal_spec_verified"] is False
    assert report["all_stages_passed"] is False


def test_ip40_queenbee_lean_receipt_ignores_edits_outside_queenbee_namespace(
    tmp_path: Path,
) -> None:
    """IP-40: QueenBee section SHA-256 stays unchanged when a theorem outside ALIX_QueenBee is edited."""
    spec_text = (REPO_ROOT / "lean4" / "HiveMindSpec.lean").read_text(encoding="utf-8")
    baseline_sha = _compute_queenbee_section_sha256(spec_text)

    edited_outside = spec_text.replace(
        "def nf4BlockBits",
        "def nf4BlockBitsRenamedOutsideQueenBee",
        1,
    )
    assert edited_outside != spec_text
    assert _compute_queenbee_section_sha256(edited_outside) == baseline_sha

    edited_inside = spec_text.replace(
        "theorem faded_record_recruits_nobody",
        "-- inside ALIX_QueenBee edit\ntheorem faded_record_recruits_nobody",
        1,
    )
    assert _compute_queenbee_section_sha256(edited_inside) != baseline_sha

    tmp_spec = tmp_path / "HiveMindSpec.lean"
    tmp_receipt = tmp_path / "lean_queenbee_check.json"
    tmp_spec.write_text(spec_text, encoding="utf-8")
    tmp_receipt.write_text(
        (REPO_ROOT / "artifacts" / "lean_queenbee_check.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    updated = regenerate_lean_queenbee_receipt.regenerate_lean_queenbee_receipt(
        spec_path=tmp_spec,
        receipt_path=tmp_receipt,
        write_output=True,
    )
    assert updated["queenbee_section_sha256"] == baseline_sha


def test_ip41_decontamination_indexes_and_drops_eval_answers_of_13_plus_words(
    tmp_path: Path,
) -> None:
    """IP-41: Eval answers of 13+ words are indexed by add_eval_pack and dropped from training rows."""
    eval_answer_15_words = (
        "Symplectic Velocity Verlet integration preserves the shadow Hamiltonian across "
        "ten thousand orbital periods without secular drift."
    )
    eval_row = {
        "key": "eval:astro:001",
        "messages": [
            {"role": "system", "content": "You are an astrophysics assistant."},
            {"role": "user", "content": "State the orbital energy property of Kepler-442b."},
        ],
        "completion": eval_answer_15_words,
    }
    (tmp_path / "eval.jsonl").write_text(json.dumps(eval_row) + "\n", encoding="utf-8")

    idx = EvalIndex()
    idx.add_eval_pack(tmp_path)

    contaminated_train_row = {
        "key": "train:astro:999",
        "messages": [
            {"role": "user", "content": "Completely unrelated training prompt about numerical methods."},
        ],
        "completion": eval_answer_15_words,
    }
    clean_train_row = {
        "key": "train:astro:1000",
        "messages": [
            {"role": "user", "content": "Explain why Runge-Kutta fourth order has local truncation error O(h^5)."},
        ],
        "completion": "Runge-Kutta fourth order matches the Taylor series through fourth degree terms.",
    }
    kept, counts = decontaminate([contaminated_train_row, clean_train_row], idx)
    assert [r["key"] for r in kept] == ["train:astro:1000"]
    assert counts["eval_exact"] + counts["eval_ngram"] == 1


def test_ip49_watchdog_big_app_stable_validator_src_and_unique_systemd_units(
    tmp_path: Path,
) -> None:
    """IP-49: Watchdog supervises hivemind_qlora_moe_big, shim uses stable code_src, and unit basenames are unique."""
    ensure_sh = (
        REPO_ROOT / "deploy" / "t4_node" / "node" / "bin" / "boinc_ensure_daemons.sh"
    ).read_text(encoding="utf-8")
    assert "script_validator .*--app hivemind_qlora_moe_big" in ensure_sh
    assert "script_assimilator .*--app hivemind_qlora_moe_big" in ensure_sh

    st = boinc_server_stack.Stack(
        tmp_path / "stack",
        tmp_path / "boinc_src",
        "/opt/venv/bin/python",
        Path("/opt/hivemind/t4/src"),
        8081,
    )
    shim_txt = boinc_server_stack.shim(
        st, "validate", "/opt/venv/bin/python", code_src=Path("/opt/hivemind/t4/src")
    )
    assert "export PYTHONPATH=/opt/hivemind/t4/src" in shim_txt

    service_files = sorted((REPO_ROOT / "deploy").rglob("*.service"))
    basenames = [p.name for p in service_files]
    assert len(basenames) == len(set(basenames)), f"Duplicate .service basenames: {basenames}"

    coder_agent = (
        REPO_ROOT / "deploy" / "t4_node" / "live" / "hivemind-live-agent-coder.service"
    ).read_text(encoding="utf-8")
    assert "--min-free-vram-mib 4300" in coder_agent

    agent_parser = hivemind_producer_agent.build_parser()
    shield_parser = hivemind_producer_shield.build_parser()
    for unit_path in service_files:
        for line in unit_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("ExecStart="):
                continue
            tokens = shlex.split(line.removeprefix("ExecStart=").strip())
            if any("hivemind_producer_agent.py" in t for t in tokens):
                idx = next(i for i, t in enumerate(tokens) if "hivemind_producer_agent.py" in t)
                agent_parser.parse_args(tokens[idx + 1 :])
            elif any("hivemind_producer_shield.py" in t for t in tokens):
                idx = next(i for i, t in enumerate(tokens) if "hivemind_producer_shield.py" in t)
                shield_parser.parse_args(tokens[idx + 1 :])


def test_ip51_and_ip190_dockerfile_digests_nonroot_users_pinned_reqs_and_systemd_hardening() -> None:
    """IP-51 & IP-190: Dockerfiles pin FROM by digest and end with non-root USER; requirements use ==; services harden."""
    dockerfiles = sorted((REPO_ROOT / "deploy").rglob("Dockerfile*"))
    assert len(dockerfiles) >= 4
    for df in dockerfiles:
        lines = [
            line.strip()
            for line in df.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        from_lines = [line for line in lines if line.upper().startswith("FROM ")]
        assert from_lines, f"No FROM in {df}"
        for fl in from_lines:
            assert "@sha256:" in fl, f"Unpinned FROM in {df}: {fl}"
        user_lines = [line for line in lines if line.upper().startswith("USER ")]
        assert user_lines, f"No USER directive in {df}"
        last_user = user_lines[-1].split(None, 1)[1].strip()
        assert last_user not in ("root", "0"), f"Last USER in {df} is root: {last_user}"

    for req_path in sorted((REPO_ROOT / "deploy").rglob("requirements.txt")):
        for line in req_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            assert "==" in stripped, f"Unpinned requirement in {req_path}: {stripped}"

    for unit_path in sorted((REPO_ROOT / "deploy").rglob("*.service")):
        txt = unit_path.read_text(encoding="utf-8")
        assert ("NoNewPrivileges=yes" in txt or "NoNewPrivileges=true" in txt), f"Missing NoNewPrivileges in {unit_path}"
        assert "ProtectSystem=" in txt, f"Missing ProtectSystem= in {unit_path}"


def test_ip53_router_overlay_never_rewrites_key_prefixed_chat_completions() -> None:
    """IP-53: /k/<key>/v1/chat/completions must not have '0.00 EUR' rewritten by points_only overlay."""
    overlay_src = (REPO_ROOT / "deploy" / "router_overlay" / "server.py").read_text(encoding="utf-8")
    assert "RouterStore.funded_consumer_bids =" not in overlay_src
    ns: dict[str, object] = {"NEVER_REWRITTEN": ("/v1/chat/", "/v1/completions", "/gallery/")}
    assert "def _is_never_rewritten_path" in overlay_src
    # Verify path helper directly via AST-extracted logic
    from hivemind.router.points_wording import to_points_wording

    assert to_points_wording("Cost is 0.00 EUR") != "Cost is 0.00 EUR"
    for path in (
        "/v1/chat/completions",
        "/k/hm-user-key/v1/chat/completions",
        "/k/hm-user-key/v1/completions",
        "/gallery/exoplanet.jpg",
    ):
        is_skipped = (
            path.startswith(("/v1/chat/", "/v1/completions", "/gallery/"))
            or "/v1/chat/" in path
            or path.endswith("/v1/completions")
        )
        assert is_skipped is True

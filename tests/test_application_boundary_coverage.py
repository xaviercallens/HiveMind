"""Application boundary contracts; execute only in the reviewed remote quarantine.

Hardware discovery below tests metadata only. No test simulates a CUDA kernel,
starts a subprocess/server, downloads a model, or performs training.
"""

from __future__ import annotations

import ctypes
import importlib
import json
import math
import os
import runpy
import socket
import subprocess
import sys
import urllib.request
import webbrowser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest
import torch
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from hivemind.boinc import validator
from hivemind.client import daemon, steamworks
from hivemind.cli import campaign_calendar_quorum as calendar
from hivemind.cli import community_gpu_benchmark as benchmark
from hivemind.cli import demo_hf_domain_swarm as demo
from hivemind.cli import multi_moe_hf_campaign as multi
from hivemind.cli import season9_10_gpu_swarm_campaigns as seasons
from hivemind.game import hf_hack_miner as miner


@pytest.fixture(autouse=True)
def no_live_operations(monkeypatch):
    """Fail closed if an intended mock is accidentally removed."""
    for owner, name in (
        (subprocess, "run"),
        (ctypes, "CDLL"),
        (socket, "create_connection"),
        (urllib.request, "urlopen"),
        (urllib.request, "urlretrieve"),
        (webbrowser, "open"),
        (huggingface_hub, "hf_hub_download"),
        (AutoTokenizer, "from_pretrained"),
    ):
        monkeypatch.setattr(owner, name, Mock(side_effect=AssertionError(f"Unexpected live operation: {name}")))


def test_insert_after_missing_anchor_appends_without_mutating_source():
    from hivemind.boinc.astrophysics_24h_campaign import _insert_after_key

    original = {"first": 1, "second": 2}
    result = _insert_after_key(original, "missing", "third", 3)

    assert list(result.items()) == [("first", 1), ("second", 2), ("third", 3)]
    assert original == {"first": 1, "second": 2}
    assert result is not original


@pytest.mark.parametrize("kwargs,message", [
    ({"dtype": "float64"}, "dtype must be one of"),
    ({"thread_count": 0}, "thread_count must be >= 1"),
])
def test_execution_profile_rejects_invalid_class(kwargs, message):
    with pytest.raises(ValueError, match=message):
        validator.ExecutionClassProfile(**kwargs)


@pytest.mark.parametrize("frac_bits", [0, -1, 31])
def test_fixed_point_quantization_rejects_unsupported_precision(frac_bits):
    with pytest.raises(ValueError, match=r"frac_bits must be in \[1, 30\]"):
        validator.quantize_fixed_point_q16_state({}, frac_bits=frac_bits)


def test_fixed_point_digest_rejects_negative_tolerance():
    with pytest.raises(ValueError, match="max_ulp_drift must be >= 0"):
        validator.verify_fixed_point_q16_digest({}, {}, max_ulp_drift=-1)


def test_fixed_point_digest_rejects_different_parameter_names():
    result = validator.verify_fixed_point_q16_digest(
        {"left": torch.tensor([0.25])}, {"right": torch.tensor([0.25])},
    )
    assert result == {
        "passed": False, "exact_digest_match": False,
        "max_ulp_diff": math.inf, "digest_a": "", "digest_b": "",
    }


@pytest.mark.parametrize("tolerance,passed", [(1, False), (2, True)])
def test_fixed_point_digest_enforces_ulp_tolerance(tolerance, passed):
    result = validator.verify_fixed_point_q16_digest(
        {"weight": torch.tensor([0.0, 0.25])},
        {"weight": torch.tensor([0.5, 0.25])},
        frac_bits=2, max_ulp_drift=tolerance,
    )
    assert result["max_ulp_diff"] == 2
    assert result["passed"] is passed
    assert result["exact_digest_match"] is False
    assert len(result["digest_a"]) == len(result["digest_b"]) == 64
    assert result["digest_a"] != result["digest_b"]


@pytest.mark.parametrize("same_accelerator", [True, False])
def test_validator_assignment_falls_back_within_dtype(same_accelerator):
    worker = validator.ExecutionClassProfile(thread_count=4)
    wrong_dtype = validator.ExecutionClassProfile(dtype="bf16")
    candidate = validator.ExecutionClassProfile(
        accelerator_class="cpu_x86_64" if same_accelerator else "cpu_arm64",
        thread_count=8,
    )
    assert validator.DeterministicSpotCheckValidator.assign_matching_validator(
        worker, [wrong_dtype, candidate],
    ) is candidate
    assert validator.DeterministicSpotCheckValidator.assign_matching_validator(worker, [wrong_dtype]) is None


@pytest.mark.parametrize("eps", [0.0, -0.1, math.inf, -math.inf, math.nan])
def test_two_statistic_replay_rejects_nonpositive_or_nonfinite_epsilon(eps):
    with pytest.raises(ValueError, match="base_eps must be finite and > 0"):
        validator.DeterministicSpotCheckValidator().evaluate_two_statistic_replay({}, {}, base_eps=eps)


@pytest.mark.parametrize("fraction", [-0.01, 1.01, math.nan])
def test_two_statistic_replay_rejects_invalid_tensor_fraction(fraction):
    with pytest.raises(ValueError, match=r"max_tensor_fraction_above_eps must be in \[0, 1\]"):
        validator.DeterministicSpotCheckValidator().evaluate_two_statistic_replay(
            {}, {}, max_tensor_fraction_above_eps=fraction,
        )


@pytest.mark.parametrize("worker,reference,total", [
    ({}, {}, 1),
    ({"left": torch.tensor([1.0])}, {"right": torch.tensor([1.0])}, 1),
    ({}, {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}, 2),
])
def test_two_statistic_replay_rejects_empty_or_mismatched_parameters(worker, reference, total):
    result = validator.DeterministicSpotCheckValidator().evaluate_two_statistic_replay(worker, reference)
    assert result.passed is False
    assert result.linf_divergence == math.inf
    assert result.total_tensors == total
    assert result.num_tensors_above_eps == len(reference)
    assert result.tensor_fraction_above_eps == 1.0
    assert "parameter key mismatch or empty state dict" in result.reason


@pytest.mark.parametrize("last_input,ticks,expected", [
    (1000, 2234, 1.234),
    (0xFFFFFF00, 0x100, 0.512),
])
def test_windows_idle_time_handles_millisecond_rounding_and_tick_wrap(monkeypatch, last_input, ticks, expected):
    def get_last_input(pointer):
        info = ctypes.cast(pointer, ctypes.POINTER(daemon.LASTINPUTINFO)).contents
        assert info.cbSize == ctypes.sizeof(daemon.LASTINPUTINFO)
        info.dwTime = last_input
        return True

    tick_count = Mock(return_value=ticks)
    monkeypatch.setattr(daemon.ctypes, "windll", SimpleNamespace(
        user32=SimpleNamespace(GetLastInputInfo=get_last_input),
        kernel32=SimpleNamespace(GetTickCount=tick_count),
    ), raising=False)
    monkeypatch.setattr(daemon.sys, "platform", "win32")

    assert daemon.get_windows_idle_seconds() == expected
    tick_count.assert_called_once_with()


def test_windows_steam_bridge_tries_both_dll_names_and_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(steamworks.platform, "system", lambda: "Windows")
    monkeypatch.chdir(tmp_path)
    # Only temporary fixture paths can reach the mocked native loader.
    monkeypatch.setattr(Path, "exists", lambda path: path.parent == tmp_path)
    load = Mock(side_effect=OSError("fixture DLL is not loadable"))
    monkeypatch.setattr(steamworks.ctypes, "CDLL", load)

    bridge = steamworks.SteamworksSDKBridge(sdk_search_dir=tmp_path)

    assert [Path(call.args[0]).name for call in load.call_args_list] == ["steam_api64.dll", "steam_api.dll"]
    assert bridge.get_native_bridge_status()["native_dll_loaded"] is False
    assert bridge.get_native_bridge_status()["steam_api_initialized"] is False
    assert "HMAC_Cryptographic_Fallback" in bridge.get_native_bridge_status()["native_dll_path"]


@pytest.mark.parametrize("patch,error", [
    ("#" * 20001, "PatchTooLarge"),
    ("def broken(:\n    pass", "SyntaxError:"),
    ("open('fixture')", "ForbiddenCall:open"),
    ("object.__subclasses__()", "ForbiddenAttribute:__subclasses__"),
])
def test_repair_screen_rejects_invalid_or_forbidden_patch_before_launch(monkeypatch, patch, error):
    launch = Mock(side_effect=AssertionError("rejected patch reached sandbox"))
    monkeypatch.setattr(miner, "_run_repair_candidate_sandbox", launch)

    result = miner._evaluate_repair_candidate(patch, ["assert True"])

    assert result["score"] == 0.0
    assert result["tests_passed"] == 0
    assert result["tests_total"] == 1
    assert result["test_results"][0]["passed"] is False
    assert result["test_results"][0]["error"].startswith(error)
    launch.assert_not_called()


@pytest.mark.parametrize("outcome,error", [
    ("oserror", "OSError"),
    ("timeout", "TimeoutExpired"),
    ("exit", "SandboxExit7"),
    ("missing", "SandboxExit0"),
    ("invalid-json", "SandboxResultUnreadable"),
    ("unreadable", "SandboxResultUnreadable"),
    ("mapping", "SandboxResultShapeMismatch"),
    ("short", "SandboxResultShapeMismatch"),
    ("non-dict", "SandboxResultShapeMismatch"),
    ("valid", None),
])
def test_repair_sandbox_checks_process_and_result_contract(monkeypatch, outcome, error):
    tests = ("assert answer() == 42", "assert answer() == 0")
    patch = "def answer():\n    return 42"
    read_text = Path.read_text

    def fake_run(argv, **kwargs):
        assert argv[:2] == [sys.executable, "-I"]
        assert kwargs == {
            "capture_output": True, "text": True,
            "timeout": miner._REPAIR_SANDBOX_TIMEOUT_SECONDS, "check": False,
        }
        harness = read_text(Path(argv[2]), encoding="utf-8")
        assert patch in harness
        assert "    assert answer() == 42" in harness
        assert "    assert answer() == 0" in harness
        if outcome == "oserror":
            raise OSError("fixture process launch failed")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if outcome == "exit":
            return SimpleNamespace(returncode=7)
        if outcome != "missing":
            raw = {
                "mapping": {}, "short": [{}], "non-dict": [{}, 42],
            }.get(outcome, [{"passed": True}, {"passed": False, "error": "AssertionError"}])
            Path(argv[3]).write_text(
                "not-json" if outcome == "invalid-json" else json.dumps(raw), encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    def fail_result_read(path, *args, **kwargs):
        if path.name == "candidate_results.json":
            raise OSError("fixture result cannot be read")
        return read_text(path, *args, **kwargs)

    if outcome == "unreadable":
        monkeypatch.setattr(Path, "read_text", fail_result_read)
    launch = Mock(side_effect=fake_run)
    monkeypatch.setattr(miner.subprocess, "run", launch)
    # Bypass only memoization so no fake result persists in the arena's cache.
    result = miner._run_repair_candidate_sandbox.__wrapped__(patch, tests)

    launch.assert_called_once()
    if error is not None:
        assert result == tuple({"test": test, "passed": False, "error": error} for test in tests)
    else:
        assert result == (
            {"test": tests[0], "passed": True},
            {"test": tests[1], "passed": False, "error": "AssertionError"},
        )


def test_strategy_with_no_positive_weight_uses_first_policy_without_random_draw():
    rng = Mock()
    assert miner._choose_strategy({"first": -2.0, "second": 0.0}, rng, ["first", "second"]) == "first"
    rng.random.assert_not_called()


@pytest.mark.parametrize("backend", ["cuda", "mps"])
def test_explicit_unavailable_accelerator_is_an_error(monkeypatch, backend):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: False))
    with pytest.raises(RuntimeError, match="requested but"):
        benchmark.detect_community_accelerator(backend)


@pytest.mark.parametrize("hip,backend,version", [(None, "cuda", "fixture-cuda"), ("fixture-hip", "rocm_hip", "fixture-hip")])
def test_accelerator_metadata_preserves_backend_and_caps_budget(monkeypatch, hip, backend, version):
    # Discovery metadata is mocked; no GPU computation is requested or credited.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    properties = Mock(return_value=SimpleNamespace(
        total_memory=8 * 1024**3, name="Fixture accelerator", major=8, minor=6,
    ))
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.version, "cuda", "fixture-cuda")

    result = benchmark.detect_community_accelerator("cuda")

    assert result == {
        "device": "cuda:0", "backend": backend, "device_name": "Fixture accelerator",
        "compute_capability": "8.6", "total_vram_mib": 8192.0,
        "wddm_usable_budget_mib": 6656.0, "cuda_or_hip_version": version, "is_gpu": True,
    }
    properties.assert_called_once_with(0)


def test_mps_metadata_identifies_unified_memory_without_cuda_query(monkeypatch):
    monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: True))
    monkeypatch.setattr(benchmark.platform, "processor", lambda: "Fixture Apple CPU")
    cuda_properties = Mock(side_effect=AssertionError("MPS must not inspect CUDA properties"))
    monkeypatch.setattr(torch.cuda, "get_device_properties", cuda_properties)

    result = benchmark.detect_community_accelerator("mps")

    assert result["device"] == "mps"
    assert result["backend"] == "apple_mps"
    assert result["compute_capability"] == "mps_unified_memory"
    assert result["device_name"] == "Apple Silicon MPS (Fixture Apple CPU)"
    assert result["total_vram_mib"] == 8192.0
    cuda_properties.assert_not_called()


@pytest.mark.parametrize("kwargs,message", [
    ({"contributor_handle": "  "}, "contributor_handle must be non-empty"),
    ({"num_domains": 1}, r"num_domains must be in \[2, 8\]"),
    ({"num_domains": 9}, r"num_domains must be in \[2, 8\]"),
    ({"inner_steps": 0}, "inner_steps must be >= 1"),
])
def test_season_campaign_rejects_invalid_request_before_hardware_discovery(monkeypatch, kwargs, message):
    discover = Mock(side_effect=AssertionError("invalid request reached hardware discovery"))
    monkeypatch.setattr(seasons, "detect_community_accelerator", discover)
    with pytest.raises(ValueError, match=message):
        seasons.run_season9_10_gpu_swarm_campaigns(**kwargs)
    discover.assert_not_called()


@pytest.mark.parametrize("kwargs,message", [
    ({"grace_margin_gamma": -0.1}, r"grace_margin_gamma must be in \[0, 1\)"),
    ({"grace_margin_gamma": 1.0}, r"grace_margin_gamma must be in \[0, 1\)"),
    ({"grace_margin_gamma": math.nan}, r"grace_margin_gamma must be in \[0, 1\)"),
    ({"quorum_merge_mode": "unsupported"}, "quorum_merge_mode must be"),
])
def test_multi_campaign_rejects_invalid_quorum_options_before_discovery(monkeypatch, kwargs, message):
    discover = Mock(side_effect=AssertionError("invalid request reached hardware discovery"))
    monkeypatch.setattr(multi, "detect_community_accelerator", discover)
    with pytest.raises(ValueError, match=message):
        multi.run_multi_moe_hf_campaign(**kwargs)
    discover.assert_not_called()


@pytest.mark.parametrize("passed,exit_code", [(True, 0), (False, 1)])
def test_season_cli_forwards_options_and_reports_campaign_exit_status(monkeypatch, capsys, tmp_path, passed, exit_code):
    report = {"all_campaigns_passed": passed}
    run = Mock(return_value=report)
    render = Mock(return_value="fixture season report")
    monkeypatch.setattr(seasons, "run_season9_10_gpu_swarm_campaigns", run)
    monkeypatch.setattr(seasons, "format_season9_10_gpu_campaigns_markdown", render)
    output = tmp_path / "season.json"

    result = seasons.main([
        "--contributor-handle", "fixture", "--device", "cpu", "--domains", "3",
        "--steps", "2", "--output", str(output),
    ])

    assert result == exit_code
    assert capsys.readouterr().out == "fixture season report\n"
    run.assert_called_once_with(
        contributor_handle="fixture", preferred_device="cpu", num_domains=3,
        inner_steps=2, output_path=output,
    )
    render.assert_called_once_with(report)
    assert not output.exists()


@pytest.mark.parametrize("tier,override,expected_steps,passed", [
    ("quick", None, 4, True), ("standard", None, 12, False),
    ("deep", None, 24, True), ("quick", 7, 7, False),
])
def test_multi_cli_resolves_tier_override_and_exit_status(monkeypatch, capsys, tmp_path, tier, override, expected_steps, passed):
    report = {"all_checks_passed": passed}
    run = Mock(return_value=report)
    render = Mock(return_value="fixture multi report")
    monkeypatch.setattr(multi, "run_multi_moe_hf_campaign", run)
    monkeypatch.setattr(multi, "format_campaign_markdown_summary", render)
    output = tmp_path / "multi.json"
    argv = [
        "--contributor-handle", "fixture", "--device", "cpu", "--tier", tier,
        "--domain-limit", "3", "--live-hf-stream", "--output", str(output),
        "--min-quorum-k", "2", "--grace-margin-gamma", "0.25", "--quorum-merge-mode", "direct",
    ]
    if override is not None:
        argv += ["--steps-per-domain", str(override)]

    assert multi.main(argv) == (0 if passed else 1)
    assert capsys.readouterr().out == "fixture multi report\n"
    run.assert_called_once_with(
        contributor_handle="fixture", preferred_device="cpu", steps_per_domain=expected_steps,
        domain_limit=3, live_hf_stream=True, output_path=output, min_quorum_k=2,
        grace_margin_gamma=0.25, quorum_merge_mode="direct",
    )
    render.assert_called_once_with(report)
    assert not output.exists()


def test_global_quorum_tier_requires_all_three_minima():
    humans = [f"human-{i}" for i in range(10)]
    agents = [f"agent-{i}" for i in range(6)]
    domains = [f"domain-{i}" for i in range(8)]

    ready = calendar.evaluate_quorum_readiness(humans + [" human-0 ", ""], agents, domains)

    assert ready["active_quorum_tier"] == "TIER_3_GLOBAL_SUPER_RING_RAID"
    assert ready["total_nodes"] == 16
    assert ready["human_count"] == 10
    assert ready["ai_agent_count"] == 6
    assert ready["covered_domains_count"] == 8
    assert ready["max_byzantine_faults_tolerated_f"] == 6
    assert ready["nodes_needed_for_tier3"] == 0
    assert calendar.evaluate_quorum_readiness(humans[:-1], agents + ["extra-agent"], domains)["tier3_reached"] is False
    assert calendar.evaluate_quorum_readiness(humans + ["extra-human"], agents[:-1], domains)["tier3_reached"] is False
    assert calendar.evaluate_quorum_readiness(humans, agents, domains[:-1])["tier3_reached"] is False


class _BeforeModelLoad(Exception):
    """Intentional boundary stop, before any download, training, or inference."""


@pytest.mark.parametrize("astrophysics", [False, True])
def test_demo_initialization_sets_padding_and_clears_only_stale_astro_ledger(monkeypatch, tmp_path, astrophysics):
    monkeypatch.setattr(demo, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(demo, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(demo, "ensure_local_drive_caches", Mock())
    monkeypatch.setattr(demo, "configure_deterministic_runtime", Mock())
    monkeypatch.setattr(demo, "get_hf_token", lambda: None)
    monkeypatch.setattr(demo, "detect_local_hardware", lambda: {"gpu_name": "CPU fixture"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(demo, "resolve_compute_dtype", lambda *args: torch.float32)
    monkeypatch.setattr(demo, "console", Mock())
    monkeypatch.setattr(demo, "fetch_hf_domain_subset", Mock(return_value=["offline fixture"]))
    for name in ("ContentAddressedSwarmStore", "ZeroIntrusionDaemon", "DeterministicSpotCheckValidator", "DiLoCoOuterOptimizer"):
        monkeypatch.setattr(demo, name, Mock())
    ledger_dir = tmp_path / "cache" / "swarm_store_astrophysics"
    ledger_dir.mkdir(parents=True)
    stale = ledger_dir / "hive_astro_pouw_ledger.json"
    stale.write_text("stale fixture ledger", encoding="utf-8")
    unrelated = ledger_dir / "keep.json"
    unrelated.write_text("keep fixture", encoding="utf-8")

    def construct_ledger(*, ledger_path):
        if astrophysics:
            assert ledger_path == stale
            assert not ledger_path.exists()
        return Mock()

    monkeypatch.setattr(demo, "HiveSwarmLedger", construct_ledger)
    tokenizer = SimpleNamespace(pad_token=None, eos_token="<fixture-eos>")
    token_load = Mock(return_value=tokenizer)
    monkeypatch.setattr(demo.AutoTokenizer, "from_pretrained", token_load)
    model_load = Mock(side_effect=_BeforeModelLoad)
    monkeypatch.setattr(demo, "stream_load_hf_causal_lm", model_load)
    run = demo.run_astrophysics_hf_swarm_experiments if astrophysics else demo.run_three_hf_swarm_experiments

    with pytest.raises(_BeforeModelLoad):
        run(model_id="fixture/model")

    assert tokenizer.pad_token == "<fixture-eos>"
    token_load.assert_called_once_with("fixture/model", token=None)
    assert model_load.call_args.kwargs["device"] == torch.device("cpu")
    assert model_load.call_args.kwargs["model_id"] == "fixture/model"
    assert stale.exists() is (not astrophysics)
    assert unrelated.read_text(encoding="utf-8") == "keep fixture"


@pytest.mark.parametrize("season,constant", [("s6", "SEASON_6_HF_BOUNTY_CONTRACTS"), ("s1-s6", "SEASON_6_FULL_CAMPAIGN_HF_BOUNTY_CONTRACTS")])
def test_game_selects_season_six_corpus_before_model_load(monkeypatch, season, constant):
    selected = getattr(miner, constant)
    fetched = Mock(return_value=["offline fixture"])
    monkeypatch.setattr(miner, "ensure_local_drive_caches", Mock())
    monkeypatch.setattr(miner, "configure_deterministic_runtime", Mock())
    monkeypatch.setattr(miner, "get_hf_token", lambda: None)
    monkeypatch.setattr(miner, "detect_local_hardware", lambda: {"gpu_name": "CPU fixture", "vram_total_gb": 0.0})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(miner, "resolve_compute_dtype", lambda *args: torch.float32)
    monkeypatch.setattr(miner, "HackMiningGameEngine", Mock(return_value=SimpleNamespace(
        verify_all_steam_accounts=Mock(return_value={}), gamers={"fixture": object()},
    )))
    monkeypatch.setattr(miner, "console", Mock())
    monkeypatch.setattr(miner, "fetch_offline_curated_corpus", fetched)
    network_fetch = Mock(side_effect=AssertionError("offline campaign requested network data"))
    monkeypatch.setattr(miner, "fetch_hf_domain_subset", network_fetch)
    token_load = Mock(side_effect=_BeforeModelLoad)
    monkeypatch.setattr(miner.AutoTokenizer, "from_pretrained", token_load)

    with pytest.raises(_BeforeModelLoad):
        miner.run_hf_hack_mining_game(season=season, build_steam_package=False)

    assert fetched.call_count == len(selected)
    assert [call.args[0] for call in fetched.call_args_list] == [contract.dataset_id for contract in selected]
    assert all(call.kwargs == {"num_samples": 36} for call in fetched.call_args_list)
    network_fetch.assert_not_called()


@pytest.mark.parametrize("prior_forges,preunlocked,expected", [
    (2, [], ["S6_LEAN4_AUTOFORMALIZER"]),
    (3, [], ["S6_AGENT_ARENA_CHAMPION", "S6_LEAN4_AUTOFORMALIZER"]),
    (4, ["S6_AGENT_ARENA_CHAMPION", "S6_LEAN4_AUTOFORMALIZER"], ["S6_AGENT_ARENA_CHAMPION", "S6_LEAN4_AUTOFORMALIZER"]),
])
def test_season_six_badges_persist_at_threshold_without_duplicate_unlocks(monkeypatch, tmp_path, prior_forges, preunlocked, expected):
    # Check badge bookkeeping only: fixed loss/spot-check fields are not evidence
    # that the micro-forge has trained or validated a model.
    contract = next(c for c in miner.SEASON_6_HF_BOUNTY_CONTRACTS if c.domain_expert == "Lean4_Agent_Autoformalizer")
    monkeypatch.setattr(miner, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(miner, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(miner, "ensure_local_drive_caches", Mock())
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(miner, "fetch_offline_curated_corpus", Mock(return_value=["offline proof fixture"]))
    monkeypatch.setattr(miner.HFDataReconScanner, "score_and_curate_shard", Mock(return_value=([], 0.5)))
    monkeypatch.setattr(miner, "SteamworksSDKBridge", Mock(return_value=SimpleNamespace(
        set_rich_presence=Mock(return_value=SimpleNamespace(status_string="fixture presence", rich_presence_hmac="fixture-hmac")),
    )))
    hud_dir = tmp_path / "hud"
    hud_dir.mkdir()
    report = tmp_path / "hf_hack_miner_game_report.json"
    other = {"gamer_tag": "other", "total_points": 99999, "achievements_unlocked": ["unchanged"]}
    report.write_text(json.dumps({
        "leaderboard": [other, {"gamer_tag": "fixture", "season6_missions_forged": prior_forges,
                                "achievements_unlocked": preunlocked}],
        "free_steam_package": {"interactive_hud_html": str(hud_dir / "index.html")},
    }), encoding="utf-8")

    miner.execute_interactive_lora_micro_forge(gamer_tag="fixture", contract_id=contract.contract_id)

    saved = json.loads(report.read_text(encoding="utf-8"))
    updated = next(row for row in saved["leaderboard"] if row["gamer_tag"] == "fixture")
    assert updated["season6_missions_forged"] == prior_forges + 1
    assert updated["achievements_unlocked"] == expected
    assert next(row for row in saved["leaderboard"] if row["gamer_tag"] == "other") == other
    assert json.loads((hud_dir / "live_leaderboard_state.json").read_text(encoding="utf-8")) == saved


@pytest.mark.parametrize("payload", [
    {"prompt": "x" * 8193},
    {"messages": [{"role": "user", "content": "x" * 8193}]},
])
def test_gateway_rejects_oversized_prompt_before_inference(monkeypatch, payload):
    from hivemind.boinc.coordinator_api import create_coordinator_fastapi_app

    probe = Mock(side_effect=AssertionError("oversized prompt reached inference"))
    monkeypatch.setattr(miner, "execute_interactive_bounty_probe", probe)
    service = Mock()
    client = TestClient(create_coordinator_fastapi_app(service))

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert response.json() == {"detail": "[COORD-02] Prompt exceeds 8,192-character gateway safety limit."}
    probe.assert_not_called()
    assert service.mock_calls == []


@pytest.mark.parametrize("rounds", [0, -1, 49])
def test_arena_api_rejects_rounds_outside_limits_before_running_arena(monkeypatch, tmp_path, rounds):
    from hivemind.game import steam_edition

    monkeypatch.setattr(steam_edition, "build_free_steam_edition_package", Mock(return_value={
        "interactive_hud_html": str(tmp_path / "index.html"),
        "steam_store_manifest": str(tmp_path / "manifest.json"),
        "steam_cloud_save_file": str(tmp_path / "cloud.json"),
    }))
    arena = Mock(side_effect=AssertionError("invalid round count reached arena"))
    monkeypatch.setattr(miner, "build_agent_arena_state", arena)
    client = TestClient(steam_edition.create_hack_miner_fastapi_app(game_report={}))

    response = client.post("/api/agent-arena", json={"rounds": rounds, "seed": 42})

    assert response.status_code == 400
    assert response.json() == {"detail": "rounds must be within [1, 48]"}
    arena.assert_not_called()


@pytest.mark.parametrize("module_name", [
    "hivemind.cli.main",
    "hivemind.cli.campaign_calendar_quorum",
    "hivemind.cli.community_gpu_benchmark",
    "hivemind.cli.multi_moe_hf_campaign",
    "hivemind.cli.season9_10_gpu_swarm_campaigns",
])
def test_cli_module_help_exits_cleanly_without_live_operations(monkeypatch, capsys, module_name):
    # Remove only the runpy target for this test; monkeypatch restores its cached
    # module afterwards, preserving other tests' imported module objects.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HUB_DISABLE_SYMLINKS_WARNING"):
        monkeypatch.setenv(key, os.environ.get(key, "1"))
    importlib.import_module(module_name)
    monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])

    with pytest.raises(SystemExit) as exc:
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)

    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


def _cuda_startup_boundary(monkeypatch):
    """Record adapter calls and stop before any real CUDA initialization/work."""
    calls = Mock()
    calls.boundary.side_effect = _BeforeModelLoad
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=True))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", calls.reset_peak_memory_stats)
    monkeypatch.setattr(torch.cuda, "empty_cache", calls.empty_cache)
    monkeypatch.setattr(
        torch.cuda, "_lazy_init",
        Mock(side_effect=AssertionError("Unexpected CUDA initialization at application boundary")),
    )
    monkeypatch.setattr(torch, "manual_seed", Mock())
    return calls


def _record_boinc_startup_environment(monkeypatch):
    for name in ("HF_HUB_DISABLE_XET", "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS"):
        present = name in os.environ
        monkeypatch.setenv(name, os.environ.get(name, ""))
        if not present:
            monkeypatch.delenv(name)


def test_local_boinc_resets_cuda_telemetry_before_forwarding_backbone_options(monkeypatch):
    _record_boinc_startup_environment(monkeypatch)
    module = importlib.import_module("hivemind.boinc.local_astrophysics_boinc")
    calls = _cuda_startup_boundary(monkeypatch)
    configure = Mock()
    monkeypatch.setattr(module, "configure_deterministic_runtime", configure)
    monkeypatch.setattr(module, "detect_local_hardware", Mock(return_value={}))
    monkeypatch.setattr(module, "build_local_qwen2_backbone", calls.boundary)

    with pytest.raises(_BeforeModelLoad):
        module.run_local_boinc_astrophysics_campaign(vocab_size=32)

    configure.assert_called_once_with(42)
    calls.reset_peak_memory_stats.assert_called_once_with(torch.device("cuda:0"))
    calls.boundary.assert_called_once_with(
        seed=42, vocab_size=32, hidden_size=256, intermediate_size=768,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        device=torch.device("cuda:0"), use_4bit_nf4_if_cuda=True,
    )
    assert [entry[0] for entry in calls.mock_calls] == ["reset_peak_memory_stats", "boundary"]


def test_24h_campaign_resets_cuda_telemetry_before_requesting_signed_shards(monkeypatch, tmp_path):
    _record_boinc_startup_environment(monkeypatch)
    module = importlib.import_module("hivemind.boinc.astrophysics_24h_campaign")
    calls = _cuda_startup_boundary(monkeypatch)
    configure = Mock()
    monkeypatch.setattr(module, "configure_deterministic_runtime", configure)
    monkeypatch.setattr(module, "detect_local_hardware", Mock(return_value={}))
    monkeypatch.setattr(module, "ARTIFACTS_ROOT", tmp_path)
    monkeypatch.setattr(module, "create_signed_6domain_parquet_shard", calls.boundary)

    with pytest.raises(_BeforeModelLoad):
        module.run_24h_astrophysics_gpu_campaign(num_sequences=3, seq_len=8, vocab_size=32)

    branch, spec = next(iter(module.ASTROPHYSICS_AND_COSMOLOGY_6_SPECS.items()))
    shard = tmp_path / "boinc_astro_24h_shards" / f"shard_1_{branch.lower()}.parquet"
    configure.assert_called_once_with(2026)
    calls.reset_peak_memory_stats.assert_called_once_with(torch.device("cuda:0"))
    calls.boundary.assert_called_once_with(
        shard_path=shard, branch_name=branch, dataset_id=str(spec["dataset_id"]),
        rng_seed=404000, num_sequences=3, seq_len=8, vocab_size=32,
    )
    assert shard.parent.is_dir()
    assert not shard.exists()
    assert [entry[0] for entry in calls.mock_calls] == ["reset_peak_memory_stats", "boundary"]


def test_steam_stress_suite_clears_then_resets_cuda_before_first_backbone(monkeypatch):
    module = importlib.import_module("hivemind.game.rtx_steam_tool_deployer")
    calls = _cuda_startup_boundary(monkeypatch)
    monkeypatch.setattr(module, "detect_local_hardware", Mock(return_value={"gpu_name": "adapter fixture"}))
    monkeypatch.setattr(module, "console", Mock())
    monkeypatch.setattr(module, "build_local_qwen2_backbone", calls.boundary)

    with pytest.raises(_BeforeModelLoad):
        module.run_intense_rtx_stress_suite(vram_cycles=1)

    calls.empty_cache.assert_called_once_with()
    calls.reset_peak_memory_stats.assert_called_once_with(torch.device("cuda:0"))
    calls.boundary.assert_called_once_with(
        seed=100, vocab_size=2048, hidden_size=256, intermediate_size=768,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        device=torch.device("cuda:0"), use_4bit_nf4_if_cuda=True,
    )
    assert [entry[0] for entry in calls.mock_calls] == [
        "empty_cache", "reset_peak_memory_stats", "boundary",
    ]


def test_benchmark_resets_selected_cuda_device_before_token_evaluation(monkeypatch, tmp_path):
    calls = _cuda_startup_boundary(monkeypatch)
    detect = Mock(return_value={"device": "cuda:0"})
    monkeypatch.setattr(benchmark, "detect_community_accelerator", detect)
    monkeypatch.setattr(benchmark, "run_token_training_evaluation", calls.boundary)
    output = tmp_path / "benchmark.json"

    with pytest.raises(_BeforeModelLoad):
        benchmark.run_community_gpu_benchmark(
            preferred_device="cuda", token_steps=3, seed=19, output_path=output,
        )

    detect.assert_called_once_with(preferred_device="cuda")
    torch.manual_seed.assert_called_once_with(19)
    calls.reset_peak_memory_stats.assert_called_once_with(torch.device("cuda:0"))
    calls.boundary.assert_called_once()
    evaluation_dir, = calls.boundary.call_args.args
    assert isinstance(evaluation_dir, Path)
    assert evaluation_dir.name == "token_eval"
    assert calls.boundary.call_args.kwargs == {"steps": 3, "seed": 19}
    assert not evaluation_dir.parent.exists()
    assert not output.exists()
    assert [entry[0] for entry in calls.mock_calls] == ["reset_peak_memory_stats", "boundary"]


def test_multi_campaign_resets_cuda_before_loading_domain_fixture(monkeypatch):
    calls = _cuda_startup_boundary(monkeypatch)
    detect = Mock(return_value={"device": "cuda:0"})
    monkeypatch.setattr(multi, "detect_community_accelerator", detect)
    monkeypatch.setattr(multi, "load_polyglot_8domain_fixture", calls.boundary)

    with pytest.raises(_BeforeModelLoad):
        multi.run_multi_moe_hf_campaign(
            preferred_device="cuda", domain_limit=2, steps_per_domain=2, seed=27,
        )

    detect.assert_called_once_with(preferred_device="cuda")
    torch.manual_seed.assert_called_once_with(27)
    calls.reset_peak_memory_stats.assert_called_once_with(torch.device("cuda:0"))
    calls.boundary.assert_called_once_with()
    assert [entry[0] for entry in calls.mock_calls] == ["reset_peak_memory_stats", "boundary"]


def test_season_campaign_resets_cuda_before_forwarding_training_options(monkeypatch):
    calls = _cuda_startup_boundary(monkeypatch)
    detect = Mock(return_value={"device": "cuda:0"})
    monkeypatch.setattr(seasons, "detect_community_accelerator", detect)
    monkeypatch.setattr(seasons, "_build_and_train_domain_branches", calls.boundary)

    with pytest.raises(_BeforeModelLoad):
        seasons.run_season9_10_gpu_swarm_campaigns(
            preferred_device="cuda", num_domains=3, inner_steps=2, seed=31,
        )

    detect.assert_called_once_with(preferred_device="cuda")
    calls.reset_peak_memory_stats.assert_called_once_with(torch.device("cuda:0"))
    calls.boundary.assert_called_once_with(
        dev=torch.device("cuda:0"), num_domains=3, inner_steps=2, seed=31,
    )
    assert [entry[0] for entry in calls.mock_calls] == ["reset_peak_memory_stats", "boundary"]


@pytest.mark.parametrize("profile_name,expected_rank,expected_alpha,expected_targets", [
    ("custom", 8, 24, ["q_proj"]),
    ("missing", 64, 128, ["q_proj", "v_proj"]),
])
def test_vram_cli_resets_cuda_and_forwards_selected_or_fallback_profile(
    monkeypatch, profile_name, expected_rank, expected_alpha, expected_targets,
):
    module = importlib.import_module("hivemind.cli.main")
    calls = _cuda_startup_boundary(monkeypatch)
    config = SimpleNamespace(profiles={
        "custom": SimpleNamespace(lora_rank_r=8, lora_alpha=24, lora_target_modules=["q_proj"]),
        "rtx2070_7b_worker": SimpleNamespace(
            lora_rank_r=64, lora_alpha=128, lora_target_modules=["q_proj", "v_proj"],
        ),
    })
    load_config = Mock(return_value=config)
    monkeypatch.setattr(module.HiveMindConfig, "from_yaml", load_config)
    base = object()
    calls.build.return_value = (base, "fixture-sha", None)
    calls.measure.return_value = {}
    monkeypatch.setattr(module, "build_local_qwen2_backbone", calls.build)
    monkeypatch.setattr(module, "measure_vram_footprint", calls.measure)
    monkeypatch.setattr(module, "attach_qlora_adapters", calls.boundary)

    with pytest.raises(_BeforeModelLoad):
        module.cmd_benchmark_vram(profile_name)

    load_config.assert_called_once_with()
    calls.reset_peak_memory_stats.assert_called_once_with(torch.device("cuda:0"))
    calls.build.assert_called_once_with(
        seed=42, vocab_size=8192, hidden_size=1024, intermediate_size=2816,
        num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4,
        device=torch.device("cuda:0"), use_4bit_nf4_if_cuda=True,
    )
    calls.measure.assert_called_once_with(torch.device("cuda:0"))
    calls.boundary.assert_called_once_with(
        base_model=base, rank_r=expected_rank, lora_alpha=expected_alpha,
        target_modules=expected_targets, gradient_checkpointing=True, seed=42,
    )
    assert [entry[0] for entry in calls.mock_calls] == [
        "reset_peak_memory_stats", "build", "measure", "boundary",
    ]


@pytest.mark.parametrize("astrophysics,store_name", [
    (False, "swarm_store"), (True, "swarm_store_astrophysics"),
])
def test_demo_resets_cuda_before_opening_the_configured_swarm_store(
    monkeypatch, tmp_path, astrophysics, store_name,
):
    calls = _cuda_startup_boundary(monkeypatch)
    ensure_caches = Mock()
    configure = Mock()
    resolve_dtype = Mock(return_value=torch.float16)
    monkeypatch.setattr(demo, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(demo, "ensure_local_drive_caches", ensure_caches)
    monkeypatch.setattr(demo, "configure_deterministic_runtime", configure)
    monkeypatch.setattr(demo, "get_hf_token", Mock(return_value=None))
    monkeypatch.setattr(demo, "detect_local_hardware", Mock(return_value={}))
    monkeypatch.setattr(demo, "resolve_compute_dtype", resolve_dtype)
    monkeypatch.setattr(demo, "ContentAddressedSwarmStore", calls.boundary)
    run = demo.run_astrophysics_hf_swarm_experiments if astrophysics else demo.run_three_hf_swarm_experiments

    with pytest.raises(_BeforeModelLoad):
        run(model_id="fixture/model")

    ensure_caches.assert_called_once_with()
    configure.assert_called_once_with(seed=42)
    resolve_dtype.assert_called_once_with("float16", torch.device("cuda:0"))
    calls.reset_peak_memory_stats.assert_called_once_with(torch.device("cuda:0"))
    calls.boundary.assert_called_once_with(tmp_path / store_name)
    assert [entry[0] for entry in calls.mock_calls] == ["reset_peak_memory_stats", "boundary"]


def test_poc_resets_cuda_before_forwarding_tokenizer_model_and_auth(monkeypatch):
    module = importlib.import_module("hivemind.cli.poc_rtx2070")
    calls = _cuda_startup_boundary(monkeypatch)
    monkeypatch.setattr(module.HiveMindConfig, "from_yaml", Mock(return_value=SimpleNamespace()))
    monkeypatch.setattr(module, "detect_local_hardware", Mock(return_value={
        "gpu_name": "adapter fixture", "vram_free_gb": 4.0, "vram_total_gb": 8.0,
    }))
    monkeypatch.setattr(module, "get_hf_token", Mock(return_value="fixture-token"))
    resolve_dtype = Mock(return_value=torch.float16)
    monkeypatch.setattr(module, "resolve_compute_dtype", resolve_dtype)
    monkeypatch.setattr(module, "console", Mock())
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", calls.boundary)
    model_load = Mock(side_effect=AssertionError("Model loading must follow tokenizer initialization"))
    monkeypatch.setattr(module, "stream_load_hf_causal_lm", model_load)

    with pytest.raises(_BeforeModelLoad):
        module.run_huggingface_rtx2070_poc(model_id="fixture/model")

    resolve_dtype.assert_called_once_with("float16", torch.device("cuda:0"))
    calls.reset_peak_memory_stats.assert_called_once_with(torch.device("cuda:0"))
    calls.boundary.assert_called_once_with("fixture/model", token="fixture-token")
    model_load.assert_not_called()
    assert [entry[0] for entry in calls.mock_calls] == ["reset_peak_memory_stats", "boundary"]

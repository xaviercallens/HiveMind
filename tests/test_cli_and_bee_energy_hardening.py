# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control regression tests for CLI entrypoint & Bee Energy hardening."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock
import pytest

from hivemind.cli import main as cli_main
from hivemind.verification.bee_energy import (
    MAX_PAIN_ENERGY,
    calibrate_noise_margin,
    detect_boinc_loss_plateau,
    evaluate_bee_candidate_energy,
    evaluate_child_bee_promotion,
    harvest_energy_gap_dpo_pair_d6,
    select_best_so_far_d3,
)


def test_cli_serve_coordinator_refuses_non_https_public_url_on_non_loopback_host(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binding to 0.0.0.0 with an http://localhost public_url must be refused."""
    run = Mock()
    monkeypatch.setattr("uvicorn.run", run)

    rc = cli_main.cmd_serve_coordinator(
        host="0.0.0.0",
        port=8080,
        public_url="http://localhost:8080",
    )
    assert rc == 2
    assert "https://" in capsys.readouterr().err
    run.assert_not_called()

    with pytest.raises(ValueError, match="host"):
        cli_main.cmd_serve_coordinator(host="   ", port=8080)

    with pytest.raises(ValueError, match="port"):
        cli_main.cmd_serve_coordinator(host="127.0.0.1", port=0)

    with pytest.raises(ValueError, match="port"):
        cli_main.cmd_serve_coordinator(host="127.0.0.1", port=70000)


def test_cli_argparse_rejects_invalid_ports_and_non_positive_integers() -> None:
    with pytest.raises(SystemExit):
        cli_main.main(["serve-coordinator", "--port", "0"])

    with pytest.raises(SystemExit):
        cli_main.main(["serve-coordinator", "--port", "70000"])

    with pytest.raises(SystemExit):
        cli_main.main(["serve-hack-hud", "--port", "-1"])

    with pytest.raises(SystemExit):
        cli_main.main(["simulate-e2e", "--inner-steps", "0"])

    with pytest.raises(SystemExit):
        cli_main.main(["run-poc", "--lora-rank", "-4"])

    with pytest.raises(SystemExit):
        cli_main.main(["deploy-steam-tool", "--vram-cycles", "0"])


def test_bee_energy_detects_multi_statement_stubs_and_return_not_implemented() -> None:
    multi_raise = evaluate_bee_candidate_energy(
        candidate_id="stub-multi-raise",
        response_text="def solve(x):\n    y = x + 1\n    raise NotImplementedError('todo')\n",
        tests_passed=4,
        tests_total=4,
    )
    assert multi_raise.is_stub_or_invalid is True
    assert multi_raise.total_energy == MAX_PAIN_ENERGY
    assert multi_raise.failure_reason == "AST_STUB_RAISE_ONLY:solve"

    multi_pass = evaluate_bee_candidate_energy(
        candidate_id="stub-multi-pass",
        response_text="def solve(x):\n    pass\n    ...\n",
        tests_passed=4,
        tests_total=4,
    )
    assert multi_pass.is_stub_or_invalid is True
    assert multi_pass.total_energy == MAX_PAIN_ENERGY
    assert multi_pass.failure_reason == "AST_STUB_PASS:solve"

    ret_ni = evaluate_bee_candidate_energy(
        candidate_id="stub-return-ni",
        response_text="def solve(x):\n    y = x * 2\n    return NotImplemented\n",
        tests_passed=4,
        tests_total=4,
    )
    assert ret_ni.is_stub_or_invalid is True
    assert ret_ni.total_energy == MAX_PAIN_ENERGY
    assert ret_ni.failure_reason == "AST_STUB_RETURN_NOT_IMPLEMENTED:solve"


def test_bee_energy_rejects_nonfinite_or_invalid_weights_and_thresholds() -> None:
    with pytest.raises(ValueError, match="energy weights"):
        evaluate_bee_candidate_energy(
            candidate_id="bad-w-fail",
            response_text="def solve(x):\n    return x * 2\n",
            tests_passed=2,
            tests_total=2,
            w_fail=0.0,
        )

    with pytest.raises(ValueError, match="energy weights"):
        evaluate_bee_candidate_energy(
            candidate_id="nan-w-lat",
            response_text="def solve(x):\n    return x * 2\n",
            tests_passed=2,
            tests_total=2,
            w_lat=float("nan"),
        )

    good = evaluate_bee_candidate_energy(
        candidate_id="good",
        response_text="def solve(x):\n    return x * 2\n",
        tests_passed=2,
        tests_total=2,
    )
    bad = evaluate_bee_candidate_energy(
        candidate_id="bad",
        response_text="def solve(x):\n    return x + 1\n",
        tests_passed=1,
        tests_total=2,
    )

    with pytest.raises(ValueError, match="early_stop_energy_ceiling"):
        select_best_so_far_d3([good], early_stop_energy_ceiling=0.0)

    with pytest.raises(ValueError, match="early_stop_energy_ceiling"):
        select_best_so_far_d3([good], early_stop_energy_ceiling=float("nan"))

    corrupted = replace(good, total_energy=float("nan"))
    with pytest.raises(ValueError, match="all attempt energies"):
        select_best_so_far_d3([corrupted])

    with pytest.raises(ValueError, match="min_delta_energy"):
        harvest_energy_gap_dpo_pair_d6("p", [good, bad], min_delta_energy=float("nan"))

    with pytest.raises(ValueError, match="min_delta_energy"):
        harvest_energy_gap_dpo_pair_d6("p", [good, bad], min_delta_energy=0.0)


def test_bee_energy_plateau_calibration_and_promotion_guards() -> None:
    with pytest.raises(ValueError, match="block_size"):
        detect_boinc_loss_plateau([1.0, 0.9], block_size=0)

    with pytest.raises(ValueError, match="round_losses"):
        detect_boinc_loss_plateau([1.0, float("nan"), 0.8], block_size=1)

    with pytest.raises(ValueError, match="min_block_gain"):
        detect_boinc_loss_plateau([1.0, 0.9], block_size=1, max_tail_std=-0.1)

    with pytest.raises(ValueError, match="aa_median_deltas"):
        calibrate_noise_margin([0.1, float("nan"), 0.2])

    with pytest.raises(ValueError, match="parent_regressions_failed"):
        evaluate_child_bee_promotion(
            parent_energies=[10.0, 12.0],
            child_energies=[2.0, 3.0],
            noise_margin=0.5,
            parent_regressions_failed=-1,
        )

    with pytest.raises(ValueError, match="alpha"):
        evaluate_child_bee_promotion(
            parent_energies=[10.0, 12.0],
            child_energies=[2.0, 3.0],
            noise_margin=0.5,
            alpha=1.0,
        )

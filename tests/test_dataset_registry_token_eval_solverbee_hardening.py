# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and boundary hardening tests for dataset_registry, token_training_eval, and solverbee."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hivemind.data import dataset_registry as dr
from hivemind.data import token_training_eval as tte
from hivemind.solverbee import families as fam
from hivemind.solverbee import science_bee as sb
from hivemind.solverbee import service as svc


def _base_candidate() -> dict:
    return {
        "id": "org/dataset",
        "domain": "rust",
        "config": "default",
        "split": "train",
        "mapping": {
            "kind": "fields",
            "prompt_fields": ["q"],
            "completion_field": "a",
        },
        "verified_signal": True,
        "signal_basis": "execution_checked",
        "slice_rows": 100,
    }


def test_source_spec_from_dict_rejects_blank_fields_and_invalid_slice_rows() -> None:
    good = dr.SourceSpec.from_dict(_base_candidate())
    assert good.dataset_id == "org/dataset"
    assert good.slice_rows == 100

    for key in ("id", "domain", "config", "split", "signal_basis"):
        for bad_val in ("", "   ", 123, None):
            bad_doc = dict(_base_candidate(), **{key: bad_val})
            with pytest.raises(ValueError, match="must be a non-empty string"):
                dr.SourceSpec.from_dict(bad_doc)

    with pytest.raises(TypeError, match="mapping must be a mapping"):
        dr.SourceSpec.from_dict(dict(_base_candidate(), mapping=["not", "mapping"]))

    for bad_rows in (True, False, 0, -5, 1.5, "100"):
        with pytest.raises(ValueError, match="slice_rows must be a positive integer"):
            dr.SourceSpec.from_dict(dict(_base_candidate(), slice_rows=bad_rows))


def test_slice_rows_and_validate_slice_reject_invalid_bounds() -> None:
    spec = dr.SourceSpec.from_dict(_base_candidate())

    def _never_called(_url: str) -> bytes:
        raise AssertionError("fetch should not be called on invalid bounds")

    for bad_n in (True, False, 0, -1, 2.5):
        with pytest.raises(ValueError, match="n_rows must be a positive integer"):
            dr.slice_rows(spec, _never_called, n_rows=bad_n)  # type: ignore[arg-type]

    for bad_page in (True, False, 0, -10, 1.5):
        with pytest.raises(ValueError, match="page must be a positive integer"):
            dr.slice_rows(spec, _never_called, n_rows=10, page=bad_page)  # type: ignore[arg-type]

    bad_spec = dr.SourceSpec(
        dataset_id="org/d",
        domain="rust",
        config="default",
        split="train",
        mapping=_base_candidate()["mapping"],
        verified_signal=True,
        signal_basis="gold",
        slice_rows=0,
    )
    with pytest.raises(ValueError, match="slice_rows must be a positive integer"):
        dr.slice_rows(bad_spec, _never_called)

    for bad_frac in (True, False, -0.01, 1.01, float("nan"), float("inf"), "0.1"):
        with pytest.raises(ValueError, match="max_unmappable_fraction"):
            dr.validate_slice(spec, [], "sys", max_unmappable_fraction=bad_frac)  # type: ignore[arg-type]


def test_eval_index_and_mixture_config_reject_bool_nan_and_invalid_bounds() -> None:
    for bad_n in (True, False, 0, -1, 1.5):
        with pytest.raises(ValueError, match="n must be an integer >= 1"):
            dr.EvalIndex(n=bad_n)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="max_doc_freq must be an integer >= 1"):
            dr.EvalIndex(max_doc_freq=bad_n)  # type: ignore[arg-type]

    for bad_total in (True, False, 0, -10, 5.5):
        with pytest.raises(ValueError, match="total_rows must be positive"):
            dr.MixtureConfig(total_rows=bad_total, domain_weights={"rust": 1.0})  # type: ignore[arg-type]

    for bad_seed in (True, False, 1.5, "42"):
        with pytest.raises(ValueError, match="seed must be an integer"):
            dr.MixtureConfig(total_rows=100, domain_weights={"rust": 1.0}, seed=bad_seed)  # type: ignore[arg-type]

    for bad_weights in (
        {},
        {"": 1.0},
        {"   ": 1.0},
        {"rust": True},
        {"rust": False},
        {"rust": float("nan")},
        {"rust": float("inf")},
        {"rust": -0.1},
        {"rust": 0.0},
    ):
        with pytest.raises(ValueError, match="domain_weights must be non-negative with a positive sum"):
            dr.MixtureConfig(total_rows=100, domain_weights=bad_weights)  # type: ignore[arg-type]

    for bad_cap in (True, False, 0.0, -0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="per_source_cap"):
            dr.MixtureConfig(total_rows=100, domain_weights={"rust": 1.0}, per_source_cap=bad_cap)  # type: ignore[arg-type]

    for bad_unv in (True, False, -0.1, 1.0, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="unverified_cap"):
            dr.MixtureConfig(total_rows=100, domain_weights={"rust": 1.0}, unverified_cap=bad_unv)  # type: ignore[arg-type]


def test_token_training_eval_rejects_bool_and_non_integer_parameters(tmp_path: Path) -> None:
    for bad_seq in (True, False, 1, 129, 16.0, "16"):
        with pytest.raises(ValueError, match="sequence_length must be between 2 and 128"):
            tte.tokenize_documents(["fn main() {}"], sequence_length=bad_seq)  # type: ignore[arg-type]

    for bad_seed in (True, False, 3.14, "42"):
        with pytest.raises(ValueError, match="seed must be an integer"):
            tte.build_model(bad_seed)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="seed must be an integer"):
            tte.run_token_training_evaluation(tmp_path / "run", steps=0, seed=bad_seed)  # type: ignore[arg-type]

    for bad_steps in (True, False, -1, 1.5, "2"):
        with pytest.raises(ValueError, match="steps must be nonnegative and learning_rate finite and positive"):
            tte.run_token_training_evaluation(tmp_path / "run", steps=bad_steps)  # type: ignore[arg-type]

    for bad_lr in (True, False, 0.0, -0.01, math.nan, math.inf, "0.01"):
        with pytest.raises(ValueError, match="steps must be nonnegative and learning_rate finite and positive"):
            tte.run_token_training_evaluation(tmp_path / "run", steps=0, learning_rate=bad_lr)  # type: ignore[arg-type]


def test_solverbee_service_bounds_tolerances_and_constructor_limits() -> None:
    base_spec = fam.pendulum_spec(0.5, 0.0, [1.0, 2.0], damping=0.1)
    valid_bytes = json.dumps(dict(base_spec, rtol=1e-6, atol=1e-8)).encode("utf-8")
    parsed = svc.validate_spec_bytes(valid_bytes)
    assert parsed["rtol"] == 1e-6

    for bad_rtol in (0.0, -1e-6, 1.5, True):
        raw = json.dumps(dict(base_spec, rtol=bad_rtol)).encode("utf-8")
        with pytest.raises(svc.SpecError, match="rtol must be a positive number"):
            svc.validate_spec_bytes(raw)

    for bad_atol in (0.0, -1e-8, 2.0, True):
        raw = json.dumps(dict(base_spec, atol=bad_atol)).encode("utf-8")
        with pytest.raises(svc.SpecError, match="atol must be a positive number"):
            svc.validate_spec_bytes(raw)

    for bad_atol_list in ([1e-8, 2.0], [0.0, 1e-8], [1e-8]):
        raw = json.dumps(dict(base_spec, atol=bad_atol_list)).encode("utf-8")
        with pytest.raises(svc.SpecError, match="atol array must have n positive entries"):
            svc.validate_spec_bytes(raw)

    for bad_bin in ("", "   ", None, 123):
        with pytest.raises(ValueError, match="binary must be a non-empty string"):
            svc.SolverBeeService(bad_bin)  # type: ignore[arg-type]

    for bad_conc in (True, False, 0, -1, 1.5):
        with pytest.raises(ValueError, match="max_concurrent must be an integer >= 1"):
            svc.SolverBeeService("/bin/true", max_concurrent=bad_conc)  # type: ignore[arg-type]

    for bad_timeout in (True, False, 0.0, -5.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="timeout_s must be a finite positive number"):
            svc.SolverBeeService("/bin/true", timeout_s=bad_timeout)  # type: ignore[arg-type]

    for bad_mem in (True, False, 0, -256, 1024.5):
        with pytest.raises(ValueError, match="mem_mb must be an integer >= 1"):
            svc.SolverBeeService("/bin/true", mem_mb=bad_mem)  # type: ignore[arg-type]


def test_science_bee_ollama_backend_atol_and_formulate_bounds() -> None:
    base_spec = fam.pendulum_spec(0.5, 0.0, [1.0, 2.0], damping=0.1)
    for bad_atol in (2.0, [1e-8, 2.0]):
        with pytest.raises(sb.SpecInvalid, match="atol"):
            sb.validate_spec(dict(base_spec, atol=bad_atol))

    for bad_timeout in (True, False, 0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(sb.BackendError, match="timeout_s must be a finite positive number"):
            sb.OllamaBackend("qwen:7b", timeout_s=bad_timeout)  # type: ignore[arg-type]

    for bad_temp in (True, False, -0.1, float("nan"), float("inf")):
        with pytest.raises(sb.BackendError, match="temperature must be a finite number >= 0"):
            sb.OllamaBackend("qwen:7b", temperature=bad_temp)  # type: ignore[arg-type]

    for bad_seed in (True, False, 1.5, "42"):
        with pytest.raises(sb.BackendError, match="seed must be an integer"):
            sb.OllamaBackend("qwen:7b", seed=bad_seed)  # type: ignore[arg-type]

    for bad_pred in (True, False, 0, -10, 12.5):
        with pytest.raises(sb.BackendError, match="num_predict must be an integer >= 1"):
            sb.OllamaBackend("qwen:7b", num_predict=bad_pred)  # type: ignore[arg-type]

    for bad_resp in (True, False, 0, -1, 64.5):
        with pytest.raises(sb.BackendError, match="max_response_bytes must be an integer >= 1"):
            sb.OllamaBackend("qwen:7b", max_response_bytes=bad_resp)  # type: ignore[arg-type]

    for bad_retries in (True, False, -1, 1.5):
        with pytest.raises(ValueError, match="max_retries must be an integer >= 0"):
            sb.formulate([], sb.FakeBackend([]), max_retries=bad_retries)  # type: ignore[arg-type]

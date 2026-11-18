# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and boundary hardening tests for solverbee (families, verifier, mcp) and dht_storage."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from hivemind.solverbee import families as fam
from hivemind.solverbee import mcp as mcp_mod
from hivemind.solverbee import service as svc_mod
from hivemind.solverbee import verifier as ver
from hivemind.swarm import dht_storage as dht


def test_families_check_params_rejects_nonpositive_rates_and_negative_damping() -> None:
    t_out = [1.0, 2.0]
    for param_name in ("alpha", "beta", "delta", "gamma"):
        for bad_val in (0.0, -0.5):
            with pytest.raises(fam.FamilyError, match=f"params.{param_name} must be > 0"):
                fam.named_spec("lotka_volterra", {param_name: bad_val}, [1.0, 1.0], t_out)

    for family, param_name, y0 in (
        ("harmonic", "zeta", [1.0, 0.0]),
        ("robertson", "k1", [1.0, 0.0, 0.0]),
        ("robertson", "k2", [1.0, 0.0, 0.0]),
        ("robertson", "k3", [1.0, 0.0, 0.0]),
        ("stiff_chain", "k", [1.0] * 10),
        ("brusselator", "a", [1.0, 1.0]),
        ("brusselator", "b", [1.0, 1.0]),
        ("van_der_pol", "mu", [1.0, 0.0]),
    ):
        with pytest.raises(fam.FamilyError, match=f"params.{param_name} must be >= 0"):
            fam.named_spec(family, {param_name: -0.1}, y0, t_out)

    with pytest.raises(fam.FamilyError, match="unknown family"):
        fam.describe_family("nonexistent_family")


def test_compile_slot_spec_decay_chain_rejects_non_list_rates_and_y0() -> None:
    for field_name in ("rates", "rates_per_min", "time_constants_s"):
        for bad_seq in ("1.0", 123, {}, []):
            with pytest.raises(fam.FamilyError, match="must be a list"):
                fam.compile_slot_spec(
                    {
                        "schema": fam.SLOT_SCHEMA,
                        "slot_family": "decay_chain",
                        "slots": {"t_out": [1.0, 2.0], field_name: bad_seq},
                    }
                )

    with pytest.raises(fam.FamilyError, match="y0 must be a list"):
        fam.compile_slot_spec(
            {
                "schema": fam.SLOT_SCHEMA,
                "slot_family": "decay_chain",
                "slots": {"t_out": [1.0, 2.0], "rates": [0.5], "y0": "not-a-list"},
            }
        )


def test_verifier_rejects_nan_duplicate_keys_and_unbounded_tolerances() -> None:
    base_spec = fam.pendulum_spec(0.5, 0.0, [1.0, 2.0], damping=0.1)
    spec_bytes = ver.canonical_json(base_spec)
    good_result = {
        "schema": ver.RESULT_SCHEMA,
        "id": base_spec["id"],
        "status": "ok",
        "t": [1.0, 2.0],
        "y": [[0.4, -0.1], [0.2, -0.1]],
        "input_sha256": hashlib.sha256(spec_bytes).hexdigest(),
    }

    nan_spec = b'{"schema":"hivemind-solver-spec/1","id":"x","problem":{"kind":"named","family":"pendulum","params":{}},"y0":[NaN,0.0],"t_out":[1.0]}'
    assert ver.verify(nan_spec, good_result)["verdict"] == "UNVERIFIED"

    dup_spec = b'{"schema":"hivemind-solver-spec/1","id":"x","id":"y","problem":{"kind":"named","family":"pendulum","params":{}},"y0":[0.5,0.0],"t_out":[1.0]}'
    assert ver.verify(dup_spec, good_result)["verdict"] == "UNVERIFIED"

    for bad_tol in ({"rtol": 2.0}, {"atol": 0.0}, {"atol": 1.5}, {"atol": [0.0, 1e-8]}, {"atol": [1e-8, 2.0]}):
        bad_s = dict(base_spec, **bad_tol)
        bad_bytes = ver.canonical_json(bad_s)
        bad_res = dict(good_result, input_sha256=hashlib.sha256(bad_bytes).hexdigest())
        cert = ver.verify(bad_bytes, bad_res, rerun=lambda _b: bad_res)
        assert cert["verdict"] == "FAILED"

    with pytest.raises(ValueError, match="problem must be a dict or None"):
        ver.certify_achieved_accuracy(spec_bytes, good_result, good_result, problem="not-a-dict")  # type: ignore[arg-type]


def test_mcp_validates_sandbox_mode_raw_line_and_read_frame_limit() -> None:
    svc = svc_mod.SolverBeeService("/bin/true", allow_unsandboxed=True)
    for bad_mode in ("", "invalid", "docker", None):
        with pytest.raises(ValueError, match="sandbox_mode"):
            mcp_mod.SolverBeeMcp(svc, sandbox_mode=bad_mode, problems_dir=None)  # type: ignore[arg-type]

    server = mcp_mod.SolverBeeMcp(svc, sandbox_mode="none", problems_dir=None)
    for bad_raw in ("not-bytes", 123, None, [b"x"]):
        with pytest.raises(TypeError, match="raw line must be bytes"):
            server.handle_line(bad_raw)  # type: ignore[arg-type]

    for bad_limit in (True, False, 0, -10, 1.5, "1024"):
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            mcp_mod.read_frame(io.BytesIO(b"hello\n"), limit=bad_limit)  # type: ignore[arg-type]


def test_dht_storage_rejects_bool_clocks_ttls_replication_chunk_sizes_and_blank_leases(tmp_path: Path) -> None:
    for bad_now in (True, False, float("nan"), float("inf"), "1.0"):
        with pytest.raises(ValueError, match="now_s must be finite"):
            dht._lease_clock(bad_now)  # type: ignore[arg-type]

    for field_name in ("alpha", "t_step_s", "rtt_s"):
        for bad_val in (True, False, "1.0"):
            kwargs = {"alpha": 2.0, "t_step_s": 1.0, "rtt_s": 0.1, field_name: bad_val}
            with pytest.raises(ValueError, match=field_name):
                dht.compute_lease_ttl_s(**kwargs)  # type: ignore[arg-type]

    for bad_rep in (True, False, 0, -1, 2.5, "3"):
        with pytest.raises(ValueError, match="replication_factor must be an integer >= 1"):
            dht.ContentAddressedSwarmStore(tmp_path / "bad_vault", replication_factor=bad_rep)  # type: ignore[arg-type]

    store = dht.ContentAddressedSwarmStore(tmp_path / "vault")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "sample.bin").write_bytes(b"hello world")

    for bad_chunk in (True, False, 0, -1, 16.5, "256"):
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            store.verify_proof_of_storage("node-1", "cid://sha256/00", bundle_dir / "sample.bin", "0" * 64, chunk_size=bad_chunk)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="chunk_size must be in"):
            store.plan_bundle_chunks(bundle_dir, chunk_size=bad_chunk)  # type: ignore[arg-type]

    for bad_cid, bad_worker in (("   ", "worker-1"), ("cid-1", "   "), (None, "worker-1"), ("cid-1", 123)):
        with pytest.raises(ValueError, match="shard_cid and worker_id must be non-empty"):
            store.grant_shard_lease(bad_cid, bad_worker, now_s=1.0)  # type: ignore[arg-type]

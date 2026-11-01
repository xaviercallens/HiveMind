# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hooks run by a real BOINC server's ``script_validator`` and ``script_assimilator`` daemons.

``script_validator --init_script "hivemind_validate files result_id"`` calls ``hivemind_validate <out files...> <id>``
and treats exit status 0 as "valid". ``script_assimilator --script "hivemind_assimilate wu_id files"`` calls
``hivemind_assimilate <wu_db_id> <out files...>`` for a workunit with a canonical result (or ``--error <mask> <id>``).

Both look up the workunit spec in ``$HIVEMIND_SPEC_DIR/<wu_name>.json`` (written by the work generator, server side,
so a volunteer cannot alter it) and write to ``$HIVEMIND_LEDGER_DIR``. BOINC's own credit is granted by the
validator daemon; the HiveMind ``verified_credit`` is recorded beside it in the ledger and the two are reported as two
different quantities.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from hivemind.boinc.real_client.bridge import Verdict
from hivemind.boinc.real_client.contributor_policy import BoincHostDirectory, replay_all_for_result
from hivemind.boinc.real_client.moe_bridge import MoEValidator, _reject
from hivemind.boinc.real_client.moe_profiles import TRAIN_APPS, profile_for_app
from hivemind.economy.compute_coin import CreditClaim, verified_credit

SPEC_DIR = Path(os.environ.get("HIVEMIND_SPEC_DIR", "hivemind_specs"))
LEDGER_DIR = Path(os.environ.get("HIVEMIND_LEDGER_DIR", "hivemind_ledger"))
SOLVER_SWEEP_APPS: tuple[str, ...] = ("hivemind_solver_sweep", "hivemind_solver")
CPU_SOLVER_PEAK_GFLOPS: float = 128.0
S1_CROSS_CPU_RTOL: float = 1e-6
S1_CROSS_CPU_ATOL: float = 1e-9


def _host_directory() -> BoincHostDirectory | None:
    """Result -> account lookup through the project database, when the validator shim says where the project is (community mode only)."""
    base, src = os.environ.get("HIVEMIND_STACK_BASE"), os.environ.get("HIVEMIND_BOINC_SRC")
    if not base or not src:
        return None
    from hivemind.boinc.real_client.server_stack import Stack

    return BoincHostDirectory(Stack(Path(base), Path(src), sys.executable, Path(__file__).resolve().parents[3]).sql)


def _meta(files: list[str]) -> tuple[dict, list[Path]]:
    paths = [Path(f) for f in files if not f.isdigit()]
    meta = json.loads(paths[0].read_text())
    return meta, paths


def wait_for_idle(quiet_s: float = 5.0, max_wait_s: float | None = None) -> float:
    """Block until nginx shows no in-flight inference request for ``quiet_s`` seconds, so the validator's GPU replay never competes
    with end users. Gives up after ``max_wait_s`` (default 30 min) so validation cannot starve forever. Returns seconds waited."""
    import urllib.request

    url = os.environ.get("HIVEMIND_IDLE_URL", "http://127.0.0.1:8091/nginx_status")
    limit = float(os.environ.get("HIVEMIND_IDLE_WAIT_S", 1800)) if max_wait_s is None else max_wait_s
    t0, quiet_since = time.time(), None
    while time.time() - t0 < limit:
        try:
            txt = urllib.request.urlopen(url, timeout=2).read().decode()
            writing = int(txt.split("Writing:")[1].split()[0]) - 1  # minus this poll
        except Exception:
            return time.time() - t0  # no status endpoint (e.g. a test box): do not block
        if writing <= 0:
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since >= quiet_s:
                break
        else:
            quiet_since = None
        time.sleep(1)
    return time.time() - t0


def _log_stderr(line: str) -> None:
    print(line, file=sys.stderr)


def _trajectory_max_rel_diff(
    y_a: Sequence[Sequence[float]],
    y_b: Sequence[Sequence[float]],
    rtol: float = S1_CROSS_CPU_RTOL,
    atol: float = S1_CROSS_CPU_ATOL,
) -> tuple[float, float]:
    """Return ``(max_abs_diff, max_bound_ratio)`` under the S-1 cross-CPU tolerance."""
    if len(y_a) != len(y_b):
        return math.inf, math.inf
    max_abs = 0.0
    max_ratio = 0.0
    for row_a, row_b in zip(y_a, y_b):
        if len(row_a) != len(row_b):
            return math.inf, math.inf
        for va, vb in zip(row_a, row_b):
            if not (isinstance(va, (int, float)) and isinstance(vb, (int, float))):
                return math.inf, math.inf
            if not (math.isfinite(va) and math.isfinite(vb)):
                return math.inf, math.inf
            err = abs(float(va) - float(vb))
            bound = rtol * max(abs(float(va)), abs(float(vb))) + atol
            ratio = err / bound if bound > 0.0 else (0.0 if err == 0.0 else math.inf)
            max_abs = max(max_abs, err)
            max_ratio = max(max_ratio, ratio)
    return max_abs, max_ratio


def validate_solver_sweep(
    spec: dict[str, Any],
    meta: dict[str, Any],
    host: str,
    peer_meta: dict[str, Any] | None = None,
) -> Verdict:
    """Validate a Wave 2 CPU solver or parameter-sweep workunit (``hivemind_solver_sweep`` / ``hivemind_solver``).

    Enforces:
    1. Workunit ID, app name, and canonical ``spec_sha256`` / ``input_sha256`` integrity.
    2. Structural finiteness, time grid match (``t_out``), and invariant drift bound across all parameter grid points.
    3. S-1 class-aware cross-CPU trajectory tolerance against ``reference_trajectories`` and/or a quorum-of-2
       replica result (``peer_meta``).
    """
    wu = str(spec.get("wu_id", ""))
    if meta.get("wu_id") != wu or meta.get("app") not in SOLVER_SWEEP_APPS:
        return _reject(wu, host, "result is not for this solver workunit/app")

    expected_sha = spec.get("spec_sha256")
    if expected_sha:
        claimed_sha = meta.get("spec_sha256") or meta.get("input_sha256")
        if claimed_sha != expected_sha:
            return _reject(wu, host, f"spec_sha256 mismatch: got {claimed_sha!r}, expected {expected_sha!r}")

    sweep_results = meta.get("sweep_results")
    if not isinstance(sweep_results, list) or not sweep_results:
        return _reject(wu, host, "sweep_results must be a non-empty list of trajectory certificates")

    expected_grid = spec.get("parameter_grid")
    if isinstance(expected_grid, list) and len(sweep_results) != len(expected_grid):
        return _reject(
            wu,
            host,
            f"sweep_results count {len(sweep_results)} != parameter_grid length {len(expected_grid)}",
        )

    expected_t_out = spec.get("t_out")
    max_allowed_drift = float(spec.get("max_invariant_drift", 1e-6))
    rtol = float(spec.get("s1_rtol", S1_CROSS_CPU_RTOL))
    atol = float(spec.get("s1_atol", S1_CROSS_CPU_ATOL))

    worst_drift = 0.0
    worst_ref_ratio = 0.0
    worst_abs_diff = 0.0

    ref_trajs = spec.get("reference_trajectories")
    peer_sweeps = peer_meta.get("sweep_results") if isinstance(peer_meta, dict) else None
    if isinstance(peer_sweeps, list) and len(peer_sweeps) != len(sweep_results):
        return _reject(wu, host, "quorum peer sweep_results length mismatch")

    for idx, item in enumerate(sweep_results):
        if not isinstance(item, dict) or item.get("status", "ok") != "ok":
            return _reject(wu, host, f"sweep point {idx} status is not 'ok'")
        if isinstance(expected_grid, list):
            p_val = item.get("param_value")
            if not isinstance(p_val, (int, float)) or abs(float(p_val) - float(expected_grid[idx])) > 1e-12:
                return _reject(wu, host, f"sweep point {idx} param_value {p_val!r} != {expected_grid[idx]!r}")
        t_vec = item.get("t")
        y_mat = item.get("y")
        if not isinstance(t_vec, list) or not isinstance(y_mat, list) or not t_vec or len(t_vec) != len(y_mat):
            return _reject(wu, host, f"sweep point {idx} malformed t/y trajectory")
        if isinstance(expected_t_out, list):
            if len(t_vec) != len(expected_t_out):
                return _reject(wu, host, f"sweep point {idx} t length {len(t_vec)} != {len(expected_t_out)}")
            for ti, (ta, tb) in enumerate(zip(t_vec, expected_t_out)):
                if not (isinstance(ta, (int, float)) and math.isfinite(ta)) or abs(float(ta) - float(tb)) > 1e-12:
                    return _reject(wu, host, f"sweep point {idx} t[{ti}] mismatch")
        for row in y_mat:
            if not isinstance(row, list) or not row or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in row
            ):
                return _reject(wu, host, f"sweep point {idx} contains non-finite state values")
        drift = float(item.get("invariant_drift", 0.0))
        if not math.isfinite(drift) or drift < 0.0 or drift > max_allowed_drift:
            return _reject(
                wu,
                host,
                f"sweep point {idx} invariant_drift {drift:.3e} exceeds bound {max_allowed_drift:.3e}",
                {"sweep_index": idx, "invariant_drift": drift, "bound": max_allowed_drift},
            )
        worst_drift = max(worst_drift, drift)

        if isinstance(ref_trajs, list) and idx < len(ref_trajs) and isinstance(ref_trajs[idx], list):
            abs_d, ratio = _trajectory_max_rel_diff(y_mat, ref_trajs[idx], rtol=rtol, atol=atol)
            worst_abs_diff = max(worst_abs_diff, abs_d)
            worst_ref_ratio = max(worst_ref_ratio, ratio)
            if ratio > 1.0:
                return _reject(
                    wu,
                    host,
                    f"sweep point {idx} exceeds S-1 reference tolerance (ratio={ratio:.3f})",
                    {"sweep_index": idx, "max_abs_diff": abs_d, "s1_ratio": ratio},
                )

        if isinstance(peer_sweeps, list):
            peer_item = peer_sweeps[idx]
            peer_y = peer_item.get("y") if isinstance(peer_item, dict) else None
            if not isinstance(peer_y, list):
                return _reject(wu, host, f"quorum peer point {idx} missing trajectory y")
            abs_d, ratio = _trajectory_max_rel_diff(y_mat, peer_y, rtol=rtol, atol=atol)
            worst_abs_diff = max(worst_abs_diff, abs_d)
            worst_ref_ratio = max(worst_ref_ratio, ratio)
            if ratio > 1.0:
                return _reject(
                    wu,
                    host,
                    f"quorum-of-2 trajectory disagreement at point {idx} (ratio={ratio:.3f})",
                    {"sweep_index": idx, "max_abs_diff": abs_d, "s1_ratio": ratio},
                )

    flops_claimed = float(meta.get("flops_claimed", len(sweep_results) * 2.5e9))
    wall_s = max(float(meta.get("wall_seconds", 1.0)), 1e-6)
    claim = CreditClaim(host, wu, flops_claimed, wall_s, CPU_SOLVER_PEAK_GFLOPS, True, True)
    detail = {
        "sweep_points": len(sweep_results),
        "worst_invariant_drift": worst_drift,
        "worst_s1_ratio": worst_ref_ratio,
        "max_abs_diff": worst_abs_diff,
        "quorum_checked": peer_sweeps is not None,
        "s1_rtol": rtol,
        "s1_atol": atol,
    }
    return Verdict(
        wu,
        host,
        True,
        f"verified {len(sweep_results)} solver sweep points: max_drift={worst_drift:.2e}, s1_ratio={worst_ref_ratio:.3f}",
        worst_abs_diff == 0.0,
        worst_abs_diff,
        verified_credit(claim).to_dict(),
        detail,
    )


def run_validation(args: list[str], mv: Any = None, log: Callable[[str], None] = _log_stderr) -> int:
    """The validation itself, shared by the one-shot ``hivemind_validate`` and the optional resident validator
    (``resident_validator.py``): both run exactly this function, so their verdicts cannot differ. ``mv`` is the validator
    object (default: a fresh ``MoEValidator``; the resident server passes one whose base model comes from a cache)."""
    meta, paths = _meta(args)
    wu = meta.get("wu_id", "")
    spec_path = SPEC_DIR / f"{wu}.json"
    if not wu or not spec_path.is_file() or "/" in wu or ".." in wu:
        log(f"unknown workunit {wu!r}")
        return 2
    spec = json.loads(spec_path.read_text())
    host = f"result-{args[-1]}" if args and args[-1].isdigit() else "host"
    if meta.get("app") in SOLVER_SWEEP_APPS:
        peer_meta = json.loads(paths[1].read_text()) if len(paths) >= 2 and paths[1].suffix == ".json" else None
        verdict = validate_solver_sweep(spec, meta, host, peer_meta=peer_meta)
    else:
        import torch

        wait_for_idle()
        if mv is None:
            mv = MoEValidator()
        if meta.get("app") in TRAIN_APPS and len(paths) >= 2:
            verdict = mv.validate_train(spec, paths[0], paths[1], host, replay_all=replay_all_for_result(args[-1] if args and args[-1].isdigit() else None, _host_directory()))
        elif meta.get("app") == "hivemind_moe_infer":
            adapter = torch.load(SPEC_DIR / f"{wu}.adapter.pt", map_location="cpu")["state"]
            verdict = mv.validate_infer(spec, paths[0], adapter, host)
        else:
            log("unknown app in result")
            return 3
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    (LEDGER_DIR / f"verdict_{wu}_{int(time.time())}.json").write_text(json.dumps(verdict.to_dict(), indent=2, sort_keys=True))
    log(f"{wu}: valid={verdict.valid} {verdict.reason}")
    return 0 if verdict.valid else 1


def validate(args: list[str]) -> int:
    """Entry point of ``hivemind_validate``. Optional (off by default): with ``--resident[=SOCKET]`` or the environment variable
    ``HIVEMIND_RESIDENT_VALIDATOR_SOCKET`` set, forward to a resident validator; if it is unreachable, warn once and run the
    one-shot path below. With neither set the behaviour is exactly the one-shot path."""
    sock = os.environ.get("HIVEMIND_RESIDENT_VALIDATOR_SOCKET")
    if args and (args[0] == "--resident" or args[0].startswith("--resident=")):
        flag, args = args[0], args[1:]
        sock = flag.partition("=")[2] or sock
        if not sock:
            print("hivemind_validate: --resident without a socket (set HIVEMIND_RESIDENT_VALIDATOR_SOCKET); running one-shot", file=sys.stderr)
    if sock:
        from hivemind.boinc.real_client.resident_validator import forward_validate

        rc = forward_validate(sock, args)
        if rc is not None:
            return rc
    return run_validation(args)


def assimilate(args: list[str]) -> int:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    if args and args[0] == "--error":
        (LEDGER_DIR / f"assimilate_error_{int(time.time())}.json").write_text(json.dumps({"error_mask": args[1], "wu_db_id": args[2]}))
        return 0
    meta, paths = _meta(args)
    wu = meta["wu_id"]
    entry: dict[str, Any] = {"wu_id": wu, "app": meta["app"], "t": time.time()}
    if meta["app"] in TRAIN_APPS:
        import torch

        from hivemind.boinc.real_client.moe_app import state_sha

        stem = profile_for_app(meta["app"])["chain_stem"]  # each app has its own chain: a rank-16 adapter cannot continue a rank-8 one
        state = torch.load(paths[1], map_location="cpu")["states"][-1]
        out = LEDGER_DIR / f"adapter_{wu}.pt"
        sha = state_sha(state)
        torch.save({"state": state, "sha256": sha, "wu_id": wu}, out)
        entry["adapter"], entry["adapter_sha256"] = out.name, sha
        entry["eval_loss_before"], entry["eval_loss_after"] = meta.get("eval_loss_before"), meta.get("eval_loss_after")
        entry["domain"] = meta.get("domain")
        entry["eval_by_domain_before"], entry["eval_by_domain_after"] = meta.get("eval_by_domain_before"), meta.get("eval_by_domain_after")
        # publish the chain head: the work generator starts the next workunit from this adapter
        tmp = LEDGER_DIR / f"{stem}.pt.tmp"
        torch.save({"state": state, "sha256": sha, "wu_id": wu}, tmp)
        tmp.replace(LEDGER_DIR / f"{stem}.pt")
        (LEDGER_DIR / f"{stem}.json").write_text(json.dumps(
            {"wu_id": wu, "sha256": sha, "eval_loss_before": meta.get("eval_loss_before"), "eval_loss_after": meta.get("eval_loss_after"), "t": time.time()}))
    elif meta["app"] in SOLVER_SWEEP_APPS:
        sweep_results = meta.get("sweep_results", [])
        cert_bytes = json.dumps(sweep_results, sort_keys=True, separators=(",", ":")).encode("utf-8")
        cert_sha = hashlib.sha256(cert_bytes).hexdigest()
        out_cert = LEDGER_DIR / f"solver_sweep_{wu}.json"
        out_cert.write_text(
            json.dumps(
                {
                    "wu_id": wu,
                    "app": meta["app"],
                    "family": meta.get("family", "van_der_pol"),
                    "parameter_name": meta.get("parameter_name", "mu"),
                    "sweep_count": len(sweep_results),
                    "certificate_sha256": cert_sha,
                    "sweep_results": sweep_results,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        entry["family"] = meta.get("family", "van_der_pol")
        entry["parameter_name"] = meta.get("parameter_name", "mu")
        entry["sweep_count"] = len(sweep_results)
        entry["certificate_sha256"] = cert_sha
        entry["certificate_file"] = out_cert.name
    else:
        entry["completions"] = meta.get("completions", [])
    with open(LEDGER_DIR / "assimilated.jsonl", "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return 0


def main() -> int:
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "validate":
        return validate(rest)
    if cmd == "assimilate":
        # script_assimilator passes: <wu_db_id> <files...> ; the helper wants the files only
        return assimilate(rest[1:] if rest and rest[0].isdigit() else rest)
    return 64


if __name__ == "__main__":
    raise SystemExit(main())

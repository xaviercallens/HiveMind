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

import json
import os
import sys
import time
from pathlib import Path

from hivemind.boinc.real_client.moe_bridge import MoEValidator

SPEC_DIR = Path(os.environ.get("HIVEMIND_SPEC_DIR", "hivemind_specs"))
LEDGER_DIR = Path(os.environ.get("HIVEMIND_LEDGER_DIR", "hivemind_ledger"))


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


def validate(args: list[str]) -> int:
    import torch

    meta, paths = _meta(args)
    wait_for_idle()
    wu = meta.get("wu_id", "")
    spec_path = SPEC_DIR / f"{wu}.json"
    if not wu or not spec_path.is_file() or "/" in wu or ".." in wu:
        print(f"unknown workunit {wu!r}", file=sys.stderr)
        return 2
    spec = json.loads(spec_path.read_text())
    mv = MoEValidator()
    host = f"result-{args[-1]}" if args and args[-1].isdigit() else "host"
    if meta.get("app") == "hivemind_qlora_moe" and len(paths) >= 2:
        verdict = mv.validate_train(spec, paths[0], paths[1], host)
    elif meta.get("app") == "hivemind_moe_infer":
        adapter = torch.load(SPEC_DIR / f"{wu}.adapter.pt", map_location="cpu")["state"]
        verdict = mv.validate_infer(spec, paths[0], adapter, host)
    else:
        print("unknown app in result", file=sys.stderr)
        return 3
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    (LEDGER_DIR / f"verdict_{wu}_{int(time.time())}.json").write_text(json.dumps(verdict.to_dict(), indent=2, sort_keys=True))
    print(f"{wu}: valid={verdict.valid} {verdict.reason}", file=sys.stderr)
    return 0 if verdict.valid else 1


def assimilate(args: list[str]) -> int:
    import torch

    from hivemind.boinc.real_client.moe_app import state_sha

    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    if args and args[0] == "--error":
        (LEDGER_DIR / f"assimilate_error_{int(time.time())}.json").write_text(json.dumps({"error_mask": args[1], "wu_db_id": args[2]}))
        return 0
    meta, paths = _meta(args)
    wu = meta["wu_id"]
    entry = {"wu_id": wu, "app": meta["app"], "t": time.time()}
    if meta["app"] == "hivemind_qlora_moe":
        state = torch.load(paths[1], map_location="cpu")["states"][-1]
        out = LEDGER_DIR / f"adapter_{wu}.pt"
        sha = state_sha(state)
        torch.save({"state": state, "sha256": sha, "wu_id": wu}, out)
        entry["adapter"], entry["adapter_sha256"] = out.name, sha
        entry["eval_loss_before"], entry["eval_loss_after"] = meta.get("eval_loss_before"), meta.get("eval_loss_after")
        entry["domain"] = meta.get("domain")
        entry["eval_by_domain_before"], entry["eval_by_domain_after"] = meta.get("eval_by_domain_before"), meta.get("eval_by_domain_after")
        # publish the chain head: the work generator starts the next workunit from this adapter
        tmp = LEDGER_DIR / "latest_adapter.pt.tmp"
        torch.save({"state": state, "sha256": sha, "wu_id": wu}, tmp)
        tmp.replace(LEDGER_DIR / "latest_adapter.pt")
        (LEDGER_DIR / "latest_adapter.json").write_text(json.dumps(
            {"wu_id": wu, "sha256": sha, "eval_loss_before": meta.get("eval_loss_before"), "eval_loss_after": meta.get("eval_loss_after"), "t": time.time()}))
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

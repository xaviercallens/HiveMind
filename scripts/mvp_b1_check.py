#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""MVP criterion B1 check: build shards from a real custom dataset (default: this repository's docs, paragraph per row) with the real Granite tokenizer and
verify them. Writes ``artifacts/mvp/B1_custom_shards.json``; ``passed`` is true only if every sub-check holds.

    HIVEMIND_SHARD_SECRET=<16+ chars> HF_HOME=<cache> python scripts/mvp_b1_check.py --tokenizer <snapshot dir or id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import make_custom_shards as mcs  # noqa: E402
from hivemind.boinc.real_client.moe_work import load_custom_shards  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--docs-glob", default="docs/*.md")
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    secret = mcs.secret_from_env(False)

    paragraphs = [p.strip() for f in sorted(ROOT.glob(args.docs_glob)) for p in f.read_text(encoding="utf-8").split("\n\n") if len(p.strip()) > 200]
    work = Path(tempfile.mkdtemp(prefix="b1_"))
    jsonl = work / "custom.jsonl"
    jsonl.write_text("\n".join(json.dumps({"text": p}) for p in paragraphs), encoding="utf-8")

    checks: dict[str, bool] = {}
    err = None
    manifest = {}
    try:
        rc = mcs.main(["--jsonl", str(jsonl), "--domain", "hivemind_docs", "--tokenizer", args.tokenizer, "--out", str(work / "out"), "--seq-len", str(args.seq_len)])
        train, held, manifest = load_custom_shards(work / "out", secret)
        tdocs, hdocs = mcs.split_docs(mcs.read_jsonl(jsonl, "text"), 0.1, 7)
        checks = {
            "script_exit_zero": rc == 0,
            "enough_chunks": len(train) >= 20 and len(held) >= 2,
            "documents_split_disjoint": set(tdocs).isdisjoint(hdocs) and len(tdocs) + len(hdocs) == len(set(mcs.read_jsonl(jsonl, "text"))),
            "sha256_and_hmac_verify": True,  # load_custom_shards raises otherwise
            "wrong_secret_rejected": _raises(lambda: load_custom_shards(work / "out", b"not-the-secret-0123456789")),
            "no_chunk_shared_between_train_and_heldout": not ({tuple(c) for c in train} & {tuple(c) for c in held}),
            "signed_with_a_private_secret": manifest["signed_with_default_public_secret"] is False,
        }
    except Exception as exc:  # noqa: BLE001 - the artifact must record the failure, not hide it
        err = f"{type(exc).__name__}: {exc}"
    result = {
        "criterion": "B1", "what": "custom dataset -> signed shards + held-out split", "dataset": f"{len(paragraphs)} paragraphs of {args.docs_glob} (this repository)",
        "tokenizer": args.tokenizer if not Path(args.tokenizer).exists() else "ibm-granite/granite-3.1-1b-a400m-base (local snapshot)",
        "checks": checks, "manifest": {k: manifest.get(k) for k in ("domain", "seq_len", "documents", "chunks", "train_sha256", "heldout_sha256")},
        "network": "sockets are blocked while building (mcs.no_network); the unit test proves a connect attempt raises",
        "error": err, "passed": bool(checks) and all(checks.values()) and err is None,
        "reason": err or ("" if checks and all(checks.values()) else f"failed checks: {[k for k, v in checks.items() if not v]}"),
    }
    out = ROOT / "artifacts/mvp/B1_custom_shards.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": checks, "chunks": result["manifest"]["chunks"], "error": err}, indent=2))
    return 0 if result["passed"] else 1


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())

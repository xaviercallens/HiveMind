# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Bring Your Own Bee: write a manifest template, sign it, and pre-check it against the live Hugging Face Hub.

    python scripts/byob_submit.py template > my_bee.json
    python scripts/byob_submit.py sign  --manifest my_bee.json --key-file ~/.hivemind/byob.key
    python scripts/byob_submit.py check --manifest my_bee.json --key-file ~/.hivemind/byob.key [--out entry.json]

``check`` runs exactly the static validation the hive runs (``hivemind.swarm.byob.static_validate``) with your own key as
the only keyring entry, then the evaluation stage. On a contributor machine no sandboxed GPU executor and no held-out
pack are configured, so evaluation reports BLOCKED: the best a self-check can show is ``stage: candidate``. The key file
must hold at least 32 bytes and must not be group/world readable (``chmod 600``). Exit code: 0 when static validation
passes, 1 on REJECT, 2 on BLOCKED.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hivemind.security.hivemind_firewall import load_secret_file  # noqa: E402
from hivemind.swarm.byob import (  # noqa: E402
    MANIFEST_SCHEMA,
    BeeManifest,
    HfHttpHubClient,
    SubmissionLedger,
    existing_bee_names,
    run_submission,
)

TEMPLATE = {
    "schema": MANIFEST_SCHEMA,
    "bee_name": "bee-your-domain",
    "contributor_id": "your-github-login",
    "hf_repo_id": "your-hf-user/your-adapter",
    "revision": "<40-hex commit sha from the Hub 'Files and versions' tab>",
    "kind": "lora_adapter",
    "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "license": "apache-2.0",
    "domain": "your-domain",
    "vram_tier_gb": 8,
    "eval_pack": "your-domain",
    "description": "One paragraph on what the bee is for and what data trained it.",
    "routing_keywords": ["keyword one", "keyword two"],
    "endpoint_url": "",
    "subfolder": "",
    "primary_file": "",
    "signature": "",
}


def _load(path: Path) -> BeeManifest:
    return BeeManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def main(argv: Sequence[str] | None = None, hub: HfHttpHubClient | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template", help="print a manifest template")
    for name in ("sign", "check"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--key-file", type=Path, required=True)
        p.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "template":
        print(json.dumps(TEMPLATE, indent=2))
        return 0
    key = load_secret_file(args.key_file)
    manifest = _load(args.manifest)
    if args.cmd == "sign":
        signed = manifest.signed(key)
        out = args.out or args.manifest
        out.write_text(json.dumps(signed.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"signed": str(out), "manifest_sha256": signed.digest()}))
        return 0
    with tempfile.TemporaryDirectory() as tmp:  # a self-check must not consume the contributor's own quota on disk
        entry = run_submission(manifest, keyring={manifest.contributor_id: key}, hub=hub or HfHttpHubClient(),
                               ledger=SubmissionLedger(Path(tmp) / "ledger.jsonl"), taken_names=existing_bee_names())
    text = json.dumps(entry, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    verdict = entry["static"]["verdict"]
    return {"ACCEPT": 0, "REJECT": 1}.get(verdict, 2)


if __name__ == "__main__":
    sys.exit(main())

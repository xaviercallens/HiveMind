#!/usr/bin/env python3
"""Re-check criterion B7 on the artifacts a previous `boinc_core_node_e2e.py` run left in its work directory.

The first e2e run asked the endpoint for 16 tokens while the batch workunit generated 24, so B7 was scored false by a
harness mismatch (the first 16 ids were identical). This script compares the full 24 ids for every prompt, using the
adapter the project assimilated and the completions the stock client returned. It is a re-check on stored artifacts,
not a fresh end-to-end run.

    <venv>/bin/python scripts/boinc_core_node_endpoint_check.py --work .../boinc/core1 --out artifacts/boinc_core_node/b7_recheck.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hivemind.boinc.real_client.moe_work import build_infer_spec  # noqa: E402
from hivemind.boinc.real_client.node_endpoint import NodeEndpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    srv = a.work / "server"
    batch = json.loads((srv / "upload" / "hm_moe_infer_001_0_0").read_text())["completions"]
    ep = NodeEndpoint(build_infer_spec("endpoint", [c["prompt"] for c in batch]), srv / "assimilated" / "hm_moe_good_001.pt")
    ep.start()
    rows = []
    for c in batch:
        body = json.dumps({"model": ep.name, "messages": [{"role": "user", "content": c["prompt"]}], "max_tokens": len(c["token_ids"])}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{ep.port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        got = json.loads(urllib.request.urlopen(req, timeout=180).read())["hivemind"]["token_ids"]
        rows.append({"prompt": c["prompt"], "n_tokens": len(got), "identical_to_batch": got == c["token_ids"]})
    out = {"bound": ep.httpd.server_address[0], "rows": rows, "B7": all(r["identical_to_batch"] for r in rows) and ep.httpd.server_address[0] == "127.0.0.1"}
    ep.stop()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if out["B7"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

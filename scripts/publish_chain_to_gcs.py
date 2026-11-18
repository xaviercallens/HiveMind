#!/usr/bin/env python3
"""Copy the BOINC QLoRA-MoE chain's new adapters, verdicts and index to a Google Cloud Storage prefix.

Read-only on the ledger. Only work units newer than ``--after-wu`` are uploaded (the earlier ones are already in
the 2026-11-05 snapshot). Adapters and verdicts are immutable per work unit, so they use ``--no-clobber``;
``assimilated.jsonl``, ``latest_adapter.json``, ``MANIFEST.sha256`` and ``README.md`` are rewritten on each run, so
a run is safe to repeat (every 15 minutes, or just before the T4 stops).

    python3 scripts/publish_chain_to_gcs.py --dest gs://BUCKET/PREFIX --dry-run
    python3 scripts/publish_chain_to_gcs.py --dest gs://BUCKET/PREFIX

The destination must be a prefix of your own that holds nothing else of value: the README and manifest are replaced.
A bucket with a public IAM binding is refused unless ``--allow-public`` is given: what is uploaded there is readable
by anyone. The base model, the training data and the signing key are never uploaded; ``latest_adapter.pt`` is not
copied either (its hash does not match the ledger's pointer).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_LEDGER = "/mnt/disks/disk-socrateai-local-1/boinc/stack/project/hmproj/hivemind_ledger"
WU_RE = re.compile(r"hmq_(\d+)")


def gcloud(*args: str, check: bool = True) -> str:
    out = subprocess.run(["gcloud", *args], capture_output=True, text=True, timeout=600)
    if check and out.returncode != 0:
        sys.exit(f"gcloud {' '.join(args[:3])} failed: {out.stderr.strip()[:300]}")
    return out.stdout


def is_public(bucket: str) -> bool:
    policy = json.loads(gcloud("storage", "buckets", "get-iam-policy", f"gs://{bucket}", "--format=json"))
    return any(m in ("allUsers", "allAuthenticatedUsers") for b in policy.get("bindings", []) for m in b.get("members", []))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def wu_number(path: Path) -> int:
    m = WU_RE.search(path.name)
    return int(m.group(1)) if m else -1


def readme(dest: str, rows: list[dict], after: int, n_files: int) -> str:
    gains = [r["eval_loss_before"] - r["eval_loss_after"] for r in rows if r.get("eval_loss_before") is not None and r.get("eval_loss_after") is not None]
    wus = sorted(int(WU_RE.search(r["wu_id"]).group(1)) for r in rows if WU_RE.search(str(r.get("wu_id", ""))))
    gain_line = (
        f"Held-out loss change per work unit (before minus after, positive = better): median {statistics.median(gains):+.4f}, "
        f"min {min(gains):+.4f}, max {max(gains):+.4f}, over {len(gains)} work units with both numbers."
        if gains
        else "No work unit in this upload has both held-out numbers."
    )
    scope = f"continuation after hmq_{after:05d}" if after > 0 else "every work unit still in the ledger"
    return f"""# HiveMind QLoRA-MoE adapter chain (BOINC), {scope}

Source: `hivemind_ledger` of the BOINC project `hmproj` on the T4 VM `socreateai-agora-hermes-node1`. Copied read-only by
`scripts/publish_chain_to_gcs.py`. Destination: `{dest}`. Work units in this upload: hmq_{wus[0]:05d} to hmq_{wus[-1]:05d} ({len(wus)}), {n_files} files, hashes in `MANIFEST.sha256`.

## What is here
- `adapters/adapter_hmq_*.pt`: QLoRA adapters of the chained work units (PyTorch pickles: load with `torch.load(path, weights_only=True)` only).
- `verdicts/verdict_*.json`: the validator's verdict per work unit, where the ledger has one.
- `assimilated.jsonl`: the whole ledger index (every work unit, with adapter sha256 and held-out losses).
- `latest_adapter.json`: the ledger's own pointer to its latest adapter.

## Read this before using any adapter
- {gain_line}
- These adapters were **not** put through the train-to-serve gate (`docs/TRAIN_TO_SERVE_GATE.md`), which requires a pooled held-out gain above 0 with a 95% interval that excludes 0. Small per-unit gains are not evidence that an adapter is good, and a single work unit's loss can move by chance.
- The base model (`granite-3.1-1b-a400m-base`) and the training data are not in this bucket.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--dest", required=True, help="gs://BUCKET/PREFIX")
    ap.add_argument("--after-wu", type=int, default=176, help="upload work units numbered above this")
    ap.add_argument("--allow-public", action="store_true", help="the bucket is readable by anyone and that is intended")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    m = re.fullmatch(r"gs://([a-z0-9._-]+)/(.+?)/?", a.dest)
    if not m:
        print("destination must look like gs://BUCKET/PREFIX")
        return 2
    bucket, prefix = m.groups()
    ledger = Path(a.ledger)
    if not (ledger / "assimilated.jsonl").is_file():
        print(f"no assimilated.jsonl in {ledger}")
        return 2
    if is_public(bucket) and not a.allow_public:
        print(f"REFUSED: gs://{bucket} has a public IAM binding: everything uploaded is readable by anyone. Pass --allow-public if that is intended.")
        return 3

    adapters = sorted((p for p in ledger.glob("adapter_hmq_*.pt") if wu_number(p) > a.after_wu), key=wu_number)
    verdicts = sorted((p for p in list(ledger.glob("verdict_*.json")) + list((ledger / "verdicts").glob("verdict_*.json")) if wu_number(p) > a.after_wu), key=wu_number)
    if not adapters:
        print(f"nothing newer than hmq_{a.after_wu:05d}")
        return 0

    rows = []
    for line in (ledger / "assimilated.jsonl").read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if wu_number(Path(str(row.get("wu_id", "")))) > a.after_wu:
            rows.append(row)

    with tempfile.TemporaryDirectory(dir=str(Path(a.ledger).parent.parent.parent.parent / "tmp") if (Path(a.ledger).parent.parent.parent.parent / "tmp").is_dir() else None) as tmp:
        stage = Path(tmp)
        files = [(p, f"adapters/{p.name}") for p in adapters] + [(p, f"verdicts/{p.name}") for p in verdicts]
        manifest = "".join(f"{sha256(p)}  {name}\n" for p, name in files)
        (stage / "MANIFEST.sha256").write_text(manifest)
        (stage / "README.md").write_text(readme(a.dest, rows, a.after_wu, len(files)))
        # the index and pointer are copied as they are
        for name in ("assimilated.jsonl", "latest_adapter.json"):
            if (ledger / name).is_file():
                (stage / name).write_bytes((ledger / name).read_bytes())
        total = sum(p.stat().st_size for p, _ in files)
        print(f"{len(adapters)} adapters + {len(verdicts)} verdicts ({total / 1e6:.1f} MB) after hmq_{a.after_wu:05d} -> {a.dest}")
        if a.dry_run:
            print("dry run: nothing uploaded")
            return 0
        for p, name in files:
            gcloud("storage", "cp", "--no-clobber", "-q", str(p), f"{a.dest}/{name}")
        for name in ("assimilated.jsonl", "latest_adapter.json", "MANIFEST.sha256", "README.md"):
            if (stage / name).is_file():
                gcloud("storage", "cp", "-q", str(stage / name), f"{a.dest}/{name}")
    listing = gcloud("storage", "ls", "-r", a.dest)
    print(f"uploaded; {len(listing.splitlines())} lines under {a.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

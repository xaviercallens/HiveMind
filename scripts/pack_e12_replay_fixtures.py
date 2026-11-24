#!/usr/bin/env python3
"""Pack the eight E12 replay-divergence fixtures (.pt) into one deterministic zip for the Zenodo record.

Why: a Zenodo record holds at most 100 files. Up to v4.1.0 the eight fixtures were carried over one file each from
v3.7.1; packing them frees seven slots for the v4.3.0 receipts without dropping any data. The fixtures are not in git;
they are downloaded from the public v4.1.0 record (10.5281/zenodo.23776302) and each one is checked against the MD5
that record publishes before it is packed.

Usage: python3 scripts/pack_e12_replay_fixtures.py --src DIR_WITH_PT_FILES --out ZIP_PATH
Writes the manifest to artifacts/experiments/E12_replay_fixtures_manifest.json (committed) and into the zip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_RECORD = "10.5281/zenodo.23776302"
EXPECTED_MD5 = json.loads((ROOT / "paper" / "zenodo_carried_over_v3.7.1.json").read_text(encoding="utf-8"))["md5"]
FIXED_DATE = (2026, 10, 24, 0, 0, 0)  # deterministic archive: same inputs give the same zip bytes


def md5(p: pathlib.Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = pathlib.Path(a.src)
    files = {}
    for name, want in sorted(EXPECTED_MD5.items()):
        got = md5(src / name)
        if got != want:
            raise SystemExit(f"{name}: md5 {got} != published {want}; refusing to pack")
        files[name] = {"md5": got, "bytes": (src / name).stat().st_size}
    manifest = {
        "archive": pathlib.Path(a.out).name,
        "purpose": "E12 replay-divergence fixtures (reference, tampered control, thread-pool, bf16, remote CPU, LoRA init), packed to respect the 100-file record cap",
        "source_record": SOURCE_RECORD,
        "files": files,
        "verified_against_published_md5": True,
    }
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:
        for name in sorted(files):
            zi = zipfile.ZipInfo(name, date_time=FIXED_DATE)
            z.writestr(zi, (src / name).read_bytes())
        zi = zipfile.ZipInfo("MANIFEST.json", date_time=FIXED_DATE)
        z.writestr(zi, json.dumps(manifest, indent=2) + "\n")
    manifest["archive_md5"] = md5(out)
    (ROOT / "artifacts" / "experiments" / "E12_replay_fixtures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"packed {len(files)} fixtures into {out} (md5 {manifest['archive_md5']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Publish a self-contained app version of the HiveMind apps through BOINC's own channel (signed files in the project's download directory).

  python3 scripts/boinc_publish_app_version.py --dist-dir DIR --version 102            # publish as BETA: only accounts with allow_beta_work receive it
  python3 scripts/boinc_publish_app_version.py --version 102 --promote                 # make it the default and retire older versions

The version carries four files per app: the launcher (main program) and the three content-hashed archives built by scripts/boinc_build_distribution.sh. Both
apps reference the SAME archive names, so the download directory stores each once and a client downloads each once. Files are signed by BOINC's
``update_versions`` with the project code-signing key. Run with the project venv (it imports the Stack helper):  <venv>/bin/python ...
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hivemind.boinc.real_client.server_stack import PLAN_CLASS, PLATFORM, Stack  # noqa: E402

APPS = {"hivemind_qlora_moe": "train", "hivemind_moe_infer": "infer"}


def newest(dist: Path, kind: str, suffix: str) -> str:
    found = sorted(dist.glob(f"hivemind_{kind}_*{suffix}"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit(f"no hivemind_{kind}_*{suffix} in {dist}: run scripts/boinc_build_distribution.sh first")
    return found[-1].name


def publish(st: Stack, dist: Path, version: int, beta: bool) -> None:
    names = {"RUNTIME": newest(dist, "runtime", ".tar.xz"), "SRC": newest(dist, "src", ".tar.xz"), "MODEL": newest(dist, "model", ".tar")}
    template = (ROOT / "deploy" / "boinc_dist" / "launcher.sh.in").read_text()
    for app, mode in APPS.items():
        vdir = st.root / "apps" / app / str(version) / f"{PLATFORM}__{PLAN_CLASS}"
        if (vdir / "version.xml").exists() and not beta:
            raise SystemExit(f"{vdir} exists: BOINC files are immutable, choose a new version number")
        vdir.mkdir(parents=True, exist_ok=True)
        exe = vdir / f"{app}_{version}_{PLATFORM}"
        text = template.replace("@MODE@", mode)
        for k, v in names.items():
            text = text.replace(f"@{k}@", v)
        exe.write_text(text)
        exe.chmod(0o755)
        for n in names.values():
            target = vdir / n
            if not target.exists():
                try:
                    os.link(dist / n, target)  # same filesystem: no second copy of gigabytes
                except OSError:
                    shutil.copy2(dist / n, target)
        files = f" <file>\n  <physical_name>{exe.name}</physical_name>\n  <main_program/>\n </file>\n" + "".join(
            f" <file>\n  <physical_name>{n}</physical_name>\n </file>\n" for n in names.values())
        (vdir / "version.xml").write_text(
            f"<version>\n{files} <plan_class>{PLAN_CLASS}</plan_class>\n <avg_ncpus>1</avg_ncpus>\n <flops>4e12</flops>\n"
            " <coproc>\n  <type>NVIDIA</type>\n  <count>1</count>\n </coproc>\n <gpu_ram>3500000000</gpu_ram>\n</version>\n"
        )
    print(st.update_versions())  # signs every file with the code-signing key, copies to download/, registers the app versions
    if beta:
        st.sql(f"use {st.dbname}; update app_version set beta=1 where version_num={version};")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack-base", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/stack"))
    ap.add_argument("--boinc-src", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/src"))
    ap.add_argument("--dist-dir", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/hivemind-dist"))
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--promote", action="store_true", help="clear the beta flag of --version and deprecate every lower version")
    a = ap.parse_args()
    st = Stack(a.stack_base, a.boinc_src, sys.executable, ROOT / "src")
    if a.promote:
        st.sql(f"use {st.dbname}; update app_version set beta=0 where version_num={a.version};")
        st.deprecate_versions_below(a.version)
        print(st.sql(f"use {st.dbname}; select id,appid,version_num,beta,deprecated from app_version order by id;"))
        return 0
    publish(st, a.dist_dir, a.version, beta=True)
    print(st.sql(f"use {st.dbname}; select id,appid,version_num,beta,deprecated from app_version order by id;"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

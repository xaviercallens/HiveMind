#!/usr/bin/env python3
"""Publish a self-contained app version of the HiveMind apps through BOINC's own channel (signed files in the project's download directory).

  python3 scripts/boinc_publish_app_version.py --dist-dir DIR --version 102            # BETA, under its own plan class (cuda_hivemind_beta)
  python3 scripts/boinc_publish_app_version.py --dist-dir DIR --version 103 --stable   # make it the default: same archives, stable plan class, retire older versions

The version carries four files per app: the launcher (main program) and the three content-hashed archives built by scripts/boinc_build_distribution.sh. Both
apps reference the SAME archive names, so the download directory stores each once and a client downloads each once. Files are signed by BOINC's
``update_versions`` with the project code-signing key. Run with the project venv (it imports the Stack helper):  <venv>/bin/python ...

WHY A BETA HAS ITS OWN PLAN CLASS. The scheduler's cache (sched_shmem.cpp) keeps only the highest version per (app, platform, plan class) and flags the lower
ones deprecated, beta or not; a host that has not opted in to beta work then skips the beta version (sched_version.cpp). A beta published under the SAME plan
class as the stable version therefore hides the stable one from the cache and nobody who has not opted in gets any work. This happened on 2026-10-26 (v102 over
v101) and was only noticed on 2026-10-28; ``assert_beta_cannot_mask`` now refuses it.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hivemind.boinc.real_client.moe_profiles import PROFILES  # noqa: E402
from hivemind.boinc.real_client.server_stack import PLAN_CLASS, PLATFORM, Stack  # noqa: E402

APPS = {
    "hivemind_qlora_moe": "train",
    "hivemind_qlora_moe_big": "train",
    "hivemind_moe_infer": "infer",
}
BETA_PLAN_CLASS = f"{PLAN_CLASS}_beta"
BETA_FLOPS_SCALE = "0.2"  # twice the stable class: a host that opted in to beta work prefers the beta version by projected speed


def plan_classes_for_app(app: str) -> tuple[str, str]:
    """Return ``(stable_plan_class, beta_plan_class)`` for ``app``."""
    for p in PROFILES.values():
        if p["app"] == app:
            stable = str(p["plan_class"])
            return stable, f"{stable}_beta"
    return PLAN_CLASS, BETA_PLAN_CLASS


def gpu_ram_mb_for_app(app: str) -> int:
    """Return the minimum GPU VRAM requirement (in MB) for ``app``."""
    for p in PROFILES.values():
        if p["app"] == app:
            return int(p["min_gpu_ram_mb"])
    return 3500


def render_version_xml(launcher: str, archives: Sequence[str], plan_class: str, gpu_ram_mb: int = 3500) -> str:
    files = f" <file>\n  <physical_name>{launcher}</physical_name>\n  <main_program/>\n </file>\n" + "".join(
        f" <file>\n  <physical_name>{n}</physical_name>\n </file>\n" for n in archives)
    return (
        f"<version>\n{files} <plan_class>{plan_class}</plan_class>\n <avg_ncpus>1</avg_ncpus>\n <flops>4e12</flops>\n"
        f" <coproc>\n  <type>NVIDIA</type>\n  <count>1</count>\n </coproc>\n <gpu_ram>{int(gpu_ram_mb) * 1_000_000}</gpu_ram>\n</version>\n"
    )


def plan_class_spec_with_beta(spec: str, stable_class: str = PLAN_CLASS, beta_class: str | None = None) -> str:
    """``spec`` (plan_class_spec.xml text) plus a ``<stable_class>_beta`` class copied from ``stable_class``, unless it already has it."""
    target_beta = beta_class or (BETA_PLAN_CLASS if stable_class == PLAN_CLASS else f"{stable_class}_beta")
    if f"<name>{target_beta}</name>" in spec:
        return spec
    block = re.search(rf"[ \t]*<plan_class>\s*<name>{re.escape(stable_class)}</name>.*?</plan_class>\n?", spec, re.S)
    if block is None:
        raise SystemExit(f"plan_class_spec.xml has no '{stable_class}' class to copy")
    beta = block.group(0).replace(f"<name>{stable_class}</name>", f"<name>{target_beta}</name>")
    beta = re.sub(r"<gpu_peak_flops_scale>[^<]*</gpu_peak_flops_scale>", f"<gpu_peak_flops_scale>{BETA_FLOPS_SCALE}</gpu_peak_flops_scale>", beta)
    return spec.replace("</plan_classes>", beta + ("" if beta.endswith("\n") else "\n") + "</plan_classes>")


def assert_beta_cannot_mask(plan_class: str, stable_plan_classes: Sequence[str]) -> None:
    """Refuse a beta whose plan class is also used by a live stable version (see the module docstring)."""
    if plan_class in stable_plan_classes:
        raise SystemExit(
            f"refusing to publish a beta under plan class '{plan_class}': a stable version already uses it, and the scheduler cache would keep only the "
            f"higher (beta) version, so hosts that have not opted in to beta work would get no work. Use '{BETA_PLAN_CLASS}'."
        )


def newest(dist: Path, kind: str, suffix: str) -> str:
    found = sorted(dist.glob(f"hivemind_{kind}_*{suffix}"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit(f"no hivemind_{kind}_*{suffix} in {dist}: run scripts/boinc_build_distribution.sh first")
    return found[-1].name


def publish(st: Stack, dist: Path, version: int, plan_class: str, beta: bool, apps: Mapping[str, str] | None = None) -> None:
    target_apps = apps or APPS
    live: list[str] = []
    if beta:
        live = st.sql(f"use {st.dbname}; select distinct plan_class from app_version where deprecated=0 and beta=0;").split()[1:]
        assert_beta_cannot_mask(plan_class, live)
        spec = st.root / "plan_class_spec.xml"
        spec_text = spec.read_text()
        spec_text = plan_class_spec_with_beta(spec_text)
        for app in target_apps:
            stable_cls, beta_cls = plan_classes_for_app(app)
            if f"<name>{stable_cls}</name>" in spec_text:
                spec_text = plan_class_spec_with_beta(spec_text, stable_class=stable_cls, beta_class=beta_cls)
        spec.write_text(spec_text)
    names = {"RUNTIME": newest(dist, "runtime", ".tar.xz"), "SRC": newest(dist, "src", ".tar.xz"), "MODEL": newest(dist, "model", ".tar")}
    template = (ROOT / "deploy" / "boinc_dist" / "launcher.sh.in").read_text()
    published_plan_classes: set[str] = set()
    for app, mode in target_apps.items():
        stable_cls, beta_cls = plan_classes_for_app(app)
        app_plan_class = (beta_cls if beta else stable_cls) if plan_class in (PLAN_CLASS, BETA_PLAN_CLASS) else plan_class
        if beta:
            assert_beta_cannot_mask(app_plan_class, live)
        published_plan_classes.add(app_plan_class)
        vdir = st.root / "apps" / app / str(version) / f"{PLATFORM}__{app_plan_class}"
        if (vdir / "version.xml").exists():
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
        (vdir / "version.xml").write_text(
            render_version_xml(exe.name, list(names.values()), app_plan_class, gpu_ram_mb=gpu_ram_mb_for_app(app))
        )
    print(st.update_versions())  # signs every file with the code-signing key, copies to download/, registers the app versions
    for cls in sorted(published_plan_classes):
        st.sql(f"use {st.dbname}; update app_version set beta={1 if beta else 0} where version_num={version} and plan_class='{cls}';")
    if not beta:  # promotion: every older version, of any plan class, is retired so the cache offers exactly this one
        st.sql(f"use {st.dbname}; update app_version set deprecated=1 where version_num<{version};")
    (st.root / "reread_db").touch()  # the feeder reloads the app versions
    print(st.sql(f"use {st.dbname}; select id,appid,version_num,plan_class,beta,deprecated from app_version order by id;"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack-base", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/stack"))
    ap.add_argument("--boinc-src", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/src"))
    ap.add_argument("--dist-dir", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/hivemind-dist"))
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--stable", action="store_true", help="publish under the stable plan class and retire every lower version (promotion); default is a beta")
    a = ap.parse_args()
    st = Stack(a.stack_base, a.boinc_src, sys.executable, ROOT / "src")
    publish(st, a.dist_dir, a.version, PLAN_CLASS if a.stable else BETA_PLAN_CLASS, beta=not a.stable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

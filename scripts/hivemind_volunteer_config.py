#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Settings for the STOCK BOINC client so a volunteer's GPU works for HiveMind only when it will not get in the way. Nothing to install: the BOINC client already has these features.

  python scripts/hivemind_volunteer_config.py gamer            # print the two files for a gaming PC
  python scripts/hivemind_volunteer_config.py gamer --game eldenring.exe --game cs2 --apply ~/.local/share/boinc   # write them into a BOINC data directory

Profiles
  gamer      GPU (and CPU) work only after the PC has been idle for a few minutes, and a work unit that is paused leaves GPU memory (the client quits it instead of keeping it loaded).
             Games also stop the work at once: the client suspends GPU work while any program named with --game is running (cc_config.xml <exclusive_gpu_app>), and keyboard/mouse
             activity suspends it too. Not a guarantee for a game that is not listed and played with a controller only: list it with --game.
  night      like gamer, and work only between 22:00 and 07:00 LOCAL time of the volunteer's PC.
  dedicated  a machine that is not used for anything else: always on, never pauses for activity. Use this only for a PC you do not use.

The files are the two the stock client reads: ``global_prefs_override.xml`` (preferences, read with ``boinccmd --read_global_prefs_override``) and ``cc_config.xml`` (client options, read with
``boinccmd --read_cc_config``). Existing files are never overwritten: with --apply the old ones are kept as ``*.hivemind-backup``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+()-]{0,63}$")


class ConfigError(ValueError):
    pass


def _default_games() -> tuple[str, ...]:
    """No universal list of games exists and guessing names would give a false sense of protection, so the default is empty: the activity rules do the work, and the volunteer names their games."""
    return ()


PROFILES = {
    "gamer": {"run_if_user_active": 0, "run_gpu_if_user_active": 0, "idle_time_to_run": 5.0, "leave_apps_in_memory": 0, "cpu_usage_limit": 50.0, "max_ncpus_pct": 50.0,
              "disk_max_used_gb": 10.0, "work_buf_min_days": 0.1, "work_buf_additional_days": 0.0},
    "night": {"run_if_user_active": 0, "run_gpu_if_user_active": 0, "idle_time_to_run": 5.0, "leave_apps_in_memory": 0, "cpu_usage_limit": 50.0, "max_ncpus_pct": 50.0,
              "disk_max_used_gb": 10.0, "work_buf_min_days": 0.1, "work_buf_additional_days": 0.0, "start_hour": 22.0, "end_hour": 7.0},
    "dedicated": {"run_if_user_active": 1, "run_gpu_if_user_active": 1, "idle_time_to_run": 0.0, "leave_apps_in_memory": 0, "cpu_usage_limit": 100.0, "max_ncpus_pct": 100.0,
                  "disk_max_used_gb": 20.0, "work_buf_min_days": 0.1, "work_buf_additional_days": 0.25},
}


def check_game(name: str) -> str:
    """A game name goes into an XML file the client reads: allow ordinary file names only (no path, no markup)."""
    if not _NAME.match(name) or ".." in name:
        raise ConfigError(f"game name {name!r} is not a plain program name (letters, digits, space . _ + ( ) -; at most 64)")
    return name


def prefs_xml(profile: str) -> str:
    if profile not in PROFILES:
        raise ConfigError(f"unknown profile {profile!r}; choose one of {', '.join(PROFILES)}")
    root = ET.Element("global_preferences")
    for key, value in PROFILES[profile].items():
        ET.SubElement(root, key).text = f"{value:g}" if isinstance(value, float) else str(value)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


LINUX_COMM_MAX = 15  # the kernel keeps only the first 15 characters of a process name; the client compares what the kernel reports


def with_linux_names(games: list[str]) -> list[str]:
    """Add the 15-character form of long names: on Linux the client sees a truncated name and would never match the full one. Harmless elsewhere."""
    out: list[str] = []
    for g in games:
        for name in (g, g[:LINUX_COMM_MAX]):
            if name not in out:
                out.append(name)
    return out


def cc_config_xml(games: list[str]) -> str:
    root = ET.Element("cc_config")
    opts = ET.SubElement(root, "options")
    ET.SubElement(opts, "suppress_net_info").text = "1"  # keep the client from reporting network details to the log: the volunteer's privacy first
    for g in with_linux_names([check_game(g) for g in games]):
        ET.SubElement(opts, "exclusive_gpu_app").text = g
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


def apply(dest: Path, prefs: str, cc: str) -> list[str]:
    if not dest.is_dir():
        raise ConfigError(f"{dest} is not a directory; this is the BOINC data directory (it holds client_state.xml)")
    done = []
    for name, text in (("global_prefs_override.xml", prefs), ("cc_config.xml", cc)):
        target = dest / name
        if target.exists():
            shutil.copy2(target, dest / f"{name}.hivemind-backup")
        target.write_text(text)
        done.append(str(target))
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", choices=sorted(PROFILES))
    ap.add_argument("--game", action="append", default=[], help="program name of a game that must pause GPU work (repeatable)")
    ap.add_argument("--apply", metavar="BOINC_DATA_DIR", help="write the files into this BOINC data directory (backups kept)")
    args = ap.parse_args(argv)
    try:
        games = [check_game(g) for g in [*_default_games(), *args.game]]
        prefs, cc = prefs_xml(args.profile), cc_config_xml(games)
        if args.apply:
            for path in apply(Path(args.apply).expanduser(), prefs, cc):
                print(f"wrote {path}")
            print("now tell the running client: boinccmd --read_global_prefs_override && boinccmd --read_cc_config")
        else:
            print(f"<!-- global_prefs_override.xml -->\n{prefs}<!-- cc_config.xml -->\n{cc}", end="")
    except ConfigError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

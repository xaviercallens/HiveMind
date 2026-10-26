#!/usr/bin/env python3
"""Build /etc/hivemind/t4-coder.env and t4-bee.env for the older producers from their running processes' environments.

One-time helper run by the owner (or on their behalf) so the systemd units can bring those producers back after a reboot. Reads
HIVEMIND_API_KEY / HIVEMIND_ENDPOINT_TOKEN from /proc/<pid>/environ of the agent and shield (same user), writes the files to a
staging directory with mode 0600, and prints ONLY lengths and whether the shield's token equals the agent's. Never prints a value.
Install the staged files with:  sudo install -m 0600 -o root -g root STAGE/t4-coder.env /etc/hivemind/t4-coder.env  (likewise bee)
"""

from __future__ import annotations

import argparse
from pathlib import Path


def procs() -> list[tuple[str, str]]:
    out = []
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                out.append((p.name, (p / "cmdline").read_bytes().replace(b"\0", b" ").decode()))
            except OSError:
                pass
    return out


def env_of(pid: str) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    return dict(x.decode(errors="replace").split("=", 1) for x in raw if b"=" in x)


def find(needles: list[str], not_needles: list[str] = ()) -> str | None:
    for pid, cmd in procs():
        if all(n in cmd for n in needles) and not any(n in cmd for n in not_needles) and "t4_capture" not in cmd:
            return pid
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    a = ap.parse_args()
    a.stage.mkdir(parents=True, exist_ok=True)
    sets = {
        "coder": (find(["hivemind_producer_agent.py", "--model coder="], ["/bee", "/u "]), find(["hivemind_producer_shield.py", "--port 8788"])),
        "bee": (find(["hivemind_producer_agent.py", "--model bee-stars="]), find(["hivemind_producer_shield.py", "--port 8789"])),
    }
    for name, (agent, shield) in sets.items():
        if not agent or not shield:
            print(f"{name}: process not found (agent={bool(agent)}, shield={bool(shield)}); skipped")
            continue
        ae, se = env_of(agent), env_of(shield)
        key, a_tok, s_tok = ae.get("HIVEMIND_API_KEY", ""), ae.get("HIVEMIND_ENDPOINT_TOKEN", ""), se.get("HIVEMIND_ENDPOINT_TOKEN", "")
        ok = bool(key) and bool(s_tok) and (a_tok == s_tok or not a_tok)
        print(f"{name}: key {len(key)} chars, shield token {len(s_tok)} chars, agent token {'same as shield' if a_tok == s_tok else ('absent' if not a_tok else 'DIFFERENT')} -> {'written' if ok else 'NOT written'}")
        if ok:
            f = a.stage / f"t4-{name}.env"
            f.write_text(f"HIVEMIND_API_KEY={key}\nHIVEMIND_ENDPOINT_TOKEN={s_tok}\n")
            f.chmod(0o600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

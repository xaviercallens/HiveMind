"""Checkpoint the volatile state of the T4 campaign so that a Spot-VM preemption does not lose it.

What is protected: the work that git ignores or that only exists in /tmp or in memory files — E15/E16 stage files and per-problem journals
(``artifacts/experiments/<EXX>/``), job logs, the queue state, the project memory notes, the watch scripts — plus a ``RESUME.md`` with the exact
restart commands and a ``MANIFEST.json`` with sha256 of every file. Committed code and the top-level ``artifacts/experiments/*.json`` reports are
already protected by the normal pushes of the feature branch.

How: git plumbing only (hash-object, a TEMPORARY index file, write-tree, commit-tree, update-ref), so the working tree, the real index and the
checked-out branch are never touched and running jobs are undisturbed. Each checkpoint is a commit on top of the previous one under
``refs/checkpoints/t4-campaign`` and is pushed fast-forward only (never forced) to ``origin`` as ``checkpoint/t4-campaign-state``. A second copy
is written as a tar.gz on the persistent data disk (last ``--keep`` archives).

Safety: file-name denylist (token, secret, key, .env, id_ed25519, credentials, ...), 25 MB cap per file, no model weights / parquet / .pt /
.safetensors / .bin, UTF-8 text-like files only for logs (tail-truncated). It never reads ~/.ssh, ~/.config or token files.

Triggers: ``--once``; ``--watch --every 1200`` (periodic); and, on a GCE VM, the metadata server (``instance/preempted``, ``maintenance-event``) is
polled every 5 s and an immediate checkpoint is taken on a preemption notice (the notice gives ~30 s).

    python3 scripts/checkpoint_state.py --once
    python3 scripts/checkpoint_state.py --watch --every 1200
    python3 scripts/checkpoint_state.py --once --dry-run        # list what would be stored
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
REF = "refs/checkpoints/t4-campaign"
REMOTE_BRANCH = "checkpoint/t4-campaign-state"
MAX_FILE = 25 * 1024 * 1024
LOG_TAIL = 400 * 1024
DENY_NAME = ("token", "secret", "credential", "passwd", "password", ".env", "id_ed25519", "id_rsa", "apikey", "api_key", ".pem", ".key")
DENY_SUFFIX = (".pt", ".pth", ".safetensors", ".bin", ".parquet", ".gguf", ".ckpt", ".npy", ".npz", ".tar", ".gz", ".zip")
SECRET_RE = re.compile(
    rb"(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{25,}|AIza[0-9A-Za-z_\-]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    rb"AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|(?i:bearer)\s+[A-Za-z0-9._\-]{25,}|(?i:zenodo[_-]?token)\s*[=:]\s*\S{10,}|"
    rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
)
SKIPPED_SECRET_LIKE: List[str] = []
JOB_DIR = Path(os.environ.get("CLAUDE_JOB_DIR", ""))
MEMORY_DIR = Path(os.environ.get("HIVEMIND_CHECKPOINT_MEMORY_DIR", ""))
DATA_DISK_DEFAULT = "/mnt/disks/disk-socrateai-local-1/hivemind_checkpoints"
META = "http://metadata.google.internal/computeMetadata/v1/instance/"

RESUME = """# Resume the T4 campaign after a Spot-VM preemption

State captured: {when} UTC from branch `{branch}` at `{head}`.

1. Get the code: `git fetch origin {branch}` and work from that branch (all code, prereg files and top-level `artifacts/experiments/*.json` reports are
   committed there). This checkpoint branch holds only the ignored/volatile state under `state/`.
2. Restore the volatile state (optional but saves GPU time): copy `state/artifacts/experiments/E15`, `E16` (stage JSONs and journals) back into
   `artifacts/experiments/` so that `--resume` skips finished stages and problems.
3. Check the GPU is idle (`nvidia-smi`), the Ollama daemon is up (`curl -s 127.0.0.1:11434/api/tags`) and the models are present
   (`ollama pull qwen2.5-coder:3b`; the 1.5b and 7b-instruct tags were already pulled). Hugging Face caches re-download automatically.
4. Relaunch the queue: `python3 scripts/run_t4_benchmark_queue.py --hours 9 --queue artifacts/experiments/T4_benchmark_queue_ext.json --log artifacts/experiments/T4_benchmark_queue_ext_log.json`.
   Jobs are resumable (`--resume`; completed stages/cells are skipped; a job killed mid-run loses only its current cell). The LoRA weights of E15 are
   NOT checkpointed (large): its train stage reruns from the rejection-sampling journal.
5. Read `state/memory/` for the full plan, decisions and caveats (project memory notes), and `state/MANIFEST.json` for file hashes.
6. Remember the rules: no merge/tag by the assistant, no Zenodo publish before the provenance lock passes, the PR #21 of GWAYA is not touched without the owner.
"""


def git(*args: str, env: Optional[Dict[str, str]] = None, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


def allowed(path: Path) -> bool:
    n = path.name.lower()
    return not (any(k in n for k in DENY_NAME) or n.endswith(DENY_SUFFIX) or path.stat().st_size > MAX_FILE and path.suffix not in (".log", ".jsonl"))


def collect(include_memory: bool = False) -> List[Tuple[str, bytes]]:
    """(path inside the checkpoint tree, content) of everything worth protecting."""
    items: List[Tuple[str, bytes]] = []

    def add(rel: str, p: Path, tail: bool = False) -> None:
        if not p.is_file() or not allowed(p):
            return
        data = p.read_bytes()
        if len(data) > (LOG_TAIL if tail else MAX_FILE):
            data = data[-LOG_TAIL:] if tail else data[:MAX_FILE]
        if SECRET_RE.search(data):  # content scan on top of the name denylist: never store a file that looks like it holds a credential or e-mail
            SKIPPED_SECRET_LIKE.append(rel)
            return
        items.append((rel, data))

    exp = ROOT / "artifacts" / "experiments"
    if exp.is_dir():
        for sub in sorted(d for d in exp.iterdir() if d.is_dir()):  # E15/, E16/, ... (git-ignored)
            for f in sorted(sub.rglob("*")):
                add(f"state/artifacts/experiments/{sub.name}/{f.relative_to(sub)}", f)
        for f in sorted(exp.glob("*partial*.json")) + sorted(exp.glob("T4_benchmark_queue*.json")):  # queue state + leftovers
            add(f"state/artifacts/experiments/{f.name}", f)
    if str(JOB_DIR) and JOB_DIR != Path(".") and JOB_DIR.is_dir():
        for f in sorted((JOB_DIR / "tmp" / "queue_logs").glob("*.log")):
            add(f"state/job_logs/{f.name}", f, tail=True)
        for f in sorted((JOB_DIR / "tmp").glob("*.py")) + sorted((JOB_DIR / "tmp").glob("*.json")):
            add(f"state/job_tmp/{f.name}", f)
    if include_memory and str(MEMORY_DIR) and MEMORY_DIR != Path(".") and MEMORY_DIR.is_dir():
        for f in sorted(MEMORY_DIR.glob("*.md")):
            add(f"state/memory/{f.name}", f)
    return items


def metadata_flag(path: str) -> Optional[str]:
    try:
        req = urllib.request.Request(META + path, headers={"Metadata-Flavor": "Google"})
        return urllib.request.urlopen(req, timeout=2).read().decode().strip()
    except Exception:
        return None


def is_private_remote(remote: str = "origin") -> bool:
    if os.environ.get("HIVEMIND_PRIVATE_REMOTE") == "1":
        return True
    try:
        url = git("remote", "get-url", remote, check=False).lower()
    except Exception:
        return False
    if not url or "github.com/xaviercallens/hivemind" in url or "github.com:xaviercallens/hivemind" in url:
        return False
    return True


def checkpoint(
    dry_run: bool,
    keep: int,
    data_disk: str,
    reason: str,
    include_memory: bool = False,
) -> Dict[str, object]:
    SKIPPED_SECRET_LIKE.clear()
    allow_mem = bool(include_memory and is_private_remote())
    items = collect(include_memory=allow_mem)
    head = git("rev-parse", "--short", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    manifest = {"written_utc": when, "reason": reason, "branch": branch, "head": head, "skipped_secret_like": list(SKIPPED_SECRET_LIKE),
                "files": {rel: hashlib.sha256(d).hexdigest() for rel, d in items}}
    items += [("state/MANIFEST.json", json.dumps(manifest, indent=1, sort_keys=True).encode()), ("RESUME.md", RESUME.format(when=when, branch=branch, head=head).encode())]
    total = sum(len(d) for _, d in items)
    if dry_run:
        return {"dry_run": True, "files": len(items), "bytes": total, "paths": [rel for rel, _ in items][:60]}
    with tempfile.TemporaryDirectory() as td:
        env = {"GIT_INDEX_FILE": str(Path(td) / "index")}
        for rel, data in items:
            blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=ROOT, input=data, capture_output=True, check=True).stdout.decode().strip()
            git("update-index", "--add", "--cacheinfo", f"100644,{blob},{rel}", env=env)
        tree = git("write-tree", env=env)
    parent = git("rev-parse", "-q", "--verify", REF, check=False)
    if parent and git("rev-parse", f"{parent}^{{tree}}") == tree:
        return {"status": "unchanged", "files": len(items), "tree": tree[:12]}
    msg = (f"checkpoint {when} UTC ({reason}): {len(items)} files, {total // 1024} KiB, from {branch}@{head}\n\n"
           "Volatile state of the T4 campaign (git-ignored experiment folders, job logs, queue state, memory notes). See RESUME.md.\n\n"
           "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01JWcvy9P3NMPXiV6BycKYyY")
    args = ["commit-tree", tree, "-m", msg] + (["-p", parent] if parent else [])
    commit = git(*args)
    git("update-ref", REF, commit)
    pushed = subprocess.run(["git", "push", "-q", "origin", f"{REF}:refs/heads/{REMOTE_BRANCH}"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    out: Dict[str, object] = {"status": "ok", "commit": commit[:12], "files": len(items), "bytes": total, "pushed": pushed.returncode == 0,
                              "push_error": pushed.stderr.strip()[:200] if pushed.returncode else None}
    try:  # second copy on the persistent data disk
        d = Path(data_disk)
        d.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for rel, data in items:
                ti = tarfile.TarInfo(rel)
                ti.size = len(data)
                ti.mtime = int(time.time())
                tf.addfile(ti, io.BytesIO(data))
        name = d / f"t4_campaign_state_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.tar.gz"
        name.write_bytes(buf.getvalue())
        for old in sorted(d.glob("t4_campaign_state_*.tar.gz"))[:-keep]:
            old.unlink()
        out["archive"] = str(name)
    except OSError as exc:
        out["archive_error"] = repr(exc)[:200]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=int, default=1200, help="seconds between periodic checkpoints in --watch mode")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep", type=int, default=6)
    ap.add_argument("--data-disk", default=DATA_DISK_DEFAULT)
    ap.add_argument(
        "--include-memory",
        action="store_true",
        help="Include MEMORY_DIR notes only when explicitly requested and the git remote is private (IP-140).",
    )
    a = ap.parse_args()
    if not (a.once or a.watch):
        ap.error("--once or --watch")
    if a.once:
        print(
            json.dumps(
                checkpoint(a.dry_run, a.keep, a.data_disk, "manual", include_memory=a.include_memory),
                indent=1,
            ),
            flush=True,
        )
        return 0
    last = 0.0
    on_gce = metadata_flag("name") is not None
    print(f"[checkpoint] watch every {a.every}s; GCE metadata {'available' if on_gce else 'not reachable'}", flush=True)
    while True:
        preempted = on_gce and (metadata_flag("preempted") == "TRUE" or (metadata_flag("maintenance-event") or "NONE") not in ("NONE", ""))
        if preempted or time.time() - last >= a.every:
            try:
                res = checkpoint(
                    a.dry_run,
                    a.keep,
                    a.data_disk,
                    "PREEMPTION NOTICE" if preempted else "periodic",
                    include_memory=a.include_memory,
                )
                print(f"[checkpoint] {time.strftime('%H:%M:%S')} {json.dumps(res)}", flush=True)
            except Exception as exc:  # a failed checkpoint must never kill the watcher
                print(f"[checkpoint] failed: {exc!r}", flush=True)
            last = time.time()
            if preempted:
                time.sleep(25)
        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Let a HiveMind inference producer also train the QLoRA MoE, without ever taking the GPU from its users.

A producer that serves inference can run a stock BOINC client attached to the project; the repository's governor (scripts/t4_governor.py,
already parameterised) keeps the client's GPU mode at ``never`` while requests are in flight and lets it train only after a quiet period,
by default 1800 s inside the European night (22:00-07:00 Europe/Paris) and 7200 s outside it. This tool looks at the producer's GPU, says which
training profiles it can run (standard >= 3500 MiB, big >= 10000 MiB), and writes the files to install:

  governor.service     systemd unit running the governor for this host
  nginx_status.conf    loopback stub_status the governor watches; the producer's inference endpoint must pass through the same nginx
  SETUP.txt            the commands, in order

The BOINC account key is never an argument or written to a file: the attach command reads it from $HIVEMIND_BOINC_ACCOUNT_KEY.
Usage: python scripts/producer_train_kit.py --server-url URL --out DIR [--gpu-mib N] [--repo DIR]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hivemind.boinc.real_client.moe_profiles import PROFILES, profiles_for_gpu  # noqa: E402

FREE_VRAM_MARGIN_MIB = 800  # beyond the profile floor, mirroring the T4's 4300 free for a 3500 floor
STATE_DIR = "/var/lib/hivemind-producer"


def parse_gpu_mib(nvidia_smi_csv: str) -> int | None:
    """Largest ``memory.total`` (MiB) from ``nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`` output, or None."""
    best: int | None = None
    for line in nvidia_smi_csv.splitlines():
        try:
            mib = int(line.rsplit(",", 1)[1].strip())
        except (IndexError, ValueError):
            continue
        best = mib if best is None else max(best, mib)
    return best


def detect_gpu_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_gpu_mib(out.stdout) if out.returncode == 0 else None


def render_governor_unit(repo: Path, gpu_mib: int, night_tz: str = "Europe/Paris", night_start_h: int = 22, night_end_h: int = 7,
                         idle_start_s: int = 1800, day_idle_s: int = 7200) -> str:
    profiles = profiles_for_gpu(gpu_mib)
    if not profiles:
        raise ValueError(f"a GPU with {gpu_mib} MiB is below the smallest training profile")
    min_free = PROFILES[profiles[-1]]["min_gpu_ram_mb"] + FREE_VRAM_MARGIN_MIB  # free VRAM needed to start: the smallest profile this GPU can run
    args = (f"--status-url http://127.0.0.1:8091/nginx_status --boinccmd /usr/bin/boinccmd --lib /usr/lib/x86_64-linux-gnu "
            f"--boinc-dir /var/lib/boinc-client --rpc-port 31416 --telemetry {STATE_DIR}/governor.jsonl --state-file {STATE_DIR}/governor_state.json "
            f"--hold-file {STATE_DIR}/HOLD --idle-start-s {idle_start_s} --day-idle-s {day_idle_s} --min-free-mib {min_free} "
            f"--night-tz {night_tz} --night-start-h {night_start_h} --night-end-h {night_end_h}")
    return (
        "[Unit]\nDescription=HiveMind producer training governor (train only when nobody is asking for inference)\n"
        "After=boinc-client.service nginx.service\nRequires=boinc-client.service\n\n"
        f"[Service]\nExecStartPre=/usr/bin/install -d {STATE_DIR}\nExecStart=/usr/bin/python3 {repo}/scripts/t4_governor.py {args}\n"
        "Restart=on-failure\nRestartSec=10\n\n[Install]\nWantedBy=multi-user.target\n"
    )


def render_nginx_status() -> str:
    return ("# Loopback-only request counter the governor polls once a second. Route the producer's inference endpoint through this nginx\n"
            "# (as on the T4) so in-flight requests are visible; other GPU users are still caught by the governor's GPU-utilisation guard.\n"
            "server {\n    listen 127.0.0.1:8091;\n    location /nginx_status { stub_status; allow 127.0.0.1; deny all; }\n}\n")


def render_setup(server_url: str, gpu_mib: int, out: Path, repo: Path) -> str:
    profiles = profiles_for_gpu(gpu_mib)
    which = ", ".join(f"{n} ({PROFILES[n]['app']}, plan class {PROFILES[n]['plan_class']})" for n in profiles)
    return (
        f"GPU memory {gpu_mib} MiB -> can train: {which}\n\n"
        "1. Install the BOINC client and nginx:   sudo apt-get install -y boinc-client nginx\n"
        f"2. Install the stub_status site:         sudo cp {out}/nginx_status.conf /etc/nginx/conf.d/ && sudo nginx -t && sudo systemctl reload nginx\n"
        "3. Put the producer's inference endpoint behind this nginx (so the governor sees requests).\n"
        "4. Hold the GPU until the governor decides:   boinccmd --set_gpu_mode never 0\n"
        f"5. Attach to the project (key from the environment, never typed here):\n"
        f"     boinccmd --project_attach {server_url} \"$HIVEMIND_BOINC_ACCOUNT_KEY\"\n"
        f"6. Install and start the governor:        sudo cp {out}/governor.service /etc/systemd/system/hivemind-producer-governor.service\n"
        "                                          sudo systemctl daemon-reload && sudo systemctl enable --now hivemind-producer-governor\n"
        f"   (the unit runs {repo}/scripts/t4_governor.py, so this repository must be checked out there)\n"
        "7. Check:  cat /var/lib/hivemind-producer/governor_state.json   -> state 'yielded' while serving, 'training' after the quiet period.\n\n"
        "Pause training at any time:  touch /var/lib/hivemind-producer/HOLD   (remove the file to resume).\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", required=True, help="the project's master URL (must be reachable from this host)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu-mib", type=int, default=0, help="override the detected GPU memory")
    ap.add_argument("--repo", type=Path, default=Path("/opt/hivemind/HiveMind"))
    a = ap.parse_args(argv)
    gpu = a.gpu_mib or detect_gpu_mib()
    if not gpu:
        print("no NVIDIA GPU detected (pass --gpu-mib to override)", file=sys.stderr)
        return 2
    if not profiles_for_gpu(gpu):
        print(f"a GPU with {gpu} MiB is below the smallest training profile ({PROFILES['standard']['min_gpu_ram_mb']} MiB)", file=sys.stderr)
        return 3
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "governor.service").write_text(render_governor_unit(a.repo, gpu))
    (a.out / "nginx_status.conf").write_text(render_nginx_status())
    (a.out / "SETUP.txt").write_text(render_setup(a.server_url, gpu, a.out, a.repo))
    print(f"wrote governor.service, nginx_status.conf, SETUP.txt to {a.out}; eligible: {', '.join(profiles_for_gpu(gpu))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

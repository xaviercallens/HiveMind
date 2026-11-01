# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Prepare and drive a private stock BOINC client data directory (anonymous platform, GPU app).

Nothing here touches a system BOINC install: the client binary is run by the current user with its own
``--dir`` and GUI-RPC port.
"""

from __future__ import annotations

import os
import re
import secrets
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from hivemind.boinc.real_client.app import APP_NAME, APP_VERSION
from hivemind.boinc.real_client.project_server import PLAN_CLASS, PLATFORM

APP_FILE = f"{APP_NAME}_{APP_VERSION}_{PLATFORM}"


def project_dir_name(master_url: str) -> str:
    """``escape_project_url`` from ``lib/url.cpp``: drop the scheme, non-[alnum . - _] -> ``_``, drop one trailing ``_``."""
    rest = master_url.split("://", 1)[-1]
    out = re.sub(r"[^A-Za-z0-9._-]", "_", rest)
    return out[:-1] if out.endswith("_") else out


def toy_app(python: str = "python3") -> dict[str, Any]:
    return {"name": APP_NAME, "version": APP_VERSION, "python": python, "module": "hivemind.boinc.real_client.app", "args": [], "env": {}, "gpu_ram_mb": 1000, "plan_class": PLAN_CLASS}


def moe_apps(venv_python: str, gpu_ram_mb: int = 3600) -> list[dict[str, Any]]:
    """The QLoRA MoE training and batch-inference apps; they need the venv with peft + bitsandbytes."""
    env = {"HF_HUB_OFFLINE": "1"}
    return [
        {"name": "hivemind_qlora_moe", "version": 100, "python": venv_python, "module": "hivemind.boinc.real_client.moe_app", "args": ["--mode", "train"], "env": env, "gpu_ram_mb": gpu_ram_mb, "plan_class": PLAN_CLASS},
        {"name": "hivemind_moe_infer", "version": 100, "python": venv_python, "module": "hivemind.boinc.real_client.moe_app", "args": ["--mode", "infer"], "env": env, "gpu_ram_mb": gpu_ram_mb, "plan_class": PLAN_CLASS},
    ]


def big_moe_app(venv_python: str, gpu_ram_mb: int = 10000) -> dict[str, Any]:
    """The 10 GB+ VRAM QLoRA MoE big-profile training app (`cuda_hivemind_big`)."""
    return {
        "name": "hivemind_qlora_moe_big",
        "version": 100,
        "python": venv_python,
        "module": "hivemind.boinc.real_client.moe_app",
        "args": ["--mode", "train"],
        "env": {"HF_HUB_OFFLINE": "1"},
        "gpu_ram_mb": gpu_ram_mb,
        "plan_class": "cuda_hivemind_big",
    }


def solver_sweep_app(python: str = "python3", script_body: str | None = None) -> dict[str, Any]:
    """CPU-only deterministic solver parameter-sweep app (`cpu_hivemind_solver`)."""
    app_spec: dict[str, Any] = {
        "name": "hivemind_solver_sweep",
        "version": 100,
        "python": python,
        "module": "hivemind.boinc.real_client.app",
        "args": [],
        "env": {},
        "gpu_ram_mb": 0,
        "plan_class": "cpu_hivemind_solver",
        "cpu_only": True,
    }
    if script_body is not None:
        app_spec["script_body"] = script_body
    return app_spec


def prepare_data_dir(
    data_dir: Path,
    master_url: str,
    src_root: Path,
    python: str = "python3",
    gpu_flops: float = 4e12,
    apps: list[dict[str, Any]] | None = None,
    authenticator: str | None = None,
    install_apps: bool = True,
) -> dict[str, Any]:
    """Client data dir. ``install_apps=False`` writes no ``app_info.xml``: the client then fetches the project's
    signed app versions itself (use with a real server and the ``authenticator`` its web stack returned)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    pdir = data_dir / "projects" / project_dir_name(master_url)
    pdir.mkdir(parents=True, exist_ok=True)
    auth = authenticator or secrets.token_hex(16)
    (data_dir / "gui_rpc_auth.cfg").write_text(secrets.token_hex(8))
    (data_dir / "cc_config.xml").write_text(
        "<cc_config>\n <log_flags><task>1</task><file_xfer>1</file_xfer><sched_ops>1</sched_ops>"
        "<coproc_debug>1</coproc_debug><cpu_sched>1</cpu_sched><task_debug>1</task_debug></log_flags>\n"
        " <options><allow_multiple_clients>1</allow_multiple_clients><use_all_gpus>1</use_all_gpus>"
        "<report_results_immediately>1</report_results_immediately></options>\n</cc_config>\n"
    )
    (data_dir / "global_prefs_override.xml").write_text(
        "<global_preferences><run_if_user_active>1</run_if_user_active><run_gpu_if_user_active>1</run_gpu_if_user_active>"
        "<max_ncpus_pct>50</max_ncpus_pct><cpu_usage_limit>100</cpu_usage_limit><work_buf_min_days>0.05</work_buf_min_days>"
        "<work_buf_additional_days>0</work_buf_additional_days><suspend_cpu_usage>0</suspend_cpu_usage></global_preferences>\n"
    )
    (data_dir / f"account_{project_dir_name(master_url)}.xml").write_text(
        f"<account><master_url>{master_url}</master_url><authenticator>{auth}</authenticator></account>\n"
    )
    # The science app: a launcher script. The client runs it in a slot directory (cwd) with the WU command line.
    if not install_apps:
        return {"project_dir": pdir, "authenticator": auth}
    xml = ["<app_info>\n"]
    for a in apps or [toy_app(python)]:
        version = int(a.get("version", APP_VERSION))
        fname = f"{a['name']}_{version}_{PLATFORM}"
        launcher = pdir / fname
        if a.get("script_body"):
            launcher.write_text(str(a["script_body"]))
        else:
            exports = "".join(f"export {k}={v}\n" for k, v in a.get("env", {}).items())
            launcher.write_text(
                "#!/bin/bash\n"
                "unset LD_LIBRARY_PATH\n"  # the private libboinc path must not leak into the python app
                f"export PYTHONPATH={src_root}\n"
                f"{exports}"
                f"exec {a['python']} -W ignore -m {a['module']} {' '.join(a.get('args', []))} \"$@\"\n"
            )
        launcher.chmod(0o755)
        raw_ram = int(a.get("gpu_ram_mb", 0))
        cpu_only = bool(a.get("cpu_only", False)) or str(a.get("plan_class", "")).startswith("cpu_") or raw_ram <= 0
        plan_class = str(a.get("plan_class", "cpu_hivemind_solver" if cpu_only else PLAN_CLASS))
        app_flops = float(a.get("flops", gpu_flops))
        if cpu_only:
            resource_lines = f"  <flops>{app_flops}</flops>\n"
        else:
            gpu_ram_bytes = raw_ram * 1_000_000 if raw_ram < 1_000_000 else raw_ram
            resource_lines = (
                f"  <flops>{app_flops}</flops><gpu_ram>{gpu_ram_bytes}</gpu_ram>\n"
                "  <coproc><type>NVIDIA</type><count>1</count></coproc>\n"
            )
        xml.append(
            f" <app><name>{a['name']}</name><user_friendly_name>{a['name']}</user_friendly_name></app>\n"
            f" <file_info><name>{fname}</name><executable/></file_info>\n"
            " <app_version>\n"
            f"  <app_name>{a['name']}</app_name><version_num>{version}</version_num><platform>{PLATFORM}</platform>\n"
            f"  <plan_class>{plan_class}</plan_class><avg_ncpus>1</avg_ncpus><max_ncpus>1</max_ncpus>\n"
            f"{resource_lines}"
            f"  <file_ref><file_name>{fname}</file_name><main_program/></file_ref>\n"
            " </app_version>\n"
        )
    xml.append("</app_info>\n")
    (pdir / "app_info.xml").write_text("".join(xml))
    return {"project_dir": pdir, "authenticator": auth}


def parse_boinccmd_cc_status(raw: str) -> dict[str, Any]:
    """Parse ``boinccmd --get_cc_status`` output into a structured dictionary."""
    out: dict[str, Any] = {
        "network_connection_status": "unknown",
        "cpu": {"suspended": False, "suspend_reason": None, "current_mode": "unknown", "perm_mode": "unknown", "perm_delay_sec": 0.0},
        "gpu": {"suspended": False, "suspend_reason": None, "current_mode": "unknown", "perm_mode": "unknown", "perm_delay_sec": 0.0},
        "network": {"suspended": False, "suspend_reason": None, "current_mode": "unknown", "perm_mode": "unknown", "perm_delay_sec": 0.0},
    }
    section: str | None = None
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("network connection status:"):
            out["network_connection_status"] = s.split(":", 1)[1].strip()
        elif s == "CPU status":
            section = "cpu"
        elif s == "GPU status":
            section = "gpu"
        elif s == "Network status":
            section = "network"
        elif section in ("cpu", "gpu", "network"):
            sec = out[section]
            if s == "not suspended":
                sec["suspended"] = False
                sec["suspend_reason"] = None
            elif s.startswith("suspended:"):
                sec["suspended"] = True
                sec["suspend_reason"] = s.split(":", 1)[1].strip()
            elif s.startswith("current mode:"):
                sec["current_mode"] = s.split(":", 1)[1].strip()
            elif s.startswith("perm mode:"):
                sec["perm_mode"] = s.split(":", 1)[1].strip()
            elif s.startswith("perm becomes current in"):
                m = re.search(r"(\d+(?:\.\d+)?)\s*sec", s)
                if m:
                    sec["perm_delay_sec"] = float(m.group(1))
    return out


def parse_boinccmd_project_status(raw: str) -> list[dict[str, Any]]:
    """Parse ``boinccmd --get_project_status`` output, preserving top-level project fields over nested ``GUI URL:`` blocks."""
    projects: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    gui_url: dict[str, str] | None = None
    float_keys = {"resource_share", "user_total_credit", "user_expavg_credit", "host_total_credit", "host_expavg_credit", "elapsed_time", "project_files_downloaded"}
    int_keys = {"nrpc_failures", "master_fetch_failures", "jobs_succeeded", "jobs_failed"}

    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("========"):
            continue
        if re.match(r"^\d+\)\s*-+", s):
            if cur is not None:
                projects.append(cur)
            cur = {"gui_urls": []}
            gui_url = None
            continue
        if cur is None:
            continue
        if s == "GUI URL:":
            gui_url = {}
            cur["gui_urls"].append(gui_url)
            continue
        if ":" in s:
            k_raw, v_raw = s.split(":", 1)
            k = k_raw.strip()
            v = v_raw.strip()
            if gui_url is not None and k in ("name", "description", "URL"):
                gui_url[k] = v
                if k == "URL":
                    gui_url = None
                continue
            gui_url = None
            norm_k = k.lower().replace(" ", "_").replace("-", "_").replace("'", "")
            if norm_k in float_keys:
                try:
                    cur[norm_k] = float(v)
                except ValueError:
                    cur[norm_k] = v
            elif norm_k in int_keys:
                try:
                    cur[norm_k] = int(v)
                except ValueError:
                    cur[norm_k] = v
            else:
                cur[norm_k] = v
    if cur is not None:
        projects.append(cur)
    return projects


def parse_boinccmd_tasks(raw: str) -> list[dict[str, Any]]:
    """Parse ``boinccmd --get_tasks`` output into a list of task dictionaries."""
    tasks: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    float_keys = {"estimated_cpu_time_remaining", "elapsed_task_time", "fraction_done", "current_cpu_time"}
    int_keys = {"app_version_num", "exit_status", "slot"}

    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("========"):
            continue
        if re.match(r"^\d+\)\s*-+", s):
            if cur is not None:
                tasks.append(cur)
            cur = {}
            continue
        if cur is None or ":" not in s:
            continue
        k_raw, v_raw = s.split(":", 1)
        norm_k = k_raw.strip().lower().replace(" ", "_").replace("-", "_")
        v = v_raw.strip()
        if norm_k in float_keys:
            try:
                cur[norm_k] = float(v)
            except ValueError:
                cur[norm_k] = v
        elif norm_k in int_keys:
            try:
                cur[norm_k] = int(v)
            except ValueError:
                cur[norm_k] = v
        else:
            cur[norm_k] = v
    if cur is not None:
        tasks.append(cur)
    return tasks


class Client:
    def __init__(self, boinc_bin: Path, lib_dir: Path, data_dir: Path, rpc_port: int) -> None:
        self.bin, self.lib, self.dir, self.port = boinc_bin, lib_dir, data_dir, rpc_port
        self.proc: subprocess.Popen | None = None
        self.log_path = data_dir / "client_stdout.log"

    def _env(self) -> dict[str, str]:
        return dict(os.environ, LD_LIBRARY_PATH=str(self.lib))

    def start(self) -> None:
        self._log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            [str(self.bin), "--dir", str(self.dir), "--gui_rpc_port", str(self.port), "--allow_multiple_clients"],
            stdout=self._log, stderr=subprocess.STDOUT, env=self._env(), cwd=str(self.dir), start_new_session=True,
        )

    def cmd(self, *args: str, timeout: int = 20) -> str:
        pw = (self.dir / "gui_rpc_auth.cfg").read_text().strip()
        cc = self.bin.with_name("boinccmd")
        r = subprocess.run([str(cc), "--host", f"127.0.0.1:{self.port}", "--passwd", pw, *args], capture_output=True, text=True, env=self._env(), timeout=timeout)
        return r.stdout + r.stderr

    def kill(self) -> None:
        """Simulate preemption: SIGKILL the client and its science apps (process group), no clean shutdown."""
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()

    def quit(self, timeout: int = 60) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        try:
            self.cmd("--quit")
            self.proc.wait(timeout=timeout)
        except Exception:
            self.kill()

    def wait_rpc(self, timeout: int = 60) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            try:
                if "poll status" in self.cmd("--get_state") or "======== Projects ========" in self.cmd("--get_state"):
                    return True
            except Exception:
                time.sleep(0.1)
            time.sleep(1)
        return False

    def get_cc_status(self) -> dict[str, Any]:
        """Return structured CPU/GPU/Network daemon status parsed from ``boinccmd --get_cc_status``."""
        return parse_boinccmd_cc_status(self.cmd("--get_cc_status"))

    def get_project_status(self) -> list[dict[str, Any]]:
        """Return structured attached project records parsed from ``boinccmd --get_project_status``."""
        return parse_boinccmd_project_status(self.cmd("--get_project_status"))

    def get_tasks(self) -> list[dict[str, Any]]:
        """Return structured task records parsed from ``boinccmd --get_tasks``."""
        return parse_boinccmd_tasks(self.cmd("--get_tasks"))

    def set_gpu_mode(self, mode: str, duration: float = 0.0) -> str:
        """Set GPU execution mode (``always``, ``auto``, or ``never``)."""
        if mode not in ("always", "auto", "never"):
            raise ValueError(f"Invalid GPU mode: {mode!r}")
        return self.cmd("--set_gpu_mode", mode, str(duration))

    def set_run_mode(self, mode: str, duration: float = 0.0) -> str:
        """Set CPU execution mode (``always``, ``auto``, or ``never``)."""
        if mode not in ("always", "auto", "never"):
            raise ValueError(f"Invalid run mode: {mode!r}")
        return self.cmd("--set_run_mode", mode, str(duration))

    def set_network_mode(self, mode: str, duration: float = 0.0) -> str:
        """Set network mode (``always``, ``auto``, or ``never``)."""
        if mode not in ("always", "auto", "never"):
            raise ValueError(f"Invalid network mode: {mode!r}")
        return self.cmd("--set_network_mode", mode, str(duration))

    def attach_project(self, master_url: str, authenticator: str) -> str:
        """Attach the client to a BOINC project at ``master_url`` using ``authenticator``."""
        return self.cmd("--project_attach", master_url, authenticator)

    def update_project(self, master_url: str) -> str:
        """Trigger an immediate scheduler RPC contact for ``master_url``."""
        return self.cmd("--project", master_url, "update")

    def suspend_project(self, master_url: str) -> str:
        """Suspend work for ``master_url``."""
        return self.cmd("--project", master_url, "suspend")

    def resume_project(self, master_url: str) -> str:
        """Resume work for ``master_url``."""
        return self.cmd("--project", master_url, "resume")

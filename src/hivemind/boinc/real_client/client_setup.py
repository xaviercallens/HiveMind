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
    return {"name": APP_NAME, "version": APP_VERSION, "python": python, "module": "hivemind.boinc.real_client.app", "args": [], "env": {}, "gpu_ram_mb": 1000}


def moe_apps(venv_python: str, gpu_ram_mb: int = 3600) -> list[dict[str, Any]]:
    """The QLoRA MoE training and batch-inference apps; they need the venv with peft + bitsandbytes."""
    env = {"HF_HUB_OFFLINE": "1"}
    return [
        {"name": "hivemind_qlora_moe", "version": 100, "python": venv_python, "module": "hivemind.boinc.real_client.moe_app", "args": ["--mode", "train"], "env": env, "gpu_ram_mb": gpu_ram_mb},
        {"name": "hivemind_moe_infer", "version": 100, "python": venv_python, "module": "hivemind.boinc.real_client.moe_app", "args": ["--mode", "infer"], "env": env, "gpu_ram_mb": gpu_ram_mb},
    ]


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
        fname = f"{a['name']}_{a['version']}_{PLATFORM}"
        launcher = pdir / fname
        exports = "".join(f"export {k}={v}\n" for k, v in a["env"].items())
        launcher.write_text(
            "#!/bin/bash\n"
            "unset LD_LIBRARY_PATH\n"  # the private libboinc path must not leak into the python app
            f"export PYTHONPATH={src_root}\n"
            f"{exports}"
            f"exec {a['python']} -W ignore -m {a['module']} {' '.join(a['args'])} \"$@\"\n"
        )
        launcher.chmod(0o755)
        xml.append(
            f" <app><name>{a['name']}</name><user_friendly_name>{a['name']}</user_friendly_name></app>\n"
            f" <file_info><name>{fname}</name><executable/></file_info>\n"
            " <app_version>\n"
            f"  <app_name>{a['name']}</app_name><version_num>{a['version']}</version_num><platform>{PLATFORM}</platform>\n"
            f"  <plan_class>{PLAN_CLASS}</plan_class><avg_ncpus>1</avg_ncpus><max_ncpus>1</max_ncpus>\n"
            f"  <flops>{gpu_flops}</flops><gpu_ram>{a['gpu_ram_mb']}</gpu_ram>\n"
            "  <coproc><type>NVIDIA</type><count>1</count></coproc>\n"
            f"  <file_ref><file_name>{fname}</file_name><main_program/></file_ref>\n"
            " </app_version>\n"
        )
    xml.append("</app_info>\n")
    (pdir / "app_info.xml").write_text("".join(xml))
    return {"project_dir": pdir, "authenticator": auth}


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
                pass
            time.sleep(1)
        return False

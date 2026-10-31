# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""A real BOINC server (BOINC 7.24 built from source) running unprivileged on loopback, driven from Python.

Everything runs as the current user: a private MariaDB (socket only, no TCP), a private Apache with mod_php on
127.0.0.1:<port>, and BOINC's own daemons (feeder, transitioner, file_deleter, ``script_validator``,
``script_assimilator``). Nothing is installed as a system service and nothing listens on a non-loopback address.

Apps are real, code-signed app versions (no anonymous platform): the stock client downloads and verifies them.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PLATFORM = "x86_64-pc-linux-gnu"
PLAN_CLASS = "cuda_hivemind"
CPU_SOLVER_PLAN_CLASS = "cpu_hivemind_solver"
from hivemind.boinc.real_client.moe_profiles import PROFILES  # noqa: E402  (plan classes of the training profiles)
APACHE_MODULES = {
    "mpm_prefork_module": "mod_mpm_prefork.so", "authz_core_module": "mod_authz_core.so", "authz_host_module": "mod_authz_host.so",
    "access_compat_module": "mod_access_compat.so", "alias_module": "mod_alias.so", "dir_module": "mod_dir.so", "mime_module": "mod_mime.so",
    "env_module": "mod_env.so", "setenvif_module": "mod_setenvif.so", "rewrite_module": "mod_rewrite.so", "cgi_module": "mod_cgi.so",
    "headers_module": "mod_headers.so", "autoindex_module": "mod_autoindex.so",  # log_config/unixd are built in on Ubuntu
}


def sh(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True, timeout: int = 600) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"{' '.join(cmd)} -> rc {r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r


class Stack:
    def __init__(self, base: Path, boinc_src: Path, venv_python: str, repo_src: Path, port: int = 8081, name: str = "hmproj") -> None:
        self.base, self.src, self.venv_python, self.repo_src, self.port, self.name = base, boinc_src, venv_python, repo_src, port, name
        self.root = base / "project" / name
        self.url = f"http://127.0.0.1:{port}"
        self.master_url = f"{self.url}/{name}/"
        self.dbname, self.dbuser = name, "hmboinc"
        self.pwfile = base / "db_password"
        for d in ("log", "run", "project"):
            (base / d).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ database
    def sql(self, statement: str) -> str:
        return sh(["mariadb", "--no-defaults", "-u", "root", "-S", "/run/mysqld/mysqld.sock", "-e", statement]).stdout

    SOCKET = "/run/mysqld/mysqld.sock"  # /run is tmpfs: `sudo install -d -o $USER /run/mysqld` after every reboot

    def start_db(self) -> bool:
        """Private MariaDB, socket only (--skip-networking: no TCP listener at all). Idempotent."""
        if Path(self.SOCKET).exists():
            return True
        if not Path("/run/mysqld").is_dir():
            raise RuntimeError("/run/mysqld is missing: run `sudo install -d -o $USER /run/mysqld` once per boot")
        subprocess.Popen(
            ["mariadbd", "--no-defaults", "--basedir=/usr", f"--datadir={self.base}/mysql/data", f"--socket={self.SOCKET}",
             f"--pid-file={self.base}/run/mariadb.pid", "--skip-networking", f"--log-error={self.base}/log/mariadb.err"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        end = time.time() + 60
        while time.time() < end:
            if sh(["mariadb", "--no-defaults", "-u", "root", "-S", self.SOCKET, "-e", "select 1"], check=False).returncode == 0:
                return True
            time.sleep(1)
        return False

    def stop_db(self) -> None:
        sh(["mariadb-admin", "--no-defaults", "-u", "root", "-S", self.SOCKET, "shutdown"], check=False)

    def make_project(self) -> None:
        if not self.pwfile.exists():
            self.pwfile.write_text(secrets.token_hex(16))
            self.pwfile.chmod(0o600)
        pw = self.pwfile.read_text().strip()
        self.sql(f"CREATE USER IF NOT EXISTS '{self.dbuser}'@'localhost' IDENTIFIED BY '{pw}'; ALTER USER '{self.dbuser}'@'localhost' IDENTIFIED BY '{pw}'; "
                 f"CREATE DATABASE IF NOT EXISTS {self.dbname}; GRANT ALL ON {self.dbname}.* TO '{self.dbuser}'@'localhost'; FLUSH PRIVILEGES;")
        env = dict(os.environ, USER=os.environ.get("USER", "callensxavier"))
        sh(["python3", "make_project", "--srcdir", str(self.src), "--project_root", str(self.root), "--key_dir", str(self.root / "keys"),
            "--url_base", self.url + "/", "--project_host", "127.0.0.1", "--db_name", self.dbname, "--db_user", self.dbuser,
            "--db_passwd", pw, "--db_host", "localhost", "--no_query", "--delete_prev_inst", "--drop_db_first", self.name, "HiveMind local BOINC server"],
           cwd=self.src / "tools", env=env, timeout=900)

    # ------------------------------------------------------------------ apache
    def write_apache(self) -> Path:
        lines = ["ServerName 127.0.0.1", f"Listen 127.0.0.1:{self.port}", f"PidFile {self.base}/run/apache.pid",
                 f"ErrorLog {self.base}/log/apache_error.log", "LogLevel warn", "ServerRoot /etc/apache2", f"Mutex file:{self.base}/run"]
        lines += [f"LoadModule {m} /usr/lib/apache2/modules/{f}" for m, f in APACHE_MODULES.items()]
        lines += ["LoadModule php_module /usr/lib/apache2/modules/libphp8.1.so", "TypesConfig /etc/mime.types", "DirectoryIndex index.php index.html",
                  "<FilesMatch \\.php$>", "  SetHandler application/x-httpd-php", "</FilesMatch>",
                  f"<Directory />\n  Require all denied\n</Directory>", f"<Directory {self.root}>\n  Options Indexes FollowSymLinks\n  AllowOverride All\n  Require all granted\n</Directory>",
                  f"DocumentRoot {self.root}/html/user", f"Include {self.root}/{self.name}.httpd.conf"]
        conf = self.base / "apache.conf"
        conf.write_text("\n".join(lines) + "\n")
        return conf

    def start_apache(self) -> None:
        sh(["apache2", "-f", str(self.base / "apache.conf"), "-k", "start"], timeout=60)

    def stop_apache(self) -> None:
        sh(["apache2", "-f", str(self.base / "apache.conf"), "-k", "stop"], check=False, timeout=60)

    # ------------------------------------------------------------------ daemons and config
    def configure(self, validator_shim: str, assimilator_shim: str, apps: list[str]) -> None:
        cfg = self.root / "config.xml"
        text = cfg.read_text()
        text = re.sub(r"<daemons>.*?</daemons>", "", text, flags=re.S)
        # make_project derives <host> from the URL ("127"); the start script matches it to the machine's hostname
        import socket

        text = re.sub(r"<host>[^<]*</host>", f"<host>{socket.gethostname().split('.')[0]}</host>", text, count=1)
        daemons = ["<daemons>", "<daemon><cmd>feeder -d 3</cmd></daemon>", "<daemon><cmd>transitioner -d 3</cmd></daemon>", "<daemon><cmd>file_deleter -d 3</cmd></daemon>"]
        for app in apps:  # same executable twice needs distinct output/pid files, or the start script skips the second
            daemons.append(f"<daemon><cmd>script_validator -d 3 --app {app} --init_script \"{validator_shim} files result_id\"</cmd>"
                           f"<output>validator_{app}.log</output><pid_file>validator_{app}.pid</pid_file></daemon>")
            daemons.append(f"<daemon><cmd>script_assimilator -d 3 --app {app} --script \"{assimilator_shim} wu_id files\"</cmd>"
                           f"<output>assimilator_{app}.log</output><pid_file>assimilator_{app}.pid</pid_file></daemon>")
        daemons.append("</daemons>")
        text = text.replace("</boinc>", "\n".join(daemons) + "\n</boinc>") if "</boinc>" in text else text + "\n".join(daemons)
        cfg.write_text(text)

    def fix_scheduler_list(self) -> None:
        """make_project leaves a commented <scheduler> line and a placeholder ``href="url"`` link in schedulers.txt:
        the client tried the placeholder first ("Couldn't resolve host name") before falling back. Write a clean list."""
        (self.root / "html" / "user" / "schedulers.txt").write_text(f"<scheduler>{self.url}/{self.name}_cgi/cgi</scheduler>\n")

    def write_plan_class_spec(self) -> None:
        """One plan class per training profile; the big one has a higher ``min_gpu_ram_mb`` so the scheduler never offers its app to a small GPU.
        Also registers ``cpu_hivemind_solver`` (Wave 2 Issue #270) as a CPU-only plan class without NVIDIA/CUDA coprocessor requirements."""
        classes = {PLAN_CLASS: 3500, **{p["plan_class"]: p["min_gpu_ram_mb"] for p in PROFILES.values()}}
        body = "".join(
            f" <plan_class>\n  <name>{name}</name>\n  <gpu_type>nvidia</gpu_type>\n  <cuda/>\n"
            f"  <min_gpu_ram_mb>{mb}</min_gpu_ram_mb>\n  <gpu_ram_used_mb>{mb}</gpu_ram_used_mb>\n  <cpu_frac>0.2</cpu_frac>\n  <avg_ncpus>1</avg_ncpus>\n"
            "  <gpu_peak_flops_scale>0.1</gpu_peak_flops_scale>\n </plan_class>\n"
            for name, mb in classes.items()
        )
        cpu_body = (
            f" <plan_class>\n  <name>{CPU_SOLVER_PLAN_CLASS}</name>\n"
            "  <cpu_frac>1.0</cpu_frac>\n  <avg_ncpus>1</avg_ncpus>\n </plan_class>\n"
        )
        (self.root / "plan_class_spec.xml").write_text("<plan_classes>\n" + body + cpu_body + "</plan_classes>\n")

    # ------------------------------------------------------------------ apps, templates, work
    def install_app(self, app: str, launcher_body: str, n_inputs: int, outputs: list[str], max_nbytes: int, version: int = 100,
                    plan_class: str = PLAN_CLASS, gpu_ram_mb: int = 3500, *, cpu_only: bool = False,
                    min_quorum: int = 1, target_nresults: int | None = None) -> None:
        is_cpu = cpu_only or plan_class == CPU_SOLVER_PLAN_CLASS
        quorum = max(1, int(min_quorum))
        target_res = max(quorum, int(target_nresults if target_nresults is not None else quorum))
        max_err = max(2, target_res)
        max_tot = max(3, target_res + 2)
        vdir = self.root / "apps" / app / str(version) / f"{PLATFORM}__{plan_class}"
        vdir.mkdir(parents=True, exist_ok=True)
        exe = vdir / f"{app}_{version}_{PLATFORM}"
        exe.write_text(launcher_body)
        exe.chmod(0o755)
        if is_cpu:
            (vdir / "version.xml").write_text(
                f"<version>\n <file>\n  <physical_name>{exe.name}</physical_name>\n  <main_program/>\n </file>\n"
                f" <plan_class>{plan_class}</plan_class>\n <avg_ncpus>1</avg_ncpus>\n <flops>1.28e11</flops>\n</version>\n"
            )
        else:
            (vdir / "version.xml").write_text(
                f"<version>\n <file>\n  <physical_name>{exe.name}</physical_name>\n  <main_program/>\n </file>\n"
                f" <plan_class>{plan_class}</plan_class>\n <avg_ncpus>1</avg_ncpus>\n <flops>4e12</flops>\n"
                f" <coproc>\n  <type>NVIDIA</type>\n  <count>1</count>\n </coproc>\n <gpu_ram>{gpu_ram_mb * 1_000_000}</gpu_ram>\n</version>\n"
            )
        tdir = self.root / "templates"
        tdir.mkdir(exist_ok=True)
        ins = "".join(f"<file_info><number>{i}</number></file_info>\n" for i in range(n_inputs))
        refs = "".join(f"<file_ref><file_number>{i}</file_number><open_name>{n}</open_name></file_ref>\n" for i, n in enumerate(["in.json", "adapter.pt"][:n_inputs]))
        (tdir / f"{app}_in").write_text(ins + "<workunit>\n" + refs + "<rsc_memory_bound>4e9</rsc_memory_bound>\n<rsc_disk_bound>2e9</rsc_disk_bound>\n"
                                        f"<delay_bound>86400</delay_bound>\n<min_quorum>{quorum}</min_quorum>\n<target_nresults>{target_res}</target_nresults>\n"
                                        f"<max_error_results>{max_err}</max_error_results>\n<max_total_results>{max_tot}</max_total_results>\n<max_success_results>{target_res}</max_success_results>\n</workunit>\n")
        outs = "".join(f"<file_info><name><OUTFILE_{i}/></name><generated_locally/><upload_when_present/><max_nbytes>{max_nbytes}</max_nbytes><url><UPLOAD_URL/></url></file_info>\n" for i in range(len(outputs)))
        orefs = "".join(f"<file_ref><file_name><OUTFILE_{i}/></file_name><open_name>{n}</open_name><copy_file/></file_ref>\n" for i, n in enumerate(outputs))
        (tdir / f"{app}_out").write_text(outs + "<result>\n" + orefs + "</result>\n")

    def install_cpu_sweep_app(
        self,
        launcher_body: str,
        *,
        app: str = "hivemind_solver_sweep",
        version: int = 100,
        min_quorum: int = 2,
        max_nbytes: int = 16_000_000,
    ) -> None:
        """Install the Wave 2 CPU-only parameter sweep app (``hivemind_solver_sweep``, quorum of 2)."""
        self.install_app(
            app=app,
            launcher_body=launcher_body,
            n_inputs=1,
            outputs=["out.json"],
            max_nbytes=max_nbytes,
            version=version,
            plan_class=CPU_SOLVER_PLAN_CLASS,
            cpu_only=True,
            min_quorum=min_quorum,
            target_nresults=min_quorum,
        )

    def project_env(self) -> dict[str, str]:
        return dict(os.environ, PYTHONPATH=f"{self.src}/py:{self.root}/py" + os.pathsep + os.environ.get("PYTHONPATH", ""), PATH=f"{self.base}/shims:{self.root}/bin:" + os.environ["PATH"])

    def register_apps(self, apps: list[str]) -> None:
        proj = (self.root / "project.xml").read_text()
        for a in apps:
            if f"<name>{a}</name>" not in proj:
                proj = proj.replace("</boinc>", f"<app>\n  <name>{a}</name>\n  <user_friendly_name>{a}</user_friendly_name>\n</app>\n</boinc>")
        (self.root / "project.xml").write_text(proj)
        sh(["bin/xadd"], cwd=self.root, env=self.project_env())
        sh(["bin/update_versions", "--no_query"] if False else ["bin/update_versions"], cwd=self.root, env=self.project_env(), check=False)

    def update_versions(self) -> str:
        """Run BOINC's interactive update_versions, answering yes (sign and add every new app version directory)."""
        r = subprocess.run(["bin/update_versions"], cwd=self.root, env=self.project_env(), input="y\n" * 60, capture_output=True, text=True, timeout=300)
        return r.stdout[-1500:]

    def deprecate_versions_below(self, version: int) -> None:
        self.sql(f"use {self.dbname}; update app_version set deprecated=1 where version_num<{version};")

    def stage(self, name: str, data: bytes) -> None:
        # create_work copies download/<name> into the hashed download hierarchy itself
        (self.root / "download").mkdir(exist_ok=True)
        (self.root / "download" / name).write_bytes(data)

    def ensure_chain_template(self, app: str = "hivemind_qlora_moe") -> str:
        """Input template for chained training workunits: in.json plus the start adapter (adapter.pt)."""
        name = f"{app}_chain_in"
        (self.root / "templates" / name).write_text(
            "<file_info><number>0</number></file_info>\n<file_info><number>1</number></file_info>\n<workunit>\n"
            "<file_ref><file_number>0</file_number><open_name>in.json</open_name></file_ref>\n"
            "<file_ref><file_number>1</file_number><open_name>adapter.pt</open_name></file_ref>\n"
            "<rsc_memory_bound>4e9</rsc_memory_bound>\n<rsc_disk_bound>2e9</rsc_disk_bound>\n<delay_bound>86400</delay_bound>\n"
            "<min_quorum>1</min_quorum>\n<target_nresults>1</target_nresults>\n<max_error_results>2</max_error_results>\n"
            "<max_total_results>2</max_total_results>\n<max_success_results>1</max_success_results>\n</workunit>\n"
        )
        return name

    def create_work(self, app: str, wu_name: str, inputs: dict[str, bytes], fpops_est: float, command_line: str = "", template: str | None = None) -> None:
        names = []
        for open_name, data in inputs.items():
            phys = f"{wu_name}_{open_name.replace('.', '_')}"
            self.stage(phys, data)
            names.append(phys)
        cmd = ["bin/create_work", "--appname", app, "--wu_name", wu_name, "--wu_template", f"templates/{template or app + '_in'}", "--result_template", f"templates/{app}_out",
               "--rsc_fpops_est", f"{fpops_est:.6g}", "--rsc_fpops_bound", f"{fpops_est * 100:.6g}"]
        if command_line:
            cmd += ["--command_line", command_line]
        sh(cmd + names, cwd=self.root, env=self.project_env())

    def start_daemons(self) -> str:
        return sh(["bin/start"], cwd=self.root, env=self.project_env(), check=False).stdout

    def stop_daemons(self) -> None:
        sh(["bin/stop"], cwd=self.root, env=self.project_env(), check=False)

    # ------------------------------------------------------------------ accounts (through the real web stack)
    def create_account(self, email: str = "node@local.invalid", name: str = "hivemind-t4") -> str:
        pw_hash = hashlib.md5(("local-secret" + email).encode()).hexdigest()
        q = urllib.parse.urlencode({"email_addr": email, "passwd_hash": pw_hash, "user_name": name})
        body = urllib.request.urlopen(f"{self.master_url}create_account.php?{q}", timeout=30).read().decode()
        m = re.search(r"<authenticator>(.*?)</authenticator>", body)
        if not m:
            raise RuntimeError(f"create_account failed: {body[:300]}")
        return m.group(1)

    def wait_http(self, timeout: float = 30) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            try:
                urllib.request.urlopen(self.master_url, timeout=3)
                return True
            except Exception:
                time.sleep(1)
        return False

    def reconcile_credits(self, ledger_dir: Path | None = None) -> dict[str, Any]:
        """Synchronize validated HiveMind cobblestone verdicts from ``ledger_dir`` into the BOINC MariaDB tables."""
        from hivemind.boinc.real_client.boinc_server_scripts import reconcile_boinc_credits

        target_dir = ledger_dir if ledger_dir is not None else (self.root / "hivemind_ledger")
        return reconcile_boinc_credits(target_dir, self.sql, dbname=self.dbname)


def launcher(venv_python: str, repo_src: Path, module: str, args: list[str], env: dict[str, str]) -> str:
    exports = "".join(f"export {k}={v}\n" for k, v in env.items())
    return f"#!/bin/bash\nunset LD_LIBRARY_PATH\nexport PYTHONPATH={repo_src}\n{exports}exec {venv_python} -W ignore -m {module} {' '.join(args)} \"$@\"\n"


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True)

# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""A minimal BOINC project server for a single local machine (127.0.0.1 only).

It speaks the wire formats a stock BOINC client uses, taken from the BOINC sources
(``client/scheduler_op.cpp``, ``client/client_types.cpp``, ``client/file_xfer.cpp``,
``sched/file_upload_handler.cpp``):

* ``GET <master_url>``                the master file, with the ``<scheduler>`` URL,
* ``POST /cgi``                       the scheduler RPC (``<scheduler_request>`` -> ``<scheduler_reply>``),
* ``POST /file_upload_handler``       the ``<data_server_request>`` upload and ``<get_file_size>`` query,
* ``GET /download/<name>``            workunit input files.

It is not the production BOINC server (no MySQL, daemons or code signing; apps run through the client's
anonymous-platform mechanism), and it is meant for tests and demonstrations. Never expose it: it binds to the
loopback interface and refuses any other bind address.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from collections.abc import Callable
from typing import Any

from hivemind.boinc.real_client.app import APP_NAME, APP_VERSION
from hivemind.boinc.real_client.bridge import Verdict, validate_result

PLATFORM = "x86_64-pc-linux-gnu"
PLAN_CLASS = "cuda_hivemind"
RESULT_FILES_UPLOADED = 5  # boinc state value reported by the client


def _x(tag: str, text: Any) -> str:
    return f"<{tag}>{text}</{tag}>"


class ProjectServer:
    def __init__(self, workdir: Path, port: int = 0, host: str = "127.0.0.1", validate=validate_result) -> None:
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("the local BOINC project server only binds to the loopback interface")
        self.dir = Path(workdir)
        for sub in ("download", "upload"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.workunits: dict[str, dict[str, Any]] = {}
        self.sched_requests: list[dict[str, Any]] = []
        self.verdicts: dict[str, Verdict] = {}
        self._validate = validate
        # app name -> callable(workunit dict, host) -> Verdict; the callable reads uploads via self.output_path
        self.app_validators: dict[str, Callable[[dict[str, Any], str], Verdict]] = {}
        # app name -> callable(workunit dict, verdict); runs only for a VALID result (BOINC's assimilator)
        self.assimilators: dict[str, Callable[[dict[str, Any], Verdict], None]] = {}
        self._threads: list[threading.Thread] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                """Silence standard BaseHTTPRequestHandler stderr logging during tests."""
                _ = len(a)

            def _send(self, body: bytes, ctype: str = "text/plain") -> None:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path.startswith("/download/"):
                    p = server.dir / "download" / Path(self.path[len("/download/") :]).name
                    return self._send(p.read_bytes(), "application/octet-stream") if p.is_file() else self.send_error(404)
                self._send(f"<scheduler>{server.url}/cgi</scheduler>\n".encode())

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path.startswith("/cgi"):
                    return self._send(server.scheduler_rpc(body.decode("utf-8", "replace")).encode())
                if self.path.startswith("/file_upload_handler"):
                    return self._send(server.upload(body).encode())
                self.send_error(404)

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.port = self.httpd.server_address[1]
        self.url = f"http://{host}:{self.port}"
        self.master_url = f"{self.url}/hivemind/"

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        for t in self._threads:
            t.join(timeout=120)

    # ------------------------------------------------------------------ work creation (BOINC create_work)
    def create_work(
        self,
        spec: dict[str, Any],
        command_line: str = "",
        flops_est: float | None = None,
        *,
        app: str = APP_NAME,
        extra_inputs: dict[str, bytes] | None = None,
        outputs: tuple[str, ...] = ("out.json",),
        max_nbytes: int = 5_000_000,
    ) -> str:
        """Register a workunit: ``spec`` is the ``in.json`` input; ``extra_inputs`` maps an open name to bytes."""
        wu = spec["wu_id"]
        inputs = {"in.json": json.dumps(spec, sort_keys=True).encode(), **(extra_inputs or {})}
        files = []
        for open_name, data in inputs.items():
            fname = f"{wu}_{open_name.replace('.', '_')}"
            (self.dir / "download" / fname).write_bytes(data)
            files.append({"name": fname, "open_name": open_name, "md5": hashlib.md5(data).hexdigest(), "nbytes": len(data)})
        with self.lock:
            self.workunits[wu] = {
                "spec": spec,
                "app": app,
                "command_line": command_line,
                "inputs": files,
                "outputs": list(outputs),
                "max_nbytes": max_nbytes,
                "flops_est": flops_est or 1e12,
                "state": "unsent",
                "result_name": f"{wu}_0",
                "host": None,
                "reported": None,
            }
        return wu

    def output_path(self, wu_id: str, open_name: str) -> Path:
        wu = self.workunits[wu_id]
        return self.dir / "upload" / f"{wu['result_name']}_{wu['outputs'].index(open_name)}"

    # ------------------------------------------------------------------ scheduler RPC
    @staticmethod
    def _parse_request(text: str) -> dict[str, Any]:
        info: dict[str, Any] = {"results": [], "raw_len": len(text)}
        m = re.search(r"<scheduler_request>.*</scheduler_request>", text, re.S)
        try:
            root = ET.fromstring(m.group(0) if m else text)
        except ET.ParseError:
            return info
        info["authenticator"] = (root.findtext("authenticator") or "").strip()
        info["hostid"] = int(root.findtext("hostid") or 0)
        info["work_req_seconds"] = float(root.findtext("work_req_seconds") or 0)
        coprocs = root.find("coprocs") if root.find("coprocs") is not None else root.find(".//host_info/coprocs")
        info["coprocs_xml"] = ET.tostring(coprocs, encoding="unicode") if coprocs is not None else ""
        info["gpu_names"] = [n.text for n in root.iter("name") if coprocs is not None and n in list(coprocs.iter("name"))]
        for r in root.findall("result"):
            info["results"].append(
                {
                    "name": r.findtext("name"),
                    "exit_status": int(r.findtext("exit_status") or 0),
                    "state": int(r.findtext("state") or 0),
                    "elapsed": float(r.findtext("final_elapsed_time") or 0),
                    "stderr": (r.findtext("stderr_out") or "")[:500],
                }
            )
        return info

    def scheduler_rpc(self, text: str) -> str:
        req = self._parse_request(text)
        out: list[str] = []
        acks: list[str] = []
        with self.lock:
            self.sched_requests.append({k: v for k, v in req.items() if k != "results"} | {"t": time.time(), "n_reported": len(req["results"])})
            for r in req["results"]:
                acks.append(_x("result_ack", _x("name", r["name"])))
                wu = next((w for w in self.workunits.values() if w["result_name"] == r["name"]), None)
                if wu is not None and wu["reported"] is None:
                    wu["reported"] = r
                    wu["state"] = "reported" if r["exit_status"] == 0 and r["state"] == RESULT_FILES_UPLOADED else "client_error"
                    if wu["state"] == "reported":
                        t = threading.Thread(target=self._validate_job, args=(wu["spec"]["wu_id"],), daemon=True)
                        self._threads.append(t)
                        t.start()
            gpu_ok = bool(req.get("gpu_names"))
            files, wus, results = [], [], []
            if req.get("work_req_seconds", 0) > 0 and gpu_ok:
                for wu in self.workunits.values():
                    if wu["state"] != "unsent":
                        continue
                    wu["state"] = "in_progress"
                    wu["host"] = f"host{req.get('hostid') or 1}"
                    spec, name = wu["spec"], wu["result_name"]
                    for f in wu["inputs"]:
                        files.append(
                            "<file_info>"
                            + _x("name", f["name"])
                            + _x("url", f"{self.url}/download/{f['name']}")
                            + _x("md5_cksum", f["md5"])
                            + _x("nbytes", f["nbytes"])
                            + "</file_info>"
                        )
                    for i, _o in enumerate(wu["outputs"]):
                        files.append(
                            "<file_info>"
                            + _x("name", f"{name}_{i}")
                            + "<generated_locally/><upload_when_present/>"
                            + _x("max_nbytes", wu["max_nbytes"])
                            + _x("url", f"{self.url}/file_upload_handler")
                            + "</file_info>"
                        )
                    wus.append(
                        "<workunit>"
                        + _x("name", spec["wu_id"])
                        + _x("app_name", wu["app"])
                        + _x("version_num", APP_VERSION)
                        + (_x("command_line", wu["command_line"]) if wu["command_line"] else "")
                        + _x("rsc_fpops_est", wu["flops_est"])
                        + _x("rsc_fpops_bound", wu["flops_est"] * 100)
                        + _x("rsc_memory_bound", 4e9)
                        + _x("rsc_disk_bound", 1e9)
                        + "".join(f"<file_ref><file_name>{f['name']}</file_name><open_name>{f['open_name']}</open_name></file_ref>" for f in wu["inputs"])
                        + "</workunit>"
                    )
                    results.append(
                        "<result>"
                        + _x("name", name)
                        + _x("wu_name", spec["wu_id"])
                        + _x("report_deadline", int(time.time()) + 86400)
                        + _x("platform", PLATFORM)
                        + _x("plan_class", PLAN_CLASS)
                        + _x("version_num", APP_VERSION)
                        + "".join(f"<file_ref><file_name>{name}_{i}</file_name><open_name>{o}</open_name></file_ref>" for i, o in enumerate(wu["outputs"]))
                        + "</result>"
                    )
                    apps_used = wu["app"]
                    break  # one GPU task per request keeps the single T4 free of contention
            out += [f"<app>{_x('name', apps_used)}{_x('user_friendly_name', apps_used)}</app>"] if results else []
            out += files + wus + results
        head = (
            "<scheduler_reply>\n"
            + _x("scheduler_version", 715)
            + _x("project_name", "HiveMind local test project")
            + _x("master_url", self.master_url)
            + _x("user_name", "local-test")
            + _x("hostid", req.get("hostid") or 1)
            + _x("request_delay", 10)
            + _x("next_rpc_delay", 0)
        )
        return head + "".join(out) + "".join(acks) + "\n</scheduler_reply>\n"

    # ------------------------------------------------------------------ file_upload_handler
    def upload(self, body: bytes) -> str:
        def reply(status: int, extra: str = "", msg: str = "") -> str:
            return f"<data_server_reply>\n<status>{status}</status>\n{extra}{_x('message', msg) if msg else ''}\n</data_server_reply>\n"

        head, sep, rest = body.partition(b"<data>\n")
        h = head.decode("utf-8", "replace")
        m = re.search(r"<get_file_size>(.*?)</get_file_size>", h)
        if m:
            p = self.dir / "upload" / Path(m.group(1)).name
            return reply(0, _x("file_size", p.stat().st_size if p.exists() else 0))
        name = re.search(r"<name>(.*?)</name>", h)
        nbytes = re.search(r"<nbytes>(.*?)</nbytes>", h)
        offset = re.search(r"<offset>(.*?)</offset>", h)
        if not (name and nbytes and sep) or ".." in name.group(1) or "/" in name.group(1):
            return reply(-1, msg="bad upload request")
        n, off = int(float(nbytes.group(1))), int(float(offset.group(1) if offset else 0))
        path = self.dir / "upload" / name.group(1)
        mode = "r+b" if path.exists() else "wb"
        with open(path, mode) as f:
            f.seek(off)
            f.write(rest[: max(n - off, 0)])
        return reply(0)

    # ------------------------------------------------------------------ validator + assimilator
    def _validate_job(self, wu_id: str) -> None:
        wu = self.workunits[wu_id]
        path = self.dir / "upload" / f"{wu['result_name']}_0"
        host = wu["host"] or "host1"
        try:
            if wu["app"] in self.app_validators:
                verdict = self.app_validators[wu["app"]](wu, host)
                if verdict.valid and wu["app"] in self.assimilators:
                    self.assimilators[wu["app"]](wu, verdict)
            else:
                doc = json.loads(path.read_text())
                verdict = self._validate(wu["spec"], doc, host)
        except Exception as exc:
            from hivemind.economy.compute_coin import CreditClaim, verified_credit

            verdict = Verdict(wu_id, host, False, f"validator error: {exc}", None, None, verified_credit(CreditClaim(host, wu_id, 0.0, 1.0, 1.0, False, False)).to_dict())
        with self.lock:
            self.verdicts[wu_id] = verdict
            wu["state"] = "validated" if verdict.valid else "invalid"
            (self.dir / "ledger.json").write_text(json.dumps({k: v.to_dict() for k, v in self.verdicts.items()}, indent=2, sort_keys=True))

    def wait_all(self, timeout: float) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                if self.workunits and all(w["state"] in ("validated", "invalid", "client_error") for w in self.workunits.values()):
                    return True
            time.sleep(1.0)
        return False

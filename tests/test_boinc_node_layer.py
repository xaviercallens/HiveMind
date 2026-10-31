"""CPU-only tests of the T4 node layer: multi-domain workunit data, per-domain evaluation, the governor's night window, idle wait."""

from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hivemind.boinc.real_client import moe_work  # noqa: E402
from hivemind.boinc.real_client.boinc_server_scripts import wait_for_idle  # noqa: E402
from hivemind.boinc.real_client.moe_app import eval_by_domain  # noqa: E402


def test_domain_registry_is_consistent_and_excludes_benchmarks() -> None:
    assert set(moe_work.DOMAIN_ORDER) == set(moe_work._SOURCES)
    joined = " ".join(s["ds"] for s in moe_work._SOURCES.values()).lower()
    assert "mbpp" not in joined and "humaneval" not in joined  # the repository's evaluation benchmarks are never training data


def test_rows_slices_across_row_groups_and_from_the_end(tmp_path: Path) -> None:
    f = tmp_path / "d.parquet"
    pq.write_table(pa.table({"text": [f"row{i}" for i in range(2500)]}), f, row_group_size=1000)
    assert [r["text"] for r in moe_work._rows(f, ["text"], 995, 10)] == [f"row{i}" for i in range(995, 1005)]  # crosses a group boundary
    assert [r["text"] for r in moe_work._rows(f, ["text"], -3, 3)] == ["row2497", "row2498", "row2499"]
    assert len(moe_work._rows(f, ["text"], 2495, 50)) == 5  # asking past the end returns what exists


def test_ids_stops_when_enough_tokens_and_skips_blank_docs() -> None:
    tok = lambda s: SimpleNamespace(input_ids=list(range(len(s))))  # noqa: E731 - one token per character
    ids = moe_work._ids(tok, ["", "  ", "a" * 10, "b" * 10, "c" * 10], need=15)
    assert 15 <= len(ids) < 40  # stopped after the second real document, did not read the third


class _Stub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, labels):  # a deterministic "loss": mean token id
        return SimpleNamespace(loss=input_ids.float().mean())


def test_eval_by_domain_per_domain_and_single_set_fallback() -> None:
    m = _Stub()
    per = eval_by_domain(m, {"eval_sets": {"a": [[1, 1]], "b": [[3, 3]], "empty": []}})
    assert per == {"a": 1.0, "b": 3.0}  # an empty domain is skipped, not divided by zero
    assert eval_by_domain(m, {"eval_tokens": [[2, 2]]}) == {"general": 2.0}  # workunits created before multi-domain evaluation


def test_governor_night_window_wraps_midnight() -> None:
    from t4_governor import in_night

    def at(h: int) -> float:  # 2026-10-25 is after the end of daylight saving: Paris = UTC+1
        return dt.datetime(2026, 10, 25, h, 0, tzinfo=dt.timezone.utc).timestamp()

    assert [in_night(at(h), "Europe/Paris", 22, 7) for h in (12, 20, 21, 23, 2, 5, 6)] == [False, False, True, True, True, True, False]
    assert in_night(at(10), "Europe/Paris", 8, 18) is True  # a non-wrapping window


def test_parse_pmon_ignores_header_and_unsampled_processes() -> None:
    from t4_governor import parse_pmon

    txt = ("# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command \n# Idx           #    C/G      %      %      %      %      %      %    name \n"
           "    0     742539     C      -      -      -      -      -      -    llama-server   \n    0     946827     C     37     12      -      -      -      -    llama-server   \n")
    assert parse_pmon(txt) == {"946827": 37}  # '-' means "no sample", which must not be read as idle (0)


def test_other_gpu_busy_without_a_boinc_task_uses_whole_gpu_utilisation() -> None:
    from t4_governor import other_gpu_busy

    assert other_gpu_busy({"procs": [], "util_pct": 80}) == (True, 80)  # e.g. a local Ollama client that never touched nginx
    assert other_gpu_busy({"procs": [], "util_pct": 3}) == (False, 3)


def test_chatlog_counts_new_completions_by_route_and_survives_rotation(tmp_path: Path) -> None:
    from t4_monitor import ChatLog

    log = tmp_path / "access.log"
    log.write_text('1.2.3.4 - - [25/Oct/2026:10:00:00 +0000] "POST /v1/chat/completions HTTP/1.1" 200 5 "-" "x"\n')
    c = ChatLog(log)
    assert c.poll() == {"ok": True, "n": 0, "by": {}}  # the first poll starts "now": history is not recounted
    with open(log, "a") as f:
        f.write('1.2.3.4 - - [25/Oct/2026:10:00:01 +0000] "POST /v1/chat/completions HTTP/1.1" 200 5 "-" "x"\n')
        f.write('1.2.3.4 - - [25/Oct/2026:10:00:02 +0000] "POST /bee/v1/chat/completions HTTP/1.1" 200 5 "-" "x"\n')
        f.write('1.2.3.4 - - [25/Oct/2026:10:00:03 +0000] "POST /u/v1/chat/completions HTTP/1.1" 499 0 "-" "x"\n')
        f.write('1.2.3.4 - - [25/Oct/2026:10:00:04 +0000] "GET /v1/models HTTP/1.1" 401 5 "-" "x"\n')  # a poll, not an inference
    assert c.poll() == {"ok": True, "n": 3, "by": {"coder:200": 1, "bee:200": 1, "u:499": 1}}
    log.rename(tmp_path / "access.log.1")  # logrotate: a new file with a new inode
    log.write_text('1.2.3.4 - - [25/Oct/2026:10:05:00 +0000] "POST /v1/chat/completions HTTP/1.1" 502 0 "-" "x"\n')
    assert c.poll()["by"] == {"coder:502": 1}


def test_terminate_tree_ends_a_process_and_its_children_quickly() -> None:
    import subprocess as sp

    from t4_governor import descendants, terminate_tree

    # a parent that spawns a child, and both ignore SIGTERM to prove the SIGKILL fallback works
    code = ("import signal, subprocess, sys, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "c = subprocess.Popen([sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\ntime.sleep(60)\n")
    p = sp.Popen([sys.executable, "-c", code])
    time.sleep(1.0)
    kids = descendants(p.pid)
    assert len(kids) >= 1
    t = time.time()
    assert terminate_tree(p.pid, grace_s=0.5) is True
    assert time.time() - t < 4.0
    p.wait(timeout=5)
    from t4_governor import pid_alive

    # "gone" means not running: an orphaned grandchild is a zombie until init reaps it (a few ms), so /proc/<pid> may still exist
    assert not any(pid_alive(k) for k in kids)
    assert pid_alive(os.getpid()) is True and pid_alive(p.pid) is False


def test_policy_drift_flags_changed_arguments_and_dropins_but_not_the_expected_state() -> None:
    from t4_monitor import policy_drift

    exp = "--idle-start-s 1800 --day-idle-s 7200 --min-free-mib 4300"
    assert policy_drift(exp, exp, [])["drift"] is False
    changed = policy_drift(exp, "--idle-start-s 5 --day-idle-s 5 --min-free-mib 3800", [])
    assert changed["drift"] is True and changed["args_match"] is False
    # the 22:08 incident: arguments may even match, but a drop-in override exists and silently wins at the next restart
    assert policy_drift(exp, exp, ["/etc/systemd/system/hivemind-t4-governor.service.d/intense-moe.conf"])["drift"] is True
    assert policy_drift(None, exp, [])["drift"] is False  # no recorded expectation (installer never ran): nothing to compare, not an alarm


def test_code_drift_flags_edited_and_missing_files_only() -> None:
    from t4_monitor import code_drift

    manifest = "aaa  scripts/t4_governor.py\nbbb  scripts/t4_monitor.py\nccc  bin/startup.sh\n"
    now = {"scripts/t4_governor.py": "aaa", "scripts/t4_monitor.py": "bbb", "bin/startup.sh": "ccc"}
    assert code_drift(manifest, now.get) == []
    now["scripts/t4_governor.py"] = "edited-by-hand"  # the 02:08Z incident: same arguments, different code
    del now["bin/startup.sh"]
    assert code_drift(manifest, now.get) == ["bin/startup.sh", "scripts/t4_governor.py"]


def test_governor_policy_comes_only_from_its_arguments() -> None:
    """Regression guard for the 02:08Z bypass: the resume rule once did min(os.environ.get("HIVEMIND_INTENSE_MOE_IDLE_S", 10), idle), which silently
    capped the owner's 1800 s / 7200 s to 10 s. The governor's policy must come from its arguments only, never from environment variables."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "t4_governor.py").read_text()
    assert "environ.get(" not in src and "getenv(" not in src and "INTENSE_MOE" not in src


def test_wait_for_idle_does_not_block_when_there_is_no_status_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIVEMIND_IDLE_URL", "http://127.0.0.1:9/nginx_status")  # nothing listens here
    t = time.time()
    wait_for_idle(quiet_s=5, max_wait_s=30)
    assert time.time() - t < 5  # a box without nginx must not stall validation


def test_client_setup_prepare_data_dir_multi_profile_and_cpu_solver(tmp_path: Path) -> None:
    from hivemind.boinc.real_client.client_setup import (
        big_moe_app,
        moe_apps,
        prepare_data_dir,
        solver_sweep_app,
    )

    apps = [*moe_apps("python3"), big_moe_app("python3"), solver_sweep_app("python3")]
    info = prepare_data_dir(
        tmp_path / "boinc_data",
        "http://127.0.0.1:8081/hmproj/",
        Path(__file__).resolve().parents[1] / "src",
        apps=apps,
        authenticator="acct-key-123",
    )
    proj_dir = info["project_dir"]
    app_info_xml = (proj_dir / "app_info.xml").read_text(encoding="utf-8")

    assert "<plan_class>cuda_hivemind</plan_class>" in app_info_xml
    assert "<plan_class>cuda_hivemind_big</plan_class>" in app_info_xml
    assert "<plan_class>cpu_hivemind_solver</plan_class>" in app_info_xml
    assert "<gpu_ram>3600000000</gpu_ram>" in app_info_xml
    assert "<gpu_ram>10000000000</gpu_ram>" in app_info_xml
    # CPU solver app block has no <coproc> or <gpu_ram>
    solver_segment = app_info_xml.split("<app_name>hivemind_solver_sweep</app_name>")[1].split("</app_version>")[0]
    assert "<coproc>" not in solver_segment and "<gpu_ram>" not in solver_segment


def test_boinccmd_output_parsers_and_client_rpc_wrappers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hivemind.boinc.real_client.client_setup import (
        Client,
        parse_boinccmd_cc_status,
        parse_boinccmd_project_status,
        parse_boinccmd_tasks,
    )

    cc_sample = (
        "network connection status: don't need connection\n"
        "CPU status\n"
        "    not suspended\n"
        "    current mode: always\n"
        "GPU status\n"
        "    suspended: user request\n"
        "    current mode: never\n"
        "    perm mode: auto\n"
        "    perm becomes current in 45 sec\n"
    )
    cc = parse_boinccmd_cc_status(cc_sample)
    assert cc["network_connection_status"] == "don't need connection"
    assert cc["cpu"]["suspended"] is False
    assert cc["cpu"]["current_mode"] == "always"
    assert cc["gpu"]["suspended"] is True
    assert cc["gpu"]["suspend_reason"] == "user request"
    assert cc["gpu"]["current_mode"] == "never"
    assert cc["gpu"]["perm_delay_sec"] == 45.0

    proj_sample = (
        "======== Projects ========\n"
        "1) -----------\n"
        "   name: HiveMind local BOINC server\n"
        "   master URL: http://127.0.0.1:8081/hmproj/\n"
        "   user_name: hivemind-t4\n"
        "   user_total_credit: 142.500000\n"
        "   host_total_credit: 142.500000\n"
        "   jobs succeeded: 19\n"
        "   jobs failed: 1\n"
        "   GUI URL:\n"
        "      name: Message boards\n"
        "      description: Correspond with other users\n"
        "      URL: http://127.0.0.1:8081/hmproj/forum_index.php\n"
    )
    projs = parse_boinccmd_project_status(proj_sample)
    assert len(projs) == 1
    assert projs[0]["name"] == "HiveMind local BOINC server"
    assert projs[0]["user_total_credit"] == 142.5
    assert projs[0]["jobs_succeeded"] == 19
    assert projs[0]["gui_urls"][0]["name"] == "Message boards"

    tasks_sample = (
        "======== Tasks ========\n"
        "1) -----------\n"
        "   name: wu-train-001_0\n"
        "   WU name: wu-train-001\n"
        "   project URL: http://127.0.0.1:8081/hmproj/\n"
        "   state: downloaded\n"
        "   scheduler state: scheduled\n"
        "   active_task_state: EXECUTING\n"
        "   fraction done: 0.625000\n"
    )
    tasks = parse_boinccmd_tasks(tasks_sample)
    assert len(tasks) == 1
    assert tasks[0]["name"] == "wu-train-001_0"
    assert tasks[0]["active_task_state"] == "EXECUTING"
    assert tasks[0]["fraction_done"] == 0.625

    client = Client(
        boinc_bin=Path("/usr/bin/boinc"),
        lib_dir=Path("/usr/lib"),
        data_dir=tmp_path,
        rpc_port=31499,
    )
    recorded: list[tuple[str, ...]] = []

    def fake_cmd(*args: str, timeout: int = 20) -> str:
        recorded.append(args)
        if args == ("--get_cc_status",):
            return cc_sample
        if args == ("--get_project_status",):
            return proj_sample
        if args == ("--get_tasks",):
            return tasks_sample
        return ""

    monkeypatch.setattr(client, "cmd", fake_cmd)
    assert client.get_cc_status()["gpu"]["current_mode"] == "never"
    assert client.get_project_status()[0]["name"] == "HiveMind local BOINC server"
    assert client.get_tasks()[0]["wu_name"] == "wu-train-001"
    client.set_gpu_mode("auto", 60)
    client.set_run_mode("always")
    client.set_network_mode("auto")
    client.update_project("http://127.0.0.1:8081/hmproj/")
    client.suspend_project("http://127.0.0.1:8081/hmproj/")
    client.resume_project("http://127.0.0.1:8081/hmproj/")
    assert ("--set_gpu_mode", "auto", "60") in recorded
    assert ("--set_run_mode", "always", "0.0") in recorded
    assert ("--project", "http://127.0.0.1:8081/hmproj/", "update") in recorded


def test_project_server_cpu_solver_vram_gating_quorum_and_web_rpc(tmp_path: Path) -> None:
    import hashlib
    import json
    import urllib.request

    from hivemind.boinc.real_client.project_server import RESULT_FILES_UPLOADED, ProjectServer

    server = ProjectServer(workdir=tmp_path / "proj_srv", port=0)
    assimilated: list[str] = []
    server.assimilators["hivemind_solver_sweep"] = lambda wu, _verdict: assimilated.append(wu["spec"]["wu_id"])
    server.start()
    try:
        sched_url = f"{server.url}/cgi"
        upload_url = f"{server.url}/file_upload_handler"

        # 1. Standard BOINC Web RPC endpoints
        cfg_xml = urllib.request.urlopen(f"{server.url}/get_project_config.php", timeout=5).read().decode()
        assert "<name>HiveMind local test project</name>" in cfg_xml and "<min_passwd_length>6</min_passwd_length>" in cfg_xml
        acct_xml = urllib.request.urlopen(
            f"{server.url}/create_account.php?email_addr=a@b.c&passwd_hash=123456&user_name=tester", timeout=5
        ).read().decode()
        assert "<authenticator>" in acct_xml
        lookup_xml = urllib.request.urlopen(
            f"{server.url}/lookup_account.php?email_addr=a@b.c&passwd_hash=123456", timeout=5
        ).read().decode()
        assert "<authenticator>" in lookup_xml

        # 2. Big MoE workunit VRAM gating (requires 10,000 MB)
        server.create_work(
            {"wu_id": "wu-big-001", "app": "hivemind_qlora_moe_big"},
            app="hivemind_qlora_moe_big",
            plan_class="cuda_hivemind_big",
        )
        small_gpu_req = (
            b"<scheduler_request><hostid>10</hostid>"
            b"<coprocs><coproc_cuda><count>1</count><name>RTX 3070</name><ram>8388608000</ram></coproc_cuda></coprocs>"
            b"<work_req_seconds>60</work_req_seconds></scheduler_request>"
        )
        reply_small = urllib.request.urlopen(
            urllib.request.Request(sched_url, data=small_gpu_req), timeout=5
        ).read().decode()
        assert "wu-big-001" not in reply_small

        large_gpu_req = (
            b"<scheduler_request><hostid>11</hostid>"
            b"<coprocs><coproc_cuda><count>1</count><name>Tesla T4</name><ram>16106127360</ram></coproc_cuda></coprocs>"
            b"<work_req_seconds>60</work_req_seconds></scheduler_request>"
        )
        reply_large = urllib.request.urlopen(
            urllib.request.Request(sched_url, data=large_gpu_req), timeout=5
        ).read().decode()
        assert "<name>wu-big-001</name>" in reply_large and "<plan_class>cuda_hivemind_big</plan_class>" in reply_large

        # 3. CPU-only solver sweep with quorum=2 dispatched to two GPU-less hosts
        sweep_spec: dict[str, object] = {
            "wu_id": "wu-sweep-q2",
            "app": "hivemind_solver_sweep",
            "parameter_grid": [0.5, 1.0],
            "t_out": [0.0, 0.5, 1.0],
            "max_invariant_drift": 1e-4,
        }
        spec_sha = hashlib.sha256(json.dumps(sweep_spec, sort_keys=True).encode()).hexdigest()
        sweep_spec["spec_sha256"] = spec_sha
        server.create_work(
            sweep_spec,
            app="hivemind_solver_sweep",
            plan_class="cpu_hivemind_solver",
            min_quorum=2,
            allow_same_host_quorum=False,
        )

        cpu_req_1 = b"<scheduler_request><hostid>201</hostid><coprocs></coprocs><work_req_seconds>60</work_req_seconds></scheduler_request>"
        cpu_req_2 = b"<scheduler_request><hostid>202</hostid><coprocs></coprocs><work_req_seconds>60</work_req_seconds></scheduler_request>"
        rep1 = urllib.request.urlopen(urllib.request.Request(sched_url, data=cpu_req_1), timeout=5).read().decode()
        rep2 = urllib.request.urlopen(urllib.request.Request(sched_url, data=cpu_req_2), timeout=5).read().decode()
        assert "<name>wu-sweep-q2_0</name>" in rep1 and "<coproc>" not in rep1
        assert "<name>wu-sweep-q2_1</name>" in rep2

        # Upload both replica outputs and report completion
        valid_out = json.dumps(
            {
                "wu_id": "wu-sweep-q2",
                "app": "hivemind_solver_sweep",
                "spec_sha256": spec_sha,
                "sweep_results": [
                    {"param_value": 0.5, "status": "ok", "t": [0.0, 0.5, 1.0], "y": [[1.0, 0.0], [0.8, -0.2], [0.5, -0.4]], "invariant_drift": 1e-6},
                    {"param_value": 1.0, "status": "ok", "t": [0.0, 0.5, 1.0], "y": [[1.0, 0.0], [0.7, -0.3], [0.4, -0.5]], "invariant_drift": 2e-6},
                ],
                "flops_claimed": 5e9,
                "wall_seconds": 1.2,
            }
        ).encode()
        for r_idx, hid in ((0, 201), (1, 202)):
            upload_msg = (
                f"<data_server_request><file_upload><file_info><name>wu-sweep-q2_{r_idx}_0</name></file_info>"
                f"<nbytes>{len(valid_out)}</nbytes><offset>0</offset><data>\n".encode()
                + valid_out
                + b"</data></file_upload></data_server_request>"
            )
            urllib.request.urlopen(urllib.request.Request(upload_url, data=upload_msg), timeout=5).read()
            report_req = (
                f"<scheduler_request><hostid>{hid}</hostid>"
                f"<result><name>wu-sweep-q2_{r_idx}</name><exit_status>0</exit_status><state>{RESULT_FILES_UPLOADED}</state></result>"
                f"<work_req_seconds>0</work_req_seconds></scheduler_request>"
            ).encode()
            urllib.request.urlopen(urllib.request.Request(sched_url, data=report_req), timeout=5).read()

        for t in list(server._threads):
            t.join(timeout=10)

        assert "wu-sweep-q2" in server.verdicts
        assert server.verdicts["wu-sweep-q2"].valid is True
        assert server.verdicts["wu-sweep-q2"].detail["quorum_checked"] is True
        assert "wu-sweep-q2" in assimilated

        status_json = json.loads(urllib.request.urlopen(f"{server.url}/api/status", timeout=5).read().decode())
        assert status_json["verdicts_valid"] == 1
    finally:
        server.stop()


def test_boinc_server_scripts_solver_validation_assimilate_and_credit_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from hivemind.boinc.real_client import boinc_server_scripts as bss

    spec_dir = tmp_path / "specs"
    ledger_dir = tmp_path / "ledger"
    spec_dir.mkdir()
    ledger_dir.mkdir()
    monkeypatch.setattr(bss, "SPEC_DIR", spec_dir)
    monkeypatch.setattr(bss, "LEDGER_DIR", ledger_dir)

    spec = {
        "wu_id": "wu-solver-101",
        "app": "hivemind_solver_sweep",
        "parameter_grid": [0.25],
        "t_out": [0.0, 1.0],
        "max_invariant_drift": 1e-5,
        "spec_sha256": "abc123sha",
    }
    (spec_dir / "wu-solver-101.json").write_text(json.dumps(spec), encoding="utf-8")

    out_a = tmp_path / "out_a.json"
    out_b = tmp_path / "out_b.json"
    payload_a = {
        "wu_id": "wu-solver-101",
        "app": "hivemind_solver_sweep",
        "spec_sha256": "abc123sha",
        "family": "lorenz",
        "parameter_name": "rho",
        "sweep_results": [
            {"param_value": 0.25, "status": "ok", "t": [0.0, 1.0], "y": [[1.0, 2.0], [1.5, 2.5]], "invariant_drift": 1e-7}
        ],
        "flops_claimed": 4e9,
        "wall_seconds": 0.8,
    }
    out_a.write_text(json.dumps(payload_a), encoding="utf-8")
    out_b.write_text(json.dumps(payload_a), encoding="utf-8")

    # Quorum-of-2 validation passes when both replicas agree within S-1 tolerance
    rc = bss.run_validation([str(out_a), str(out_b), "77"])
    assert rc == 0

    # Assimilate writes solver_sweep_<wu>.json, latest_solver_sweep.json, and assimilated.jsonl
    assert bss.assimilate([str(out_a)]) == 0
    latest_sweep = json.loads((ledger_dir / "latest_solver_sweep.json").read_text(encoding="utf-8"))
    assert latest_sweep["wu_id"] == "wu-solver-101" and latest_sweep["family"] == "lorenz"

    # Reconcile credits into fake BOINC SQL runner idempotently
    sql_queries: list[str] = []

    def fake_sql(stmt: str) -> str:
        sql_queries.append(stmt)
        if "select userid, hostid, granted_credit" in stmt:
            return "userid\thostid\tgranted_credit\n1\t2\t0.000000\n"
        return ""

    first_sync = bss.reconcile_boinc_credits(ledger_dir, fake_sql, dbname="hmproj")
    assert first_sync["synced_count"] == 1 and first_sync["already_synced_count"] == 0
    assert any("update result set granted_credit" in q for q in sql_queries)

    second_sync = bss.reconcile_boinc_credits(ledger_dir, fake_sql, dbname="hmproj")
    assert second_sync["synced_count"] == 0 and second_sync["already_synced_count"] == 1


def test_coordinator_real_boinc_ledger_bridge_and_fastapi_routes(tmp_path: Path) -> None:
    import json
    from fastapi.testclient import TestClient

    from hivemind.boinc.coordinator_api import BOINCCoordinatorService, create_coordinator_fastapi_app
    from hivemind.config import HiveMindConfig

    ledger_dir = tmp_path / "boinc_ledger"
    ledger_dir.mkdir()
    (ledger_dir / "assimilated.jsonl").write_text(
        json.dumps(
            {
                "wu_id": "moe-001",
                "app": "hivemind_qlora_moe",
                "domain": "code",
                "eval_loss_after": 1.12,
                "eval_by_domain_after": {"code": 1.12, "math": 1.45},
            }
        )
        + "\n"
        + json.dumps(
            {
                "wu_id": "sweep-001",
                "app": "hivemind_solver_sweep",
                "family": "van_der_pol",
                "sweep_count": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (ledger_dir / "latest_adapter.json").write_text(
        json.dumps({"wu_id": "moe-001", "sha256": "deadbeef", "eval_loss_after": 1.12}),
        encoding="utf-8",
    )
    (ledger_dir / "latest_solver_sweep.json").write_text(
        json.dumps({"wu_id": "sweep-001", "family": "van_der_pol", "sweep_count": 4}),
        encoding="utf-8",
    )
    (ledger_dir / "verdict_moe-001_1700000001.json").write_text(
        json.dumps(
            {
                "wu_id": "moe-001",
                "host": "result-501",
                "valid": True,
                "credit": {"cobblestones": 12.5, "flops_granted": 2.5e12},
            }
        ),
        encoding="utf-8",
    )
    (ledger_dir / "verdict_moe-002_1700000002.json").write_text(
        json.dumps(
            {
                "wu_id": "moe-002",
                "host": "result-502",
                "valid": False,
                "credit": {"cobblestones": 0.0, "flops_granted": 0.0},
            }
        ),
        encoding="utf-8",
    )

    cfg = HiveMindConfig()
    init_lora = {"lora_A": torch.zeros(4, 8), "lora_B": torch.zeros(8, 4)}
    coord = BOINCCoordinatorService(
        config=cfg,
        initial_lora_state=init_lora,
        base_model_sha256="sha256-test",
        shards_dir=tmp_path / "shards",
        boinc_ledger_dir=ledger_dir,
    )
    app = create_coordinator_fastapi_app(coord)
    client = TestClient(app)

    summary_resp = client.get("/boinc/v1/ledger/summary")
    assert summary_resp.status_code == 200
    summary = summary_resp.json()
    assert summary["assimilated_total"] == 2
    assert summary["assimilated_by_app"] == {"hivemind_qlora_moe": 1, "hivemind_solver_sweep": 1}
    assert summary["latest_eval_by_domain"] == {"code": 1.12, "math": 1.45}
    assert summary["verdicts_valid"] == 1 and summary["verdicts_rejected"] == 1
    assert summary["verified_cobblestones_total"] == 12.5

    sync_resp = client.post(
        "/boinc/v1/ledger/sync",
        json={"host_to_steam_id64": {"result-501": "76561198000000501"}},
    )
    assert sync_resp.status_code == 200
    sync_data = sync_resp.json()
    assert sync_data["ingested_verdicts"] == 2
    assert coord.reputation_ledger["76561198000000501"].validated_workunits == 1
    assert coord.reputation_ledger["76561198000000501"].contributed_tflops == 2.5
    assert coord.reputation_ledger["boinc:result-502"].flagged_byzantine is True

    # Calling sync a second time is idempotent
    sync_again = client.post("/boinc/v1/ledger/sync").json()
    assert sync_again["ingested_verdicts"] == 0

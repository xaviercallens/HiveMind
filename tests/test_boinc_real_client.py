"""Tier-1 tests of the real-BOINC-client integration: no GPU, no BOINC binary needed."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hivemind.boinc.real_client.app import resolve_filename, run  # noqa: E402
from hivemind.boinc.real_client.bridge import validate_result  # noqa: E402
from hivemind.boinc.real_client.client_setup import project_dir_name, prepare_data_dir  # noqa: E402
from hivemind.boinc.real_client.project_server import ProjectServer  # noqa: E402

SPEC = {"wu_id": "wu_t", "seed": 3, "d_in": 64, "d_out": 64, "rank": 4, "batch": 32, "steps": 60, "lr": 2.0, "spot_check_interval": 20}


def _run(spec=SPEC, tmp: Path | None = None, **kw):
    return run(spec, "cpu", (tmp or Path(".")) / "ck.pt", **kw)


def test_project_dir_name_matches_boinc_escape_rule() -> None:
    assert project_dir_name("http://127.0.0.1:8765/hivemind/") == "127.0.0.1_8765_hivemind"
    assert project_dir_name("https://example.org/") == "example.org"


def _sub(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    return d


def test_app_is_deterministic_and_learns(tmp_path: Path) -> None:
    a, b = _run(tmp=_sub(tmp_path, "a")), _run(tmp=_sub(tmp_path, "b"))
    assert a["final_digest"] == b["final_digest"] and a["micro_digests"] == b["micro_digests"]
    assert a["final_loss"] < a["initial_loss"] * 0.5


def test_checkpoint_resume_is_bit_identical(tmp_path: Path) -> None:
    ref = _run(tmp=_sub(tmp_path, "ref"))
    d = _sub(tmp_path, "pre")
    with pytest.raises(SystemExit) as e:
        _run(tmp=d, crash_after_step=40)
    assert e.value.code == 75
    resumed = _run(tmp=d)
    assert resumed["resumed_from_step"] == 40
    assert resumed["final_digest"] == ref["final_digest"]


def test_resolve_filename_symlink_and_soft_link_file(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_text("{}")
    (tmp_path / "link").symlink_to(real)
    (tmp_path / "soft").write_text(f"<soft_link>{real}</soft_link>")
    assert resolve_filename("link", tmp_path) == real
    assert resolve_filename("soft", tmp_path) == real
    assert resolve_filename("plain", tmp_path) == tmp_path / "plain"


def test_validator_accepts_honest_and_rejects_forged(tmp_path: Path) -> None:
    honest = _run(tmp=_sub(tmp_path, "h"))
    v = validate_result(SPEC, honest, "h1", replay_devices=("cpu",))
    assert v.valid and v.same_class_exact is True and v.grant["cobblestones"] >= 0.0 and v.grant["capped_by"] == "verified"
    forged = _run(tmp=_sub(tmp_path, "bad"), inject_fault=0.01)
    v2 = validate_result(SPEC, forged, "h2", replay_devices=("cpu",))
    assert not v2.valid and v2.grant["cobblestones"] == 0.0 and "differs" in v2.reason


def test_validator_rejects_wrong_workunit_and_garbage_state(tmp_path: Path) -> None:
    r = _run(tmp=tmp_path)
    assert not validate_result({**SPEC, "wu_id": "other"}, r, "h", replay_devices=("cpu",)).valid
    r2 = {**r, "state_b64": {"a": "AAAA", "b": "AAAA"}}
    assert not validate_result(SPEC, r2, "h", replay_devices=("cpu",)).valid


def test_server_refuses_non_loopback(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ProjectServer(tmp_path, host="0.0.0.0")


def _req(hostid=0, gpu=True, results="") -> str:
    co = "<coprocs><coproc_cuda><count>1</count><name>Tesla T4</name></coproc_cuda></coprocs>" if gpu else ""
    return f"<scheduler_request><authenticator>a</authenticator><hostid>{hostid}</hostid><work_req_seconds>100</work_req_seconds>{co}{results}</scheduler_request>"


def test_scheduler_rpc_gives_work_only_with_gpu_and_acks_results(tmp_path: Path) -> None:
    srv = ProjectServer(tmp_path, validate=lambda s, d, h: None)
    srv.create_work({**SPEC})
    no_gpu = srv.scheduler_rpc(_req(gpu=False))
    assert "<workunit>" not in no_gpu
    rep = srv.scheduler_rpc(_req())
    assert rep.count("<workunit>") == 1 and rep.count("<result>") == 1
    assert re.search(r"<url>http://127\.0\.0\.1:\d+/file_upload_handler</url>", rep)
    assert "<command_line>" not in rep
    again = srv.scheduler_rpc(_req())
    assert "<workunit>" not in again  # not handed out twice
    done = srv.scheduler_rpc(_req(results="<result><name>wu_t_0</name><exit_status>1</exit_status><state>3</state></result>"))
    assert "<result_ack><name>wu_t_0</name></result_ack>" in done
    assert srv.workunits["wu_t"]["state"] == "client_error"


def test_upload_handler_roundtrip_and_size_query(tmp_path: Path) -> None:
    srv = ProjectServer(tmp_path, validate=lambda s, d, h: None)
    payload = b'{"x": 1}'
    body = (
        b"<data_server_request>\n<file_upload>\n<file_info>\n<name>out_a</name>\n<xml_signature>\nzz\n</xml_signature>\n"
        b"<max_nbytes>100</max_nbytes>\n</file_info>\n<nbytes>8</nbytes>\n<md5_cksum>m</md5_cksum>\n<offset>0</offset>\n<data>\n" + payload
    )
    assert "<status>0</status>" in srv.upload(body)
    assert (tmp_path / "upload" / "out_a").read_bytes() == payload
    q = srv.upload(b"<data_server_request><get_file_size>out_a</get_file_size></data_server_request>")
    assert "<file_size>8</file_size>" in q
    assert "<status>-1</status>" in srv.upload(body.replace(b"out_a", b"../evil"))


def test_download_md5_matches_workunit_input(tmp_path: Path) -> None:
    srv = ProjectServer(tmp_path)
    srv.create_work(SPEC)
    f = srv.workunits["wu_t"]["inputs"][0]
    data = (tmp_path / "download" / f["name"]).read_bytes()
    assert hashlib.md5(data).hexdigest() == f["md5"] and json.loads(data)["wu_id"] == "wu_t"


def test_prepare_data_dir_writes_anonymous_platform_gpu_app(tmp_path: Path) -> None:
    info = prepare_data_dir(tmp_path / "c", "http://127.0.0.1:9/hivemind/", Path("/src"))
    ai = (info["project_dir"] / "app_info.xml").read_text()
    assert "<type>NVIDIA</type>" in ai and "cuda" in ai and "<main_program/>" in ai
    assert (info["project_dir"] / "hivemind_inner_loop_100_x86_64-pc-linux-gnu").stat().st_mode & 0o100
    assert (tmp_path / "c" / "account_127.0.0.1_9_hivemind.xml").read_text().count("<authenticator>") == 1

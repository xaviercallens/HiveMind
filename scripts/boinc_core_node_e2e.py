#!/usr/bin/env python3
"""BOINC as a HiveMind core node: QLoRA MoE training + batch inference + local endpoint, on a stock client and the T4.

Criteria: artifacts/boinc_core_node/ACCEPTANCE.md (B1-B8). Run with the venv that has peft + bitsandbytes:

    <venv>/bin/python scripts/boinc_core_node_e2e.py --boinc-root .../boinc/root --work .../boinc/core1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from hivemind.boinc.real_client.client_setup import Client, moe_apps, prepare_data_dir  # noqa: E402
from hivemind.boinc.real_client.moe_bridge import MoEValidator  # noqa: E402
from hivemind.boinc.real_client.moe_work import build_infer_spec, build_train_spec  # noqa: E402
from hivemind.boinc.real_client.node_endpoint import NodeEndpoint  # noqa: E402
from hivemind.boinc.real_client.project_server import ProjectServer  # noqa: E402

FLOPS_EST = 4.8e14  # ~120 s at the declared 4e12 FLOPS, so the client's elapsed-time bound (100x) never bites
PROMPTS = ["The Mixture of Experts model", "Volunteer computing with BOINC means"]


def gpu_procs() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout
    # only processes this test may have left behind (its venv python / the BOINC client); a foreign GPU user such as
    # another session's llama-server is recorded separately and does not count against the criterion
    return [ln for ln in out.splitlines() if ln.strip() and not ln.startswith(f"{os.getpid()},") and ("boinc" in ln or "python" in ln)]


def gpu_foreign() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if ln.strip() and "boinc" not in ln and "python" not in ln]


def wait_for(cond, timeout: float, every: float = 1.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(every)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boinc-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/boinc_core_node/e2e_report.json")
    ap.add_argument("--rpc-port", type=int, default=31417)
    a = ap.parse_args()

    report: dict = {"gpu_busy_before": gpu_procs(), "criteria": {}}
    srv = ProjectServer(a.work / "server")
    (srv.dir / "assimilated").mkdir(exist_ok=True)
    mv = MoEValidator()

    def val_train(wu: dict, host: str):
        w = wu["spec"]["wu_id"]
        return mv.validate_train(wu["spec"], srv.output_path(w, "out.json"), srv.output_path(w, "out.pt"), host)

    def val_infer(wu: dict, host: str):
        w = wu["spec"]["wu_id"]
        adapter = torch.load(srv.dir / "download" / f"{w}_adapter_pt", map_location="cpu")["state"]
        return mv.validate_infer(wu["spec"], srv.output_path(w, "out.json"), adapter, host)

    def assimilate(wu: dict, verdict) -> None:
        from hivemind.boinc.real_client.moe_app import state_sha

        w = wu["spec"]["wu_id"]
        state = torch.load(srv.output_path(w, "out.pt"), map_location="cpu")["states"][-1]
        torch.save({"state": state, "sha256": state_sha(state), "wu_id": w, "verdict": verdict.to_dict()}, srv.dir / "assimilated" / f"{w}.pt")

    srv.app_validators = {"hivemind_qlora_moe": val_train, "hivemind_moe_infer": val_infer}
    srv.assimilators = {"hivemind_qlora_moe": assimilate}
    srv.start()
    prepare_data_dir(a.work / "client", srv.master_url, ROOT / "src", apps=moe_apps(sys.executable))
    client = Client(a.boinc_root / "usr/bin/boinc", a.boinc_root / "usr/lib/x86_64-linux-gnu", a.work / "client", a.rpc_port)

    good = build_train_spec("hm_moe_good_001")
    lazy = {**build_train_spec("hm_moe_lazy_001", shard_index=1)}
    kw = {"app": "hivemind_qlora_moe", "outputs": ("out.json", "out.pt"), "max_nbytes": 200_000_000, "flops_est": FLOPS_EST}
    srv.create_work(good, **kw)
    srv.create_work(lazy, command_line="--lazy", **kw)
    client.start()
    report["client_rpc_up"] = client.wait_rpc(90)
    report["stage_a_finished"] = srv.wait_all(1500)

    # --- B5 preemption: same data as `good` under a new id, SIGKILL after the first segment checkpoint
    pre = {**good, "wu_id": "hm_moe_pre_001"}
    srv.create_work(pre, **kw)
    client.cmd("--project", srv.master_url, "update")
    report["pre_ckpt_seen"] = wait_for(lambda: any((client.dir / "slots").glob("*/hivemind_moe_ckpt.pt")), 600, 0.5)
    time.sleep(2)
    client.kill()
    time.sleep(2)
    report["pre_app_survivors"] = subprocess.run(["pgrep", "-f", "real_client.moe_app"], capture_output=True, text=True).stdout.split()
    client.start()
    report["pre_client_restarted"] = client.wait_rpc(90)
    report["stage_pre_finished"] = srv.wait_all(1500)

    # --- B6 batch inference with the assimilated adapter
    good_art = srv.dir / "assimilated" / "hm_moe_good_001.pt"
    if good_art.exists():
        adapter_bytes = good_art.read_bytes()
        inf = build_infer_spec("hm_moe_infer_001", PROMPTS)
        srv.create_work(inf, app="hivemind_moe_infer", extra_inputs={"adapter.pt": adapter_bytes}, outputs=("out.json",), flops_est=FLOPS_EST)
        client.cmd("--project", srv.master_url, "update")
        report["stage_infer_finished"] = srv.wait_all(900)
    client.quit()
    srv.stop()
    time.sleep(2)

    # --- B7 interactive endpoint vs batch output
    ep_result = None
    if good_art.exists() and "hm_moe_infer_001" in srv.verdicts:
        batch = json.loads(srv.output_path("hm_moe_infer_001", "out.json").read_text())["completions"][0]["token_ids"]
        ep = NodeEndpoint(build_infer_spec("endpoint", PROMPTS), good_art)
        ep.start()
        import urllib.request

        req = urllib.request.Request(f"http://127.0.0.1:{ep.port}/v1/chat/completions", data=json.dumps({"model": ep.name, "messages": [{"role": "user", "content": PROMPTS[0]}], "max_tokens": 24}).encode(), headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        listen = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
        ep_result = {"endpoint_token_ids": resp["hivemind"]["token_ids"], "batch_token_ids": batch, "text": resp["choices"][0]["message"]["content"], "bound": ep.httpd.server_address[0], "port": ep.port, "ss_lines": [ln for ln in listen.splitlines() if f":{ep.port} " in ln or f":{srv.port} " in ln]}
        ep.stop()
    report["endpoint"] = ep_result

    del mv
    torch.cuda.empty_cache()
    time.sleep(2)
    report["gpu_busy_after"] = gpu_procs()
    report["gpu_foreign_after"] = gpu_foreign()
    report["stray_boinc"] = subprocess.run(["pgrep", "-f", "usr/bin/boinc"], capture_output=True, text=True).stdout.split()
    report["verdicts"] = {k: v.to_dict() for k, v in srv.verdicts.items()}

    # --- score
    v, c = srv.verdicts, report["criteria"]
    up = {w: [(srv.output_path(w, o).stat().st_size if srv.output_path(w, o).exists() else 0) for o in ("out.json", "out.pt")] for w in ("hm_moe_good_001",)}
    report["uploaded_sizes_bytes"] = up
    meta = json.loads(srv.output_path("hm_moe_good_001", "out.json").read_text()) if srv.output_path("hm_moe_good_001", "out.json").exists() else {}
    report["good_meta"] = {k: meta.get(k) for k in ("device_name", "eval_loss_before", "eval_loss_after", "wall_seconds", "peak_vram_mib", "trainable_params", "state_sha256")}
    g = v.get("hm_moe_good_001")
    c["B1"] = bool(meta.get("device_name") == "Tesla T4" and up["hm_moe_good_001"][0] > 0 and up["hm_moe_good_001"][1] > 10_000_000)
    c["B2"] = bool(g and g.valid and g.detail.get("max_abs_divergence", 1) <= 1e-4)
    c["B3"] = bool(g and g.grant["cobblestones"] > 0 and g.grant["capped_by"] == "verified" and good_art.exists())
    lz = v.get("hm_moe_lazy_001")
    c["B4"] = bool(lz and not lz.valid and lz.grant["cobblestones"] == 0)
    pm = json.loads(srv.output_path("hm_moe_pre_001", "out.json").read_text()) if srv.output_path("hm_moe_pre_001", "out.json").exists() else {}
    report["pre_meta"] = {"resumed_from_segment": pm.get("resumed_from_segment"), "state_sha_equal_to_uninterrupted": pm.get("state_sha256") == meta.get("state_sha256")}
    c["B5"] = bool(pm.get("resumed_from_segment", 0) > 0 and pm.get("state_sha256") == meta.get("state_sha256") and v.get("hm_moe_pre_001") and v["hm_moe_pre_001"].valid)
    c["B6"] = bool(v.get("hm_moe_infer_001") and v["hm_moe_infer_001"].valid)
    c["B7"] = bool(ep_result and ep_result["endpoint_token_ids"] == ep_result["batch_token_ids"] and ep_result["bound"] == "127.0.0.1")
    c["B8"] = not report["gpu_busy_after"] and not report["stray_boinc"] and srv.httpd.server_address[0] == "127.0.0.1"
    report["PASS"] = all(c.values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    (a.out.parent / "client_stdout.log").write_text(client.log_path.read_text(errors="replace")[-80000:])
    print(json.dumps({"criteria": c, "PASS": report["PASS"]}, indent=2))
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

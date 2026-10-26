"""Live monitor for end users of the public HiveMind services: real traffic from request logs plus synthetic probes.

Two parts, each written as one JSON line to ``--out``:

1. ``traffic``: Cloud Run request logs of the three public services since ``--since`` (default: the last hour). Counts per
   path group and status, chat-completion latency (p50, p95, max), registrations, 402 and 5xx counts, and browser versus
   scripted callers. No IP address, key, prompt or account is read or written.
2. ``probes``: one persistent probe account (``--key-file``) asks each service a question per tier whose answer is
   checked: the Bee expert must classify a held-out star as in the NIGHT-MOE-1 run, the coder must define ``add(a, b)``
   returning ``a + b`` (checked on the syntax tree, the code is never executed), auto must answer 17 * 23 = 391. Every
   checker is first run on a known-wrong answer and must reject it, or the probe run is marked INVALID. An unknown model
   must be refused (404) and a request without a key must get 401.

With ``--admin-key-file`` the record also gets ``users``: real and test accounts counted apart (total, created and active in the window),
read from the router's admin endpoint because request logs cannot tell a test account from a real one.

Exit status: 0 when every probe passes, 1 when a probe fails, 2 when a checker control fails (the run proves nothing).
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

SERVICES: dict[str, str] = {
    "ruche": "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app",
    "tribute": "https://hivemind-at-home-cfhmhmvv5a-uc.a.run.app",
    "router": "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
}
USER_PATH_PREFIXES: tuple[str, ...] = ("/v1/", "/signup", "/login")
CLOUD_RUN_NAMES: dict[str, str] = {"ruche": "hivemind-at-home-ruche", "tribute": "hivemind-at-home", "router": "hivemind-openrouter"}
BEE_ITEM_PATH: str = "artifacts/experiments/BEE_STARS_probe_item.json"


def check_bee(text: str, gold_reply: str) -> bool:
    return text.strip() == gold_reply


def check_coder(text: str) -> bool:
    """True when the reply defines add(a, b) whose body is a single `return a + b` (either operand order)."""
    code = text
    if "```" in code:
        blocks = code.split("```")
        code = blocks[1] if len(blocks) > 1 else code
        code = code.removeprefix("python").removeprefix("py")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "add" and [a.arg for a in node.args.args] == ["a", "b"]:
            rets = [n for n in node.body if isinstance(n, ast.Return)]
            if len(rets) == 1 and isinstance(rets[0].value, ast.BinOp) and isinstance(rets[0].value.op, ast.Add):
                names = {getattr(rets[0].value.left, "id", None), getattr(rets[0].value.right, "id", None)}
                if names == {"a", "b"}:
                    return True
    return False


def check_arith(text: str) -> bool:
    return "391" in text.replace(",", "").split()[:3] or text.strip().rstrip(".") == "391"


def checker_controls(gold_reply: str) -> dict[str, bool]:
    """Each checker must accept its right answer and reject a near-miss; a checker that cannot fail proves nothing."""
    return {
        "bee_accepts_gold": check_bee(gold_reply, gold_reply),
        "bee_rejects_wrong": not check_bee("Class: Z", gold_reply),
        "coder_accepts_right": check_coder("```python\ndef add(a, b):\n    return a + b\n```"),
        "coder_rejects_minus": not check_coder("```python\ndef add(a, b):\n    return a - b\n```"),
        "coder_rejects_prose": not check_coder("Here is a function that adds two numbers."),
        "arith_accepts_391": check_arith("391"),
        "arith_rejects_381": not check_arith("381"),
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def traffic_summary(since: datetime) -> dict[str, Any]:
    """Aggregate Cloud Run request logs of the public services; returns counts only."""
    names = " OR ".join(f'"{n}"' for n in CLOUD_RUN_NAMES.values())
    flt = (f'resource.type="cloud_run_revision" AND resource.labels.service_name=({names}) AND '
           f'timestamp>="{since.strftime("%Y-%m-%dT%H:%M:%SZ")}" AND httpRequest.requestUrl:("/v1/" OR "/signup" OR "/login")')
    proc = subprocess.run(["gcloud", "logging", "read", flt, "--limit", "5000", "--format=json"], capture_output=True, text=True,
                          timeout=180, check=False)
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()[-300:]}
    rows = json.loads(proc.stdout or "[]")
    by_service: dict[str, Any] = {}
    for svc_key, svc_name in CLOUD_RUN_NAMES.items():
        mine = [r for r in rows if r.get("resource", {}).get("labels", {}).get("service_name") == svc_name
                and r.get("httpRequest", {}).get("requestUrl", "").split(".run.app")[-1].startswith(USER_PATH_PREFIXES)]
        statuses: Counter[str] = Counter()
        chat_latency: list[float] = []
        browser = scripted = registrations = 0
        for r in mine:
            h = r.get("httpRequest", {})
            path = h.get("requestUrl", "").split(".run.app")[-1].split("?")[0]
            if not path.startswith(USER_PATH_PREFIXES):
                continue  # e.g. the dashboard's own /api/v1/... polling, which the log filter's substring match lets in
            group = "chat" if path.startswith("/v1/chat/completions") else "register" if "/register" in path else \
                "heartbeat" if "heartbeat" in path else "account" if path.startswith("/v1/hivemind/") else path.split("/")[1] or "/"
            status = int(h.get("status", 0))
            statuses[f"{group}:{status}"] += 1
            if group == "register" and status == 200:
                registrations += 1
            if group == "chat":
                chat_latency.append(float(str(h.get("latency", "0s")).rstrip("s") or 0.0))
            if group not in ("heartbeat",):
                if "Mozilla" in h.get("userAgent", ""):
                    browser += 1
                else:
                    scripted += 1
        by_service[svc_key] = {
            "requests": len(mine), "status": dict(sorted(statuses.items())), "registrations": registrations,
            "chat": {"n": len(chat_latency), "p50_s": _percentile(chat_latency, 0.5), "p95_s": _percentile(chat_latency, 0.95),
                     "max_s": max(chat_latency) if chat_latency else None},
            "http_402": sum(n for k, n in statuses.items() if k.endswith(":402")),
            "http_5xx": sum(n for k, n in statuses.items() if k.split(":")[1].startswith("5")),
            "browser_requests": browser, "scripted_requests": scripted,
        }
    return {"since_utc": since.isoformat(), "log_rows": len(rows), "services": by_service}


def user_counts(client: httpx.Client, base: str, admin_key: str, window_s: float) -> dict[str, Any]:
    """Real and test accounts counted apart, from the router's admin endpoint: the request logs cannot tell them apart."""
    reply = client.get(base + "/v1/hivemind/admin/users", params={"window_s": window_s}, headers={"X-Admin-Key": admin_key})
    if reply.status_code != 200:
        return {"error": f"HTTP {reply.status_code}"}
    return dict(reply.json())


def _chat(client: httpx.Client, base: str, key: str, model: str, messages: list[dict[str, str]], stream: bool) -> tuple[int, str, float]:
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 96, "stream": stream}
    headers = {"Authorization": f"Bearer {key}"}
    t0 = time.monotonic()
    if not stream:
        r = client.post(base + "/v1/chat/completions", json=body, headers=headers)
        text = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else r.text[:200]
        return r.status_code, text, time.monotonic() - t0
    parts: list[str] = []
    with client.stream("POST", base + "/v1/chat/completions", json=body, headers=headers) as s:
        status = s.status_code
        for line in s.iter_lines():
            if line.startswith("data: {"):
                choices = json.loads(line[6:]).get("choices") or [{}]
                parts.append((choices[0].get("delta") or {}).get("content") or "")
            elif status != 200:
                parts.append(line)
    return status, "".join(parts), time.monotonic() - t0


def run_probes(key: str, bee_item: dict[str, Any], services: dict[str, str]) -> dict[str, Any]:
    gold_reply = bee_item["expected_reply"]
    controls = checker_controls(gold_reply)
    if not all(controls.values()):
        return {"valid": False, "controls": controls, "results": []}
    cases: list[tuple[str, list[dict[str, str]], Callable[[str], bool], bool]] = [
        ("hivemind/bee-stars", bee_item["messages"], lambda t: check_bee(t, gold_reply), False),
        ("hivemind/coder", [{"role": "user", "content": "Write a Python function add(a, b) that returns a + b. Code only."}], check_coder, True),
        ("hivemind/auto", [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}], check_arith, False),
    ]
    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=180) as client:
        for svc, base in services.items():
            for model, messages, check, stream in cases:
                try:
                    status, text, secs = _chat(client, base, key, model, messages, stream)
                except httpx.HTTPError as exc:
                    status, text, secs = 0, f"{type(exc).__name__}", 0.0
                results.append({"service": svc, "model": model, "stream": stream, "status": status, "secs": round(secs, 2),
                                "ok": status == 200 and check(text), "reply_head": text[:80]})
            unknown = client.post(base + "/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                                  json={"model": "hivemind/no-such-model", "messages": [{"role": "user", "content": "hi"}]})
            nokey = client.post(base + "/v1/chat/completions", json={"model": "hivemind/auto", "messages": [{"role": "user", "content": "hi"}]})
            results.append({"service": svc, "model": "control:unknown-model", "status": unknown.status_code, "ok": unknown.status_code == 404})
            results.append({"service": svc, "model": "control:no-key", "status": nokey.status_code, "ok": nokey.status_code == 401})
        me = client.get(services["router"] + "/v1/hivemind/me", headers={"Authorization": f"Bearer {key}"})
    balance = me.json().get("balance") if me.status_code == 200 else None
    return {"valid": True, "controls": controls, "results": results, "probe_balance_hmt": balance}


T4_HOST: tuple[str, str] = ("socreateai-agora-hermes-node1", "us-east4-b")
T4_TELEMETRY: str = "/mnt/disks/disk-socrateai-local-1/hivemind_telemetry"
# Runs on the VM: reads the governor/monitor telemetry written by the T4 node's own services; prints one JSON object.
_T4_READER: str = r"""
import json, time, collections
T = "%s"; since = %f
latest = json.load(open(T + "/latest.json"))
ev = collections.Counter(); yields_ms = []; real_ms = []; real_ended = []; states = collections.Counter(); last_state = None
for line in open(T + "/governor.jsonl"):
    try: r = json.loads(line)
    except ValueError: continue
    if r.get("t", 0) < since: continue
    e = r.get("event"); ev[e] += 1
    if e == "snapshot": states[r.get("state")] += 1; last_state = r.get("state")
    if e == "yield" and r.get("gpu_freed_ms") is not None:  # field names of t4_governor.py
        yields_ms.append(r["gpu_freed_ms"])
        # a hand-back only counts as a measurement when a BOINC GPU task was actually running and a request caused it
        if r.get("had_boinc_gpu_task") and r.get("trigger") == "nginx": real_ms.append(r["gpu_freed_ms"]); real_ended.append(r.get("ended_by_governor"))
units = latest.get("units", {})
print(json.dumps({"governor": latest.get("governor"), "events": dict(ev), "snapshot_states": dict(states), "yield_ms": yields_ms, "real_yield_ms": real_ms, "real_yield_ended_by_governor": real_ended,
                  "qlora_progress": latest.get("qlora_progress"), "gpu": latest.get("gpu"), "nginx": latest.get("nginx"),
                  "units_inactive": sorted(k for k, v in units.items() if v.get("enabled") == "enabled" and v.get("active") != "active"),
                  "gateway_live_tiers": (latest.get("gateway") or {}).get("live_tiers"), "telemetry_age_s": round(time.time() - latest.get("t", 0), 1),
                  "hold_file": __import__("os").path.exists(T + "/HOLD")}))
"""


def t4_summary(since: datetime) -> dict[str, Any]:
    """Serving/training state of the T4 node from its own telemetry: governor state over the window, yields to inference and
    their latency, QLoRA workunits assimilated, VRAM and processes, and managed units that are enabled but not running."""
    code = _T4_READER % (T4_TELEMETRY, since.timestamp())
    proc = subprocess.run(["gcloud", "compute", "ssh", T4_HOST[0], "--zone", T4_HOST[1], "--command", f"python3 -c '{code}'"],
                          capture_output=True, text=True, timeout=240, check=False)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        return {"error": (proc.stderr.strip() or "no output")[-300:]}
    return json.loads(lines[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key-file", type=Path, required=True, help="file holding the probe account's hm-... key")
    parser.add_argument("--out", type=Path, required=True, help="JSONL file; one line is appended per run")
    parser.add_argument("--since", default="", help="ISO time for the traffic window (default: one hour ago)")
    parser.add_argument("--no-probes", action="store_true", help="traffic summary only")
    parser.add_argument("--admin-key-file", type=Path, help="file holding the router's admin key: adds real and test user counts to the record")
    parser.add_argument("--t4", action="store_true", help="add the T4 node's serving/training state (via gcloud compute ssh)")
    args = parser.parse_args()
    since = datetime.fromisoformat(args.since.replace("Z", "+00:00")) if args.since else datetime.now(timezone.utc) - timedelta(hours=1)
    record: dict[str, Any] = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "traffic": traffic_summary(since)}
    code = 0
    if not args.no_probes:
        bee_item = json.loads(Path(BEE_ITEM_PATH).read_text(encoding="utf-8"))
        probes = run_probes(args.key_file.read_text(encoding="utf-8").strip(), bee_item, SERVICES)
        record["probes"] = probes
        if not probes["valid"]:
            code = 2
        elif not all(r["ok"] for r in probes["results"]):
            code = 1
        passed = sum(r["ok"] for r in probes["results"])
        record["probe_summary"] = f"{passed}/{len(probes['results'])} passed" if probes["valid"] else "INVALID: checker control failed"
    if args.admin_key_file is not None:
        window_s = max(0.0, (datetime.now(timezone.utc) - since).total_seconds())
        with httpx.Client(timeout=30) as client:
            record["users"] = user_counts(client, SERVICES["router"], args.admin_key_file.read_text(encoding="utf-8").strip(), window_s)
    if args.t4:
        record["t4"] = t4_summary(since)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    t = record["traffic"]
    if "services" in t:
        for svc, s in t["services"].items():
            print(f"{svc:8s} requests {s['requests']:4d}  chat {s['chat']['n']:3d} (p50 {s['chat']['p50_s']}, p95 {s['chat']['p95_s']} s)  "
                  f"402 {s['http_402']}  5xx {s['http_5xx']}  registrations {s['registrations']}  browser {s['browser_requests']}")
    else:
        print("traffic:", t)
    if "users" in record:
        u = record["users"]
        if "error" in u:
            print("users: unavailable:", u["error"])
        else:
            for kind in ("real", "test"):
                print(f"users {kind:4s} total {u[kind]['total']:4d}  created in window {u[kind]['created']:4d}  active in window {u[kind]['active']:4d}")
    if "t4" in record:
        n = record["t4"]
        if "error" in n:
            print("t4: unavailable:", n["error"])
        else:
            g = n.get("governor") or {}
            gpu = n.get("gpu") or {}
            print(f"t4: now {g.get('state')} (BOINC GPU {g.get('boinc_gpu_mode')}, hold file {n.get('hold_file')}) | window states {n['snapshot_states']} | "
                  f"yields {len(n['yield_ms'])} (real: with a BOINC task running, caused by a request: {n['real_yield_ms']} ms, ended_by_governor {n['real_yield_ended_by_governor']}; max of all {max(n['yield_ms'], default=None)} ms) | QLoRA assimilated {(n.get('qlora_progress') or {}).get('assimilated')} | "
                  f"VRAM {gpu.get('used_mib')}/{gpu.get('total_mib')} MiB util {gpu.get('util_pct')}% | inactive units {n['units_inactive']}")
    if "probes" in record:
        print("probes:", record["probe_summary"])
        for r in record["probes"]["results"]:
            if not r["ok"]:
                print("  FAIL", r)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

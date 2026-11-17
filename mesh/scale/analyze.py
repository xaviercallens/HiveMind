#!/usr/bin/env python3
"""MESH-SCALE-1 analysis. Python standard library only.

usage: analyze.py RUN_DIR [RUN_DIR ...] [--out PATH] [--prereg PATH]

Reads run directories written by run_scale.sh and computes, per run: registration time, submitted /
replied / lost / duplicates, e2e and overhead percentiles, throughput, the kill outcome, peak RSS per
process, and OutOfMemoryError flags. Acceptance criteria A1-A4 are evaluated exactly as pre-registered in
artifacts/experiments/MESH_SCALE_prereg.json, and only for N=1000 runs under the registered conditions.
Every other run is NOT_RUN for acceptance and is labelled smoke or non-prereg.

Operational definitions (fixed here, before any prereg run; see the amendment notes in the report):
- overhead = e2eMs - serviceMs, over tasks with an ok reply, serviceMs known, and no attempt routed to a
  killed bee (any ASSIGN to a killed bee, at any time).
- A3 set = tasks a killed bee dropped (DROP line) or received after its kill (ASSIGN at or after KILL).
  This is the set "assigned to killed bees, answered after reassignment". The prereg's first-attempt
  variant (ASSIGN attempt 1 to a killed bee, any time) is reported next to it, not used for the verdict.
- A1 looks at the latest HostAnnounce of each host received at or before last bee JVM launch + 60 s.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys

EXPERIMENT = "MESH-SCALE-1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_PREREG = os.path.join(REPO, "artifacts", "experiments", "MESH_SCALE_prereg.json")
DEFAULT_OUT = os.path.join(REPO, "artifacts", "experiments", "MESH_SCALE_report.json")
PREREG_LEVELS = (100, 300, 1000)

PREREG_RATE = 200
PREREG_DURATION = 120
PREREG_KILL_AT = 60
PREREG_KILL_FRACTION = 0.10
PREREG_BEE_JVMS = 4
PREREG_HOSTS = 2
PREREG_BEE_XMX = 768
PREREG_HOST_XMX = 512
PREREG_KAFKA_XMX = 512
PREREG_MAX_CONCURRENT = 4
PREREG_LATENCY = 50
PREREG_JITTER = 10
PREREG_HEARTBEAT = 5000
PREREG_ATTEMPT_TIMEOUT = 10000
PREREG_TIER = "coder"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj):
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(values, p):
    """Nearest-rank percentile. None when there are no values."""
    if not values:
        return None
    s = sorted(values)
    k = max(1, math.ceil(p / 100.0 * len(s)))
    return s[k - 1]


def mb_from_xmx(text):
    m = re.fullmatch(r"\s*(\d+)\s*([mMgG]?)\s*", str(text))
    if not m:
        return None
    n = int(m.group(1))
    return n * 1024 if m.group(2) in ("g", "G") else n


def parse_events(event_dir):
    assigns = []   # (taskId, attempt, beeId, ms)
    drops = []     # (taskId, beeId, attempt, phase, ms)
    kills = {}     # beeId -> ms
    for path in sorted(glob.glob(os.path.join(event_dir, "*.log"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if not p:
                    continue
                if p[0] == "ASSIGN" and len(p) == 5:
                    assigns.append((p[1], int(p[2]), p[3], int(p[4])))
                elif p[0] == "DROP" and len(p) == 6:
                    drops.append((p[1], p[2], int(p[3]), p[4], int(p[5])))
                elif p[0] == "KILL" and len(p) == 3:
                    kills[p[1]] = int(p[2])
    return assigns, drops, kills


def parse_peaks(path):
    peaks = {}
    kafka = {}
    if not os.path.exists(path):
        return peaks, kafka
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"^kafka pid=(\S+) xmx=(\S+) vmhwm_kb_lifetime=(\S+)$", line)
            if m:
                kafka = {"pid": m.group(1), "xmx_text": m.group(2), "vmhwm_kb_lifetime": m.group(3)}
                continue
            m = re.match(r"^(\S+) (\S+) vmhwm_kb=(\S+) (?:alive_at_measure=(\d)|sampled=(\d).*)$", line)
            if m:
                peaks[m.group(1)] = {
                    "pid": m.group(2),
                    "vmhwm_kb": None if m.group(3) == "NA" else int(m.group(3)),
                    "alive_at_measure": (m.group(4) == "1") if m.group(4) is not None else True,
                }
    return peaks, kafka


def parse_oom(path):
    total = 0
    per_file = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"^(\S+) OutOfMemoryError=(\d+)", line.strip())
                if m:
                    per_file[m.group(1)] = int(m.group(2))
                    total += int(m.group(2))
    return total, per_file


def prereg_conditions(meta, achieved_rate, load_duration):
    checks = {
        "rate_tps": meta.get("rate_tps") == PREREG_RATE,
        "duration_s": meta.get("duration_s") == PREREG_DURATION,
        "kill_at_s": meta.get("kill_at_s") == PREREG_KILL_AT,
        "kill_fraction": meta.get("kill_fraction") == PREREG_KILL_FRACTION,
        "bee_jvms": meta.get("bee_jvms") == PREREG_BEE_JVMS,
        "hosts": meta.get("hosts") == PREREG_HOSTS,
        "bee_xmx_mb": meta.get("bee_xmx_mb") == PREREG_BEE_XMX,
        "host_xmx_mb": meta.get("host_xmx_mb") == PREREG_HOST_XMX,
        "kafka_xmx_mb": mb_from_xmx(meta.get("kafka_xmx_mb")) == PREREG_KAFKA_XMX,
        "max_concurrent_per_bee": meta.get("max_concurrent_per_bee") == PREREG_MAX_CONCURRENT,
        "sim_latency_ms": meta.get("sim_latency_ms") == PREREG_LATENCY,
        "sim_jitter_ms": meta.get("sim_jitter_ms") == PREREG_JITTER,
        "heartbeat_ms": meta.get("heartbeat_ms") == PREREG_HEARTBEAT,
        "attempt_timeout_ms": meta.get("attempt_timeout_ms") == PREREG_ATTEMPT_TIMEOUT,
        "tier": meta.get("tier") == PREREG_TIER,
        "load_duration_measured_s": load_duration is not None and abs(load_duration - PREREG_DURATION) < 1.0,
        "offered_rate_met": achieved_rate is not None and achieved_rate >= 0.98 * PREREG_RATE,
    }
    return all(checks.values()), checks


def verdict_a1(n, meta, announces):
    last = meta.get("last_bee_launch_ms")
    if last is None or not announces:
        return {"verdict": "NOT_RUN", "measured": "no host.announce or no launch time"}
    deadline = last + 60_000
    latest = {}
    for a in announces:  # file order is receive order
        if a["recvMs"] <= deadline:
            latest[a["hostId"]] = a
    per_host = {h: a["liveBees"] for h, a in sorted(latest.items())}
    ok = len(per_host) >= PREREG_HOSTS and all(v >= n for v in per_host.values())
    return {"verdict": "PASS" if ok else "FAIL",
            "measured": {"live_bees_per_host_at_60s": per_host, "hosts_seen": len(per_host)}}


def verdict_a2(n_lost, n_dup, n_submitted, n_replied, p95, p99):
    if p95 is None:
        return {"verdict": "NOT_RUN", "measured": "no overhead samples"}
    ok = (n_lost == 0 and n_dup == 0 and n_replied == n_submitted and p95 <= 250 and p99 <= 1000)
    return {"verdict": "PASS" if ok else "FAIL",
            "measured": {"lost": n_lost, "duplicates": n_dup, "replied": n_replied, "submitted": n_submitted,
                         "overhead_p95_ms": p95, "overhead_p99_ms": p99}}


def verdict_a3(a3_set, rows):
    if not a3_set:
        return {"verdict": "NOT_RUN", "measured": "no tasks were assigned to killed bees (kill not observed)"}
    ok_count = sum(1 for t in a3_set if rows.get(t, {}).get("ok") is True)
    frac = ok_count / len(a3_set)
    return {"verdict": "PASS" if frac >= 0.99 else "FAIL",
            "measured": {"killed_assigned_tasks": len(a3_set), "answered_ok": ok_count,
                         "fraction_ok": round(frac, 6)}}


def verdict_a4(oom_total, early_exits, kafka_xmx_mb):
    clauses = {
        "no_oom": oom_total == 0,
        "no_early_exit": not early_exits,
        "kafka_heap_512mb": kafka_xmx_mb == PREREG_KAFKA_XMX,
    }
    if kafka_xmx_mb is None:
        verdict = "NOT_RUN"
    else:
        verdict = "PASS" if all(clauses.values()) else "FAIL"
    return {"verdict": verdict,
            "measured": {"oom_total": oom_total, "early_exits": early_exits,
                         "kafka_xmx_mb": kafka_xmx_mb, "clauses": clauses,
                         "note": "heap use within -Xmx is not measured directly; RSS is reported"}}


def analyze_run(run_dir):
    meta_path = os.path.join(run_dir, "run.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    n = meta["n_bees"]
    rows = load_jsonl(os.path.join(run_dir, "loadgen.jsonl"))
    summaries = [r for r in rows if r.get("summary")]
    tasks = [r for r in rows if not r.get("summary")]
    summary = summaries[-1] if summaries else {}
    by_id = {t["taskId"]: t for t in tasks}
    announces = load_jsonl(os.path.join(run_dir, "announce.jsonl"))
    assigns, drops, kills = parse_events(os.path.join(run_dir, "events"))
    peaks, kafka = parse_peaks(os.path.join(run_dir, "peak_rss.txt"))
    oom_total, oom_files = parse_oom(os.path.join(run_dir, "oom_grep.txt"))

    killed = set(kills)
    kill_ms = min(kills.values()) if kills else None
    load_start = summary.get("startMs")
    expected_kill = None if load_start is None else load_start + meta["kill_at_s"] * 1000
    affected_any = {a[0] for a in assigns if a[2] in killed}
    first_attempt_killed = {a[0] for a in assigns if a[2] in killed and a[1] == 1}
    a3_set = {d[0] for d in drops if d[1] in killed}
    a3_set |= {a[0] for a in assigns if a[2] in killed and a[3] >= kills[a[2]]}

    replied_rows = [t for t in tasks if t.get("replyMs") is not None]
    ok_rows = [t for t in replied_rows if t.get("ok") is True]
    e2e = [t["e2eMs"] for t in replied_rows]
    overhead = [t["e2eMs"] - t["serviceMs"] for t in ok_rows
                if t.get("serviceMs") is not None and t["taskId"] not in affected_any]
    service_missing = sum(1 for t in ok_rows if t.get("serviceMs") is None)
    lost = sum(1 for t in tasks if t.get("lost") is True)
    duration = summary.get("durationS", meta["duration_s"])
    achieved = summary.get("achievedRateTps")

    # Registration: first moment when every host has seen N live bees (recorder view).
    reg_ms = None
    latest = {}
    for a in announces:
        latest[a["hostId"]] = a
        if len(latest) >= meta["hosts"] and all(v["liveBees"] >= n for v in latest.values()):
            reg_ms = a["recvMs"]
            break
    last_launch = meta.get("last_bee_launch_ms")
    registration_s = None if (reg_ms is None or last_launch is None) else (reg_ms - last_launch) / 1000.0

    # Diagnostics (not verdict inputs): Kafka publish failures in the JVM logs, and retries that did not
    # touch a killed bee (a lost assignment shows up as a reassignment after attemptTimeoutMs).
    publish_errors = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "logs", "*.log"))):
        with open(path, encoding="utf-8", errors="replace") as f:
            c = sum(1 for line in f if "Error publishing to topic" in line)
        if c:
            publish_errors[os.path.basename(path)[:-4]] = c
    retries_not_killed = sum(1 for t in tasks if (t.get("attempts") or 0) > 1 and t["taskId"] not in affected_any)
    warm = None if load_start is None else load_start + 10_000
    after_warm = [t["e2eMs"] - t["serviceMs"] for t in ok_rows
                  if t.get("serviceMs") is not None and t["taskId"] not in affected_any
                  and warm is not None and t["submitMs"] >= warm]

    kafka_xmx = mb_from_xmx(meta.get("kafka_xmx_mb"))
    conds_ok, conds = prereg_conditions(meta, achieved, duration)
    if n == 1000 and conds_ok:
        label = "prereg-1000"
    elif conds_ok and n in PREREG_LEVELS:
        label = "prereg-level-%d" % n
    elif conds_ok:
        label = "smoke"
    else:
        label = "non-prereg"

    non_loadgen = {k: v for k, v in peaks.items() if k != "loadgen"}
    early_exits = sorted(k for k, v in non_loadgen.items() if not v.get("alive_at_measure", True))

    result = {
        "run_id": meta["run_id"],
        "run_dir": run_dir,
        "label": label,
        "n_bees": n,
        "prereg_conditions": conds_ok,
        "prereg_condition_checks": conds,
        "registration": {
            "recorder_all_hosts_at_n_s": registration_s,
            "status_exit_code": meta.get("status_exit_code"),
            "status_end_minus_last_launch_s": (
                None if last_launch is None else (meta["status_end_ms"] - last_launch) / 1000.0),
        },
        "load": {
            "rate_target_tps": meta["rate_tps"],
            "duration_s": duration,
            "achieved_submit_rate_tps": achieved,
            "submitted": len(tasks),
            "submitted_summary": summary.get("submitted"),
            "replied": len(replied_rows),
            "ok": len(ok_rows),
            "not_ok": len(replied_rows) - len(ok_rows),
            "lost": lost,
            "submit_failures": summary.get("submitFailures"),
            "duplicates": summary.get("duplicates"),
            "service_ms_missing": service_missing,
            "throughput_tps": len(replied_rows) / duration if duration else None,
        },
        "latency_ms": {
            "e2e_p50": pct(e2e, 50), "e2e_p95": pct(e2e, 95), "e2e_p99": pct(e2e, 99),
            "overhead_p50": pct(overhead, 50), "overhead_p95": pct(overhead, 95),
            "overhead_p99": pct(overhead, 99), "overhead_samples": len(overhead),
        },
        "kill": {
            "killed_bees": len(killed),
            "kill_ms_first": kill_ms,
            "expected_kill_ms": expected_kill,
            "kill_lag_ms": None if (kill_ms is None or expected_kill is None) else kill_ms - expected_kill,
            "tasks_touching_killed_bees_any_attempt": len(affected_any),
            "a3_set_size": len(a3_set),
            "a3_answered_ok": sum(1 for t in a3_set if by_id.get(t, {}).get("ok") is True),
            "first_attempt_on_killed_size": len(first_attempt_killed),
            "first_attempt_on_killed_answered_ok": sum(
                1 for t in first_attempt_killed if by_id.get(t, {}).get("ok") is True),
        },
        "peak_rss_mb": {k: (None if v["vmhwm_kb"] is None else round(v["vmhwm_kb"] / 1024.0, 1))
                        for k, v in sorted(peaks.items())},
        "kafka": {"pid": kafka.get("pid"), "xmx_mb": kafka_xmx,
                  "vmhwm_mb_lifetime": (None if not kafka.get("vmhwm_kb_lifetime", "NA").isdigit()
                                        else round(int(kafka["vmhwm_kb_lifetime"]) / 1024.0, 1))},
        "oom": {"total": oom_total, "per_file": oom_files},
        "early_exits": early_exits,
        "diagnostics_not_verdict_inputs": {
            "kafka_publish_error_lines": sum(publish_errors.values()),
            "kafka_publish_error_lines_per_log": publish_errors,
            "retries_not_touching_killed_bees": retries_not_killed,
            "overhead_p95_after_first_10s_of_load": pct(after_warm, 95),
            "overhead_p99_after_first_10s_of_load": pct(after_warm, 99),
            "overhead_samples_after_first_10s_of_load": len(after_warm),
        },
    }

    if label == "prereg-1000":
        result["acceptance_at_1000"] = {
            "A1": verdict_a1(n, meta, announces),
            "A2": verdict_a2(lost, summary.get("duplicates", 0), len(tasks), len(replied_rows),
                             result["latency_ms"]["overhead_p95"], result["latency_ms"]["overhead_p99"]),
            "A3": verdict_a3(a3_set, by_id),
            "A4": verdict_a4(oom_total, early_exits, kafka_xmx),
        }
        result["a3_first_attempt_variant"] = {
            "size": len(first_attempt_killed),
            "fraction_ok": (round(result["kill"]["first_attempt_on_killed_answered_ok"] / len(first_attempt_killed), 6)
                            if first_attempt_killed else None),
        }
    else:
        reason = ("N=1000 but non-prereg conditions" if n == 1000 else "acceptance is defined for N=1000 only")
        if n == 1000 and not conds_ok:
            reason = "N=1000 but run conditions differ from the prereg: " + ", ".join(
                k for k, v in conds.items() if not v)
        result["acceptance_at_1000"] = {k: {"verdict": "NOT_RUN", "measured": reason}
                                        for k in ("A1", "A2", "A3", "A4")}
    return result


def combine(runs):
    """Per criterion: FAIL if any N=1000 prereg run fails, PASS if all pass, else NOT_RUN."""
    prereg_runs = [r for r in runs if r["label"] == "prereg-1000"]
    out = {}
    for crit in ("A1", "A2", "A3", "A4"):
        if not prereg_runs:
            out[crit] = {"verdict": "NOT_RUN", "measured": "no N=1000 run under the prereg conditions"}
            continue
        verdicts = [r["acceptance_at_1000"][crit]["verdict"] for r in prereg_runs]
        if "FAIL" in verdicts:
            v = "FAIL"
        elif all(x == "PASS" for x in verdicts):
            v = "PASS"
        else:
            v = "NOT_RUN"
        out[crit] = {"verdict": v, "per_run": {r["run_id"]: r["acceptance_at_1000"][crit]
                                              for r in prereg_runs}}
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--prereg", default=DEFAULT_PREREG)
    args = ap.parse_args(argv)

    runs = [analyze_run(os.path.abspath(d)) for d in args.run_dirs]
    report = {
        "experiment_id": EXPERIMENT,
        "prereg_path": os.path.relpath(args.prereg, REPO),
        "prereg_sha256": sha256_file(args.prereg),
        "runs": runs,
        "acceptance_at_1000": combine(runs),
    }
    report["overall"] = (
        "NOT_RUN" if all(v["verdict"] == "NOT_RUN" for v in report["acceptance_at_1000"].values())
        else ("FAIL" if any(v["verdict"] == "FAIL" for v in report["acceptance_at_1000"].values())
              else ("PASS" if all(v["verdict"] == "PASS" for v in report["acceptance_at_1000"].values())
                    else "NOT_RUN")))
    report["receipt"] = {
        "sha256": canonical_sha256(report),
        "method": "sha256 of canonical JSON (sort_keys, separators , and :, ascii) of this object without the receipt field",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    for r in runs:
        ld = r["load"]
        print("%s  label=%s  N=%d  submitted=%s replied=%s ok=%s lost=%s dup=%s  reg=%ss  "
              "overhead p95=%s p99=%s  e2e p50=%s  tps=%.1f" % (
                  r["run_id"], r["label"], r["n_bees"], ld["submitted"], ld["replied"], ld["ok"], ld["lost"],
                  ld["duplicates"], r["registration"]["recorder_all_hosts_at_n_s"],
                  r["latency_ms"]["overhead_p95"], r["latency_ms"]["overhead_p99"],
                  r["latency_ms"]["e2e_p50"], ld["throughput_tps"] or 0.0))
        k = r["kill"]
        print("    kill: bees=%s a3_set=%s a3_ok=%s first_attempt_set=%s first_attempt_ok=%s" % (
            k["killed_bees"], k["a3_set_size"], k["a3_answered_ok"],
            k["first_attempt_on_killed_size"], k["first_attempt_on_killed_answered_ok"]))
    print("acceptance_at_1000 (N=1000 prereg runs only):",
          {k: v["verdict"] for k, v in report["acceptance_at_1000"].items()})
    print("overall:", report["overall"], " receipt:", report["receipt"]["sha256"])
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

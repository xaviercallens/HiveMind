#!/usr/bin/env python3
"""Read-only report on the chained QLoRA-MoE training: rounds, baseline, per-domain held-out loss, plateau, and when the rounds ran.

Sources: the assimilator's ledger (``assimilated.jsonl``, ``latest_adapter.json``) and the BOINC database (round counts, run times).
Held-out sets are fixed (8 sequences x 128 tokens per domain, never used for training), so ``eval_by_domain_after`` is comparable across rounds.
Only rounds that recorded ALL domains are compared (the earliest rounds were evaluated on ``general`` alone, a different metric).
Standard library only.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import statistics as st
import subprocess
from pathlib import Path


def utc(t: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc)


def in_night(t: float) -> bool:  # 22:00-07:00 Europe/Paris
    from zoneinfo import ZoneInfo

    h = dt.datetime.fromtimestamp(t, ZoneInfo("Europe/Paris")).hour
    return h >= 22 or h < 7


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/stack/project/hmproj/hivemind_ledger"))
    a = ap.parse_args()
    rows = [json.loads(x) for x in (a.ledger / "assimilated.jsonl").read_text().splitlines() if x.strip()]
    tr = [r for r in rows if r.get("app") == "hivemind_qlora_moe" and r["wu_id"].startswith("hmq_")]
    head = json.loads((a.ledger / "latest_adapter.json").read_text())
    ndom = collections.Counter(len(r.get("eval_by_domain_after") or {}) for r in tr)
    full = [r for r in tr if len(r.get("eval_by_domain_after") or {}) >= 6]
    print(f"rounds assimilated (hmq_*): {len(tr)}; domains recorded per round: {dict(ndom)}; rounds with all domains: {len(full)}")
    t0, t1 = tr[0]["t"], tr[-1]["t"]
    print(f"span: {utc(t0):%d %b %H:%M}Z -> {utc(t1):%d %b %H:%M}Z ({(t1 - t0) / 3600:.1f} h); rounds finished inside the 22:00-07:00 Paris window: "
          f"{sum(in_night(r['t']) for r in tr)} / {len(tr)}")
    q = ("use hmproj; select sum(outcome=1 and validate_state=1), sum(validate_state=2), sum(outcome<>1 and server_state=5), "
         "round(avg(case when validate_state=1 then elapsed_time end)) from result where name regexp '^hmq_';")
    out = subprocess.run(["mariadb", "--no-defaults", "-u", "root", "-S", "/run/mysqld/mysqld.sock", "-N", "-B", "-e", q], capture_output=True, text=True).stdout.split()
    if len(out) == 4:
        print(f"BOINC results: valid {out[0]}, invalid {out[1]}, failed/aborted {out[2]}; mean run time of a valid round {out[3]} s")

    first = full[0]
    base = first["eval_by_domain_before"]  # state at the start of the first all-domain round (NOT the untrained model: earlier rounds had trained on 'general')
    head_row = next(r for r in full if r["wu_id"] == head["wu_id"])
    hd, latest = head_row["eval_by_domain_after"], full[-1]["eval_by_domain_after"]
    print(f"\nbaseline = state before {first['wu_id']} (the first all-domain round; {tr.index(first)} earlier round(s) had trained on 'general' only, so this is NOT the pure base model)")
    print(f"head = {head['wu_id']} (the adapter the next round starts from; the 'ratchet' only advances it when a round did not get worse); latest round = {full[-1]['wu_id']}")
    print(f"{'domain':9s} {'baseline':>9s} {'head':>8s} {'gain':>8s} {'gain %':>7s} {'latest':>8s}")
    for d in base:
        print(f"{d:9s} {base[d]:9.4f} {hd[d]:8.4f} {base[d] - hd[d]:8.4f} {100 * (base[d] - hd[d]) / base[d]:6.1f}% {latest[d]:8.4f}")
    mb, mh = st.mean(base.values()), st.mean(hd.values())
    print(f"{'MEAN':9s} {mb:9.4f} {mh:8.4f} {mb - mh:8.4f} {100 * (mb - mh) / mb:6.1f}% {st.mean(latest.values()):8.4f}")

    seq = [st.mean(r["eval_by_domain_after"].values()) for r in full]
    print(f"\nmean (6-domain) held-out loss after each round, blocks of 10 rounds ({len(full)} rounds):")
    for i in range(0, len(seq), 10):
        blk = seq[i : i + 10]
        print(f"  rounds {i + 1:3d}-{i + len(blk):3d}: mean {st.mean(blk):.4f}  min {min(blk):.4f}  max {max(blk):.4f}")
    print(f"std over the last 20 rounds: {st.pstdev(seq[-20:]):.4f}; best single round {min(seq):.4f} ({full[seq.index(min(seq))]['wu_id']}); "
          f"change from the first 10 rounds to the last 10: {st.mean(seq[:10]) - st.mean(seq[-10:]):+.4f}")
    print(f"rounds whose loss was lower than the previous round's: {sum(1 for p, c in zip(seq, seq[1:]) if c < p)} / {len(seq) - 1}")

    def split(rs):
        tgt, oth = [], []
        for r in rs:
            b, af, dom = r["eval_by_domain_before"], r["eval_by_domain_after"], r.get("domain")
            if dom in b:
                tgt.append(b[dom] - af[dom])
                oth += [b[d] - af[d] for d in b if d != dom]
        return (st.mean(tgt) if tgt else float("nan")), (st.mean(oth) if oth else float("nan"))

    print("\nwithin a round (positive = the held-out loss went DOWN): target-domain change vs the average change on the other five domains")
    for name, rs in (("first 30 rounds", full[:30]), ("last 30 rounds", full[-30:])):
        t, o = split(rs)
        print(f"  {name}: target {t:+.4f}   others {o:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

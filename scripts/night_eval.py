#!/usr/bin/env python3
"""Evaluate ONE model on ONE night pack (``eval.jsonl`` from scripts/night_pack_build.py) through a llama-server OpenAI endpoint.

Deterministic setup (see the E17 amendment): temperature 0, fixed seed, cache_prompt false, one request at a time. Scoring by ``kind``:
``numeric`` (last 'Final answer: N'), ``mcq`` (last 'Answer: X'), ``label`` (last 'Verdict: ...' / 'Class: X'). An unparsable reply is WRONG and is
counted in the summary. With ``--baseline JOURNAL`` the summary adds a paired exact McNemar test against that journal (same keys required).
Usage: python scripts/night_eval.py --endpoint http://127.0.0.1:8291 --pack DIR --tag base --out DIR [--baseline OUT/base_journal.jsonl] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import httpx

MAX_TOKENS = {"numeric": 512, "mcq": 12, "label": 24}


def parse(kind: str, text: str) -> str | None:
    if kind == "numeric":
        m = re.findall(r"final answer\s*[:=]\s*\$?\s*(-?[\d,]*\.?\d+)", text, re.I)
        return m[-1].replace(",", "") if m else None
    if kind == "mcq":
        m = re.findall(r"answer\s*[:=]\s*\(?([ABCD])\b", text, re.I)
        return m[-1].upper() if m else None
    m = re.findall(r"(?:verdict|class)\s*[:=]\s*(confirmed|false positive|[OBAFGKM])\b", text, re.I)
    return m[-1].upper() if m else None


def correct(kind: str, got: str | None, gold: str) -> bool:
    if got is None:
        return False
    if kind == "numeric":
        try:
            return abs(float(got) - float(gold)) < 1e-6
        except ValueError:
            return False
    return got.upper() == gold.upper()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p for discordant pairs b (only A right) and c (only B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="base")
    ap.add_argument("--api-key-env", default="", help="name of an environment variable holding a bearer key (e.g. for the HiveMind router)")
    args = ap.parse_args()
    rows = [json.loads(line) for line in (Path(args.pack) / "eval.jsonl").read_text().split("\n") if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    journal = out / f"{args.tag}_journal.jsonl"
    done = {json.loads(line)["key"]: json.loads(line) for line in journal.read_text().splitlines()} if journal.exists() else {}
    headers = {"Authorization": f"Bearer {os.environ[args.api_key_env]}"} if args.api_key_env else {}
    client = httpx.Client(headers=headers)
    t0 = time.time()
    for r in rows:
        if r["key"] in done:
            continue
        body = {"model": args.model, "messages": r["messages"], "temperature": 0, "seed": 7, "max_tokens": MAX_TOKENS[r["kind"]], "cache_prompt": False}
        text = client.post(f"{args.endpoint}/v1/chat/completions", json=body, timeout=900).json()["choices"][0]["message"]["content"] or ""
        got = parse(r["kind"], text)
        rec = {"key": r["key"], "kind": r["kind"], "gold": r["gold"], "got": got, "ok": correct(r["kind"], got, r["gold"]), "unparsable": got is None, "reply_head": text[:120]}
        done[r["key"]] = rec
        with journal.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    recs = [done[r["key"]] for r in rows]
    k = sum(x["ok"] for x in recs)
    summary = {"tag": args.tag, "pack": Path(args.pack).name, "n": len(recs), "correct": k, "accuracy": round(k / max(1, len(recs)), 4), "wilson95": wilson(k, len(recs)),
               "unparsable": sum(x["unparsable"] for x in recs), "wall_s": round(time.time() - t0)}
    if args.baseline:
        base = {json.loads(line)["key"]: json.loads(line) for line in Path(args.baseline).read_text().splitlines()}
        if set(done) - set(base) and any(x["key"] not in base for x in recs):
            raise SystemExit("BLOCKED: baseline journal does not cover the same keys")
        b = sum(1 for x in recs if x["ok"] and not base[x["key"]]["ok"])  # only this model right
        c = sum(1 for x in recs if not x["ok"] and base[x["key"]]["ok"])  # only baseline right
        summary.update({"vs_baseline": {"only_this_right": b, "only_baseline_right": c, "mcnemar_exact_p": round(mcnemar_exact(b, c), 5)}})
    (out / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

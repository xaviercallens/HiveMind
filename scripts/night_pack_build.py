#!/usr/bin/env python3
"""Build the data packs for the night expert-training run (CPU only, no GPU, nothing trained here).

One pack per subject under ``--out/<pack>/``: ``train.jsonl`` ({key, messages, completion}), ``eval.jsonl`` ({key, messages, gold, kind})
and ``manifest.json`` (source, licence text as read, counts, sha256 of every file, leakage checks, baselines). Plan and hypotheses:
``artifacts/experiments/NIGHT_MOE_prereg.json`` and ``docs/NIGHT_MOE_TRAINING_PLAN.md``.

Packs: ``maths`` (GSM8K train, GSM8K test eval), ``physics_transfer`` and ``astro_transfer`` (eval only, MMLU), ``exoplanet`` (NASA Kepler objects of
interest, split by star), ``stars`` (NASA Exoplanet Archive host stars, split by star), ``cyber`` (Trendyol instruction data, filtered; eval on MMLU
computer security and CTI-Bench MCQ, eval only). A pack that cannot be built reports BLOCKED with the reason; nothing is simulated.

Usage: python scripts/night_pack_build.py --out DIR [--only maths,exoplanet] [--cache DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SEED = 20261024
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
HF = "https://huggingface.co/datasets"
SYS_MATH = "You solve school-level maths and physics word problems. Think step by step, then end with one final line: 'Final answer: <number>'."
SYS_MCQ = "You answer multiple-choice questions. Reply with only 'Answer: <letter>'."
SYS_KOI = "You vet transit signals from the Kepler mission. Reply with exactly 'Verdict: CONFIRMED' or 'Verdict: FALSE POSITIVE'."
SYS_STAR = "You classify stars. Reply with exactly 'Class: X' where X is one of O, B, A, F, G, K, M."
SYS_CYBER = "You are a defensive cyber-security assistant. Give accurate, concise, defensive guidance."
DUAL_USE = re.compile(r"exploit|payload|malware|ransomware|reverse shell|keylogger|phishing|shellcode|0-?day|zero-day|crack|jailbreak|bypass|evad|botnet|backdoor|rootkit|c2 |command and control", re.I)

KOI_FEATURES = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq", "koi_insol", "koi_model_snr", "koi_impact", "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag"]
# Deliberately NOT used: koi_fpflag_* (the vetting flags that define a false positive), koi_pdisposition, koi_score, koi_disposition itself.
KOI_LABEL = {"CONFIRMED": "CONFIRMED", "FALSE POSITIVE": "FALSE POSITIVE"}
SPEC_CLASSES = "OBAFGKM"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, cache: Path, *, tries: int = 4, binary: bool = False, byte_range: str | None = None) -> bytes:
    """GET with a cache and polite retries (the NASA front end sometimes answers with a Cloudflare challenge page)."""
    key = hashlib.sha256((url + (byte_range or "")).encode()).hexdigest()[:24]
    path = cache / key
    if path.exists():
        return path.read_bytes()
    last = ""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hivemind-night-pack/1.0", **({"Range": f"bytes={byte_range}"} if byte_range else {})})
            data = urllib.request.urlopen(req, timeout=120).read()
            if not binary and (data[:15].lower().startswith(b"<!doctype html") or data[:6] == b"<!--[i"):
                raise RuntimeError("got an HTML challenge page instead of data")
            cache.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return data
        except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(8 * (attempt + 1))
    raise RuntimeError(f"BLOCKED fetching {url[:120]}: {last}")


def tap_csv(query: str, cache: Path) -> list[dict[str, str]]:
    url = TAP + "?" + urllib.parse.urlencode({"query": query, "format": "csv"})
    return list(csv.DictReader(io.StringIO(fetch(url, cache).decode("utf-8"))))


def hf_parquet_rows(repo: str, filename: str, cache: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    data = fetch(f"{HF}/{repo}/resolve/main/{filename}", cache, binary=True)
    return pq.read_table(io.BytesIO(data)).to_pylist()


def hf_card_license(repo: str, cache: Path) -> str:
    text = fetch(f"{HF}/{repo}/raw/main/README.md", cache).decode("utf-8", "replace")
    m = re.match(r"---\n(.*?)\n---", text, re.S)
    front = m.group(1) if m else ""
    lic = re.findall(r"^license(?:_name)?:\s*(.*)$", front, re.M)
    return lic[0].strip() if lic else "NO LICENCE FIELD IN CARD"


def stable_bucket(text: str, buckets: int = 100) -> int:
    return int(hashlib.sha256(f"{SEED}:{text}".encode()).hexdigest(), 16) % buckets


def jsonl_line(row: dict) -> str:
    """One JSON object per line. U+2028/U+2029 and friends are escaped: str.splitlines() (used by readers) would cut a line there."""
    return json.dumps(row, ensure_ascii=False).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029") + "\n"


def write_pack(out: Path, name: str, train: list[dict], evals: list[dict], manifest: dict) -> dict:
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    keys_tr = {r["key"] for r in train}
    keys_ev = {r["key"] for r in evals}
    prompts_tr = {json.dumps(r["messages"], sort_keys=True) for r in train}
    prompts_ev = {json.dumps(r["messages"], sort_keys=True) for r in evals}
    leak = {"key_overlap": len(keys_tr & keys_ev), "identical_prompt_overlap": len(prompts_tr & prompts_ev)}
    if leak["key_overlap"] or leak["identical_prompt_overlap"]:
        raise RuntimeError(f"BLOCKED: {name} has train/eval leakage {leak}")
    if train:  # negative control: the same prompts with the completions dealt out to other prompts (a model that learns from content must not improve)
        comps = [r["completion"] for r in train]
        random.Random(SEED + 1).shuffle(comps)
        if len(set(comps)) > 1 and all(c == r["completion"] for c, r in zip(comps, train)):
            comps = comps[1:] + comps[:1]
        shuffled = [{**r, "completion": c} for r, c in zip(train, comps)]
        (d / "train_shuffled.jsonl").write_text("".join(jsonl_line(r) for r in shuffled))
        fixed_points = sum(1 for a, b in zip(train, shuffled) if a["completion"] == b["completion"])
    for fname, rows in (("train.jsonl", train), ("eval.jsonl", evals)):
        (d / fname).write_text("".join(jsonl_line(r) for r in rows))
    manifest = {**manifest, "control_shuffled_same_completion_rows": (fixed_points if train else None), "pack": name, "n_train": len(train), "n_eval": len(evals), "leakage": leak,
                "sha256": {f: sha256_file(d / f) for f in ("train.jsonl", "eval.jsonl")}, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ---------------------------------------------------------------- maths (+ physics/astro transfer eval)
def build_maths(out: Path, cache: Path, n_train: int = 1000, n_eval: int = 200) -> dict:
    rng = random.Random(SEED)
    tr = hf_parquet_rows("openai/gsm8k", "main/train-00000-of-00001.parquet", cache)
    te = hf_parquet_rows("openai/gsm8k", "main/test-00000-of-00001.parquet", cache)
    if {r["question"] for r in tr} & {r["question"] for r in te}:
        raise RuntimeError("BLOCKED: GSM8K train and test share a question")
    rng.shuffle(tr)
    rng.shuffle(te)

    def final(ans: str) -> str:
        return ans.split("####")[-1].strip().replace(",", "")

    def body(ans: str) -> str:
        return re.sub(r"<<[^>]*>>", "", ans.split("####")[0]).strip()

    train = [{"key": f"gsm8k-train-{i}", "messages": [{"role": "system", "content": SYS_MATH}, {"role": "user", "content": r["question"]}],
              "completion": f"{body(r['answer'])}\nFinal answer: {final(r['answer'])}"} for i, r in enumerate(tr[:n_train])]
    evals = [{"key": f"gsm8k-test-{i}", "kind": "numeric", "gold": final(r["answer"]),
              "messages": [{"role": "system", "content": SYS_MATH}, {"role": "user", "content": r["question"]}]} for i, r in enumerate(te[:n_eval])]
    return write_pack(out, "maths", train, evals, {"source": "openai/gsm8k", "licence_as_read": hf_card_license("openai/gsm8k", cache),
                                                    "use": "SFT on gold step-by-step solutions; numeric final-answer match", "subset_seed": SEED})


def build_mmlu_eval(out: Path, cache: Path, pack: str, subjects: list[str], sys_prompt: str = SYS_MCQ) -> dict:
    evals = []
    for subj in subjects:
        for i, r in enumerate(hf_parquet_rows("cais/mmlu", f"{subj}/test-00000-of-00001.parquet", cache)):
            opts = "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(r["choices"]))
            evals.append({"key": f"mmlu-{subj}-{i}", "kind": "mcq", "gold": "ABCD"[int(r["answer"])],
                          "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"{r['question']}\n{opts}"}]})
    return write_pack(out, pack, [], evals, {"source": "cais/mmlu " + ",".join(subjects), "licence_as_read": hf_card_license("cais/mmlu", cache),
                                              "use": "evaluation only (transfer test), no training"})


# ---------------------------------------------------------------- exoplanet (Kepler objects of interest)
def numpy_logreg_baseline(xs_tr, ys_tr, xs_ev, ys_ev) -> dict:
    """Classical reference: standardised log-features + logistic regression by gradient descent (numpy only)."""
    import numpy as np

    def prep(rows, mu=None, sd=None):
        a = np.array(rows, dtype=float)
        a = np.sign(a) * np.log1p(np.abs(a))
        if mu is None:
            mu, sd = a.mean(0), a.std(0) + 1e-9
        return (a - mu) / sd, mu, sd

    xtr, mu, sd = prep(xs_tr)
    xev, _, _ = prep(xs_ev, mu, sd)
    ytr = np.array(ys_tr, dtype=float)
    w, b = np.zeros(xtr.shape[1]), 0.0
    for _ in range(2000):
        p = 1 / (1 + np.exp(-(xtr @ w + b)))
        w -= 0.1 * (xtr.T @ (p - ytr) / len(ytr) + 1e-3 * w)
        b -= 0.1 * float((p - ytr).mean())
    pred = (1 / (1 + np.exp(-(xev @ w + b))) > 0.5).astype(int)
    y = np.array(ys_ev)
    pos, neg = y == 1, y == 0
    return {"accuracy": float((pred == y).mean()), "balanced_accuracy": float(((pred[pos] == 1).mean() + (pred[neg] == 0).mean()) / 2), "n_eval": int(len(y))}


def _fmt(v: str, nd: int = 3) -> str:
    """Plain decimal text (no scientific notation below 1e7): 6360, 2.48, 0.0123."""
    try:
        f = float(v)
    except ValueError:
        return "unknown"
    if f == 0:
        return "0"
    if abs(f) >= 1e7:
        return f"{f:.3e}"
    if abs(f) >= 100:
        return f"{f:.0f}"
    return f"{f:.{nd}g}" if abs(f) >= 1e-3 else f"{f:.2e}"


def build_exoplanet(out: Path, cache: Path, n_train: int = 1600, n_eval: int = 600) -> dict:
    cols = ["kepid", "kepoi_name", "koi_disposition"] + KOI_FEATURES
    rows = tap_csv(f"select {','.join(cols)} from cumulative where koi_disposition in ('CONFIRMED','FALSE POSITIVE')", cache)
    rows = [r for r in rows if all(r[c] not in ("", None) for c in KOI_FEATURES)]
    rng = random.Random(SEED)
    split = {r["kepoi_name"]: ("eval" if stable_bucket(r["kepid"]) < 25 else "train") for r in rows}  # whole stars go to one side
    tr_rows = [r for r in rows if split[r["kepoi_name"]] == "train"]
    ev_rows = [r for r in rows if split[r["kepoi_name"]] == "eval"]
    if {r["kepid"] for r in tr_rows} & {r["kepid"] for r in ev_rows}:
        raise RuntimeError("BLOCKED: a star is on both sides of the exoplanet split")
    rng.shuffle(tr_rows)
    rng.shuffle(ev_rows)
    base = numpy_logreg_baseline([[float(r[c]) for c in KOI_FEATURES] for r in tr_rows], [int(r["koi_disposition"] == "CONFIRMED") for r in tr_rows],
                                 [[float(r[c]) for c in KOI_FEATURES] for r in ev_rows[:n_eval]], [int(r["koi_disposition"] == "CONFIRMED") for r in ev_rows[:n_eval]])
    names = {"koi_period": "orbital period (days)", "koi_duration": "transit duration (hours)", "koi_depth": "transit depth (ppm)", "koi_prad": "planet radius (Earth radii)",
             "koi_teq": "equilibrium temperature (K)", "koi_insol": "insolation (Earth flux)", "koi_model_snr": "transit signal-to-noise", "koi_impact": "impact parameter",
             "koi_steff": "star Teff (K)", "koi_slogg": "star log g", "koi_srad": "star radius (solar)", "koi_kepmag": "Kepler magnitude"}

    def prompt(r: dict) -> list[dict]:
        text = "Kepler transit signal:\n" + "\n".join(f"- {names[c]}: {_fmt(r[c])}" for c in KOI_FEATURES) + "\nIs this a real planet?"
        return [{"role": "system", "content": SYS_KOI}, {"role": "user", "content": text}]

    # train on a balanced draw so the model cannot win by always saying the majority class
    pos = [r for r in tr_rows if r["koi_disposition"] == "CONFIRMED"][: n_train // 2]
    neg = [r for r in tr_rows if r["koi_disposition"] == "FALSE POSITIVE"][: n_train // 2]
    train = [{"key": r["kepoi_name"], "messages": prompt(r), "completion": f"Verdict: {KOI_LABEL[r['koi_disposition']]}"} for r in pos + neg]
    rng.shuffle(train)
    evals = [{"key": r["kepoi_name"], "kind": "label", "gold": KOI_LABEL[r["koi_disposition"]], "messages": prompt(r)} for r in ev_rows[:n_eval]]
    return write_pack(out, "exoplanet", train, evals, {
        "source": "NASA Exoplanet Archive TAP table 'cumulative' (Kepler objects of interest)", "licence_as_read": "NASA Exoplanet Archive data are public; acknowledgement requested (read the archive's acknowledgement page before publishing)",
        "excluded_columns_to_avoid_label_leakage": ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec", "koi_pdisposition", "koi_score", "koi_disposition"],
        "split": "whole stars (kepid) hashed to one side; CANDIDATE rows excluded", "eval_class_balance": {k: sum(1 for r in ev_rows[:n_eval] if r["koi_disposition"] == k) for k in KOI_LABEL},
        "classical_baseline_numpy_logreg_on_same_features": base})


# ---------------------------------------------------------------- stars (host-star spectral class)
def build_stars(out: Path, cache: Path) -> dict:
    cols = ["hostname", "st_spectype", "st_teff", "st_logg", "st_rad", "st_mass", "st_met", "sy_bmag", "sy_vmag"]
    rows = tap_csv(f"select {','.join(cols)} from stellarhosts where st_spectype is not null and st_teff is not null and st_logg is not null", cache)
    feats = ["st_teff", "st_logg", "st_rad", "st_mass", "st_met", "sy_bmag", "sy_vmag"]
    seen: dict[str, dict] = {}
    for r in rows:
        cls = r["st_spectype"].strip()[:1].upper()
        if cls in SPEC_CLASSES and r["hostname"] not in seen:
            seen[r["hostname"]] = {**r, "cls": cls}
    items = list(seen.values())
    tr_rows = [r for r in items if stable_bucket(r["hostname"]) >= 25]
    ev_rows = [r for r in items if stable_bucket(r["hostname"]) < 25]
    names = {"st_teff": "effective temperature (K)", "st_logg": "log g", "st_rad": "radius (solar)", "st_mass": "mass (solar)", "st_met": "metallicity [Fe/H]", "sy_bmag": "B magnitude", "sy_vmag": "V magnitude"}

    def prompt(r: dict) -> list[dict]:
        lines = [f"- {names[c]}: {_fmt(r[c])}" for c in feats if r[c] not in ("", None)]
        return [{"role": "system", "content": SYS_STAR}, {"role": "user", "content": "Star:\n" + "\n".join(lines) + "\nSpectral class?"}]

    def rule(teff: float) -> str:  # the textbook temperature thresholds: a reference, not a competitor to beat by construction
        for lim, c in ((30000, "O"), (10000, "B"), (7500, "A"), (6000, "F"), (5200, "G"), (3700, "K")):
            if teff >= lim:
                return c
        return "M"

    rule_acc = sum(rule(float(r["st_teff"])) == r["cls"] for r in ev_rows) / max(1, len(ev_rows))
    majority = max(SPEC_CLASSES, key=lambda c: sum(r["cls"] == c for r in tr_rows))
    maj_acc = sum(r["cls"] == majority for r in ev_rows) / max(1, len(ev_rows))
    train = [{"key": "star-" + r["hostname"], "messages": prompt(r), "completion": f"Class: {r['cls']}"} for r in tr_rows]
    evals = [{"key": "star-" + r["hostname"], "kind": "label", "gold": r["cls"], "messages": prompt(r)} for r in ev_rows]
    return write_pack(out, "stars", train, evals, {
        "source": "NASA Exoplanet Archive TAP table 'stellarhosts' (observed spectral type from the literature)", "licence_as_read": "NASA Exoplanet Archive data are public; acknowledgement requested",
        "caveat": "planet hosts are a biased sample (few O, B, A stars); Teff is a feature, so a temperature rule is a strong reference",
        "class_counts_train": {c: sum(r["cls"] == c for r in tr_rows) for c in SPEC_CLASSES}, "class_counts_eval": {c: sum(r["cls"] == c for r in ev_rows) for c in SPEC_CLASSES},
        "reference_teff_rule_accuracy_on_eval": rule_acc, "reference_majority_class": majority, "reference_majority_accuracy_on_eval": maj_acc})


# ---------------------------------------------------------------- cyber
def build_cyber(out: Path, cache: Path, n_train: int = 1200, n_cti: int = 400) -> dict:
    raw = fetch(f"{HF}/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset/resolve/main/CyberSec-Dataset_escaped.jsonl", cache, binary=True, byte_range="0-60000000")
    lines = raw.decode("utf-8", "replace").split("\n")[:-1]  # the last line is cut by the byte range
    rng = random.Random(SEED)
    cands, dropped = [], 0
    for ln in lines:
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        u, a = r.get("user", ""), r.get("assistant", "")
        if not u or not a or len(a) > 2600 or len(u) > 500:
            continue
        if DUAL_USE.search(u) or DUAL_USE.search(a):
            dropped += 1
            continue
        cands.append((u, a))
    rng.shuffle(cands)
    train = [{"key": f"cyber-{i}", "messages": [{"role": "system", "content": SYS_CYBER}, {"role": "user", "content": u}], "completion": a} for i, (u, a) in enumerate(cands[:n_train])]
    evals = []
    for i, r in enumerate(hf_parquet_rows("cais/mmlu", "computer_security/test-00000-of-00001.parquet", cache)):
        opts = "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(r["choices"]))
        evals.append({"key": f"mmlu-computer_security-{i}", "kind": "mcq", "gold": "ABCD"[int(r["answer"])], "messages": [{"role": "system", "content": SYS_MCQ}, {"role": "user", "content": f"{r['question']}\n{opts}"}]})
    cti = list(csv.DictReader(io.StringIO(fetch(f"{HF}/AI4Sec/cti-bench/resolve/main/cti-mcq.tsv", cache).decode("utf-8")), delimiter="\t"))
    rng.shuffle(cti)
    for i, r in enumerate(cti[:n_cti]):
        opts = "\n".join(f"{L}. {r['Option ' + L]}" for L in "ABCD")
        evals.append({"key": f"cti-mcq-{i}", "kind": "mcq", "gold": r["GT"].strip(), "messages": [{"role": "system", "content": SYS_MCQ}, {"role": "user", "content": f"{r['Question']}\n{opts}"}]})
    return write_pack(out, "cyber", train, evals, {
        "source": "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset (train, first 60 MB, LLM-generated, no verifier) + cais/mmlu computer_security + AI4Sec/cti-bench cti-mcq (eval only)",
        "licence_as_read": {"Trendyol": hf_card_license("Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset", cache), "mmlu": hf_card_license("cais/mmlu", cache), "cti-bench": hf_card_license("AI4Sec/cti-bench", cache) + " (non-commercial, share-alike: evaluation only, nothing redistributed; owner to confirm)"},
        "dual_use_filter": {"pattern": DUAL_USE.pattern, "rows_dropped": dropped}, "eval_mix": {"mmlu_computer_security": sum(1 for e in evals if e["key"].startswith("mmlu")), "cti_mcq": sum(1 for e in evals if e["key"].startswith("cti"))}})


BUILDERS = {"maths": build_maths, "exoplanet": build_exoplanet, "stars": build_stars, "cyber": build_cyber,
            "physics_transfer": lambda o, c: build_mmlu_eval(o, c, "physics_transfer", ["high_school_physics", "college_physics", "conceptual_physics"]),
            "astro_transfer": lambda o, c: build_mmlu_eval(o, c, "astro_transfer", ["astronomy"])}

SYS_CODE = "You are a careful Python programming assistant. Reply with executable Python code that passes all unit tests."


def _parse_verifiable_reply(kind: str, text: str) -> str | None:
    if kind == "numeric":
        m = re.findall(r"final answer\s*[:=]\s*\$?\s*(-?[\d,]*\.?\d+)", text, re.I)
        return m[-1].replace(",", "") if m else None
    if kind == "mcq":
        m = re.findall(r"answer\s*[:=]\s*\(?([ABCD])\b", text, re.I)
        return m[-1].upper() if m else None
    m = re.findall(r"(?:verdict|class)\s*[:=]\s*(confirmed|false positive|[OBAFGKM])\b", text, re.I)
    return m[-1].upper() if m else None


def _is_verified_reply(kind: str, text: str, gold: str) -> bool:
    got = _parse_verifiable_reply(kind, text)
    if got is None:
        return False
    if kind == "numeric":
        try:
            return abs(float(got) - float(gold)) < 1e-6
        except ValueError:
            return False
    return got.upper() == gold.upper()


def filter_verified_self_solutions(
    candidates: list[dict[str, Any]],
    *,
    kind: str = "numeric",
    cross_domain_completions: list[str] | None = None,
) -> dict[str, Any]:
    """Rejection-sample model self-solutions verified against ground-truth answers (E15 recipe, HM-S14-02 / #166).

    Gold-text fine-tuning on terse GSM8K answers hurt Qwen2.5-Coder-7B (0.785 -> 0.715, 130 vs 157 on the 173 items
    both parsed). This function filters `k` model-generated candidate completions per training prompt, retains only
    samples whose parsed final answer matches `gold`, selects the most concise verified self-solution that preserves
    native step-by-step reasoning, and constructs a cross-domain control rather than a 0-scoring shuffled numeric control.
    """
    verified_train: list[dict[str, Any]] = []
    rejected_prompts: list[str] = []
    total_samples = 0
    verified_samples = 0
    for row in candidates:
        key = str(row["key"])
        gold = str(row["gold"])
        messages = list(row["messages"])
        samples = [str(s) for s in row.get("samples", [])]
        total_samples += len(samples)
        passing = [s.strip() for s in samples if _is_verified_reply(kind, s, gold)]
        verified_samples += len(passing)
        if not passing:
            rejected_prompts.append(key)
            continue
        chosen = min(passing, key=lambda s: (len(s), s))
        verified_train.append(
            {
                "key": key,
                "messages": messages,
                "completion": chosen,
                "gold": gold,
                "verified_candidates_count": len(passing),
                "total_candidates_count": len(samples),
            }
        )
    cross_control: list[dict[str, Any]] = []
    if cross_domain_completions and verified_train:
        for i, row in enumerate(verified_train):
            cross_control.append(
                {
                    "key": row["key"],
                    "messages": row["messages"],
                    "completion": str(cross_domain_completions[i % len(cross_domain_completions)]),
                }
            )
    return {
        "kind": kind,
        "prompts_total": len(candidates),
        "prompts_accepted": len(verified_train),
        "prompts_rejected": len(rejected_prompts),
        "rejected_keys": rejected_prompts,
        "samples_total": total_samples,
        "samples_verified": verified_samples,
        "acceptance_rate": round(len(verified_train) / max(1, len(candidates)), 4),
        "train_rows": verified_train,
        "cross_domain_control_rows": cross_control,
    }


def execute_code_candidate_tests(
    code: str,
    tests: list[str],
    *,
    setup: str = "",
    timeout_s: float = 5.0,
) -> tuple[bool, str]:
    """Verify a Python code completion against executable unit tests in an isolated subprocess (HM-S14-07 / #169)."""
    import ast
    import subprocess

    stripped = code.strip()
    if stripped.startswith("```"):
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", stripped, re.S)
        if blocks:
            stripped = blocks[-1].strip()
    if not stripped or not tests:
        return False, "empty_code_or_tests"
    try:
        tree = ast.parse(stripped)
    except SyntaxError as exc:
        return False, f"syntax_error:{exc.msg}"
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in tree.body):
        return False, "no_callable_definition"
    script_parts = [setup.strip(), stripped, *tests, "print('HIVEMIND_CODE_TESTS_OK')"]
    harness = "\n\n".join(p for p in script_parts if p)
    try:
        res = subprocess.run(
            [sys.executable, "-I", "-B", "-c", harness],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if res.returncode == 0 and "HIVEMIND_CODE_TESTS_OK" in res.stdout:
        return True, "passed"
    err = (res.stderr or res.stdout).strip().splitlines()
    return False, (err[-1][:160] if err else f"exit_{res.returncode}")


def filter_verified_cyber_mcq_targets(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter cyber training rows strictly to verifier-backed MCQ targets and drop unverified free-text or dual-use rows (HM-S14-07 / #169)."""
    verified: list[dict[str, Any]] = []
    for row in candidates:
        u = str(row.get("question", "")).strip()
        completion = str(row.get("completion", "")).strip()
        gold = str(row.get("gold", "")).strip().upper()
        if not u or not completion or gold not in {"A", "B", "C", "D"}:
            continue
        if DUAL_USE.search(u) or DUAL_USE.search(completion):
            continue
        if not _is_verified_reply("mcq", completion, gold):
            continue
        verified.append(
            {
                "key": str(row["key"]),
                "messages": [
                    {"role": "system", "content": SYS_MCQ},
                    {"role": "user", "content": u},
                ],
                "completion": completion,
                "gold": gold,
            }
        )
    return verified


def build_verified_code_expert_pack(
    out: Path,
    *,
    train_problems: list[dict[str, Any]],
    eval_problems: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a `code` expert pack whose training and evaluation rows carry executable Python unit tests (HM-S14-07 / #169)."""
    train_rows: list[dict[str, Any]] = []
    for p in train_problems:
        tests = [str(t) for t in p.get("tests", [])]
        setup = str(p.get("setup", ""))
        comp = str(p["completion"])
        ok, reason = execute_code_candidate_tests(comp, tests, setup=setup)
        if not ok:
            raise RuntimeError(f"BLOCKED: code training row {p['key']} failed executable tests ({reason})")
        prompt_text = f"{p['prompt'].strip()}\nYour code must pass:\n{tests[0]}"
        train_rows.append(
            {
                "key": str(p["key"]),
                "messages": [{"role": "system", "content": SYS_CODE}, {"role": "user", "content": prompt_text}],
                "completion": comp,
                "tests": tests,
            }
        )
    eval_rows: list[dict[str, Any]] = []
    for p in eval_problems:
        tests = [str(t) for t in p.get("tests", [])]
        if not tests:
            raise RuntimeError(f"BLOCKED: code eval row {p['key']} has no executable tests")
        prompt_text = f"{p['prompt'].strip()}\nYour code must pass:\n{tests[0]}"
        eval_rows.append(
            {
                "key": str(p["key"]),
                "kind": "code",
                "gold": json.dumps(tests),
                "tests": tests,
                "messages": [{"role": "system", "content": SYS_CODE}, {"role": "user", "content": prompt_text}],
            }
        )
    return write_pack(
        out,
        "code",
        train_rows,
        eval_rows,
        {
            "source": "verified executable Python unit-test problems (HM-S14-07 / #169)",
            "licence_as_read": "Apache-2.0 / MIT verified problem suite",
            "use": "RFT/SFT on 100% test-verified Python solutions; evaluated by isolated test execution",
            "verified_train_rows": len(train_rows),
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="")
    ap.add_argument("--only", default=",".join(BUILDERS))
    args = ap.parse_args()
    out = Path(args.out)
    cache = Path(args.cache) if args.cache else out / "_cache"
    status = {}
    for name in args.only.split(","):
        try:
            m = BUILDERS[name](out, cache)
            status[name] = {"status": "BUILT", "n_train": m["n_train"], "n_eval": m["n_eval"]}
        except Exception as exc:  # report, never hide: a pack that cannot be built is BLOCKED
            status[name] = {"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"[:300]}
        print(name, status[name], flush=True)
    (out / "build_status.json").write_text(json.dumps(status, indent=1))
    return 0 if all(v["status"] == "BUILT" for v in status.values()) else 3


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""E28b (IP-JEPA-02): 10-seed LLM-JEPA statistical power, loss-dropout (alpha=0.50, 0.75) & co-tuning study.

Extends Pilot E28 (arXiv:2509.14252) from 5 seeds to 10 seeds ({82, 23, 37, 84, 4, 11, 19, 29, 41, 53})
with a standard cosine learning-rate schedule (3e-3 -> 3e-4) and tests Section 5.2 of Huang, LeCun &
Balestriero (2025): random batch-level JEPA-loss dropout at rate alpha in {0.50, 0.75} with and without
their co-tuning rule ``lambda * (1 - alpha) = const``.

Pre-registered predictions (recorded before execution):
  * P28b_gain:    ``jepa_full`` (lambda=1, k=1, alpha=0) beats ``ntp`` on paired held-out exact match
                  across 10 seeds (paired one-sided t-test p < 0.05) AND beats ``jepa_shuf`` in 10/10 seeds.
  * P28b_dropout: ``jepa_drop75_cotuned`` (alpha=0.75, lambda=4.0, nominal cost 1.25x) retains >= 95% of
                  ``jepa_full``'s held-out exact match while keeping pure-training wall-clock overhead <= 1.25x NTP.
  * P28b_cotune:  Descriptive paired comparison of ``jepa_drop75_cotuned`` (lambda=4.0) vs.
                  ``jepa_drop75_naive`` (lambda=1.0) at alpha=0.75.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hivemind.swarm.llm_jepa import (  # noqa: E402
    JepaConfig,
    cotuned_lambda,
    llm_jepa_loss,
    pack_two_views,
    relative_epoch_cost,
    view_gap_singular_values,
)

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
OPS = ["copy", "reverse", "sort", "rotate"]
SPECIAL = ["<pad>", "<sep>", "<eos>", "<pred>", "of"]
VOCAB = SPECIAL + OPS + WORDS + [f"d{i}" for i in range(10)]
TOK = {w: i for i, w in enumerate(VOCAB)}
PAD, SEP, EOS, PRED = TOK["<pad>"], TOK["<sep>"], TOK["<eos>"], TOK["<pred>"]

DEFAULT_SEEDS = [82, 23, 37, 84, 4, 11, 19, 29, 41, 53]

ARMS_SPEC: dict[str, dict[str, Any]] = {
    "ntp": {"lam": 0.0, "k_pred": 1, "loss_dropout": 0.0, "shuffle_views": False},
    "jepa_full": {"lam": 1.0, "k_pred": 1, "loss_dropout": 0.0, "shuffle_views": False},
    "jepa_drop50_cotuned": {"lam": cotuned_lambda(1.0, 0.0, 0.50), "k_pred": 1, "loss_dropout": 0.50, "shuffle_views": False},
    "jepa_drop75_cotuned": {"lam": cotuned_lambda(1.0, 0.0, 0.75), "k_pred": 1, "loss_dropout": 0.75, "shuffle_views": False},
    "jepa_drop75_naive": {"lam": 1.0, "k_pred": 1, "loss_dropout": 0.75, "shuffle_views": False},
    "jepa_shuf": {"lam": 1.0, "k_pred": 1, "loss_dropout": 0.0, "shuffle_views": True},
}


def apply_op(op: str, digits: list[int]) -> list[int]:
    if op == "copy":
        return list(digits)
    if op == "reverse":
        return digits[::-1]
    if op == "sort":
        return sorted(digits)
    if op == "rotate":
        return digits[1:] + digits[:1]
    raise ValueError(op)


def make_pairs(n: int, rng: random.Random, seen: set | None = None) -> list[tuple[list[int], list[int]]]:
    out, keys = [], set()
    while len(out) < n:
        op = rng.choice(OPS)
        digits = [rng.randrange(10) for _ in range(rng.randint(3, 6))]
        key = (op, tuple(digits))
        if key in keys or (seen is not None and key in seen):
            continue
        keys.add(key)
        text = [TOK[op], TOK["of"]] + [TOK[WORDS[d]] for d in digits]
        code = [TOK[f"d{d}"] for d in apply_op(op, digits)]
        out.append((text, code))
    return out


def build_model(seed: int, device: str):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=len(VOCAB),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=PAD,
        bos_token_id=SEP,
        eos_token_id=EOS,
        tie_word_embeddings=True,
    )
    cfg._attn_implementation = "eager"
    return LlamaForCausalLM(cfg).to(device)


def lm_batch(pairs, device: str):
    rows = [t + [SEP] + c + [EOS] for t, c in pairs]
    seq_len = max(len(r) for r in rows)
    ids = torch.full((len(rows), seq_len), PAD, dtype=torch.long)
    labels = torch.full((len(rows), seq_len), -100, dtype=torch.long)
    att = torch.zeros((len(rows), seq_len), dtype=torch.long)
    for i, ((t, _), r) in enumerate(zip(pairs, rows)):
        ids[i, : len(r)] = torch.tensor(r)
        att[i, : len(r)] = 1
        start = len(t) + 1
        labels[i, start : len(r)] = torch.tensor(r[start:])
    return ids.to(device), att.to(device), labels.to(device)


@torch.no_grad()
def batched_exact_match(model, pairs, device: str, max_new: int = 8) -> float:
    """Left-padded batched greedy decoding (exact equivalent to single-example greedy under attention_mask)."""
    model.train(False)
    by_len: dict[int, list[tuple[list[int], list[int]]]] = {}
    for t, c in pairs:
        by_len.setdefault(len(t), []).append((t, c))
    ok = 0
    for _, group in by_len.items():
        seqs = [t + [SEP] for t, _ in group]
        cur = torch.tensor(seqs, dtype=torch.long, device=device)
        finished = [False] * len(group)
        gen: list[list[int]] = [[] for _ in group]
        for _ in range(max_new):
            logits = model(input_ids=cur).logits[:, -1, :]
            nxt = logits.argmax(dim=-1)
            for i, tok_id in enumerate(nxt.tolist()):
                if not finished[i]:
                    if tok_id == EOS:
                        finished[i] = True
                    else:
                        gen[i].append(tok_id)
            if all(finished):
                break
            cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
        for (_, c), g in zip(group, gen):
            ok += int(g == c)
    model.train(True)
    return ok / len(pairs)


@torch.no_grad()
def held_out_nll(model, pairs, device: str) -> float:
    model.train(False)
    ids, att, labels = lm_batch(pairs, device)
    loss = float(model(input_ids=ids, attention_mask=att, labels=labels).loss.float().cpu())
    model.train(True)
    return loss


@torch.no_grad()
def view_embeddings(model, pairs, k_pred: int, device: str):
    model.train(False)
    ids, mask, tl, cl = pack_two_views([t for t, _ in pairs], [c for _, c in pairs], PRED, k_pred, PAD)
    h = model(input_ids=ids.to(device), attention_mask=mask.to(device), output_hidden_states=True).hidden_states[-1]
    r = torch.arange(h.shape[0], device=device)
    model.train(True)
    return h[r, tl.to(device)].cpu(), h[r, cl.to(device)].cpu()


def one_sided_t_pvalue(t_stat: float, df: int) -> float:
    """Exact one-sided upper-tail p-value P(T_df >= t_stat) via incomplete beta / Simpson quadrature."""
    if not math.isfinite(t_stat):
        return 0.0 if t_stat > 0 else 1.0
    # Student-t PDF integrated on [-|t|, |t|] with 2000-step Simpson rule (accurate to < 1e-10 for df >= 2)
    a = abs(t_stat)
    n_steps = 2000
    h = a / n_steps
    coeff = math.gamma((df + 1) / 2.0) / (math.sqrt(df * math.pi) * math.gamma(df / 2.0))

    def pdf(x: float) -> float:
        return coeff * ((1.0 + (x * x) / df) ** (-(df + 1) / 2.0))

    integral = pdf(0.0) + pdf(a)
    for i in range(1, n_steps):
        integral += (4.0 if i % 2 == 1 else 2.0) * pdf(i * h)
    cdf_abs = 0.5 + (h / 3.0) * integral
    cdf = cdf_abs if t_stat >= 0 else 1.0 - cdf_abs
    return max(0.0, min(1.0, 1.0 - cdf))


def paired_stats(a: list[float], b: list[float]) -> dict[str, Any]:
    d = [x - y for x, y in zip(a, b)]
    m = statistics.mean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    t = m / (sd / math.sqrt(len(d))) if sd > 0 else (math.inf if m > 0 else 0.0)
    p_one = one_sided_t_pvalue(t, max(1, len(d) - 1))
    return {
        "mean_diff": round(m, 4),
        "sd_diff": round(sd, 4),
        "n": len(d),
        "wins": sum(x > 0 for x in d),
        "ties": sum(x == 0 for x in d),
        "losses": sum(x < 0 for x in d),
        "t_stat": round(t, 4) if math.isfinite(t) else str(t),
        "p_one_sided": round(p_one, 6),
        "all_positive": all(x > 0 for x in d),
    }


def run_arm(arm: str, seed: int, train, test, args, device: str) -> dict[str, Any]:
    spec = ARMS_SPEC[arm]
    model = build_model(seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_batches = args.epochs * math.ceil(len(train) / args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_batches, eta_min=args.lr * 0.1)
    cfg = JepaConfig(lam=spec["lam"], k_pred=spec["k_pred"], loss_dropout=spec["loss_dropout"])
    order_rng, drop_rng = random.Random(seed), random.Random(seed + 1)
    steps, jepa_steps = 0, 0
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.epochs):
        idx = list(range(len(train)))
        order_rng.shuffle(idx)
        for b in range(0, len(idx), args.batch):
            pairs = [train[i] for i in idx[b : b + args.batch]]
            ids, att, labels = lm_batch(pairs, device)
            lm = model(input_ids=ids, attention_mask=att, labels=labels).loss
            codes = [c for _, c in pairs]
            if spec["shuffle_views"] and len(codes) > 1:
                codes = codes[1:] + codes[:1]
            loss, log = llm_jepa_loss(lm, model, [t for t, _ in pairs], codes, PRED, cfg, rng=drop_rng, pad_id=PAD)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            steps += 1
            jepa_steps += int(log["jepa_active"])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    test_em = batched_exact_match(model, test, device)
    train_em = batched_exact_match(model, train[:200], device)
    test_nll = held_out_nll(model, test, device)
    te, ce = view_embeddings(model, test[:128], spec["k_pred"], device)
    svs = [round(x, 4) for x in view_gap_singular_values(te, ce, top=5)]
    return {
        "arm": arm,
        "seed": seed,
        "lam": spec["lam"],
        "loss_dropout": spec["loss_dropout"],
        "nominal_cost_mult": relative_epoch_cost(spec["loss_dropout"]) if spec["lam"] > 0 else 1.0,
        "steps": steps,
        "jepa_active_steps": jepa_steps,
        "train_seconds": round(train_s, 3),
        "train_exact_match": round(train_em, 4),
        "test_exact_match": round(test_em, 4),
        "test_nll": round(test_nll, 4),
        "view_gap_top5_singular_values": svs,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--arms", nargs="+", default=list(ARMS_SPEC.keys()))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/experiments/E28b_llm_jepa_dropout_cotuning.json")
    a = ap.parse_args(argv)

    data_rng = random.Random(2509_14252)
    train = make_pairs(a.n_train, data_rng)
    seen = {(VOCAB[t[0]], tuple(WORDS.index(VOCAB[x]) for x in t[2:])) for t, _ in train}
    test = make_pairs(a.n_test, data_rng, seen=seen)

    if a.device.startswith("cuda"):
        w_m = build_model(0, a.device)
        w_ids, w_att, w_lbl = lm_batch(train[:4], a.device)
        w_m(input_ids=w_ids, attention_mask=w_att, labels=w_lbl).loss.backward()
        torch.cuda.synchronize()
        del w_m

    runs = []
    for seed in a.seeds:
        for arm in a.arms:
            r = run_arm(arm, seed, train, test, a, a.device)
            runs.append(r)
        row = {r["arm"]: r["test_exact_match"] for r in runs if r["seed"] == seed}
        print(f"[seed {seed:2d}] " + " | ".join(f"{k}={100*v:5.2f}%" for k, v in row.items()), flush=True)

    by_em = {arm: [r["test_exact_match"] for r in runs if r["arm"] == arm] for arm in a.arms}
    by_nll = {arm: [r["test_nll"] for r in runs if r["arm"] == arm] for arm in a.arms}
    by_sec = {arm: [r["train_seconds"] for r in runs if r["arm"] == arm] for arm in a.arms}
    by_sv1 = {arm: [r["view_gap_top5_singular_values"][0] for r in runs if r["arm"] == arm] for arm in a.arms}

    ntp_mean_sec = statistics.mean(by_sec["ntp"]) if "ntp" in by_sec else 1.0
    summary = {}
    for arm in a.arms:
        summary[arm] = {
            "exact_match_mean": round(statistics.mean(by_em[arm]), 4),
            "exact_match_sd": round(statistics.stdev(by_em[arm]), 4) if len(by_em[arm]) > 1 else 0.0,
            "test_nll_mean": round(statistics.mean(by_nll[arm]), 4),
            "view_gap_sv1_mean": round(statistics.mean(by_sv1[arm]), 4),
            "train_seconds_mean": round(statistics.mean(by_sec[arm]), 3),
            "measured_time_ratio_vs_ntp": round(statistics.mean(by_sec[arm]) / ntp_mean_sec, 3),
            "per_seed_exact_match": by_em[arm],
        }

    comparisons = {}
    pairs_to_test = [
        ("jepa_full_minus_ntp", "jepa_full", "ntp"),
        ("jepa_drop50_cotuned_minus_ntp", "jepa_drop50_cotuned", "ntp"),
        ("jepa_drop75_cotuned_minus_ntp", "jepa_drop75_cotuned", "ntp"),
        ("jepa_drop75_naive_minus_ntp", "jepa_drop75_naive", "ntp"),
        ("jepa_drop75_cotuned_minus_naive", "jepa_drop75_cotuned", "jepa_drop75_naive"),
        ("jepa_full_minus_shuf", "jepa_full", "jepa_shuf"),
    ]
    for name, lhs, rhs in pairs_to_test:
        if lhs in by_em and rhs in by_em:
            comparisons[name] = paired_stats(by_em[lhs], by_em[rhs])

    g = comparisons.get("jepa_full_minus_ntp", {})
    s = comparisons.get("jepa_full_minus_shuf", {})
    p28b_gain_supported = bool(g.get("mean_diff", 0) > 0 and g.get("p_one_sided", 1.0) < 0.05 and s.get("all_positive"))
    d75 = summary.get("jepa_drop75_cotuned", {})
    jf = summary.get("jepa_full", {})
    retention = (d75.get("exact_match_mean", 0.0) / jf["exact_match_mean"]) if jf.get("exact_match_mean") else 0.0
    p28b_dropout_supported = bool(retention >= 0.95 and d75.get("measured_time_ratio_vs_ntp", 99.0) <= 1.25)

    out: dict[str, Any] = {
        "experiment": "E28b",
        "ticket": "IP-JEPA-02",
        "title": "10-seed LLM-JEPA statistical power, random loss-dropout (alpha=0.50, 0.75) & lambda*(1-alpha) co-tuning",
        "reference": "Huang, LeCun, Balestriero, arXiv:2509.14252 (Section 3.1 & Section 5.2)",
        "hypotheses": {
            "P28b_gain": {"supported": p28b_gain_supported, "rule": "jepa_full > ntp paired one-sided p < 0.05 AND jepa_full > jepa_shuf in 10/10 seeds"},
            "P28b_dropout": {"supported": p28b_dropout_supported, "retention_vs_jepa_full": round(retention, 4), "measured_time_ratio_vs_ntp": d75.get("measured_time_ratio_vs_ntp")},
        },
        "hardware": {
            "device": a.device,
            "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "config": {"seeds": a.seeds, "n_train": a.n_train, "n_test": a.n_test, "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "lr_schedule": "cosine_to_0.1x"},
        "summary": summary,
        "comparisons": comparisons,
        "runs": runs,
        "date": time.strftime("%Y-%m-%d"),
    }
    out["receipt_sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"summary": {k: {m: v[m] for m in ("exact_match_mean", "exact_match_sd", "view_gap_sv1_mean", "measured_time_ratio_vs_ntp")} for k, v in summary.items()}, "comparisons": comparisons, "hypotheses": out["hypotheses"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

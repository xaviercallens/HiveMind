#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""E28 pilot: LLM-JEPA (arXiv:2509.14252) vs. plain next-token training on a small two-view task.

The task has two natural views of the same content, as the paper requires:
    Text = "<op> of <digit words>"   e.g. "reverse of seven two nine"
    Code = the resulting digit string e.g. "9 2 7"   (separate code-side tokens)
Ops: copy, reverse, sort, rotate (left by one). A tiny Llama-architecture LM is trained from random init
(the paper's "pretraining" setting, Table 2) on a deliberately small training set so that the baseline does
not saturate, then evaluated by greedy exact match on held-out pairs.

Arms (same seeds, same data order, same steps):
    ntp          L_LLM only
    jepa         L_LLM + lambda * (1 - cos(Pred(Enc(Text)), Enc(Code)))   (k [PRED] tokens, block-causal mask)
    jepa_shuf    same as jepa but the Code view is taken from another example (negative control: if this helps
                 as much as jepa, the gain is generic regularisation, not view prediction)

    python scripts/run_e28_llm_jepa_pilot.py --seeds 82 23 37 --out artifacts/experiments/E28_llm_jepa_pilot.json

Writes the per-seed accuracies, paired differences and the view-gap singular values (paper Fig. 3 diagnostic).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hivemind.swarm.llm_jepa import JepaConfig, llm_jepa_loss, pack_two_views, view_gap_singular_values  # noqa: E402

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
OPS = ["copy", "reverse", "sort", "rotate"]
SPECIAL = ["<pad>", "<sep>", "<eos>", "<pred>", "of"]
VOCAB = SPECIAL + OPS + WORDS + [f"d{i}" for i in range(10)]
TOK = {w: i for i, w in enumerate(VOCAB)}
PAD, SEP, EOS, PRED = TOK["<pad>"], TOK["<sep>"], TOK["<eos>"], TOK["<pred>"]


def apply(op: str, digits: list[int]) -> list[int]:
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
        code = [TOK[f"d{d}"] for d in apply(op, digits)]
        out.append((text, code))
    return out


def build_model(seed: int, device: str):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(VOCAB), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
                      pad_token_id=PAD, bos_token_id=SEP, eos_token_id=EOS, tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    return LlamaForCausalLM(cfg).to(device)


def lm_batch(pairs, device):
    rows = [t + [SEP] + c + [EOS] for t, c in pairs]
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), PAD, dtype=torch.long)
    labels = torch.full((len(rows), L), -100, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    for i, ((t, c), r) in enumerate(zip(pairs, rows)):
        ids[i, :len(r)] = torch.tensor(r)
        att[i, :len(r)] = 1
        start = len(t) + 1  # predict Code tokens + EOS only (Text -> Code fine-tuning loss)
        labels[i, start:len(r)] = torch.tensor(r[start:])
    return ids.to(device), att.to(device), labels.to(device)


@torch.no_grad()
def exact_match(model, pairs, device, max_new: int = 8) -> float:
    model.train(False)
    ok = 0
    for t, c in pairs:
        seq = t + [SEP]
        for _ in range(max_new):
            logits = model(input_ids=torch.tensor([seq], device=device)).logits[0, -1]
            nxt = int(logits.argmax())
            if nxt == EOS:
                break
            seq.append(nxt)
        ok += int(seq[len(t) + 1:] == c)
    model.train()
    return ok / len(pairs)


@torch.no_grad()
def view_embeddings(model, pairs, k_pred, device):
    model.train(False)
    ids, mask, tl, cl = pack_two_views([t for t, _ in pairs], [c for _, c in pairs], PRED, k_pred, PAD)
    h = model(input_ids=ids.to(device), attention_mask=mask.to(device), output_hidden_states=True).hidden_states[-1]
    r = torch.arange(h.shape[0], device=device)
    model.train()
    return h[r, tl.to(device)].cpu(), h[r, cl.to(device)].cpu()


def run_arm(arm: str, seed: int, train, test, args, device) -> dict:
    model = build_model(seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    cfg = JepaConfig(lam=args.lam if arm != "ntp" else 0.0, k_pred=args.k_pred, loss_dropout=args.loss_dropout)
    order_rng, drop_rng, shuf_rng = random.Random(seed), random.Random(seed + 1), random.Random(seed + 2)
    steps, jepa_steps, t0 = 0, 0, time.time()
    curve = []
    for epoch in range(args.epochs):
        idx = list(range(len(train)))
        order_rng.shuffle(idx)
        for b in range(0, len(idx), args.batch):
            pairs = [train[i] for i in idx[b:b + args.batch]]
            ids, att, labels = lm_batch(pairs, device)
            lm = model(input_ids=ids, attention_mask=att, labels=labels).loss
            codes = [c for _, c in pairs]
            if arm == "jepa_shuf":
                codes = codes[1:] + codes[:1] if len(codes) > 1 else codes
                shuf_rng.random()
            loss, log = llm_jepa_loss(lm, model, [t for t, _ in pairs], codes, PRED, cfg, rng=drop_rng, pad_id=PAD)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
            jepa_steps += int(log["jepa_active"])
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            curve.append({"epoch": epoch + 1, "test_exact_match": round(exact_match(model, test, device), 4)})
    te, ce = view_embeddings(model, test[:128], args.k_pred, device)
    return {"arm": arm, "seed": seed, "steps": steps, "jepa_active_steps": jepa_steps, "seconds": round(time.time() - t0, 2),
            "train_exact_match": round(exact_match(model, train[:200], device), 4),
            "test_exact_match": curve[-1]["test_exact_match"], "best_test_exact_match": max(p["test_exact_match"] for p in curve),
            "curve": curve, "view_gap_top5_singular_values": [round(x, 4) for x in view_gap_singular_values(te, ce, top=5)]}


def paired(a: list[float], b: list[float]) -> dict:
    d = [x - y for x, y in zip(a, b)]
    m = statistics.mean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    t = m / (sd / math.sqrt(len(d))) if sd > 0 else (math.inf if m > 0 else 0.0)
    return {"mean_diff": round(m, 4), "sd_diff": round(sd, 4), "n": len(d), "t_stat": round(t, 3) if math.isfinite(t) else str(t),
            "all_positive": all(x > 0 for x in d)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[82, 23, 37])
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--k-pred", type=int, default=1)
    ap.add_argument("--loss-dropout", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--arms", nargs="+", default=["ntp", "jepa", "jepa_shuf"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/experiments/E28_llm_jepa_pilot.json")
    a = ap.parse_args(argv)

    data_rng = random.Random(2509_14252)
    train = make_pairs(a.n_train, data_rng)
    test = make_pairs(a.n_test, data_rng, seen={(VOCAB[t[0]], tuple(WORDS.index(VOCAB[x]) for x in t[2:])) for t, _ in train})
    runs = []
    for seed in a.seeds:
        for arm in a.arms:
            r = run_arm(arm, seed, train, test, a, a.device)
            print(json.dumps({k: r[k] for k in ("arm", "seed", "test_exact_match", "best_test_exact_match", "train_exact_match", "seconds")}), flush=True)
            runs.append(r)
    by = {arm: [r["test_exact_match"] for r in runs if r["arm"] == arm] for arm in a.arms}
    summary = {arm: {"mean": round(statistics.mean(v), 4), "sd": round(statistics.stdev(v), 4) if len(v) > 1 else 0.0, "per_seed": v} for arm, v in by.items()}
    comparisons = {}
    if "jepa" in by and "ntp" in by:
        comparisons["jepa_minus_ntp"] = paired(by["jepa"], by["ntp"])
    if "jepa_shuf" in by and "ntp" in by:
        comparisons["jepa_shuf_minus_ntp"] = paired(by["jepa_shuf"], by["ntp"])
    if "jepa" in by and "jepa_shuf" in by:
        comparisons["jepa_minus_jepa_shuf"] = paired(by["jepa"], by["jepa_shuf"])
    jm = comparisons.get("jepa_minus_ntp", {})
    ctrl = comparisons.get("jepa_minus_jepa_shuf", {})
    supported = bool(jm.get("all_positive")) and bool(ctrl.get("all_positive"))
    out = {
        "experiment": "E28", "title": "LLM-JEPA pilot: two-view embedding objective vs. next-token training",
        "reference": "Huang, LeCun, Balestriero, arXiv:2509.14252 (LLM-JEPA)",
        "pre_registered_prediction": "P28a: jepa > ntp on held-out exact match in every seed AND jepa > jepa_shuf in every seed",
        "p28a_supported": supported,
        "scope_note": "Tiny Llama-architecture model (2 layers, d=64) trained from random init on a synthetic two-view task; "
                      "tests the objective and the block-causal implementation, not a transfer claim to Granite-MoE or QLoRA.",
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "data": {"n_train": len(train), "n_test": len(test), "ops": OPS, "digits_len": [3, 6], "data_seed": 2509_14252},
        "hardware": {"device": a.device, "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
                     "torch": torch.__version__, "python": platform.python_version()},
        "summary_test_exact_match": summary, "comparisons": comparisons, "runs": runs,
        "date": time.strftime("%Y-%m-%d"),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "comparisons": comparisons, "p28a_supported": supported}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

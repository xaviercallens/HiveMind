# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HN1: does the H-Net boundary signal (low cosine between adjacent hidden states) exist in Qwen2.5-0.5B?

Pre-registered in ``artifacts/experiments/HN1_boundary_signal_prereg.json`` (written before any run).

  run:  PYTHONPATH=src python scripts/run_experiment_HN1_boundary_signal.py run --windows 8

The score is s_t = -cos(h_t, h_{t-1}); the label is "this token starts a word" (leading space or newline). AUROC of the
score against the label, pooled per text set, is the main measure. Surprisal from the same forward pass gives the
mechanism check; a randomly initialised model of the same class is the negative control; newline tokens are the
positive control. The helpers below are pure and tested in ``tests/test_hn1_helpers.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
EXP_DIR = PROJECT_ROOT / "artifacts" / "experiments"
PREREG = EXP_DIR / "HN1_boundary_signal_prereg.json"
REPORT = EXP_DIR / "HN1_boundary_signal_report.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B"
WINDOW = 256

TEXT_SETS: dict[str, tuple[str, ...]] = {
    "english": ("docs/RESEARCH_PROGRAM.md",),
    "french": ("docs/CROWDGPT_GPU_LOSS_CHECK.md",),
    "python": ("src/hivemind/queenbee/dance.py", "src/hivemind/boinc/node_federation_plan.py"),
    "lean": ("lean4/HiveMindSpec.lean",),
}


def rank_average(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based) with ties sharing their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Mann-Whitney AUROC: P(score of a positive > score of a negative), ties count one half. 0.5 when one class is empty."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    n_pos = sum(1 for b in labels if b)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rank_average(scores)
    rank_sum_pos = sum(r for r, b in zip(ranks, labels) if b)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (Pearson on average ranks); 0.0 when either side is constant."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need two sequences of equal length >= 2")
    rx, ry = rank_average(x), rank_average(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0.0 or vy == 0.0:
        return 0.0
    return cov / (vx * vy) ** 0.5


def adjacent_cosine(hidden: Any) -> list[float]:
    """cos(h_t, h_{t-1}) for t = 1..T-1 from a (T, d) tensor; zero vectors give 0.0."""
    import torch

    h = hidden.float()
    a, b = h[1:], h[:-1]
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1)
    cos = torch.where(den > 0, num / den.clamp_min(1e-12), torch.zeros_like(num))
    return [float(v) for v in cos]


def word_start_labels(token_strings: Sequence[str]) -> list[bool]:
    """True where a token's text begins a word: a leading space marker (Ġ) or a newline marker (Ċ) or a literal space/newline."""
    return [s.startswith(("Ġ", "Ċ", " ", "\n")) for s in token_strings]


def newline_labels(token_strings: Sequence[str]) -> list[bool]:
    return [s.startswith(("Ċ", "\n")) for s in token_strings]


def load_set_text(name: str) -> str:
    parts: list[str] = []
    for rel in TEXT_SETS[name]:
        p = PROJECT_ROOT / rel
        if p.exists():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def windows_of(ids: list[int], window: int, max_windows: int) -> list[list[int]]:
    out = [ids[i : i + window] for i in range(0, len(ids) - window + 1, window)]
    return out[:max_windows]


def evaluate_model(model: Any, tok: Any, windows: list[list[int]], layer: int) -> dict[str, list[float] | list[bool]]:
    """Scores, labels and surprisals pooled over windows for one model at one hidden layer (plus the last layer)."""
    import torch

    scores_mid: list[float] = []
    scores_last: list[float] = []
    starts: list[bool] = []
    newlines: list[bool] = []
    nll: list[float] = []
    for ids in windows:
        x = torch.tensor([ids])
        with torch.no_grad():
            out = model(input_ids=x, output_hidden_states=True)
        hs = out.hidden_states
        scores_mid.extend(-c for c in adjacent_cosine(hs[layer][0]))
        scores_last.extend(-c for c in adjacent_cosine(hs[-1][0]))
        logp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
        nll.extend(float(-logp[t, ids[t + 1]]) for t in range(len(ids) - 1))
        toks = tok.convert_ids_to_tokens(ids)
        starts.extend(word_start_labels(toks)[1:])
        newlines.extend(newline_labels(toks)[1:])
    return {"scores_mid": scores_mid, "scores_last": scores_last, "word_start": starts, "newline": newlines, "nll": nll}


def summarise(ev: dict[str, Any], threshold: float | None) -> dict[str, Any]:
    s_mid, s_last = ev["scores_mid"], ev["scores_last"]
    res: dict[str, Any] = {
        "n_positions": len(s_mid),
        "word_start_rate": sum(ev["word_start"]) / max(1, len(s_mid)),
        "auroc_word_start_mid": auroc(s_mid, ev["word_start"]),
        "auroc_word_start_last": auroc(s_last, ev["word_start"]),
        "auroc_newline_mid": auroc(s_mid, ev["newline"]),
        "spearman_score_surprisal_mid": spearman(s_mid, ev["nll"]),
        "mean_cos_mid": -sum(s_mid) / max(1, len(s_mid)),
    }
    if threshold is not None:
        res["boundary_density_at_english_median"] = sum(1 for s in s_mid if s > threshold) / max(1, len(s_mid))
    return res


def run(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()
    config = AutoConfig.from_pretrained(MODEL_ID)
    random_model = AutoModelForCausalLM.from_config(config).eval()
    layer = config.num_hidden_layers // 2

    sets: dict[str, Any] = {}
    english_median: float | None = None
    for name in TEXT_SETS:
        text = load_set_text(name)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        wins = windows_of(ids, WINDOW, args.windows)
        if not wins:
            sets[name] = {"status": "NOT_RUN", "reason": f"fewer than {WINDOW} tokens available"}
            continue
        ev = evaluate_model(model, tok, wins, layer)
        if name == "english":
            english_median = sorted(ev["scores_mid"])[len(ev["scores_mid"]) // 2]
        trained = summarise(ev, english_median)
        rnd = summarise(evaluate_model(random_model, tok, wins, layer), None)
        sets[name] = {
            "status": "MEASURED",
            "n_windows": len(wins),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "trained": trained,
            "random_init": {k: rnd[k] for k in ("auroc_word_start_mid", "auroc_newline_mid", "mean_cos_mid")},
        }

    measured = {k: v for k, v in sets.items() if v.get("status") == "MEASURED"}
    h1_hits = [k for k, v in measured.items() if v["trained"]["auroc_word_start_mid"] > 0.60]
    h2_hits = [k for k, v in measured.items() if v["trained"]["spearman_score_surprisal_mid"] > 0.10]
    pos_ok = [k for k, v in measured.items() if v["trained"]["auroc_newline_mid"] >= v["trained"]["auroc_word_start_mid"]]
    neg_ok = [k for k, v in measured.items() if 0.45 <= v["random_init"]["auroc_word_start_mid"] <= 0.55]
    h3 = None
    if "python" in measured and "english" in measured:
        h3 = abs(measured["python"]["trained"]["boundary_density_at_english_median"] - measured["english"]["trained"]["boundary_density_at_english_median"])
    verdict = {
        "H1_supported": len(h1_hits) >= 3,
        "H1_sets_above_0.60": h1_hits,
        "H2_supported": len(h2_hits) == len(measured) and bool(measured),
        "H2_sets_above_0.10": h2_hits,
        "H3_density_gap_python_vs_english": h3,
        "H3_supported": (h3 is not None and h3 > 0.10),
        "positive_control_passed_sets": pos_ok,
        "positive_control_valid": len(pos_ok) == len(measured) and bool(measured),
        "negative_control_passed_sets": neg_ok,
        "negative_control_valid": len(neg_ok) == len(measured) and bool(measured),
    }
    verdict["experiment_valid"] = verdict["positive_control_valid"] and verdict["negative_control_valid"]
    report = {
        "schema_version": "hivemind-experiment-HN1/v1.0",
        "experiment_id": "HN1-boundary-signal",
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "model": MODEL_ID,
        "hidden_layer_mid": layer,
        "window_tokens": WINDOW,
        "device": "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "elapsed_s": round(time.time() - t0, 1),
        "sets": sets,
        "verdict": verdict,
    }
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"elapsed_s": report["elapsed_s"], "verdict": verdict}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--windows", type=int, default=8)
    r.add_argument("--threads", type=int, default=8)
    r.set_defaults(func=run)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

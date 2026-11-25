# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Figures for the Bi-Layer Honeybee section of the foundation paper (sec:bee_agent_mesh_and_diurnal_scaling).

    python paper/scripts_figures/make_bilayer_figures.py [--screen-tsv PATH]

Writes paper/figures/fig_bilayer_{architecture,recruitment,abc_loop}.pdf. Figure 2 is an ILLUSTRATIVE mean-field
simulation of the paper's recruitment equations (beta = 5 is an assumption; the paper does not fix beta). Figure 3b
plots the MEASURED autoresearch screen of 21 Nov 2026 (paper/data/autoresearch_moe_results_2026-11-21.tsv, a copy of
the table committed with PR #948).
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3 (aqua: labels always shown)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

# Constants exactly as stated in the paper (Eqs. waggle_duration, mcl_recruitment, fitness_score; Theorem abc_scout).
TAU0_S, GATE, ALPHA_C, ALPHA_N, THETA_ABANDON, L_LIMIT = 1.20, 0.35, 0.60, 0.40, 0.50, 2


def box(ax, x, y, w, h, text, edge, fill="#ffffff", size=8.0, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", linewidth=1.3,
                                edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, weight=weight, wrap=True)


def arrow(ax, a, b, color=INK2, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, linewidth=1.2, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Layer 1 band
    ax.add_patch(FancyBboxPatch((0.01, 0.56), 0.98, 0.42, boxstyle="round,pad=0.01,rounding_size=0.02",
                                linewidth=1.0, edgecolor=BLUE, facecolor="#eef4fc"))
    ax.text(0.03, 0.945, "Layer 1: network consensus (who works on what)", fontsize=9.5, color=INK, weight="bold")
    ax.text(0.03, 0.905, "biology: scouts choose a nest site by dancing; a quorum settles the choice", fontsize=7.8,
            color=INK2)
    l1 = ["Task arrives\non the mesh", "Scout scores\nfit Q", "Waggle dance\nlasts tau0 x Q",
          "Recruit if\nAffinity x Q\n>= 0.35", "Quorum:\ntask locked\nto a group"]
    w, gap, x0 = 0.150, 0.045, 0.035
    xs = [x0 + i * (w + gap) for i in range(len(l1))]
    for txt, x in zip(l1, xs):
        box(ax, x, 0.63, w, 0.20, txt, BLUE, size=7.8)
    for x1, x2 in zip(xs, xs[1:]):
        arrow(ax, (x1 + w, 0.73), (x2, 0.73))
    # Layer 2 band
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.42, boxstyle="round,pad=0.01,rounding_size=0.02",
                                linewidth=1.0, edgecolor=ORANGE, facecolor="#fdf1ec"))
    ax.text(0.03, 0.405, "Layer 2: local execution (how the task gets solved)", fontsize=9.5, color=INK, weight="bold")
    ax.text(0.03, 0.365, "biology: employed, onlooker and scout bees (Artificial Bee Colony)", fontsize=7.8, color=INK2)
    b2 = [("Employed bees\nexploit the resident\nMoE experts", 0.045),
          ("Observer bees\nf = 0.6 C + 0.4 N\nabandon if f < 0.5\nfor L = 2 cycles", 0.385),
          ("Scout bees\ncall external tools\n(solver, search)", 0.725)]
    for txt, x in b2:
        box(ax, x, 0.10, 0.23, 0.22, txt, ORANGE, size=7.8)
    arrow(ax, (0.275, 0.21), (0.385, 0.21))
    arrow(ax, (0.615, 0.21), (0.725, 0.21))
    ax.text(0.67, 0.245, "source\nexhausted", fontsize=7, color=INK2, ha="center")
    arrow(ax, (0.84, 0.10), (0.16, 0.10), rad=-0.12, ls="--")
    ax.text(0.975, 0.365, "dashed: tool results ('nectar') cached and reused", fontsize=7.2, color=INK2, ha="right",
            va="center")
    # Coupling between the two layers, kept to the right and left edges so the labels never cross boxes
    arrow(ax, (xs[-1] + w / 2, 0.63), (xs[-1] + w / 2, 0.445), color=INK)
    ax.text(xs[-1] + w / 2 - 0.012, 0.50, "quorum hands the task\nto a worker group", fontsize=7.2, color=INK,
            ha="right", va="center")
    arrow(ax, (0.16, 0.32), (xs[1] + w / 2, 0.63), color=AQUA, ls="--")
    ax.text(0.085, 0.49, "outcome updates\nreputation\n(next quorum)", fontsize=7.2, color=INK, ha="center")
    fig.savefig(FIG / "fig_bilayer_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_recruitment(beta: float = 5.0) -> None:
    """Mean-field version of Eqs. waggle_duration, mcl_recruitment and quorum_condition for two candidate tasks."""
    q = {"A": 0.90, "B": 0.60}
    aff = 0.8
    total, theta = 100.0, 30.0  # 100 units of reputation weight; quorum at 30
    rate = 0.06
    committed = {"A": 1.0, "B": 1.0}
    hist = {"A": [], "B": []}
    lock = None
    for t in range(80):
        free = total - sum(committed.values())
        # visibility of a task = committed weight x dance duration (tau0 * Q): good sites are advertised longer
        vis = {k: committed[k] * TAU0_S * q[k] for k in q}
        z = {k: math.exp(beta * aff * q[k]) * vis[k] for k in q}
        s = sum(z.values())
        for k in q:
            if aff * q[k] >= GATE:  # the admission gate of the paper
                committed[k] += rate * free * z[k] / s
            hist[k].append(committed[k])
        if lock is None and committed["A"] >= theta:
            lock = t
    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=200)
    ax.plot(hist["A"], color=BLUE, linewidth=2)
    ax.plot(hist["B"], color=ORANGE, linewidth=2)
    ax.axhline(theta, color=INK2, linewidth=0.9, linestyle=(0, (3, 3)))
    ax.text(78, theta + 1.2, "quorum threshold", fontsize=7.5, color=INK2, ha="right")
    ax.annotate("task A (Q = 0.90)", (len(hist["A"]) - 1, hist["A"][-1]), xytext=(-4, 4), textcoords="offset points",
                ha="right", fontsize=8, color=INK)
    ax.annotate("task B (Q = 0.60)", (len(hist["B"]) - 1, hist["B"][-1]), xytext=(-4, 4), textcoords="offset points",
                ha="right", fontsize=8, color=INK)
    if lock is not None:
        ax.axvline(lock, color=INK2, linewidth=0.8)
        ax.text(lock + 1, 4, f"lock at step {lock}", fontsize=7.5, color=INK2)
    ax.set_xlabel("recruitment step")
    ax.set_ylabel("committed reputation weight")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_bilayer_recruitment.pdf")
    plt.close(fig)


def fig_abc_loop(screen_tsv: pathlib.Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=200, gridspec_kw={"width_ratios": [1, 1.15]})
    # (a) the theorem: stagnating fitness -> counter reaches L -> scout -> fresh nectar
    f = [0.62, 0.66, 0.44, 0.41, 0.72, 0.78]
    c, cs = 0, []
    for v in f:
        c = 0 if v >= THETA_ABANDON else c + 1
        cs.append(c)
    xs = list(range(1, len(f) + 1))
    a.plot(xs, f, color=BLUE, linewidth=2, marker="o", markersize=4)
    a.axhline(THETA_ABANDON, color=INK2, linewidth=0.9, linestyle=(0, (3, 3)))
    a.text(1, THETA_ABANDON + 0.02, "abandon threshold 0.5", fontsize=7, color=INK2)
    scout = next(i for i, v in enumerate(cs) if v >= L_LIMIT) + 1
    a.axvspan(scout - 0.35, scout + 0.35, color=ORANGE, alpha=0.18, linewidth=0)
    a.text(scout, 0.25, f"counter = {L_LIMIT}\n-> scout", ha="center", fontsize=7, color=INK)
    a.text(scout + 0.9, 0.84, "tool result\ninjected", ha="center", fontsize=7, color=INK2)
    a.set_ylim(0.2, 0.92)
    a.set_xlabel("cycle t")
    a.set_ylabel("observer fitness f(x_t)")
    a.set_title("(a) illustration of the theorem", fontsize=8.5, color=INK)
    # (b) measured: the 21 Nov autoresearch screen ran the same loop on training recipes
    rows = [r for r in csv.DictReader(open(screen_tsv), delimiter="\t") if r.get("J_midrow_nll")]
    names, js, kept = [], [], []
    for r in rows:
        if r["trial"] in ("T2", "T2b", "T7"):
            continue
        names.append(r["trial"])
        js.append(float(r["J_midrow_nll"]))
        kept.append(r["status"].startswith(("keep", "champion")))
    for i, (n, j, k) in enumerate(zip(names, js, kept)):
        b.bar(i, j - 4.15, bottom=4.15, color=AQUA if k else ORANGE, width=0.6)
        lab = "baseline" if n == "T1" else ("kept" if k else "discarded")
        b.text(i, j + 0.004, f"{j:.3f}\n{lab}", ha="center", fontsize=6.8, color=INK)
    b.axhline(4.286385, color=INK2, linewidth=0.9, linestyle=(0, (3, 3)))
    b.text(len(names) - 0.6, 4.2895, "base Crowd-v1", fontsize=7, color=INK2, ha="right")
    b.set_xticks(range(len(names)), names)
    b.set_ylim(4.15, 4.33)
    b.set_ylabel("independent mid-row NLL (lower = better)")
    b.set_title("(b) measured: recipe screen, 21 Nov 2026", fontsize=8.5, color=INK)
    for x in (a, b):
        x.grid(axis="y", color=GRID, linewidth=0.6)
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_bilayer_abc_loop.pdf")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-tsv", type=pathlib.Path,
                    default=ROOT / "paper/data/autoresearch_moe_results_2026-11-21.tsv")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    fig_architecture()
    fig_recruitment()
    fig_abc_loop(args.screen_tsv)
    print("wrote", [p.name for p in sorted(FIG.glob("fig_bilayer_*.pdf"))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

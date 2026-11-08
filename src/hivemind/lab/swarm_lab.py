# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HiveMind Swarm Lab ("SwarmLab-1"): a deterministic toy of a DiLoCo-style swarm with quorum policies, stragglers and non-IID data.

WHAT THIS IS NOT: it is an EDUCATIONAL, QUALITATIVE ANALOGY of an effect measured in the real E13 experiment (Qwen2.5-0.5B, 4 workers, one data domain each,
K=2-of-4 quorum, a persistent straggler). It is NOT a re-implementation of E13, its defaults were NOT tuned to reproduce E13's numbers, and its numbers must
never be quoted as results about real LLM training. Pure standard library (no numpy, torch, GPU or network); no import of ``hivemind.swarm``.

THE CONTRACT (every operation order below is normative; a JavaScript port must match ``simulate`` to 1e-9)
-----------------------------------------------------------------------------------------------------------
PRNG. mulberry32 on uint32 (JS: ``>>>0`` and ``Math.imul``)::

    state = (state + 0x6D2B79F5) & 0xFFFFFFFF; t = state
    t = imul(t ^ (t >> 15), t | 1)
    t ^= (t + imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF
    return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296        # uniform u in [0, 1)

Gaussian, one per call, no caching: u1 = next(); u2 = next(); z = sqrt(-2 ln(max(u1, 1e-300))) * cos(2*pi*u2) with ``2*pi`` evaluated first.
Seeds are taken mod 2^32. Streams: model construction = Rng(seed); worker i = Rng(seed + 1 + i), consumed ONLY in rounds where worker i is alive
(per round: 1 uniform for the jitter, then H*d gaussians, step-major, component-minor), independent of K, policy and straggle.

Model. d = ``dim`` (16), M domains. Construction draws, in order: c_shared[k] = gauss for k < d; then for each domain i: a_i[k] = 0.5 + 1.5*uniform (k < d),
then v[k] = gauss (k < d), u_i = v / |v| (|v| = sqrt of the sum of squares in k order; v unchanged if |v| < 1e-300); c_i = c_shared + spread * u_i; finally v[k] = gauss (k < d), w = v / |v|, and theta_0 = c_shared + offset * w.
L_i(theta) = 0.5 * sum_k a_ik (theta_k - c_ik)^2 (sum in k order); global loss = (sum_i L_i, i ascending) / M. With offset = 0 the start is a 'pretrained' point (shared structure learned, only the domain-specific parts remain);
outer buffer 0.

Round r = 1..R. Every alive worker starts from the global theta: for each of H steps, for k in 0..d-1: g = a_ik*(th_k - c_ik) + sigma*gauss(); th_k = th_k - eta*g.
delta_i = theta - th (per component). Arrival t_i = 1 + 0.2*uniform + penalty. Penalty ``penalty`` (2.0) applies in active rounds only: (r-1) % straggle_every == 0
(default 1 = every round; 2 gives E13's rounds 1,3,5,...). 'persistent': worker M-1. 'rotating': the n-th active round (n from 0) penalises worker M-1 - (n mod (M-1)) (never worker 0).
Both need M >= 2. Churn: worker 1 is absent (alive=False, no draws, no delta) in churn_start..churn_end (both 0 = off; needs M >= 2).

Quorum. K_eff = min(K, alive). Sort alive by (t, id); tK = K_eff-th arrival; deadline = tK*(1+grace); close = min(deadline, max alive t).
'fair' (S6, MAX_EXCLUSION=1): workers alive with excl >= 1 and t > close are REQUIRED; close = max(close, their max t). Admitted = alive with t <= close (ascending id),
late = the other alive workers. excl[i] = 0 if admitted, +1 if late, 0 if absent. Round time = close; total_time = sum of round times (the cost of fairness).
Policies for late deltas: 'discard' drops them; 'fair' drops them too (the wait is the remedy); 'deferred' (S7, MAX_STALENESS=2) queues (delta, base round r).

Merge. Items = fresh admitted deltas (ascending id, weight 1.0) followed, under 'deferred', by the queue from earlier rounds (ascending (base round, id)) with
s = r - base_round, weight 1/(1+s), dropped when s > 2 (counted in ``dropped_stale``); then the queue is replaced by this round's late deltas. W = sum w_j (item order),
S_k = sum_j w_j*D_j[k] (item order), mean_k = S_k / W. 'direct': g = mean. 'rda' (repo definition): r = (sum_j w_j*|D_j|) / |S| (1.0 if |S| < 1e-300), g_k = r * mean_k.
Outer Nesterov SGD (lr 0.7, momentum 0.9), vector-wise: buf = mu*buf + g; theta = theta - lr*(g + mu*buf).

Metrics. loss_curve = [L(theta_0), L after rounds 1..R]. Synchronous reference = same scenario with K=M, straggle 'none', policy 'discard', churn off, same seed.
retention = (L0 - L_final) / (L0 - L_sync), None when |L0 - L_sync| <= 1e-12 (mirrors E13's rho). ``contract_digest`` = first 16 hex chars of sha256 of
",".join("%.9f" % v for v in loss_curve) (UTF-8).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any

MASK = 0xFFFFFFFF
MAX_EXCLUSION = 1  # 'fair': a worker excluded this many consecutive rounds is required next round (E13 S6)
MAX_STALENESS = 2  # 'deferred': a carried delta older than this many rounds is dropped (E13 S7)
POLICIES = ("discard", "fair", "deferred")
STRAGGLES = ("none", "persistent", "rotating")
NONMONETARY_NOTICE = (
    "Notice: $HIVE / PTS are internal, non-monetary, non-transferable accounting units; no money or financial payment is or will ever be involved. "
    "(See LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md.)"
)


class Rng:
    """mulberry32 with uniform and Box-Muller gaussian (see the module docstring for the exact algorithm)."""

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & MASK

    def uniform(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & MASK
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & MASK
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & MASK)) & MASK
        return ((t ^ (t >> 14)) & MASK) / 4294967296.0

    def gauss(self) -> float:
        u1 = self.uniform()
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(max(u1, 1e-300))) * math.cos(2.0 * math.pi * u2)


@dataclass
class Scenario:
    """All knobs of one run. Defaults are documented qualitative toy values, not calibrated to any real experiment."""

    M: int = 4
    K: int = 3
    rounds: int = 12
    H: int = 10
    eta: float = 0.1
    sigma: float = 0.2
    dim: int = 16
    spread: float = 2.0
    offset: float = 0.0
    seed: int = 1
    merge: str = "direct"
    policy: str = "discard"
    straggle: str = "none"
    penalty: float = 2.0
    straggle_every: int = 1
    grace: float = 0.1
    churn_start: int = 0
    churn_end: int = 0
    outer_lr: float = 0.7
    outer_momentum: float = 0.9

    def validate(self) -> "Scenario":
        # NaN fails every comparison below, so range checks alone would accept it and the run would report final_loss nan.
        for name in ("eta", "sigma", "spread", "offset", "penalty", "grace", "outer_lr", "outer_momentum"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)!r}")
        if not 1 <= self.M <= 6:
            raise ValueError("M must be in 1..6")
        if not 1 <= self.K <= self.M:
            raise ValueError("K must be in 1..M")
        if self.rounds < 1 or self.H < 1 or self.straggle_every < 1:
            raise ValueError("rounds, H and straggle_every must be >= 1")
        if not 2 <= self.dim <= 64:
            raise ValueError("dim must be in 2..64")
        if self.eta <= 0 or self.sigma < 0 or self.spread < 0 or self.offset < 0 or self.penalty < 0 or self.grace < 0:
            raise ValueError("eta must be > 0 and sigma, spread, penalty, grace must be >= 0")
        if self.outer_lr <= 0 or not 0 <= self.outer_momentum < 1:
            raise ValueError("outer_lr must be > 0 and outer_momentum in [0, 1)")
        if self.merge not in ("direct", "rda"):
            raise ValueError("merge must be 'direct' or 'rda'")
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if self.straggle not in STRAGGLES:
            raise ValueError(f"straggle must be one of {STRAGGLES}")
        if self.straggle != "none" and self.M < 2:
            raise ValueError("a straggler needs M >= 2")
        if (self.churn_start, self.churn_end) != (0, 0):
            if self.M < 2 or not 1 <= self.churn_start <= self.churn_end <= self.rounds:
                raise ValueError("churn needs M >= 2 and 1 <= churn_start <= churn_end <= rounds (both 0 = off)")
        return self


PRESETS: dict[str, dict[str, Any]] = {
    "iid-sync": {"M": 4, "K": 4, "spread": 0.0, "offset": 3.0, "straggle": "none", "policy": "discard"},
    "persistent-straggler": {"M": 4, "K": 3, "straggle": "persistent", "policy": "discard"},
    "rotating-straggler": {"M": 4, "K": 3, "straggle": "rotating", "policy": "discard"},
    "churn": {"M": 4, "K": 3, "straggle": "persistent", "policy": "fair", "churn_start": 3, "churn_end": 6},
}


def _as_scenario(sc: "Scenario | dict[str, Any]") -> Scenario:
    return (sc if isinstance(sc, Scenario) else Scenario(**sc)).validate()


def _norm(v: list[float]) -> float:
    acc = 0.0
    for x in v:
        acc += x * x
    return math.sqrt(acc)


def build_domains(sc: Scenario) -> tuple[list[list[float]], list[list[float]], list[float]]:
    """Return (a, c, c_shared): per-domain curvatures and optima plus the shared structure, drawn from Rng(seed) in the documented order."""
    rng, d = Rng(sc.seed), sc.dim
    shared = [rng.gauss() for _ in range(d)]
    a_all, c_all = [], []
    for _ in range(sc.M):
        a = [0.5 + 1.5 * rng.uniform() for _ in range(d)]
        v = [rng.gauss() for _ in range(d)]
        n = _norm(v)
        u = v if n < 1e-300 else [x / n for x in v]
        a_all.append(a)
        c_all.append([shared[k] + sc.spread * u[k] for k in range(d)])
    v = [rng.gauss() for _ in range(d)]
    n = _norm(v)
    w = v if n < 1e-300 else [x / n for x in v]
    return a_all, c_all, [shared[k] + sc.offset * w[k] for k in range(d)]


def domain_loss(theta: list[float], a: list[float], c: list[float]) -> float:
    acc = 0.0
    for k in range(len(theta)):
        e = theta[k] - c[k]
        acc += a[k] * e * e
    return 0.5 * acc


def losses(theta: list[float], a_all: list[list[float]], c_all: list[list[float]]) -> tuple[float, list[float]]:
    per = [domain_loss(theta, a_all[i], c_all[i]) for i in range(len(a_all))]
    tot = 0.0
    for x in per:
        tot += x
    return tot / len(per), per


def merge_deltas(items: list[tuple[list[float], float]], mode: str) -> list[float]:
    """Weighted merge of (delta, weight) items in the given order; 'rda' multiplies the weighted mean by r = sum w|D| / |sum w D| (>= 1)."""
    d = len(items[0][0])
    wsum, s, num = 0.0, [0.0] * d, 0.0
    for delta, w in items:
        wsum += w
        num += w * _norm(delta)
        for k in range(d):
            s[k] += w * delta[k]
    mean = [s[k] / wsum for k in range(d)]
    if mode == "direct":
        return mean
    sn = _norm(s)
    r = 1.0 if sn < 1e-300 else num / sn
    return [r * m for m in mean]


def carried_weights(queue: list[dict[str, Any]], rnd: int) -> tuple[list[dict[str, Any]], int]:
    """Queue entries usable in round ``rnd`` as (entry with weight, n dropped as too stale), ordered by (base round, worker)."""
    keep, dropped = [], 0
    for e in sorted(queue, key=lambda e: (e["base"], e["worker"])):
        s = rnd - e["base"]
        if s > MAX_STALENESS:
            dropped += 1
        else:
            keep.append({**e, "w": 1.0 / (1.0 + s)})
    return keep, dropped


def _penalised(sc: Scenario, rnd: int) -> int:
    if sc.straggle == "none" or (rnd - 1) % sc.straggle_every != 0:
        return -1
    if sc.straggle == "persistent":
        return sc.M - 1
    return sc.M - 1 - (((rnd - 1) // sc.straggle_every) % (sc.M - 1))


def retention(l0: float, l_final: float, l_sync: float) -> float | None:
    """E13's rho = (L0 - L_cell) / (L0 - L_sync); None when the synchronous reference did not improve on L0."""
    den = l0 - l_sync
    return None if abs(den) <= 1e-12 else (l0 - l_final) / den


def _run(sc: Scenario) -> dict[str, Any]:
    a_all, c_all, shared = build_domains(sc)
    d, M = sc.dim, sc.M
    rngs = [Rng(sc.seed + 1 + i) for i in range(M)]
    theta, buf = list(shared), [0.0] * d
    l0, per0 = losses(theta, a_all, c_all)
    curve, rows, queue, excl, total_t = [l0], [], [], [0] * M, 0.0
    for rnd in range(1, sc.rounds + 1):
        absent = 1 if sc.churn_end and sc.churn_start <= rnd <= sc.churn_end else -1
        alive = [i for i in range(M) if i != absent]
        slow, arr, deltas = _penalised(sc, rnd), {}, {}
        for i in alive:
            rng = rngs[i]
            arr[i] = 1.0 + 0.2 * rng.uniform() + (sc.penalty if i == slow else 0.0)
            th = list(theta)
            for _ in range(sc.H):
                for k in range(d):
                    g = a_all[i][k] * (th[k] - c_all[i][k]) + sc.sigma * rng.gauss()
                    th[k] = th[k] - sc.eta * g
            deltas[i] = [theta[k] - th[k] for k in range(d)]
        order = sorted(alive, key=lambda i: (arr[i], i))
        t_k = arr[order[min(sc.K, len(alive)) - 1]]
        close = min(t_k * (1.0 + sc.grace), max(arr.values()))
        required = [i for i in alive if sc.policy == "fair" and excl[i] >= MAX_EXCLUSION and arr[i] > close]
        for i in required:
            close = max(close, arr[i])
        admitted = [i for i in alive if arr[i] <= close]
        late = [i for i in alive if arr[i] > close]
        items = [(deltas[i], 1.0) for i in admitted]
        n_carried = dropped = 0
        if sc.policy == "deferred":
            ready_entries = [e for e in queue if rnd >= int(e.get("ready_round", e["base"] + 1))]
            still_in_flight = [e for e in queue if rnd < int(e.get("ready_round", e["base"] + 1))]
            usable, dropped = carried_weights(ready_entries, rnd)
            items += [(e["delta"], e["w"]) for e in usable]
            n_carried = len(usable)
            delay_rounds = max(1, int(round(sc.penalty)) - 1) if sc.penalty > 2.5 else 1
            queue = still_in_flight + [
                {"delta": deltas[i], "base": rnd, "ready_round": rnd + delay_rounds, "worker": i} for i in late
            ]
        for i in range(M):
            excl[i] = 0 if (i in admitted or i not in alive) else excl[i] + 1
        g = merge_deltas(items, sc.merge)
        for k in range(d):
            buf[k] = sc.outer_momentum * buf[k] + g[k]
            theta[k] = theta[k] - sc.outer_lr * (g[k] + sc.outer_momentum * buf[k])
        total_t += close
        loss, per = losses(theta, a_all, c_all)
        curve.append(loss)
        rows.append({"round": rnd, "loss": loss, "domain_losses": per, "alive": alive, "admitted": admitted, "late": late, "required": required,
                     "carried": n_carried, "dropped_stale": dropped, "round_time": close})
    return {"initial_loss": l0, "initial_domain_losses": per0, "loss_curve": curve, "rounds": rows, "total_time": total_t, "final_loss": curve[-1],
            "final_domain_losses": rows[-1]["domain_losses"]}


def _digest(curve: list[float]) -> str:
    return hashlib.sha256(",".join("%.9f" % v for v in curve).encode("utf-8")).hexdigest()[:16]


def contract_digest(scenario: "Scenario | dict[str, Any]") -> str:
    """Short hash (16 hex chars) of the loss curve rounded to 9 decimals: the value a JavaScript port compares against."""
    return _digest(_run(_as_scenario(scenario))["loss_curve"])


def simulate(scenario: "Scenario | dict[str, Any]") -> dict[str, Any]:
    """Run one scenario; returns a JSON-serialisable dict (curve, per-round rows, retention against the synchronous reference, total simulated time)."""
    sc = _as_scenario(scenario)
    res = _run(sc)
    ref = dataclasses.replace(sc, K=sc.M, straggle="none", policy="discard", churn_start=0, churn_end=0)
    l_sync = _run(ref)["final_loss"]
    res.update({"scenario": dataclasses.asdict(sc), "sync_final_loss": l_sync, "retention": retention(res["initial_loss"], res["final_loss"], l_sync),
                "digest": _digest(res["loss_curve"])})
    return res


def compare_policies(base: "Scenario | dict[str, Any] | None" = None, policies: tuple[str, ...] = POLICIES,
                     straggles: tuple[str, ...] = STRAGGLES) -> list[dict[str, Any]]:
    """Table rows (policy x straggle mode) with retention, final loss, total simulated time and late/carried counts; other fields from ``base``."""
    sc = _as_scenario(base if base is not None else Scenario())
    rows = []
    for st in straggles:
        for pol in policies:
            r = simulate(dataclasses.replace(sc, straggle=st, policy=pol))
            rows.append({"policy": pol, "straggle": st, "retention": r["retention"], "final_loss": r["final_loss"], "total_time": r["total_time"],
                         "late_total": sum(len(x["late"]) for x in r["rounds"]), "carried_total": sum(x["carried"] for x in r["rounds"])})
    return rows


# ------------------------------------------------------------------------------------------------ CLI
def sparkline(values: list[float], width: int = 48) -> str:
    """ASCII sparkline (' .:-=+*#%@', high = large) of a series, resampled to at most ``width`` points."""
    chars = " .:-=+*#%@"
    step = max(1, math.ceil(len(values) / width))
    vs = values[::step]
    lo, hi = min(vs), max(vs)
    span = hi - lo
    return "".join(chars[0 if span <= 0 else min(len(chars) - 1, int((v - lo) / span * (len(chars) - 1) + 0.5))] for v in vs)


def _fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _scenario_from_args(args: argparse.Namespace) -> Scenario:
    base = dict(PRESETS[args.preset]) if getattr(args, "preset", None) else {}
    for f in dataclasses.fields(Scenario):
        v = getattr(args, f.name, None)
        if v is not None:
            base[f.name] = v
    return Scenario(**base).validate()


def _add_scenario_flags(p: argparse.ArgumentParser) -> None:
    for f in dataclasses.fields(Scenario):
        default = f.default
        kw: dict[str, Any] = {"type": type(default), "default": None, "help": f"(default {default!r})"}
        if f.name == "merge":
            kw["choices"] = ("direct", "rda")
        elif f.name == "policy":
            kw["choices"] = POLICIES
        elif f.name == "straggle":
            kw["choices"] = STRAGGLES
        p.add_argument("--" + f.name.replace("_", "-"), dest=f.name, **kw)


def _cmd_simulate(args: argparse.Namespace) -> int:
    res = simulate(_scenario_from_args(args))
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    print(f"SwarmLab-1 (qualitative toy, not E13)  digest={res['digest']}")
    print(f"{'round':>5} {'loss':>10} {'admitted':<14} {'late':<10} {'carried':>7} {'time':>6}")
    for r in res["rounds"]:
        print(f"{r['round']:>5} {r['loss']:>10.4f} {str(r['admitted']):<14} {str(r['late']):<10} {r['carried']:>7} {r['round_time']:>6.2f}")
    print(f"loss  [{sparkline(res['loss_curve'])}]  {res['initial_loss']:.3f} -> {res['final_loss']:.3f}")
    print(f"sync reference final loss {res['sync_final_loss']:.4f}   retention {_fmt(res['retention'])}   total simulated time {res['total_time']:.2f}")
    print("per-domain final losses: " + ", ".join(f"{x:.3f}" for x in res["final_domain_losses"]))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    rows = compare_policies(_scenario_from_args(args))
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print("SwarmLab-1 (qualitative toy, not E13): retention vs the synchronous reference")
    print(f"{'straggle':<11} {'policy':<9} {'retention':>9} {'final loss':>10} {'total time':>10} {'late':>5} {'carried':>7}")
    for r in rows:
        print(f"{r['straggle']:<11} {r['policy']:<9} {_fmt(r['retention']):>9} {r['final_loss']:>10.4f} {r['total_time']:>10.2f} {r['late_total']:>5} {r['carried_total']:>7}")
    return 0


def _cmd_presets(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps(PRESETS, indent=2))
    else:
        for name, p in PRESETS.items():
            print(f"{name:<22} {p}")
    return 0


LEVELS = (
    ("Level 1: persistent straggler", {"M": 4, "straggle": "persistent"}),
    ("Level 2: rotating straggler", {"M": 4, "straggle": "rotating"}),
    ("Level 3: a worker leaves, a straggler stays", {"M": 4, "straggle": "persistent", "churn_start": 3, "churn_end": 6}),
)


def _ask(prompt: str, valid: list[str], auto: str, use_auto: bool) -> str:
    if use_auto:
        return auto
    while True:
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            return auto
        if ans in valid:
            return ans
        print(f"  please answer one of: {', '.join(valid)}")


def _cmd_play(args: argparse.Namespace) -> int:
    print("Quorum Quest (a qualitative toy; not the real E13 experiment). Pick K and a late-delta policy; your learning score is the retention of the synchronous gain.")
    total = 0
    for name, fixed in LEVELS:
        print(f"\n{name}  (M={fixed['M']} workers; K=1..{fixed['M']}; policies: {', '.join(POLICIES)})")
        k = int(_ask("  K> ", [str(i) for i in range(1, fixed["M"] + 1)], str(fixed["M"] - 1), args.auto))
        pol = _ask("  policy> ", list(POLICIES), "deferred", args.auto)
        res = simulate(Scenario(K=k, policy=pol, seed=args.seed, **fixed))
        rho = res["retention"]
        score = 0 if rho is None else int(round(100 * min(1.0, max(0.0, rho))))
        total += score
        print(f"  K={k} policy={pol}: retention {_fmt(rho)}, simulated time {res['total_time']:.1f}, learning score {score}/100")
    print(f"\nTotal learning score: {total}/{100 * len(LEVELS)} (a learning score, not PTS or $HIVE).")
    print(NONMONETARY_NOTICE)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="swarm-lab", description="HiveMind Swarm Lab: a deterministic, GPU-free QUALITATIVE toy of quorum policy x stragglers x non-IID data "
                                 "(an educational analogy of E13, NOT a reproduction of its numbers).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("simulate", help="run one scenario")
    _add_scenario_flags(p)
    p.add_argument("--preset", choices=sorted(PRESETS), help="start from a named scenario (explicit flags override it)")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.set_defaults(fn=_cmd_simulate)
    p = sub.add_parser("compare", help="policies x straggle modes table (policy/straggle flags are ignored)")
    _add_scenario_flags(p)
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_compare)
    p = sub.add_parser("presets", help="list the named scenarios")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=_cmd_presets)
    p = sub.add_parser("play", help="Quorum Quest, a small terminal game")
    p.add_argument("--auto", action="store_true", help="non-interactive: use scripted choices")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(fn=_cmd_play)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

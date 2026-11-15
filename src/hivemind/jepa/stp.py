# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Semantic Tube Prediction (STP; Huang, LeCun, Balestriero, arXiv:2602.22617) for HiveMind training paths.

The paper's loss, used for its reported results (Section 4: "pick (random) indices s<r<t, and compute
1-cos(h_t-h_r, h_r-h_s)" on the last-layer hidden states):

    L = L_NTP + lam * mean[ 1 - cos(h_t - h_r, h_r - h_s) ]

It needs neither paired views nor an extra forward pass: the hidden states of the normal training pass are reused, so
the cost is about that of plain fine-tuning. The paper evaluates fine-tuning only.

HiveMind specifics:
- Packed raw-token streams (BOINC, DiLoCo, CrowdGPT) join unrelated documents with EOS. A trajectory across an EOS is
  not a trajectory of one text, so triples are drawn inside a single document segment. Rows without a segment of at
  least `min_segment` tokens contribute nothing.
- Indices come from a hash of (seed, step, row, k), never a device RNG, so a validator replaying a segment computes the
  same loss.
- A straight trajectory (h_i = a + i v) satisfies STP while discarding information. The paper reports the term at about
  1.4 with lam = 0 and 0.6 with lam = 0.08, never near 0. `health()` flags values under `collapse_floor`.

`random_span_loss` ports the different variant found in the released code (stp.py, --linear=random_span). The paper
does not describe it; it is provided for comparison only and is not assumed equivalent.
"""

from __future__ import annotations

from typing import Sequence

from hivemind.jepa.spec import JepaSpec, step_uniform

Triple = tuple[int, int, int]


def document_segments(ids: Sequence[int], eos_id: int, min_len: int) -> list[tuple[int, int]]:
    """Half-open [a, b) token ranges of the documents in a packed row (EOS tokens excluded), at least `min_len` long.
    eos_id = -1 treats the whole row as one document."""
    if min_len < 3:
        raise ValueError("min_len must be >= 3")
    ids = [int(x) for x in ids]
    if eos_id < 0:
        return [(0, len(ids))] if len(ids) >= min_len else []
    out, a = [], 0
    for i, t in enumerate(ids + [eos_id]):
        if t == eos_id:
            if i - a >= min_len:
                out.append((a, i))
            a = i + 1
    return out


def sample_triples(segments: Sequence[tuple[int, int]], n: int, seed: int, step: int, row: int) -> list[Triple]:
    """n deterministic triples s < r < t, each inside one segment (segment chosen with probability proportional to its
    length). For each: s uniform in [a, b-3], t uniform in [s+2, b-1], r uniform in (s, t)."""
    if not segments:
        return []
    lengths = [b - a for a, b in segments]
    total = sum(lengths)
    out: list[Triple] = []
    for k in range(n):
        u = step_uniform(seed, step, row, 4 * k) * total
        j = 0
        while j < len(lengths) - 1 and u >= lengths[j]:
            u -= lengths[j]
            j += 1
        a, b = segments[j]
        s = a + int(step_uniform(seed, step, row, 4 * k + 1) * (b - 2 - a))
        t = s + 2 + int(step_uniform(seed, step, row, 4 * k + 2) * (b - s - 2))
        r = s + 1 + int(step_uniform(seed, step, row, 4 * k + 3) * (t - s - 1))
        out.append((s, r, t))
    return out


def geodesic_loss(hidden, triples_per_row: Sequence[Sequence[Triple]]):
    """mean over all triples of 1 - cos(h_t - h_r, h_r - h_s). hidden: [B, T, D] last-layer states. Returns (loss, n);
    loss is None when no row has a triple."""
    import torch
    import torch.nn.functional as F

    rows, s, r, t = [], [], [], []
    for i, trips in enumerate(triples_per_row):
        for a, b, c in trips:
            rows.append(i); s.append(a); r.append(b); t.append(c)
    if not rows:
        return None, 0
    dev = hidden.device
    ri = torch.tensor(rows, device=dev)
    hs = hidden[ri, torch.tensor(s, device=dev)].float()
    hr = hidden[ri, torch.tensor(r, device=dev)].float()
    ht = hidden[ri, torch.tensor(t, device=dev)].float()
    return 1.0 - F.cosine_similarity(ht - hr, hr - hs, dim=-1).mean(), len(rows)


def stp_term(spec: JepaSpec, hidden, input_ids, step: int):
    """The STP term for a batch of packed raw-token rows. Returns (loss or None, info dict)."""
    if spec.mode != "stp":
        raise ValueError("stp_term needs spec.mode == 'stp'")
    if spec.stp_variant != "geodesic":
        raise ValueError("raw packed rows use the paper's geodesic variant; random_span needs explicit view ranges")
    trips = []
    for i in range(input_ids.shape[0]):
        segs = document_segments(input_ids[i].tolist(), spec.eos_id, spec.min_segment)
        trips.append(sample_triples(segs, spec.stp_samples, spec.seed, step, i))
    loss, n = geodesic_loss(hidden, trips)
    return loss, {"triples": n, "rows_without_segment": sum(1 for x in trips if not x)}


def health(value: float, spec: JepaSpec) -> str:
    """'ok', or 'collapse' when the term falls under spec.collapse_floor (possible straight-line shortcut)."""
    return "collapse" if value < spec.collapse_floor else "ok"


# ---------------------------------------------------------------------------------- released-code variant (comparison)


def _disp(hidden, row: int, pieces: Sequence[tuple[int, int]]):
    """Sum of displacements h[v-1] - h[u-1] over token ranges [u, v) (as in the released get_embeddings)."""
    tot = None
    for u, v in pieces:
        if v <= u:
            continue
        if u < 1:
            raise ValueError("a range must start at token >= 1 (needs the state before it)")
        d = hidden[row, v - 1].float() - hidden[row, u - 1].float()
        tot = d if tot is None else tot + d
    return tot


def _split(segments: Sequence[tuple[int, int]], p: int, q: int):
    """Split the concatenation of `segments` (combined offsets) into before [0,p), patch [p,q), after [q,end)."""
    before, patch, after, off = [], [], [], 0
    for a, b in segments:
        n = b - a
        for lo, hi, dst in ((0, p, before), (p, q, patch), (q, 10**12, after)):
            x0, x1 = max(lo, off), min(hi, off + n)
            if x1 > x0:
                dst.append((a + x0 - off, a + x1 - off))
        off += n
    return before, patch, after


def random_span_cos(hidden, row: int, segments: Sequence[tuple[int, int]], p: int, q: int, weighting: str | None = None):
    """Released-code variant, one sample: cos(disp(before) + disp(after), disp(patch)) for a patch [p, q) of the
    concatenated `segments` (e.g. the user and assistant messages, skipping the template tokens between them).
    Returns (cos, weight); combine samples with `random_span_aggregate`. The weight follows the released get_weights
    ('cosine_like', 'jaccard_like', or None = 1)."""
    import torch.nn.functional as F

    full = sum(b - a for a, b in segments)
    if not 0 <= p < q <= full or q - p >= full:
        raise ValueError("need 0 <= p < q <= total and a non-empty rest")
    before, patch, after = _split(segments, p, q)
    rest = [x for x in (_disp(hidden, row, before), _disp(hidden, row, after)) if x is not None]
    rest_v = rest[0] if len(rest) == 1 else rest[0] + rest[1]
    patch_v = _disp(hidden, row, patch)
    plen, rlen = q - p, full - (q - p)
    if weighting is None:
        w = 1.0
    elif weighting == "cosine_like":
        w = 2.0 * rlen * plen / (rlen * rlen + plen * plen)
    elif weighting == "jaccard_like":
        w = 1.0 - abs(rlen - plen) / (rlen + plen)
    else:
        raise ValueError("weighting must be None, 'cosine_like' or 'jaccard_like'")
    return F.cosine_similarity(rest_v, patch_v, dim=-1), w


def random_span_aggregate(cosines: Sequence, weights: Sequence[float]):
    """1 - sum(w * cos) / sum(w), as the released compute_loss."""
    import torch

    c = torch.stack([x.float() for x in cosines])
    w = torch.tensor([float(x) for x in weights], device=c.device)
    return 1.0 - (c * w).sum() / w.sum()

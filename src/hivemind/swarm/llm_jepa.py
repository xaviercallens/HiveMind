# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""LLM-JEPA auxiliary objective for HiveMind QLoRA rounds (IP-JEPA-01).

Reference: H. Huang, Y. LeCun, R. Balestriero, "LLM-JEPA: Large Language Models Meet Joint Embedding
Predictive Architectures", arXiv:2509.14252 (v2, 2025). Upstream code: https://github.com/rbalestr-lab/llm-jepa.

The objective (their eq. 2) keeps the next-token loss and adds an embedding-space term between two *views*
of the same content (Text, Code)::

    L = L_LLM(Text -> Code) + lambda * (1 - cos(Pred(Enc(Text)), Enc(Code)))

* ``Enc`` is the last-layer hidden state of the last token of a view.
* ``Pred`` is tied to the LM itself: ``k`` ``[PRED]`` tokens are appended to Text and the embedding of the
  last one is used (``k = 0`` means ``Pred`` is the identity).
* Both views are packed in one sequence with a *block-causal* additive mask so that neither view attends to
  the other; this gets both embeddings in one extra forward pass.
* Random JEPA-loss dropout ``alpha`` skips the extra pass on a fraction of batches (per-epoch cost
  ``2 - alpha`` times plain fine-tuning); the paper's co-tuning rule keeps ``lambda * (1 - alpha)`` constant.

This module only depends on ``torch`` and works with any Hugging Face causal LM that accepts a 4-D additive
``attention_mask`` and ``output_hidden_states=True``. Training adds cost; inference is unchanged, so the
served adapter (``hivemind/granite-moe``, ``coder-lite``) keeps its latency.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


@dataclass(frozen=True)
class JepaConfig:
    """Hyper-parameters of the auxiliary term (paper grid: ``k in 0..4``, ``lambda in {0.5, 1, 2, 4}``)."""

    lam: float = 1.0
    k_pred: int = 1
    loss_dropout: float = 0.0  # alpha: probability that a batch skips the JEPA term
    metric: str = "cosine"  # paper ablation: cosine > MSE >> l2 / InfoNCE

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError("lambda must be >= 0")
        if self.k_pred < 0:
            raise ValueError("k_pred must be >= 0")
        if not 0.0 <= self.loss_dropout < 1.0:
            raise ValueError("loss_dropout must be in [0, 1)")
        if self.metric not in {"cosine", "mse", "l2"}:
            raise ValueError(f"unknown metric {self.metric!r}")


def cotuned_lambda(base_lam: float, base_dropout: float, new_dropout: float) -> float:
    """Keep ``lambda * (1 - alpha)`` constant when changing the loss-dropout rate (paper Section 5.2)."""
    if not (0.0 <= base_dropout < 1.0 and 0.0 <= new_dropout < 1.0):
        raise ValueError("dropout rates must be in [0, 1)")
    return base_lam * (1.0 - base_dropout) / (1.0 - new_dropout)


def relative_epoch_cost(loss_dropout: float) -> float:
    """Per-epoch forward cost relative to plain fine-tuning: ``2 - alpha`` (paper Section 5.2)."""
    if not 0.0 <= loss_dropout <= 1.0:
        raise ValueError("loss_dropout must be in [0, 1]")
    return 2.0 - loss_dropout


def block_causal_mask(seq_len: int, blocks: Sequence[tuple[int, int]], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Additive ``(seq_len, seq_len)`` mask: causal inside each ``(start, size)`` block, ``-inf`` elsewhere.

    Positions not covered by a block (padding) attend only to themselves so softmax never sees an all-``-inf`` row.
    """
    mask = torch.full((seq_len, seq_len), NEG_INF, dtype=dtype)
    covered = torch.zeros(seq_len, dtype=torch.bool)
    for start, size in blocks:
        if size <= 0 or start < 0 or start + size > seq_len:
            raise ValueError(f"block ({start}, {size}) outside sequence of length {seq_len}")
        if covered[start:start + size].any():
            raise ValueError("blocks overlap")
        tri = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
        block = torch.zeros(size, size, dtype=dtype)
        block[tri] = NEG_INF
        mask[start:start + size, start:start + size] = block
        covered[start:start + size] = True
    for i in torch.nonzero(~covered).flatten().tolist():
        mask[i, i] = 0.0
    return mask


def embedding_distance(pred: torch.Tensor, target: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    """Mean distance between predicted and target view embeddings, shape ``(batch, hidden)``."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {tuple(pred.shape)} vs {tuple(target.shape)}")
    if metric == "cosine":
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    if metric == "mse":
        return F.mse_loss(pred, target)
    if metric == "l2":
        return torch.linalg.vector_norm(pred - target, dim=-1).mean()
    raise ValueError(f"unknown metric {metric!r}")


def pack_two_views(text_ids: Sequence[Sequence[int]], code_ids: Sequence[Sequence[int]], pred_token_id: int, k_pred: int,
                   pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ``Text + [PRED]*k`` and ``Code`` into one row per example with a block-causal mask.

    Returns ``(input_ids, attention_mask_4d, text_last_idx, code_last_idx)``; the mask has shape
    ``(batch, 1, L, L)`` and is additive (0 = attend, -inf = blocked).
    """
    if len(text_ids) != len(code_ids) or not text_ids:
        raise ValueError("need the same, non-zero number of Text and Code views")
    rows, spans = [], []
    for t, c in zip(text_ids, code_ids):
        if not t or not c:
            raise ValueError("empty view")
        t_full = list(t) + [pred_token_id] * k_pred
        rows.append(t_full + list(c))
        spans.append(((0, len(t_full)), (len(t_full), len(c))))
    seq_len = max(len(r) for r in rows)
    ids = torch.full((len(rows), seq_len), pad_id, dtype=torch.long)
    masks = []
    t_last, c_last = [], []
    for i, (row, (tspan, cspan)) in enumerate(zip(rows, spans)):
        ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        masks.append(block_causal_mask(seq_len, [tspan, cspan]))
        t_last.append(tspan[0] + tspan[1] - 1)
        c_last.append(cspan[0] + cspan[1] - 1)
    return ids, torch.stack(masks).unsqueeze(1), torch.tensor(t_last), torch.tensor(c_last)


def jepa_term(model: Any, text_ids: Sequence[Sequence[int]], code_ids: Sequence[Sequence[int]], pred_token_id: int,
              cfg: JepaConfig, pad_id: int = 0) -> torch.Tensor:
    """One extra forward pass returning ``d(Pred(Enc(Text)), Enc(Code))`` for a batch of view pairs."""
    device = next(model.parameters()).device
    ids, mask, t_last, c_last = pack_two_views(text_ids, code_ids, pred_token_id, cfg.k_pred, pad_id)
    out = model(input_ids=ids.to(device), attention_mask=mask.to(device=device, dtype=next(model.parameters()).dtype),
                output_hidden_states=True)
    h = out.hidden_states[-1]
    rows = torch.arange(h.shape[0], device=device)
    return embedding_distance(h[rows, t_last.to(device)], h[rows, c_last.to(device)], cfg.metric)


def llm_jepa_loss(lm_loss: torch.Tensor, model: Any, text_ids: Sequence[Sequence[int]], code_ids: Sequence[Sequence[int]],
                  pred_token_id: int, cfg: JepaConfig, rng: random.Random | None = None,
                  pad_id: int = 0) -> tuple[torch.Tensor, dict[str, float]]:
    """``L_LLM + lambda * d`` with random loss dropout; returns the total loss and a small log dict."""
    rng = rng or random
    if cfg.lam == 0.0 or (cfg.loss_dropout > 0.0 and rng.random() < cfg.loss_dropout):
        return lm_loss, {"lm": float(lm_loss.detach()), "jepa": float("nan"), "jepa_active": 0.0}
    d = jepa_term(model, text_ids, code_ids, pred_token_id, cfg, pad_id)
    return lm_loss + cfg.lam * d, {"lm": float(lm_loss.detach()), "jepa": float(d.detach()), "jepa_active": 1.0}


def view_gap_singular_values(text_emb: torch.Tensor, code_emb: torch.Tensor, top: int = 10) -> list[float]:
    """Top singular values of ``Enc(Text) - Enc(Code)`` (paper Fig. 3 diagnostic: lower = narrower mapping)."""
    s = torch.linalg.svdvals((text_emb - code_emb).float())
    return [float(x) for x in s[:top]]

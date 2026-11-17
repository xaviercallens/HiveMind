# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""LLM-JEPA on paired views (Huang, LeCun, Balestriero, arXiv:2509.14252), with the anti-collapse options measured
on Crowd-v1 (HiveMind PR #907).

    L = L_NTP(prompt -> answer) + lam * d(h_view1, h_view2)

h_view is the last-layer state at the last token of each view, computed in passes where the views do not attend to
each other. Measured on the T4 (LoRA r64, NL-RX-SYNTH): with lam = 1 and no anti-collapse, 1 - cos fell to about 5e-4
and general held-out loss rose by +0.90 nat, with no accuracy gain. Stop-gradient on the target view, batch centering,
or InfoNCE each removed that damage. Defaults therefore use stop-gradient.
"""

from __future__ import annotations

from hivemind.jepa.spec import JepaSpec


def pair_term(spec: JepaSpec, h_view1, h_view2, *, infonce: bool = False, temperature: float = 0.07):
    """h_view1, h_view2: [B, D] embeddings of the two views. Returns the auxiliary loss (before multiplying by lam)."""
    import torch
    import torch.nn.functional as F

    if spec.mode != "pairs":
        raise ValueError("pair_term needs spec.mode == 'pairs'")
    a, b = h_view1.float(), h_view2.float()
    if spec.stopgrad:
        b = b.detach()
    if spec.center:
        mu = torch.cat([a, b]).mean(0, keepdim=True)
        a, b = a - mu, b - mu
    if infonce:
        logits = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).t() / temperature
        return F.cross_entropy(logits, torch.arange(a.shape[0], device=a.device))
    return 1.0 - F.cosine_similarity(a, b, dim=-1).mean()


def last_token_states(hidden, attention_mask):
    """[B, T, D] states and a right-padded [B, T] mask -> [B, D] state at each row's last real token."""
    import torch

    idx = attention_mask.long().sum(dim=1) - 1
    if bool((idx < 0).any()):
        raise ValueError("a row has no real token")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]

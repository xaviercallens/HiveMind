# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hugging Face causal LMs: next-token loss plus the JEPA-style term from ONE forward pass.

`output_hidden_states=True` returns every layer's output (and keeps it alive), which costs memory on a shared GPU.
Here the decoder body is called once, the last hidden state is taken directly, and the LM head is applied only where
labels exist. Works on a plain model or a PEFT-wrapped one (`get_base_model()`).
"""

from __future__ import annotations

from hivemind.jepa.spec import JepaSpec, apply_aux
from hivemind.jepa.stp import health, stp_term


def _base(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def ntp_and_stp(model, input_ids, labels, spec: JepaSpec, step: int, attention_mask=None):
    """Returns (total_loss, info). info: ntp, aux (float or None), triples, health.

    labels follow the HF convention (-100 = ignored, shifted inside here). With spec.mode == 'none', or on a
    loss-dropout step, total_loss is the plain next-token loss computed from the same pass."""
    import torch.nn.functional as F

    base = _base(model)
    hidden = base.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    sel = labels[:, 1:] != -100
    logits = base.lm_head(hidden[:, :-1][sel]).float()
    ntp = F.cross_entropy(logits, labels[:, 1:][sel])
    info = {"ntp": float(ntp.detach()), "aux": None, "triples": 0, "health": "n/a"}
    if spec.mode != "stp" or not apply_aux(spec, step):
        return ntp, info
    aux, meta = stp_term(spec, hidden, input_ids, step)
    info["triples"] = meta["triples"]
    if aux is None:
        return ntp, info
    info["aux"] = float(aux.detach())
    info["health"] = health(info["aux"], spec)
    return ntp + spec.lam * aux, info

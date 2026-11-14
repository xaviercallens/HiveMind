"""autoresearch-style train.py for Crowd-v1 + LLM-JEPA on NL-RX-SYNTH (after karpathy/autoresearch, March 2026).

Protocol (program.md): FIXED wall-clock training budget (BUDGET_S), then the FIXED evaluation below. Single metric:
  val_ce  = mean next-token CE (nats/token) on the answer tokens of the 500-example dev split (train lines 7500..7999,
            never trained on; the test set is not used here). Lower is better.
Guard rail: far_loss (E0 "far" windows, general LM) must stay <= SFT-baseline far_loss + 0.05, or the run is discarded.
The experiment is selected by EXPERIMENT (one hypothesis = one small change). Everything runs under t4guard.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/mnt/disks/disk-socrateai-local-1/crowdgpt_exp")
sys.path.insert(0, str(ROOT / "scripts"))
import night_jepa as nj  # noqa: E402  (model, LoRA, batching, far_loss: unchanged from the P1 pilot)
import t4guard  # noqa: E402

BUDGET_S = float(__import__('os').environ.get('AR_BUDGET_S', 600))             # fixed training wall-clock budget (karpathy: 300 s on an H100; the T4 is ~10x slower)
N_TRAIN, N_DEV = 7500, 500
BS, ACCUM, LR, RANK = 8, 2, 2e-4, 64
SEED = 0

# ------------------------------------------------------------------------------------------- the experiment knob
EXPERIMENTS = {
    "sft":         dict(jepa=False),
    "jepa":        dict(jepa=True, lam=1.0),                      # P1 baseline objective
    "h1_low_lam":  dict(jepa=True, lam=0.1, dropout=0.5),          # H1: much weaker JEPA pressure + loss dropout
    "h2_stopgrad": dict(jepa=True, lam=1.0, stopgrad=True),        # H2: stop-gradient on the target (answer) view
    "h3_center":   dict(jepa=True, lam=1.0, center=True),          # H3: cosine after removing the batch mean
    "h4_infonce":  dict(jepa=True, lam=1.0, infonce=True),         # H4: in-batch contrastive (negatives), T = 0.07
    "h6_midlayer": dict(jepa=True, lam=1.0, layer=16),             # H6: align layer-16 states, not the LM-head input
    # post-hoc, exploratory (added 09:02 after all 5 screens; not part of the pre-registered selection)
    "x_h1h2": dict(jepa=True, lam=0.1, dropout=0.5, stopgrad=True),  # low pressure + no damage to the answer side
    "x_h1h3": dict(jepa=True, lam=0.1, dropout=0.5, center=True),    # low pressure + anti-anisotropy
}


def jepa_loss(cfg, eq, ea):
    eq, ea = eq.float(), ea.float()
    if cfg.get("stopgrad"):
        ea = ea.detach()
    if cfg.get("center"):
        mu = torch.cat([eq, ea]).mean(0, keepdim=True)
        eq, ea = eq - mu, ea - mu
    if cfg.get("infonce"):
        logits = F.normalize(eq, dim=-1) @ F.normalize(ea, dim=-1).t() / 0.07
        return F.cross_entropy(logits, torch.arange(len(eq), device=eq.device))
    return 1.0 - F.cosine_similarity(eq, ea, dim=-1).mean()


def view_states(model, ids, mask, vs, layer=None):
    """Packed-view pass; final (post-ln_f) states, or the residual stream after block `layer` (stops early)."""
    if layer is None:
        return model.hidden(ids, mask=mask, ckpt=True, view_starts=vs)
    m = model.m
    cos, sin = nj.cm._rope(int(m.config["headDim"]), ids.shape[1], ids.device)
    with torch.no_grad():
        ic = ids.cpu()
        prev = torch.cat((torch.zeros_like(ic[:, :1]), ic[:, :-1]), dim=1)
        prev[torch.arange(ic.shape[0]), vs] = 0
        e = model.engram.table((prev * 1_000_003 + ic) % model.engram.table.num_embeddings).cuda()
    x = m.wte(ids) + e
    for blk in m.blocks[:layer]:
        def f(x, blk=blk):
            x = x + nj.attn_forward(blk.attn, blk.ln_1(x), cos, sin, mask)
            return x + blk.mlp(blk.ln_2(x))
        x = torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False)
    return x


# ------------------------------------------------------------------------------------------- FIXED evaluation
@torch.no_grad()
def eval_val_ce(model, dev):
    tot, n = 0.0, 0
    for i in range(0, len(dev), 16):
        t4guard.wait_idle()
        ids, lab = nj.batch_sft(dev[i:i + 16])
        ids, lab = ids.cuda(), lab.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            h = model.hidden(ids)
        sel = lab != -100
        tot += F.cross_entropy(model.logits_at(h[sel]), lab[sel], reduction="sum").item()
        n += int(sel.sum())
    return tot / n


def main() -> int:
    name = sys.argv[1]
    cfg = EXPERIMENTS[name]
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    allx = nj.load(ROOT / "data/nlrx/synth_train.jsonl", tok)
    train, dev = allx[:N_TRAIN], allx[N_TRAIN:N_TRAIN + N_DEV]
    t4guard.DEADLINE = float(sys.argv[2]) if len(sys.argv) > 2 else None
    with t4guard.GpuLock():
        t4guard.wait_for_room(2800)
        t4guard.cap_process(max_cap_mib=2700)
        model = nj.Model(RANK, alpha=float(RANK))
        params = model.lora
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda")
        order, step, t_train, jlast = [], 0, 0.0, None
        while t_train < BUDGET_S:
            t4guard.wait_idle()
            t0 = time.time()
            frac = t_train / BUDGET_S  # time-based schedule, as in autoresearch
            for g in opt.param_groups:
                g["lr"] = LR * min(1.0, (frac + 1e-9) / 0.03) * 0.5 * (1 + math.cos(math.pi * frac))
            opt.zero_grad(set_to_none=True)
            use_j = cfg["jepa"] and (cfg.get("dropout", 0.0) == 0.0 or random.random() >= cfg["dropout"])
            for _ in range(ACCUM):
                if len(order) < BS:
                    order = list(range(len(train))); random.shuffle(order)
                ex = [train[i] for i in order[:BS]]; order = order[BS:]
                ids, lab = nj.batch_sft(ex)
                ids, lab = ids.cuda(), lab.cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    h = model.hidden(ids, ckpt=True)
                    sel = lab != -100
                    loss = F.cross_entropy(model.logits_at(h[sel]), lab[sel])
                    if use_j:
                        vids, vmask, ip, ia, vs = nj.batch_views(ex)
                        hv = view_states(model, vids.cuda(), vmask.cuda(), vs, cfg.get("layer"))
                        r = torch.arange(len(ex))
                        j = jepa_loss(cfg, hv[r, ip], hv[r, ia])
                        loss = loss + cfg["lam"] * j
                        jlast = float(j.detach())
                scaler.scale(loss / ACCUM).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            t_train += time.time() - t0
            step += 1
            t4guard.check()
        val_ce = eval_val_ce(model, dev)
        far = nj.far_loss(model)
        pred = nj.evaluate(model, dev[:300])
        dec = [tok.decode(p, skip_special_tokens=True).strip() for p in pred]
        dev_acc = 100.0 * sum(d == t[2] for d, t in zip(dec, dev[:300])) / 300
        out = {"experiment": name, "cfg": cfg, "val_ce": round(val_ce, 6), "far_loss": round(far, 4),
               "dev_acc300": round(dev_acc, 2), "steps": step, "train_s": round(t_train, 1), "jepa_last": jlast,
               "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    print(json.dumps(out))
    print(f"val_ce: {out['val_ce']:.6f}")
    print(f"far_loss: {out['far_loss']:.4f}")
    print(f"dev_acc300: {out['dev_acc300']:.2f}")
    print(f"peak_vram_mb: {out['peak_vram_mb']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

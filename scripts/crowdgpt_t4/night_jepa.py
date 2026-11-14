"""P1 night pilot: LoRA SFT vs LoRA SFT + LLM-JEPA on NL-RX-SYNTH, Crowd-v1 (rev 5ac3916, engram on), shared T4.

Arms (paper recipe adapted to LoRA on a 5 GB VRAM budget):
  sft  : loss = CE(answer | prompt)                       on the full sequence [prompt ; answer ; EOS]
  jepa : loss = CE(answer | prompt) + lam * (1 - cos(h_prompt_last, h_answer_last))
         CE from the normal causal pass (as the reference finetune.py: main pass on the full conversation);
         the two embeddings from a second pass where [prompt | answer] are packed with a block-causal mask
         (no cross-view attention), last-token final hidden state (after ln_f), k = 0 predictor tokens, no stop-grad.
Eval: greedy decoding on the first N test examples (fixed, pre-registered N), exact string match as in the reference
evaluate.py; plus mean next-token CE on the 12 "far" windows of E0 (general-loss regression check).
Guard: t4guard (free VRAM >= 4300 MiB, process cap, pause on in-flight inference, lock); hard deadline.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hivemind.swarm import crowdgpt_loss_check as cm  # noqa: E402  (engram-aware)
import t4guard  # noqa: E402

ROOT = Path("/mnt/disks/disk-socrateai-local-1/crowdgpt_exp")
HF = ROOT / "hf"
EOS = 151643  # <|endoftext|>
PROMPT = "Convert natural language to regular expression.\nInput: {q}\nRegex:"


# ------------------------------------------------------------------------------------------------------------ model

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.s = alpha / r

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.t()) @ self.B.t() * self.s).to(x.dtype)


def attn_forward(at, x, cos, sin, mask):
    b, t, c = x.size()
    q = at.wq(x).view(b, t, at.nh, at.hd).transpose(1, 2)
    k = at.wk(x).view(b, t, at.nkv, at.hd).transpose(1, 2)
    v = at.wv(x).view(b, t, at.nkv, at.hd).transpose(1, 2)
    ct, st = cos[:, :, :t, :], sin[:, :, :t, :]
    q = (q * ct + cm._rotate_half(q) * st).to(v.dtype)
    k = (k * ct + cm._rotate_half(k) * st).to(v.dtype)
    rep = at.nh // at.nkv
    k = k.unsqueeze(2).expand(b, at.nkv, rep, t, at.hd).reshape(b, at.nh, t, at.hd)
    v = v.unsqueeze(2).expand(b, at.nkv, rep, t, at.hd).reshape(b, at.nh, t, at.hd)
    if mask is None:
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return at.wo(o.transpose(1, 2).contiguous().view(b, t, c))


class Model:
    def __init__(self, r: int, alpha: float):
        m = cm.CrowdV1(cm.CROWD_V1_ENGRAM_CONFIG)
        m.load_flat(cm.load_flat_safetensors(HF / "model_bf16.safetensors").float(),
                    engram=cm.load_flat_safetensors(HF / "engram_bf16.safetensors").float())
        self.engram = m.engram.half().eval()  # CPU, frozen
        m.engram = None
        m = m.half().cuda().eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.lora = []
        for blk in m.blocks:
            for parent, names in ((blk.attn, ("wq", "wk", "wv", "wo")), (blk.mlp, ("w1", "w2", "w3"))):
                for n in names:
                    l = LoRALinear(getattr(parent, n), r, alpha)
                    setattr(parent, n, l)
                    self.lora += [l.A, l.B]
        self.m = m

    def hidden(self, idx, mask=None, ckpt=False, view_starts=None):
        """view_starts: [B] index where view 2 begins in a packed row; the engram bigram's previous token is reset to 0
        there, so view 2 gets exactly the engram input of a separate pass (no cross-view leak through the bigram)."""
        m = self.m
        cos, sin = cm._rope(int(m.config["headDim"]), idx.shape[1], idx.device)
        with torch.no_grad():
            ic = idx.cpu()
            prev = torch.cat((torch.zeros_like(ic[:, :1]), ic[:, :-1]), dim=1)
            if view_starts is not None:
                prev[torch.arange(ic.shape[0]), view_starts.cpu()] = 0
            rows = (prev * 1_000_003 + ic) % self.engram.table.num_embeddings
            e = self.engram.table(rows).cuda()
        x = m.wte(idx) + e
        for blk in m.blocks:
            def f(x, blk=blk):
                x = x + attn_forward(blk.attn, blk.ln_1(x), cos, sin, mask)
                return x + blk.mlp(blk.ln_2(x))
            x = torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False) if ckpt else f(x)
        return m.ln_f(x)

    def logits_at(self, h):
        w = self.m.lm_head.weight
        return F.linear(h.to(w.dtype), w).float()


# ------------------------------------------------------------------------------------------------------------- data

def load(path, tok, n=None):
    out = []
    for i, ln in enumerate(open(path)):
        if n is not None and i >= n:
            break
        msg = json.loads(ln)["messages"]
        q, a = msg[1]["content"], msg[2]["content"]
        p_ids = tok(PROMPT.format(q=q)).input_ids
        a_ids = tok(" " + a).input_ids
        q_ids = tok(q).input_ids  # JEPA view 1, as the reference (user message only, no instruction/template)
        out.append((p_ids, a_ids, a, q_ids))
    return out


def batch_sft(ex):
    T = max(len(e[0]) + len(e[1]) + 1 for e in ex)
    ids = torch.full((len(ex), T), EOS, dtype=torch.long)
    lab = torch.full((len(ex), T), -100, dtype=torch.long)
    for i, (p, a, *_r) in enumerate(ex):
        s = p + a + [EOS]
        ids[i, :len(s)] = torch.tensor(s)
        lab[i, len(p) - 1:len(s) - 1] = torch.tensor(a + [EOS])  # position t predicts token t+1
    return ids, lab


def batch_views(ex, k=0, pred_id=None):
    """[query | answer] packed, block-causal per example; returns ids, bool mask [B,1,T,T], idx_query_last, idx_ans_last,
    view-2 start positions (for the engram reset)."""
    T = max(len(e[3]) + k + len(e[1]) for e in ex)
    ids = torch.full((len(ex), T), EOS, dtype=torch.long)
    mask = torch.zeros(len(ex), 1, T, T, dtype=torch.bool)
    ip, ia, vs = [], [], []
    for i, (_p, a, _s, p) in enumerate(ex):
        p = p + [int(pred_id)] * k if k else p  # k predictor tokens after the query (paper: [PRED])
        n1, n2 = len(p), len(a)
        ids[i, :n1 + n2] = torch.tensor(p + a)
        mask[i, 0, :n1, :n1] = torch.ones(n1, n1, dtype=torch.bool).tril()
        mask[i, 0, n1:n1 + n2, n1:n1 + n2] = torch.ones(n2, n2, dtype=torch.bool).tril()
        ip.append(n1 - 1)
        ia.append(n1 + n2 - 1)
        vs.append(n1)
    mask[:, 0] |= torch.eye(T, dtype=torch.bool)  # padding rows attend to themselves (no all-False rows -> no NaN)
    return ids, mask, torch.tensor(ip), torch.tensor(ia), torch.tensor(vs)


def view_last(model, seqs):
    """One right-padded pass over standalone views; final hidden state at each sequence's last real token."""
    T = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), T), EOS, dtype=torch.long)
    am = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, x in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x); am[i, :len(x)] = 1
    h = model.hidden(ids.cuda(), ckpt=True, attention_mask=am.cuda())
    return h[torch.arange(len(seqs)), torch.tensor([len(x) - 1 for x in seqs])]


# ------------------------------------------------------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(model, test, bs=48, max_new=32, deadline=None):
    by_len = {}
    for i, (p, *_r) in enumerate(test):
        by_len.setdefault(len(p), []).append(i)
    pred = [None] * len(test)
    for L, idxs in sorted(by_len.items()):
        for j in range(0, len(idxs), bs):
            if deadline and time.time() > deadline:
                raise TimeoutError("deadline during eval")
            t4guard.wait_idle()
            sel = idxs[j:j + bs]
            seq = torch.tensor([test[i][0] for i in sel]).cuda()
            done = torch.zeros(len(sel), dtype=torch.bool, device="cuda")
            gen = []
            for _ in range(max_new):
                with torch.autocast("cuda", dtype=torch.float16):
                    h = model.hidden(seq)[:, -1]
                nxt = model.logits_at(h).argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
                gen.append(nxt)
                done |= nxt == EOS
                seq = torch.cat([seq, nxt[:, None]], 1)
                if bool(done.all()):
                    break
            g = torch.stack(gen, 1).tolist()
            for r, i in enumerate(sel):
                toks = g[r][:g[r].index(EOS)] if EOS in g[r] else g[r]
                pred[i] = toks
    return pred


@torch.no_grad()
def far_loss(model):
    toks = np.frombuffer((ROOT / "data/chunk_1026_200000000_10485760.bin").read_bytes(), dtype=np.uint32)
    starts = cm.heldout_windows(len(toks), 2048, 12, 20261113)
    ls = []
    for s in starts:
        t4guard.wait_idle()
        w = torch.from_numpy(toks[s:s + 2049].astype(np.int64))[None].cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            h = model.hidden(w[:, :-1])
        tot = 0.0
        for i in range(0, 2048, 256):
            tot += F.cross_entropy(model.logits_at(h[0, i:i + 256]), w[0, i + 1:i + 257], reduction="sum").item()
        ls.append(tot / 2048)
    return float(np.mean(ls))


# ------------------------------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["sft", "jepa"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8, help="micro-batch")
    ap.add_argument("--accum", type=int, default=2, help="micro-batches per optimizer step (effective batch = bs * accum)")
    ap.add_argument("--need-mib", type=float, default=3000, help="VRAM this run needs above the guard")
    ap.add_argument("--cap-mib", type=float, default=2700, help="hard per-process cap (worst case: inference 5.1 GB + ollama 2.4 GB + other jobs 0.75 GB + 4.3 GB floor)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--model", choices=["crowd", "qwen"], default="crowd")
    ap.add_argument("--k-pred", type=int, default=0, help="predictor tokens appended to the query view (paper: k)")
    ap.add_argument("--pred-id", type=int, default=151648, help="<|box_start|>: absent from Crowd data, in both vocabs")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=None, help="probe: stop after this many steps")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--deadline", type=float, default=None, help="unix time; abort cleanly after it")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    train = load(ROOT / "data/nlrx/synth_train.jsonl", tok, a.n_train)
    test = load(ROOT / "data/nlrx/synth_test.jsonl", tok, a.n_test)
    res = {"model": a.model, "k_pred": a.k_pred if a.arm == "jepa" else 0, "arm": a.arm, "seed": a.seed, "epochs": a.epochs, "bs": a.bs, "accum": a.accum, "lr": a.lr, "lam": a.lam if a.arm == "jepa" else 0.0,
           "rank": a.rank, "n_train": len(train), "n_test": len(test), "revision": "5ac3916", "status": "running"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    t4guard.DEADLINE = a.deadline
    with t4guard.GpuLock():
        res["waited_for_room_s"] = round(t4guard.wait_for_room(a.need_mib))
        res["process_cap_mib"] = round(t4guard.cap_process(max_cap_mib=a.cap_mib))
        if a.model == "qwen":
            import hf_model
            model = hf_model.QwenModel(a.rank, alpha=float(a.rank))
        else:
            model = Model(a.rank, alpha=float(a.rank))
        params = model.lora
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        steps = int(math.ceil(len(train) / (a.bs * a.accum)) * a.epochs)
        if a.max_steps:
            steps = min(steps, a.max_steps)
        warm = max(1, int(0.03 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
        scaler = torch.amp.GradScaler("cuda")
        order = []
        t0 = time.time(); log = []; step = 0; prev_train_s = 0.0
        ck = a.out.with_suffix(".ckpt.pt")

        def save_ckpt():
            tmp = ck.with_suffix(".tmp")
            torch.save({"lora": [x.detach().cpu() for x in params], "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "step": step, "order": order, "py_rng": random.getstate(),
                        "torch_rng": torch.get_rng_state(), "log": log, "train_s": prev_train_s + time.time() - t0}, tmp)
            tmp.replace(ck)

        if ck.exists():
            st = torch.load(ck, weights_only=False)
            with torch.no_grad():
                for x, v in zip(params, st["lora"]):
                    x.copy_(v.to(x.device))
            opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
            step, order, log, prev_train_s = st["step"], st["order"], st["log"], st["train_s"]
            random.setstate(st["py_rng"]); torch.set_rng_state(st["torch_rng"])
            res["resumed_from_step"] = res.get("resumed_from_step", []) + [step]
            print(json.dumps({"resumed_from_step": step}), flush=True)
        try:
          while step < steps:
              t4guard.wait_idle()
              if a.deadline and time.time() > a.deadline:
                  res["status"] = "deadline_train"; break
              opt.zero_grad(set_to_none=True)
              for _micro in range(a.accum):
                  if len(order) < a.bs:
                      order = list(range(len(train))); random.shuffle(order)
                  ex = [train[i] for i in order[:a.bs]]; order = order[a.bs:]
                  ids, lab = batch_sft(ex)
                  ids, lab = ids.cuda(), lab.cuda()
                  with torch.autocast("cuda", dtype=torch.float16):
                      h = model.hidden(ids, ckpt=True)
                      sel = lab != -100
                      ce = F.cross_entropy(model.logits_at(h[sel]), lab[sel])
                      loss = ce
                      j = None
                      if a.arm == "jepa":
                          r = torch.arange(len(ex))
                          if a.model == "qwen":  # separate passes (reference default mode), right-padded
                              eq = view_last(model, [e[3] + [a.pred_id] * a.k_pred for e in ex])
                              ea = view_last(model, [e[1] for e in ex])
                          else:
                              vids, vmask, ip, ia, vs = batch_views(ex, a.k_pred, a.pred_id)
                              hv = model.hidden(vids.cuda(), mask=vmask.cuda(), ckpt=True, view_starts=vs)
                              eq, ea = hv[r, ip], hv[r, ia]
                          j = 1.0 - F.cosine_similarity(eq.float(), ea.float(), dim=-1).mean()
                          loss = ce + a.lam * j
                  scaler.scale(loss / a.accum).backward()
              scaler.unscale_(opt)
              torch.nn.utils.clip_grad_norm_(params, 1.0)
              scaler.step(opt); scaler.update(); sched.step()
              step += 1
              t4guard.check()
              if step % 100 == 0:
                  save_ckpt()
              if step % 25 == 0 or step == steps:
                  log.append({"step": step, "ce": round(ce.item(), 4), "jepa": None if j is None else round(j.item(), 4),
                              "t": round(time.time() - t0, 1), "free_mib": round(t4guard.free_mib())})
                  print(json.dumps(log[-1]), flush=True)
        except t4guard.DeadlineReached:
            save_ckpt(); res["status"] = "deadline_train"; a.out.write_text(json.dumps(res, indent=1)); return 0
        except t4guard.YieldToTraining as e:
            save_ckpt(); res["status"] = "yield"; res["guard"] = str(e); a.out.write_text(json.dumps(res, indent=1))
            print(json.dumps({"yield_at_step": step, "why": str(e)}), flush=True)
            return 4
        except t4guard.GuardAbort as e:
            save_ckpt()
            res["status"] = "guard_abort"; res["guard"] = str(e); a.out.write_text(json.dumps(res, indent=1))
            print(json.dumps({"guard_abort_at_step": step, "why": str(e)}), flush=True)
            return 3
        if step >= steps:
            save_ckpt()
        res["train_seconds"] = round(prev_train_s + time.time() - t0, 1)
        res["steps"] = step
        res["sec_per_step"] = round(res["train_seconds"] / max(1, step), 3)
        res["peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        res["log"] = log
        a.out.write_text(json.dumps(res, indent=1))
        if a.skip_eval or res.get("status") == "deadline_train":
            res["status"] = res.get("status") if res.get("status") != "running" else "trained_no_eval"
            a.out.write_text(json.dumps(res, indent=1))
            return 0
        t1 = time.time()
        try:
            pred = evaluate(model, test, deadline=a.deadline)
        except (TimeoutError, t4guard.DeadlineReached):
            res["status"] = "deadline_eval"; a.out.write_text(json.dumps(res, indent=1)); return 0
        except t4guard.YieldToTraining as e:
            res["status"] = "yield_eval"; res["guard"] = str(e); a.out.write_text(json.dumps(res, indent=1)); return 4
        except t4guard.GuardAbort as e:  # training is checkpointed at its end: a rerun goes straight to eval
            res["status"] = "guard_abort_eval"; res["guard"] = str(e); a.out.write_text(json.dumps(res, indent=1)); return 3
        dec = [tok.decode(p, skip_special_tokens=True).strip() for p in pred]  # as reference evaluate.py
        correct = [d == t[2] for d, t in zip(dec, test)]
        res["accuracy"] = round(100.0 * sum(correct) / len(test), 2)
        # secondary, exploratory (added 23:33 after sft_s0; primary metric unchanged): prediction starts with the answer
        # (the paper used prefix matching when its model "struggled to terminate")
        res["accuracy_prefix"] = round(100.0 * sum(d.startswith(t[2]) for d, t in zip(dec, test)) / len(test), 2)
        res["predictions"] = dec
        res["eval_seconds"] = round(time.time() - t1, 1)
        res["samples"] = [{"gt": t[2], "pred": d} for d, t in list(zip(dec, test))[:20]]
        try:
            res["far_loss"] = round(far_loss(model), 4)
        except (t4guard.GuardAbort, t4guard.DeadlineReached) as e:
            res["far_loss"] = None; res["far_loss_error"] = str(e)
        res["min_free_mib"] = round(t4guard.free_mib())
        res["status"] = "done"
        a.out.write_text(json.dumps(res, indent=1))
        print("RESULT", json.dumps({k: res[k] for k in ("arm", "seed", "accuracy", "far_loss", "train_seconds", "eval_seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E0: what is Crowd-v1's real held-out loss? (pinned revision, T4, guarded)

Measures mean next-token CE (nats) on fixed 2048-token windows from three sources:
  seen     : chunk 1026 bytes [0, 10 MiB)       (pool_start chunk 1026 on 2026-11-12; slot-0 sub-slice starts at offset 0)
  far      : chunk 1026 at byte offset 200 MB   (likely not reached by a pool: slot * 10 MiB offsets)
  wikitext : wikitext-2-raw-v1 test, Qwen3 tokenizer (external text)
for:  Crowd-v1 fp32 / fp16 / bf16 with engram, fp16 without engram, and Qwen/Qwen3-0.6B (same tokenizer) as reference.
Writes results JSON. No training.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hivemind.swarm import crowdgpt_loss_check as cm  # noqa: E402  (engram-aware)
import t4guard  # noqa: E402

ROOT = Path("/mnt/disks/disk-socrateai-local-1/crowdgpt_exp")
HF = ROOT / "hf"
DATA = ROOT / "data"
WIN = 2048


def chunk_slice(idx: int, off: int, nbytes: int) -> np.ndarray:
    p = DATA / f"chunk_{idx:04d}_{off}_{nbytes}.bin"
    if not p.exists():
        import requests
        DATA.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/Vxtzq/CrowdGPT/resolve/main/chunks/chunk_{idx:04d}.bin"
        r = requests.get(url, headers={"Range": f"bytes={off}-{off + nbytes - 1}"}, timeout=300)
        r.raise_for_status()
        p.write_bytes(r.content)
    return np.frombuffer(p.read_bytes(), dtype=np.uint32)


def wikitext_tokens() -> np.ndarray:
    p = DATA / "wikitext2_test_qwen3.npy"
    if p.exists():
        return np.load(p)
    import pandas as pd
    from transformers import AutoTokenizer
    snap = next((Path.home() / ".cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots").iterdir())
    text = "".join(pd.read_parquet(snap / "wikitext-2-raw-v1/test-00000-of-00001.parquet")["text"].tolist())
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    ids = np.array(tok(text)["input_ids"], dtype=np.uint32)
    DATA.mkdir(parents=True, exist_ok=True)
    np.save(p, ids)
    return ids


def windows(tokens: np.ndarray, n: int, seed: int) -> list[np.ndarray]:
    starts = cm.heldout_windows(len(tokens), WIN, n, seed)
    return [tokens[s:s + WIN + 1].astype(np.int64) for s in starts]


def chunked_ce(h: torch.Tensor, head_w: torch.Tensor, tgt: torch.Tensor, chunk: int = 256) -> float:
    tot = 0.0
    for i in range(0, h.shape[1], chunk):
        lg = F.linear(h[:, i:i + chunk], head_w).float()
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt[:, i:i + chunk].reshape(-1), reduction="sum").item()
    return tot / tgt.numel()


class Crowd:
    def __init__(self, dtype: torch.dtype, engram: bool):
        cfg = cm.CROWD_V1_ENGRAM_CONFIG if engram else cm.CROWD_V1_CONFIG
        m = cm.CrowdV1(cfg)
        flat = cm.load_flat_safetensors(HF / "model_bf16.safetensors").float()
        eng = cm.load_flat_safetensors(HF / "engram_bf16.safetensors").float() if engram else None
        m.load_flat(flat, engram=eng)
        del flat, eng
        self.engram = m.engram  # stays on CPU (as in client.py)
        m.engram = None
        self.m = m.to("cuda", dtype=dtype).eval()
        if self.engram is not None:
            self.engram = self.engram.to(dtype)
        self.dtype = dtype

    @torch.no_grad()
    def loss(self, w: np.ndarray) -> float:
        x = torch.from_numpy(w[:-1])[None].cuda()
        y = torch.from_numpy(w[1:])[None].cuda()
        m = self.m
        cos, sin = cm._rope(int(m.config["headDim"]), x.shape[1], x.device)
        h = m.wte(x)
        if self.engram is not None:
            h = h + self.engram(x.cpu()).to("cuda", dtype=h.dtype)
        for blk in m.blocks:
            h = blk(h, cos, sin)
        h = m.ln_f(h)
        return chunked_ce(h, m.lm_head.weight, y)

    def close(self):
        del self.m, self.engram
        torch.cuda.empty_cache()


class Qwen:
    def __init__(self):
        from transformers import AutoModelForCausalLM
        self.m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def loss(self, w: np.ndarray) -> float:
        x = torch.from_numpy(w[:-1])[None].cuda()
        y = torch.from_numpy(w[1:])[None].cuda()
        h = self.m.model(input_ids=x).last_hidden_state
        return chunked_ce(h, self.m.lm_head.weight, y)

    def close(self):
        del self.m
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out", type=Path, default=ROOT / "results/e0_sanity.json")
    a = ap.parse_args()
    sets = {
        "seen_chunk1026_off0": windows(chunk_slice(1026, 0, 10 * 2**20), a.n, a.seed),
        "far_chunk1026_off200MB": windows(chunk_slice(1026, 200_000_000, 10 * 2**20), a.n, a.seed),
        "wikitext2_test": windows(wikitext_tokens(), a.n, a.seed),
    }
    models = [
        ("crowd_v1_fp16_engram", lambda: Crowd(torch.float16, True)),
        ("crowd_v1_fp16_no_engram", lambda: Crowd(torch.float16, False)),
        ("crowd_v1_fp32_engram", lambda: Crowd(torch.float32, True)),
        ("crowd_v1_bf16_engram", lambda: Crowd(torch.bfloat16, True)),
        ("qwen3_0.6b_fp16", Qwen),
    ]
    out = {"revision": "5ac3916611c13400bae7d953fae1752c01a1d1f2", "window": WIN, "n_windows": a.n, "seed": a.seed,
           "uniform_nats": math.log(151669), "results": {}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with t4guard.GpuLock():
        cap = t4guard.cap_process()
        out["process_cap_mib"] = round(cap)
        for name, ctor in models:
            t4guard.wait_idle()
            t0 = time.time()
            mdl = ctor()
            res = {}
            for sname, ws in sets.items():
                ls = []
                for w in ws:
                    t4guard.wait_idle()
                    ls.append(mdl.loss(w))
                res[sname] = {"mean": float(np.mean(ls)), "std": float(np.std(ls)), "per_window": [round(v, 4) for v in ls]}
            res["peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
            res["seconds"] = round(time.time() - t0, 1)
            res["min_free_mib_seen"] = round(t4guard.free_mib())
            mdl.close()
            torch.cuda.reset_peak_memory_stats()
            out["results"][name] = res
            print(name, {k: (round(v["mean"], 3) if isinstance(v, dict) else v) for k, v in res.items()}, flush=True)
            a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

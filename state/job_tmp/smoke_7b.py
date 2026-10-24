"""Smoke test: Qwen2.5-Coder-7B fp16 on the T4 (embeddings on CPU), chunked NLL, no-op attention hook."""
import glob, json, os, sys, time
import torch
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

hub = os.path.expanduser("~/.cache/huggingface/hub")
name = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-Coder-7B-Instruct"
path = glob.glob(f"{hub}/models--Qwen--{name}/snapshots/*")[0]
print("transformers", transformers.__version__, "RAM GB", os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, flush=True)
tok = AutoTokenizer.from_pretrained(path)
pq = glob.glob(f"{hub}/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet")[0]
text = "\n\n".join(pd.read_parquet(pq)["text"].tolist())
ids = tok(text, return_tensors="pt").input_ids
print("test tokens", ids.shape, flush=True)

big = "7B" in name
dm = {"model.embed_tokens": "cpu", "model.layers": 0, "model.norm": 0, "model.rotary_emb": 0, "lm_head": 0} if big else {"": 0}
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map=dm)
model.eval()
print("load s", round(time.time() - t0, 1), "alloc GiB", torch.cuda.memory_allocated() / 2**30, flush=True)
print("attn impl", model.config._attn_implementation, flush=True)

from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
calls = {"n": 0}

def kvq(module, query, key, value, attention_mask, **kw):
    calls["n"] += 1
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)

AttentionInterface.register("kvq", kvq)


@torch.no_grad()
def nll(window):
    x = window.unsqueeze(0)
    h = model.model(input_ids=x).last_hidden_state
    tot, n = 0.0, 0
    for s in range(0, h.shape[1] - 1, 256):
        e = min(s + 256, h.shape[1] - 1)
        lg = model.lm_head(h[:, s:e]).float()
        y = x[:, s + 1 : e + 1].to(lg.device)
        tot += torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="sum").item()
        n += e - s
    return tot / n


torch.cuda.reset_peak_memory_stats()
res = {}
for L in (1024, 2048):
    w = ids[0, :L]
    t0 = time.time()
    a = nll(w)
    res[L] = {"nll_sdpa": a, "ppl": float(torch.tensor(a).exp()), "s": round(time.time() - t0, 2), "peak_GiB": torch.cuda.max_memory_allocated() / 2**30}
    print(L, res[L], flush=True)
model.set_attn_implementation("kvq")
b = nll(ids[0, :1024])
print("hook calls", calls["n"], "nll_hook", b, "sdpa", res[1024]["nll_sdpa"], "exact_equal", b == res[1024]["nll_sdpa"], flush=True)
# fp32 key/value range per layer (overflow headroom check)
print(json.dumps({"ok": True}))

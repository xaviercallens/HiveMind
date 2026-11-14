"""P0: per-domain held-out loss, Crowd-v1 (rev 5ac3916 + engram) vs Qwen3-0.6B, same Qwen3 tokenizer. Inference only.

Documents of each domain are concatenated with <|endoftext|>, then 8 windows of 2048 tokens are drawn (fixed seed).
Report the GAP (Crowd - Qwen) per domain, not absolute losses: Qwen has likely seen all of these sources, and Crowd-v1 was
trained on fineweb, so fineweb-edu probably overlaps its training data. No French source is cached (not measured).
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import e0_heldout_sanity as e0  # noqa: E402
import t4guard  # noqa: E402

H = Path.home() / ".cache/huggingface/hub"
ROOT = Path("/mnt/disks/disk-socrateai-local-1/crowdgpt_exp")
N, SEED, EOS = 8, 20261114, 151643


def snap(name):
    return next((H / name / "snapshots").iterdir())


def texts():
    fw = pd.read_parquet(snap("datasets--HuggingFaceFW--fineweb-edu") / "sample/10BT/000_00000.parquet", columns=["text"])["text"][:400].tolist()
    code = []
    with gzip.open(next(snap("datasets--codeparrot--codeparrot-clean-valid").glob("*.json.gz")), "rt") as f:
        for line in f:
            code.append(json.loads(line)["content"])
            if len(code) >= 300:
                break
    nm = pd.read_parquet(snap("datasets--AI-MO--NuminaMath-CoT") / "data/test-00000-of-00001.parquet")
    math = (nm["problem"] + "\n\n" + nm["solution"]).tolist()[:300]
    sq = pd.read_parquet(snap("datasets--allenai--sciq") / "data/test-00000-of-00001.parquet")
    sci = (sq["support"] + "\n\nQuestion: " + sq["question"] + "\nAnswer: " + sq["correct_answer"]).tolist()
    return {"fineweb_edu (web, educational)": fw, "code_python (codeparrot)": code, "math (NuminaMath-CoT)": math,
            "science (SciQ)": sci}


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    sets = {}
    for name, docs in texts().items():
        ids = []
        for d in docs:
            ids += tok(d).input_ids + [EOS]
        sets[name] = e0.windows(np.array(ids, dtype=np.uint32), N, SEED)
    sets["wikitext2 (encyclopedic)"] = e0.windows(e0.wikitext_tokens(), N, SEED)
    sets["crowd_chunk1026_far (own data)"] = e0.windows(e0.chunk_slice(1026, 200_000_000, 10 * 2**20), N, SEED)
    out = {"revision": "5ac3916", "n_windows": N, "window": 2048, "results": {}}
    with t4guard.GpuLock():
        t4guard.wait_for_room(2800)
        t4guard.cap_process(max_cap_mib=2700)
        for mname, ctor in (("crowd_v1", lambda: e0.Crowd(e0.torch.float16, True)), ("qwen3_0.6b", e0.Qwen)):
            m = ctor()
            for sname, ws in sets.items():
                ls = []
                for w in ws:
                    t4guard.wait_idle()
                    ls.append(m.loss(w))
                out["results"].setdefault(sname, {})[mname] = round(float(np.mean(ls)), 4)
            m.close()
    for s, r in out["results"].items():
        r["gap"] = round(r["crowd_v1"] - r["qwen3_0.6b"], 4)
    p = ROOT / "results/p0_domains.json"
    p.write_text(json.dumps(out, indent=1))
    for s, r in sorted(out["results"].items(), key=lambda kv: -kv[1]["gap"]):
        print(f"{s:36s} crowd {r['crowd_v1']:.3f}  qwen {r['qwen3_0.6b']:.3f}  gap {r['gap']:+.3f}")


if __name__ == "__main__":
    main()

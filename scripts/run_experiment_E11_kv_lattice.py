"""Experiment E11: lattice KV-cache codecs vs FP8 / scalar at matched measured bits on real models (RQ10, HM-S10-05).

Pre-registered in ``artifacts/experiments/E11_kv_lattice_prereg.json``. One model per process (GPU memory):

    python3 scripts/run_experiment_E11_kv_lattice.py --model qwen0.5b
    python3 scripts/run_experiment_E11_kv_lattice.py --model qwen7b
    python3 scripts/run_experiment_E11_kv_lattice.py --model gemma4-e4b
    python3 scripts/run_experiment_E11_kv_lattice.py --aggregate

Keys and values are intercepted where the stock SDPA attention receives them (post-RoPE, before ``repeat_kv``,
i.e. what the cache stores): the registered ``sdpa`` attention function is wrapped in place, so mask construction
and every other code path are the stock ones. The FP8 arm calls ``FP8PagedKVCache._quantize_page_fp8`` unchanged;
the lattice arms use ``hivemind.compress.kv_lattice`` (torch ports of the OptimalEnergyAI decoders + A₂).

Built for a 16 GB T4: the 7B Qwen runs in fp16 with only its input-embedding table on the CPU; Gemma-4-E4B runs
in bf16 with its 5.25 GiB per-layer-embedding table on the CPU; the logits are computed in 256-token chunks.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hivemind.compress.kv_lattice import BLOCK, DECODERS_T, is_a2_point, lattice_quantize, nearest_a2_t  # noqa: E402
from hivemind.compress.lattice_quantizer import nearest_e8, nearest_zn  # noqa: E402
from hivemind.core.btm_moe import FP8PagedKVCache  # noqa: E402

EXP_DIR = PROJECT_ROOT / "artifacts" / "experiments"
PREREG_PATH = EXP_DIR / "E11_kv_lattice_prereg.json"
REPORT_PATH = EXP_DIR / "E11_kv_lattice_report.json"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))

WINDOW = 1024
WINDOWS_PER_SEED = 8
SEEDS = (1, 2, 3)
RATES = (2, 3, 4)
LATTICES = ("Z", "A2", "E8")
TRANSFORMS = ("none", "klt")
CALIB_VECTORS_PER_LAYER = 4096  # (token, head) vectors per layer in the calibration sample
LOGIT_CHUNK = 256

MODELS: Dict[str, Dict[str, str]] = {
    "qwen0.5b": {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "float16"},
    "qwen7b": {"repo": "Qwen/Qwen2.5-Coder-7B-Instruct", "dtype": "float16"},
    "gemma4-e4b": {"repo": "google/gemma-4-E4B", "dtype": "bfloat16"},
}


ACTIVE_TRANSFORMS: Tuple[str, ...] = TRANSFORMS  # main() adds "kltc" for the exploratory E11b configurations


def cell_names() -> List[str]:
    cells = ["none", "fp8"] + [f"int@{b}" for b in RATES]
    cells += [f"{k}@{r}|{t}" for k in LATTICES for t in ACTIVE_TRANSFORMS for r in RATES]
    return cells + ["A2@8|none"]


# --------------------------------------------------------------------------- model wrapper


class CPUEmbedding(nn.Module):
    """Keeps a large embedding table in host RAM; looks up on the CPU and returns the rows on the GPU."""

    def __init__(self, emb: nn.Module, device: torch.device) -> None:
        super().__init__()
        self.emb = emb
        self.out_device = device

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(input_ids.to("cpu")).to(self.out_device)


class LoadedModel:
    def __init__(self, key: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        spec = MODELS[key]
        self.key = key
        snap = sorted(glob.glob(str(HUB / ("models--" + spec["repo"].replace("/", "--")) / "snapshots" / "*")))
        if not snap:
            raise FileNotFoundError(spec["repo"])
        self.path = snap[0]
        self.snapshot = Path(self.path).name
        self.dtype = getattr(torch, spec["dtype"])
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.dev = torch.device("cuda:0")
        self.bos: Optional[int] = None
        self.softcap: Optional[float] = None
        self.kv_source: Dict[int, int] = {}  # layer -> layer whose K/V it reads (Gemma KV sharing)
        if key.startswith("qwen"):
            dm = {"": 0} if key == "qwen0.5b" else {"model.embed_tokens": "cpu", "model.layers": 0, "model.norm": 0, "model.rotary_emb": 0, "lm_head": 0}
            m = AutoModelForCausalLM.from_pretrained(self.path, dtype=self.dtype, device_map=dm, attn_implementation="sdpa")
            self.body, self.head = m.model, m.lm_head
            n = m.config.num_hidden_layers
            self.kv_source = {i: i for i in range(n)}
            self.model = m
        else:
            from transformers import AutoModelForImageTextToText

            m = AutoModelForImageTextToText.from_pretrained(self.path, dtype=self.dtype, device_map={"": "cpu"}, attn_implementation="sdpa")
            lm = m.model.language_model
            for name in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
                if hasattr(m.model, name):
                    setattr(m.model, name, None)
            ple = lm.embed_tokens_per_layer
            lm.embed_tokens_per_layer = None
            lm.to(self.dev)
            lm.embed_tokens_per_layer = CPUEmbedding(ple, self.dev)
            self.body, self.head = lm, m.lm_head.to(self.dev)
            cfg = lm.config
            self.softcap = getattr(cfg, "final_logit_softcapping", None)
            self.bos = self.tok.bos_token_id
            n = cfg.num_hidden_layers
            first_shared = n - getattr(cfg, "num_kv_shared_layers", 0)
            for i in range(n):
                if i < first_shared:
                    self.kv_source[i] = i
                else:
                    t = cfg.layer_types[i]
                    self.kv_source[i] = max(j for j in range(first_shared) if cfg.layer_types[j] == t)
            self.model = m
        self.model.eval()
        self.num_layers = n

    def tokens(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    @torch.no_grad()
    def nll_sum(self, window: torch.Tensor) -> Tuple[float, int]:
        """Sum of next-token NLL over the window's 1023 predictions (a BOS, if the model has one, is context only)."""
        x = window.unsqueeze(0)
        off = 0
        if self.bos is not None:
            x = torch.cat([torch.tensor([[self.bos]]), x], dim=1)
            off = 1
        x = x.to(self.dev)
        h = self.body(input_ids=x, use_cache=False).last_hidden_state
        tot, n = 0.0, 0
        for s in range(off, h.shape[1] - 1, LOGIT_CHUNK):
            e = min(s + LOGIT_CHUNK, h.shape[1] - 1)
            lg = self.head(h[:, s:e]).float()
            if self.softcap:
                lg = torch.tanh(lg / self.softcap) * self.softcap
            y = x[:, s + 1 : e + 1]
            tot += float(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="sum"))
            n += e - s
        return tot, n


# --------------------------------------------------------------------------- KV hook and codecs


class KVState:
    def __init__(self) -> None:
        self.codec: Optional[Callable[[int, int, torch.Tensor], torch.Tensor]] = None  # (layer, 0=K/1=V, x) -> x_hat
        self.capture: Optional[Dict[Tuple[int, int], List[torch.Tensor]]] = None
        self.stats: Optional[Dict[str, List[float]]] = None  # filled when a codec measures rate/MSE
        self.kv_source: Dict[int, int] = {}


STATE = KVState()


def install_hook() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    orig = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(orig, "_e11_wrapped", False):
        return

    def kv_hook(module, query, key, value, attention_mask, **kw):
        layer = int(getattr(module, "layer_idx", -1))
        if STATE.capture is not None and STATE.kv_source.get(layer, layer) == layer:
            STATE.capture.setdefault((layer, 0), []).append(key.detach().float().cpu())
            STATE.capture.setdefault((layer, 1), []).append(value.detach().float().cpu())
        if STATE.codec is not None:
            key = STATE.codec(layer, 0, key)
            value = STATE.codec(layer, 1, value)
        return orig(module, query, key, value, attention_mask, **kw)

    kv_hook._e11_wrapped = True  # type: ignore[attr-defined]
    ALL_ATTENTION_FUNCTIONS["sdpa"] = kv_hook


def fp8_codec(x: torch.Tensor) -> torch.Tensor:
    q, s = FP8PagedKVCache._quantize_page_fp8(x)
    return FP8PagedKVCache._dequantize_page_fp8(q, s, target_dtype=x.dtype)


def int_codec(x: torch.Tensor, bits: int) -> torch.Tensor:
    xf = x.float()
    mn = xf.amin(dim=-1, keepdim=True).half().float()
    mx = xf.amax(dim=-1, keepdim=True).half().float()
    levels = 2**bits - 1
    step = ((mx - mn) / levels).clamp_min(1e-8)
    q = torch.clamp(torch.round((xf - mn) / step), 0, levels)
    return (q * step + mn).to(x.dtype)


def _rel_mse(x: torch.Tensor, xh: torch.Tensor) -> float:
    xf = x.double()
    return float(((xf - xh.double()) ** 2).sum() / (xf**2).sum().clamp_min(1e-30))


class LatticeCodec:
    """RMS-normalise each (token, head) vector (fp16 side value), optional per-(layer, head) KLT, quantise on scale·L."""

    def __init__(self, kind: str, transform: str, scales: Tuple[float, float], klt: Dict[Tuple[int, int], Dict[str, torch.Tensor]]) -> None:
        self.kind, self.transform, self.scales, self.klt = kind, transform, scales, klt

    def __call__(self, layer: int, kv: int, x: torch.Tensor) -> torch.Tensor:
        tbl = self.klt.get((STATE.kv_source.get(layer, layer), kv))
        xf = transform_fwd(self.transform, tbl, x.double())
        rms = xf.pow(2).mean(dim=-1, keepdim=True).sqrt().half().double().clamp_min(1e-12)
        measure = STATE.stats is not None and STATE.kv_source.get(layer, layer) == layer
        q, rate = lattice_quantize(self.kind, xf / rms, self.scales[kv], measure_rate=measure)
        xh = transform_inv(self.transform, tbl, q * rms)
        if measure:
            STATE.stats.setdefault("rate", []).append(rate)
            STATE.stats.setdefault("coords", []).append(float(x.numel()))
            STATE.stats.setdefault("rel_mse", []).append(_rel_mse(x, xh))
        return xh.to(x.dtype)


class ScalarCodec:
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.fn = fn

    def __call__(self, layer: int, kv: int, x: torch.Tensor) -> torch.Tensor:
        xh = self.fn(x)
        if STATE.stats is not None and STATE.kv_source.get(layer, layer) == layer:
            STATE.stats.setdefault("coords", []).append(float(x.numel()))
            STATE.stats.setdefault("rel_mse", []).append(_rel_mse(x, xh))
        return xh


# --------------------------------------------------------------------------- calibration


def transform_fwd(transform: str, tbl: Optional[Dict[str, torch.Tensor]], x: torch.Tensor) -> torch.Tensor:
    """none | klt (registered E11) | kltc (exploratory E11b: subtract the calibrated per-channel mean, then KLT).

    The mean is a fixed per-(layer, head) table, not per-token side information."""
    if transform == "none":
        return x
    if transform == "kltc":
        x = x - tbl["mu"].unsqueeze(1)
    return torch.einsum("bhtd,hde->bhte", x, tbl["u"])


def transform_inv(transform: str, tbl: Optional[Dict[str, torch.Tensor]], y: torch.Tensor) -> torch.Tensor:
    if transform == "none":
        return y
    x = torch.einsum("bhte,hde->bhtd", y, tbl["u"])
    return x + tbl["mu"].unsqueeze(1) if transform == "kltc" else x


def klt_bases(capture: Dict[Tuple[int, int], List[torch.Tensor]]) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
    out = {}
    for key, chunks in capture.items():
        x = torch.cat(chunks, dim=2).double()  # (1, H, T, d)
        bases, means = [], []
        for h in range(x.shape[1]):
            v = x[0, h]
            mu = v.mean(dim=0, keepdim=True)
            v = v - mu
            cov = v.T @ v / max(1, v.shape[0] - 1)
            evals, evecs = torch.linalg.eigh(cov)
            bases.append(evecs[:, torch.argsort(evals, descending=True)])
            means.append(mu[0].half().double())  # stored as an fp16 table
        out[key] = {"u": torch.stack(bases), "mu": torch.stack(means)}  # u: (H, d, d) columns = eigenvectors; mu: (H, d)
    return out


def pooled_rows(capture, klt, kv: int, transform: str, kind: str, gen: torch.Generator) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    """Per non-shared layer: (lattice rows, in-head group index, coordinate weight) of the normalised (and transformed)
    calibration vectors, at most CALIB_VECTORS_PER_LAYER vectors per layer.

    The rate is measured per layer and averaged with coordinate weights, exactly as in evaluation (a coder keeps
    per-layer tables). The first execution pooled all layers into one histogram; the entropy of that mixture
    exceeds the mean per-layer entropy, so its scales realised ~0.3 bit/dim less than the target in evaluation
    (kept as E11_first_execution_qwen0.5b.json)."""
    from hivemind.compress.kv_lattice import to_rows

    out = []
    for (layer, k), chunks in sorted(capture.items()):
        if k != kv:
            continue
        x = transform_fwd(transform, klt.get((layer, kv)), torch.cat(chunks, dim=2).double())
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().half().double().clamp_min(1e-12)
        v = (x / rms).reshape(-1, x.shape[-1])
        if len(v) > CALIB_VECTORS_PER_LAYER:
            v = v[torch.randperm(len(v), generator=gen)[:CALIB_VECTORS_PER_LAYER]]
        r, c = to_rows(kind, v)
        out.append((r, c, float(x.numel())))
    return out


def calibrate_scale(kind: str, layers: List[Tuple[torch.Tensor, torch.Tensor, float]], target: float, tol: float = 0.02, iters: int = 60) -> Tuple[float, float]:
    from hivemind.compress.kv_lattice import factorised_rate_t

    layers = [(r.cuda(), c.cuda(), w) for r, c, w in layers]
    total = sum(w for _, _, w in layers)

    def rate_at(s: float) -> float:
        return sum(w * factorised_rate_t(kind, DECODERS_T[kind](r / s), c) for r, c, w in layers) / total

    lo, hi = math.log(1e-3), math.log(1e2)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = rate_at(math.exp(mid))
        if abs(r - target) <= tol:
            return math.exp(mid), r
        if r > target:
            lo = mid
        else:
            hi = mid
    s = math.exp(0.5 * (lo + hi))
    return s, rate_at(s)


# --------------------------------------------------------------------------- controls


def decoder_controls() -> Dict[str, Any]:
    rng = np.random.default_rng(2026)
    q = rng.normal(scale=2.0, size=(20000, 8))
    t = rng.integers(-3, 4, size=(5000, 8)).astype(float)
    t[:, ::2] += 0.5
    q = np.concatenate([q, t])
    xt = torch.from_numpy(q).cuda()
    z_ok = bool(np.array_equal(DECODERS_T["Z"](xt).cpu().numpy(), nearest_zn(q)))
    e8_ok = bool(np.array_equal(DECODERS_T["E8"](xt).cpu().numpy(), nearest_e8(q)))
    a = rng.uniform(-3.0, 3.0, size=(5000, 2))
    c = nearest_a2_t(torch.from_numpy(a).cuda()).cpu()
    member = bool(is_a2_point(c).all())
    s3 = math.sqrt(3.0)
    pts = np.array([(i + 0.5 * j, 0.5 * s3 * j) for i, j in itertools.product(range(-16, 17), repeat=2)])
    pts = pts[np.einsum("ij,ij->i", pts, pts) <= 36.0]
    d_min = ((a[:, None, :] - pts[None]) ** 2).sum(-1).min(axis=1)
    mism = int(np.sum(np.abs(((a - c.numpy()) ** 2).sum(-1) - d_min) > 1e-9))
    return {"Z_cuda_equals_numpy": z_ok, "E8_cuda_equals_numpy": e8_ok, "n_queries_8d": len(q), "A2_members": member, "A2_mismatches": mism,
            "n_queries_A2": len(a), "passed": bool(z_ok and e8_ok and member and mism == 0)}


# --------------------------------------------------------------------------- run one model


def _hub_file(repo_dir: str, pattern: str) -> str:
    hits = sorted(glob.glob(str(HUB / repo_dir / "snapshots" / "*" / pattern)))
    if not hits:
        raise FileNotFoundError(f"{repo_dir}/{pattern} not in the HF cache")
    return hits[0]


def dataset_texts(name: str) -> Tuple[str, str]:
    """(calibration text, evaluation text), disjoint. 'wikitext2' is the pre-registered E11 data; the others are E11b."""
    import pandas as pd

    if name == "wikitext2":
        f = lambda s: "\n\n".join(pd.read_parquet(_hub_file("datasets--Salesforce--wikitext", f"wikitext-2-raw-v1/{s}-*.parquet"))["text"].tolist())  # noqa: E731
        return f("validation"), f("test")
    if name == "fineweb-edu":
        docs = pd.read_parquet(_hub_file("datasets--HuggingFaceFW--fineweb-edu", "sample/10BT/000_00000.parquet"), columns=["text"])["text"]
        return "\n\n".join(docs.iloc[:40].tolist()), "\n\n".join(docs.iloc[1000:1600].tolist())
    if name == "gsm8k":
        g = lambda s: "\n\n".join((q + "\n" + a) for q, a in pd.read_parquet(_hub_file("datasets--openai--gsm8k", f"main/{s}-*.parquet"))[["question", "answer"]].itertuples(index=False))  # noqa: E731
        return g("train")[:60000], g("test")
    if name == "code":
        import gzip

        rows = [json.loads(line) for line in gzip.open(_hub_file("datasets--codeparrot--codeparrot-clean-valid", "file-000000000054.json.gz"), "rt")]
        code = [r["content"] for r in rows if r.get("content")]
        return "\n\n".join(code[:30]), "\n\n".join(code[200:900])
    # second E11b wave (2026-10-24): news, narrative fiction, biomedical science, French
    if name == "news":
        cal = pd.read_parquet(_hub_file("datasets--abisee--cnn_dailymail", "3.0.0/validation-*.parquet"), columns=["article"])["article"]
        ev = pd.read_parquet(_hub_file("datasets--abisee--cnn_dailymail", "3.0.0/test-*.parquet"), columns=["article"])["article"]
        return "\n\n".join(cal.iloc[:40].tolist()), "\n\n".join(ev.iloc[:700].tolist())
    if name == "fiction":
        txt = pd.read_parquet(_hub_file("datasets--EleutherAI--lambada_openai", "en/test/en.parquet"), columns=["text"])["text"]
        return "\n\n".join(txt.iloc[:300].tolist()), "\n\n".join(txt.iloc[1000:4500].tolist())
    if name == "pubmed":
        cal = pd.read_parquet(_hub_file("datasets--ccdv--pubmed-summarization", "section/validation-*.parquet"), columns=["article"])["article"]
        ev = pd.read_parquet(_hub_file("datasets--ccdv--pubmed-summarization", "section/test-*.parquet"), columns=["article"])["article"]
        return "\n\n".join(cal.iloc[:10].tolist()), "\n\n".join(ev.iloc[:160].tolist())
    if name == "wiki-fr":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(_hub_file("datasets--wikimedia--wikipedia", "20231101.fr/train-00000-of-00017.parquet"))
        arts = pf.read_row_group(0, columns=["text"]).column("text").to_pylist()
        return "\n\n".join(arts[:20]), "\n\n".join(arts[100:250])
    raise KeyError(name)


def window_indices(n_windows: int, seed: int) -> List[int]:
    return [int((3 * j + seed - 1) * n_windows // 24) for j in range(WINDOWS_PER_SEED)]


def make_codec(cell: str, scales: Dict[str, Tuple[float, float]], klt) -> Optional[Callable]:
    if cell == "none":
        return None
    if cell == "fp8":
        return ScalarCodec(fp8_codec)
    if cell.startswith("int@"):
        b = int(cell.split("@")[1])
        return ScalarCodec(lambda x, b=b: int_codec(x, b))
    kind_rate, transform = cell.split("|")
    kind = kind_rate.split("@")[0]
    return LatticeCodec(kind, transform, scales[cell], klt)


def run_model(key: str, dataset: str = "wikitext2") -> Dict[str, Any]:
    started = time.perf_counter()
    torch.manual_seed(0)
    lm = LoadedModel(key)
    STATE.kv_source = lm.kv_source
    load_s = round(time.perf_counter() - started, 1)
    print(f"[{key}] loaded in {load_s}s; GPU alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    calib_text, eval_text = dataset_texts(dataset)
    test = lm.tokens(eval_text)
    val = lm.tokens(calib_text)
    if len(val) < 2 * WINDOW:
        raise ValueError(f"calibration text of {dataset} has {len(val)} tokens < 2 windows of {WINDOW}")
    n_windows = len(test) // WINDOW
    if n_windows < 24:
        raise ValueError(f"evaluation text of {dataset} has only {n_windows} windows of {WINDOW}")
    win = lambda i: test[i * WINDOW : (i + 1) * WINDOW]  # noqa: E731

    # hook identity: stock SDPA NLL of evaluation window 0 before the wrapper exists, then through it with no codec
    w0 = window_indices(n_windows, 1)[0]
    stock = lm.nll_sum(win(w0))
    install_hook()
    hooked = lm.nll_sum(win(w0))
    hook_identity = {"stock_nll_sum": stock[0], "hooked_nll_sum": hooked[0], "bitwise_equal": stock[0] == hooked[0], "passed": stock[0] == hooked[0]}
    print(f"[{key}] hook identity {hook_identity}", flush=True)

    # calibration capture on validation windows 0 and 1
    STATE.capture = {}
    for i in range(2):
        lm.nll_sum(val[i * WINDOW : (i + 1) * WINDOW])
    capture = STATE.capture
    STATE.capture = None
    klt = klt_bases(capture)
    klt_dev = {k: {n: t.to(lm.dev) for n, t in v.items()} for k, v in klt.items()}
    gen = torch.Generator().manual_seed(2026)
    scales: Dict[str, Tuple[float, float]] = {}
    calibration: Dict[str, Any] = {}
    for kind, transform in itertools.product(LATTICES, ACTIVE_TRANSFORMS):
        rows_kv = [pooled_rows(capture, klt, kv, transform, kind, gen) for kv in (0, 1)]
        for r in RATES + ((8,) if kind == "A2" and transform == "none" else ()):
            cell = f"{kind}@{r}|{transform}"
            sk, rk = calibrate_scale(kind, rows_kv[0], float(r))
            sv, rv = calibrate_scale(kind, rows_kv[1], float(r))
            scales[cell] = (sk, sv)
            calibration[cell] = {"scale_K": sk, "scale_V": sv, "rate_K_on_calibration": rk, "rate_V_on_calibration": rv}
        del rows_kv
        torch.cuda.empty_cache()
    del capture
    print(f"[{key}] calibrated {len(scales)} lattice cells", flush=True)

    head_dims = sorted({int(t["u"].shape[-1]) for t in klt.values()})
    cells = cell_names()
    runs: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for wi, w in enumerate(window_indices(n_windows, seed)):
            x = win(w)
            for cell in cells:
                STATE.codec = make_codec(cell, scales, klt_dev)
                STATE.stats = {} if wi == 0 and cell != "none" else None
                t0 = time.perf_counter()
                tot, n = lm.nll_sum(x)
                row = {"seed": seed, "window": w, "cell": cell, "nll_sum": tot, "n_tokens": n, "seconds": round(time.perf_counter() - t0, 3)}
                if STATE.stats:
                    st = STATE.stats
                    wts = st["coords"]
                    row["rel_mse"] = float(np.average(st["rel_mse"], weights=wts))
                    if "rate" in st:
                        row["rate_bits_per_dim"] = float(np.average(st["rate"], weights=wts))
                runs.append(row)
                STATE.codec, STATE.stats = None, None
            base = next(r for r in runs if r["seed"] == seed and r["window"] == w and r["cell"] == "none")
            print(f"[{key}] seed {seed} window {w}: base NLL {base['nll_sum'] / base['n_tokens']:.4f}", flush=True)

    return {
        "model": key,
        "dataset": dataset,
        "window": WINDOW,
        "windows_per_seed": WINDOWS_PER_SEED,
        "repo": MODELS[key]["repo"],
        "snapshot": lm.snapshot,
        "dtype": MODELS[key]["dtype"],
        "num_layers": lm.num_layers,
        "kv_shared_layers": sum(1 for i, s in lm.kv_source.items() if i != s),
        "head_dims": head_dims,
        "bos_prefixed": lm.bos is not None,
        "n_test_tokens": int(len(test)),
        "n_windows": n_windows,
        "window_indices": {str(s): window_indices(n_windows, s) for s in SEEDS},
        "calibration": calibration,
        "controls": {"hook_identity": hook_identity, "decoders": decoder_controls()},
        "runs": runs,
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "load_seconds": load_s,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }


# --------------------------------------------------------------------------- aggregate and decide


def mean_ci(xs: List[float]) -> Tuple[float, float]:
    """Mean and 95% t half-width with df = 2 (same rule as run_experiment_E5_toy.mean_ci, inlined because importing
    that script pulls the swarm package and its optional QUIC dependency)."""
    vals = [float(v) for v in xs]
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return m, 4.302652729911275 * sd / math.sqrt(len(vals))


def per_seed_dnll(raw: Dict[str, Any], cell: str) -> List[float]:
    out = []
    for s in SEEDS:
        rows = [r for r in raw["runs"] if r["seed"] == s]
        c = [r for r in rows if r["cell"] == cell]
        b = [r for r in rows if r["cell"] == "none"]
        out.append(sum(r["nll_sum"] for r in c) / sum(r["n_tokens"] for r in c) - sum(r["nll_sum"] for r in b) / sum(r["n_tokens"] for r in b))
    return out


def decide_model(raw: Dict[str, Any]) -> Dict[str, Any]:
    cells = {}
    present = list(dict.fromkeys(r["cell"] for r in raw["runs"]))
    for cell in present[1:]:
        d = per_seed_dnll(raw, cell)
        m, h = mean_ci(d)
        rows = [r for r in raw["runs"] if r["cell"] == cell and "rel_mse" in r]
        rates = [r["rate_bits_per_dim"] for r in rows if "rate_bits_per_dim" in r]
        cells[cell] = {"dnll_mean": m, "dnll_ci95": h, "dnll_per_seed": d, "rel_mse": statistics.fmean(r["rel_mse"] for r in rows) if rows else None,
                       "rate_bits_per_dim": statistics.fmean(rates) if rates else None}
    base = [r for r in raw["runs"] if r["cell"] == "none"]
    base_nll = sum(r["nll_sum"] for r in base) / sum(r["n_tokens"] for r in base)

    def matched(cell: str, target: float) -> bool:
        r = cells[cell]["rate_bits_per_dim"]
        return r is not None and abs(r - target) <= 0.1

    for cell, c in cells.items():
        if "@" in cell and "|" in cell:
            c["rate_matched"] = matched(cell, float(cell.split("@")[1].split("|")[0]))

    def diff(a: str, b: str) -> Tuple[float, float, List[float]]:
        d = [x - y for x, y in zip(cells[a]["dnll_per_seed"], cells[b]["dnll_per_seed"])]
        m, h = mean_ci(d)
        return m, h, d

    # P11a
    p11a_cells = {}
    gains, losses, nulls, valid = 0, 0, 0, 0
    for r in RATES:
        z, a2 = f"Z@{r}|none", f"A2@{r}|none"
        ok = cells[z]["rate_matched"] and cells[a2]["rate_matched"]
        m, h, d = diff(z, a2)
        st = "invalid (rate not matched)" if not ok else ("A2 better" if m - h > 0 else ("A2 worse" if m + h < 0 else "CI includes 0"))
        p11a_cells[f"R={r}"] = {"z_minus_a2_mean": m, "ci95": h, "per_seed": d, "status": st}
        if ok:
            valid += 1
            gains += st == "A2 better"
            losses += st == "A2 worse"
            nulls += st == "CI includes 0"
    if valid == 0:
        p11a = "invalid (no matched rate)"
    elif gains == valid:
        p11a = "supported"
    elif losses > 0 or nulls == valid:
        p11a = "refuted"
    else:
        p11a = "inconclusive"

    # P11b
    p11b_cells = {}
    none_beats, klt_beats, klt_null, valid_b = 0, 0, 0, 0
    for r in RATES:
        for t in TRANSFORMS:
            z, e8 = f"Z@{r}|{t}", f"E8@{r}|{t}"
            ok = cells[z]["rate_matched"] and cells[e8]["rate_matched"]
            m, h, d = diff(z, e8)
            st = "invalid (rate not matched)" if not ok else ("E8 better" if m - h > 0 else ("E8 worse" if m + h < 0 else "CI includes 0"))
            p11b_cells[f"R={r}|{t}"] = {"z_minus_e8_mean": m, "ci95": h, "per_seed": d, "status": st}
            if ok and t == "none":
                none_beats += st == "E8 better"
            if ok and t == "klt":
                valid_b += 1
                klt_beats += st == "E8 better"
                klt_null += st != "E8 better"
    # the registered refutation clause 'E8 beats Z without the transform at some matched R' is decidable from the
    # rate-matched 'none' cells alone, so it is applied even when the KLT cells are not rate-matched
    if none_beats > 0:
        p11b = "refuted (E8 already beats Z without the transform)"
    elif valid_b == 0:
        p11b = "invalid (no matched rate)"
    elif klt_beats == valid_b:
        p11b = "supported"
    elif klt_null == valid_b:
        p11b = "refuted (the transform does not make E8 beat Z)"
    else:
        p11b = "inconclusive"

    # controls
    pos = cells["int@2"]
    pos_ok = pos["dnll_mean"] - pos["dnll_ci95"] > 0
    m, h, d = diff("A2@8|none", "fp8")
    neg1_ok = (abs(m) <= h) or abs(m) <= 0.002
    hook_ok = raw["controls"]["hook_identity"]["passed"]
    dec_ok = raw["controls"]["decoders"]["passed"]
    base_ok = math.isfinite(base_nll)
    # descriptive frontier: does a 4-bit lattice reach FP8 at ~half its bytes?
    frontier = {}
    for c4 in ("A2@4|none", "E8@4|klt", "Z@4|none", "int@4"):
        fm, fh, fd = diff(c4, "fp8")
        frontier[c4] = {"minus_fp8_mean": fm, "ci95": fh, "per_seed": fd, "reaches_fp8": abs(fm) <= fh}
    valid_model = bool(base_ok and hook_ok)
    return {
        "baseline_nll": base_nll,
        "baseline_ppl": math.exp(base_nll),
        "cells": cells,
        "P11a": {"cells": p11a_cells, "status": p11a if valid_model else "model invalid"},
        "P11b": {"cells": p11b_cells, "status": p11b if valid_model else "model invalid"},
        "controls": {
            "positive_int2_raises_dnll": {"dnll_mean": pos["dnll_mean"], "ci95": pos["dnll_ci95"], "passed": pos_ok},
            "negative_A2at8_equals_fp8": {"diff_mean": m, "ci95": h, "per_seed": d, "passed": neg1_ok},
            "negative_hook_identity": raw["controls"]["hook_identity"],
            "negative_decoders": raw["controls"]["decoders"],
            "baseline_finite": base_ok,
        },
        "controls_all_passed": bool(pos_ok and neg1_ok and hook_ok and dec_ok and base_ok),
        "valid_model": valid_model,
        "descriptive_frontier_vs_fp8": frontier,
    }


def aggregate() -> int:
    prereg_sha = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    per_model, raws = {}, {}
    for key in MODELS:
        p = EXP_DIR / f"E11_raw_{key}.json"
        if not p.exists():
            per_model[key] = {"status": "not run"}
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        raws[key] = raw
        per_model[key] = {"raw_file": p.name, "raw_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "repo": raw["repo"], "snapshot": raw["snapshot"],
                          "dtype": raw["dtype"], "head_dims": raw["head_dims"], "num_layers": raw["num_layers"], "kv_shared_layers": raw["kv_shared_layers"],
                          "peak_gpu_gib": raw["peak_gpu_gib"], "elapsed_seconds": raw["elapsed_seconds"], **decide_model(raw)}

    def overall(pred: str) -> str:
        sts = [v[pred]["status"] for v in per_model.values() if v.get("valid_model")]
        if not sts:
            return "no valid model"
        if any(s.startswith("refuted") for s in sts):
            return "refuted (on " + ", ".join(k for k, v in per_model.items() if v.get("valid_model") and v[pred]["status"].startswith("refuted")) + ")"
        if all(s == "supported" for s in sts):
            return "supported"
        return "mixed/inconclusive"

    report = {
        "schema_version": "hivemind-experiment-E11-kv-lattice/v1.0",
        "experiment_id": "E11-kv-lattice",
        "prereg_sha256": prereg_sha,
        "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__, "python": platform.python_version()},
        "models": per_model,
        "analysis": {"P11a": {"status": overall("P11a"), "per_model": {k: v["P11a"]["status"] for k, v in per_model.items() if "P11a" in v}},
                     "P11b": {"status": overall("P11b"), "per_model": {k: v["P11b"]["status"] for k, v in per_model.items() if "P11b" in v}}},
        "controls_all_passed": all(v.get("controls_all_passed", False) for v in per_model.values() if v.get("status") != "not run"),
        "scope_note": prereg["scope"],
    }
    try:
        import transformers

        report["hardware"]["transformers"] = transformers.__version__
    except ImportError:
        pass
    payload = json.dumps({k: v for k, v in report.items() if k != "receipt_sha256"}, sort_keys=True).encode("utf-8")
    report["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot(per_model)
    print(json.dumps({"P11a": report["analysis"]["P11a"], "P11b": report["analysis"]["P11b"], "controls_all_passed": report["controls_all_passed"],
                      "receipt": report["receipt_sha256"]}, indent=2))
    return 0 if report["controls_all_passed"] else 1


def plot(per_model: Dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Qwen2.5-Coder-7B is INVALID (negative control failed, scientific lead review 2026-10-24): not plotted
    done = [k for k, v in per_model.items() if "cells" in v and k != "qwen7b"]
    if not done:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(done), figsize=(4.6 * len(done), 3.9), squeeze=False)
    style = {"Z|none": ("#e67e22", "o", "-"), "A2|none": ("#27ae60", "s", "-"), "E8|none": ("#2980b9", "^", "-"),
             "Z|klt": ("#e67e22", "o", "--"), "A2|klt": ("#27ae60", "s", "--"), "E8|klt": ("#2980b9", "^", "--")}
    for ax, key in zip(axes[0], done):
        cells = per_model[key]["cells"]
        for kind, t in itertools.product(LATTICES, TRANSFORMS):
            col, mk, ls = style[f"{kind}|{t}"]
            xs = [cells[f"{kind}@{r}|{t}"]["rate_bits_per_dim"] for r in RATES]
            ys = [max(cells[f"{kind}@{r}|{t}"]["dnll_mean"], 1e-5) for r in RATES]
            ax.plot(xs, ys, marker=mk, ls=ls, color=col, label=f"{kind}{' + KLT' if t == 'klt' else ''}")
        ax.plot(RATES, [max(cells[f"int@{b}"]["dnll_mean"], 1e-5) for b in RATES], marker="x", color="#7f8c8d", label="INT (min-max, fixed rate)")
        ax.axhline(max(cells["fp8"]["dnll_mean"], 1e-5), color="#c0392b", lw=1, ls=":", label="FP8 (8 bits)")
        ax.set_yscale("log")
        ax.set_xlabel("measured bits/dim (payload)")
        ax.set_ylabel("ΔNLL vs fp16/bf16 (nats/token)")
        ax.set_title(f"E11 {key} (3 seeds)")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig8_e11_kv_lattice.{ext}", dpi=160)
    plt.close(fig)


def main() -> int:
    global WINDOW, WINDOWS_PER_SEED, ACTIVE_TRANSFORMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "fineweb-edu", "gsm8k", "code", "news", "fiction", "pubmed", "wiki-fr"])
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--windows-per-seed", type=int, default=WINDOWS_PER_SEED)
    a = ap.parse_args()
    if a.aggregate:
        return aggregate()
    if not a.model:
        ap.error("--model or --aggregate")
    WINDOW, WINDOWS_PER_SEED = a.window, a.windows_per_seed
    registered = a.dataset == "wikitext2" and a.window == 1024 and a.windows_per_seed == 8
    if not registered:
        ACTIVE_TRANSFORMS = TRANSFORMS + ("kltc",)
    raw = run_model(a.model, a.dataset)
    raw["prereg_sha256"] = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    raw["registered_configuration"] = registered
    # only the pre-registered configuration feeds the E11 decision; everything else is the exploratory E11b sweep
    out = EXP_DIR / (f"E11_raw_{a.model}.json" if registered else f"E11b_raw_{a.model}_{a.dataset}_w{a.window}.json")
    out.write_text(json.dumps(raw, indent=1), encoding="utf-8")
    print(f"[{a.model}] wrote {out.name} in {raw['elapsed_seconds']}s, peak {raw['peak_gpu_gib']} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

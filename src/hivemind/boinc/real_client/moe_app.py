# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""QLoRA MoE science apps for a stock BOINC client: ``hivemind_qlora_moe`` (training) and ``hivemind_moe_infer``.

Training: the workunit is ``K`` segments of ``S`` inner steps run through the repository's own
:func:`hivemind.core.diloco_inner.replay_inner_segment` (AdamW restarts at each segment start, which makes
checkpoint/resume and spot-check replay exact by construction). The model is NF4 on the attention projections
(bitsandbytes, fp16 compute) with fp32 LoRA adapters; the MoE experts stay frozen fp16 (they are not ``nn.Linear``).

Inference: greedy decoding of a prompt file with the assimilated adapter; greedy makes the result checkable.

Heavy imports (torch, transformers, peft, bitsandbytes) are deferred so that this module imports in any environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from hivemind.boinc.real_client.app import boinc_finish, class_fingerprint, resolve_filename
from hivemind.boinc.real_client.moe_profiles import TRAIN_APPS

# hivemind.config.ensure_local_drive_caches() (run by importing the training code) re-points HF_HOME/HF_HUB_CACHE
# at a repo-local .cache, which hides the models cached in the user's real cache. Capture the real path first and
# pass it explicitly as cache_dir.
HF_CACHE = os.environ.get("HIVEMIND_REAL_HF_CACHE") or str(Path.home() / ".cache/huggingface/hub")
TRAIN_APP = "hivemind_qlora_moe"
INFER_APP = "hivemind_moe_infer"
APP_VERSION = 100
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def tokens_sha(tokens: list[list[int]]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def build_mixed_domain_train_tokens(
    target_tokens: list[list[int]],
    replay_pools_by_domain: dict[str, list[list[int]]] | None,
    *,
    target_domain: str = "general",
    replay_ratio: float = 0.2,
    rng_seed: int = 42,
) -> list[list[int]]:
    """Construct an 80/20 mixed-domain token batch schedule to prevent cross-domain forgetting on shared attention LoRAs.

    Replaces ``round(len(target_tokens) * replay_ratio)`` sequences at deterministic indices with sequences sampled
    uniformly across non-target domains in ``replay_pools_by_domain``.
    """
    if not target_tokens or not replay_pools_by_domain or replay_ratio <= 0.0:
        return [list(seq) for seq in target_tokens]
    other_domains = sorted(d for d, pool in replay_pools_by_domain.items() if d != target_domain and pool)
    if not other_domains:
        return [list(seq) for seq in target_tokens]
    clamped_ratio = min(0.5, max(0.0, float(replay_ratio)))
    n_total = len(target_tokens)
    n_replay = max(1, int(round(n_total * clamped_ratio))) if n_total >= 2 else 0
    rng = random.Random(int(rng_seed))
    mixed = [list(seq) for seq in target_tokens]
    replay_positions = sorted(rng.sample(range(n_total), k=min(n_replay, n_total)))
    for idx, pos in enumerate(replay_positions):
        dom = other_domains[idx % len(other_domains)]
        pool = replay_pools_by_domain[dom]
        chosen = pool[rng.randrange(len(pool))]
        mixed[pos] = list(chosen)
    return mixed


def cosine_segment_lr(
    base_lr: float,
    segment_idx: int,
    total_segments: int,
    *,
    round_idx: int = 0,
    total_rounds: int = 100,
    min_lr_scale: float = 0.2,
) -> float:
    """Cosine learning-rate schedule across chained BOINC QLoRA rounds and inner segments."""
    tot_steps = max(1, int(total_rounds) * max(1, int(total_segments)))
    cur_step = min(tot_steps, max(0, int(round_idx) * max(1, int(total_segments)) + int(segment_idx)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * cur_step / tot_steps))
    scale = float(min_lr_scale) + (1.0 - float(min_lr_scale)) * cosine
    return float(base_lr) * scale


def load_peft_model(spec: dict[str, Any], adapter_state: dict[str, Any] | None = None):
    """NF4 attention + fp32 LoRA, deterministic init from ``spec['init_seed']``."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from hivemind.core.quantization import configure_deterministic_runtime, load_lora_state_dict

    q = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    )
    base = AutoModelForCausalLM.from_pretrained(spec["model_id"], quantization_config=q, device_map={"": 0}, cache_dir=HF_CACHE)
    configure_deterministic_runtime(int(spec["init_seed"]))
    cfg = LoraConfig(r=int(spec["rank"]), lora_alpha=int(spec["alpha"]), target_modules=list(LORA_TARGETS), task_type="CAUSAL_LM", lora_dropout=0.0)
    model = get_peft_model(base, cfg)
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
    if adapter_state is not None:
        load_lora_state_dict(model, adapter_state)
    return model


def eval_by_domain(model, spec: dict[str, Any]) -> dict[str, float]:
    """Held-out loss per domain (``eval_sets``); falls back to the single ``eval_tokens`` set of older workunits."""
    sets = spec.get("eval_sets") or {"general": spec["eval_tokens"]}
    return {d: eval_loss(model, toks) for d, toks in sets.items() if toks}


def eval_loss(model, tokens, batch: int = 4) -> float:
    import torch

    model.eval()
    dev = next(model.parameters()).device
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(tokens), batch):
            ids = torch.tensor(tokens[i : i + batch], dtype=torch.long, device=dev)
            tot += float(model(input_ids=ids, labels=ids.clone()).loss) * ids.size(0)
            n += ids.size(0)
    model.train()
    return tot / max(n, 1)


def state_sha(state: dict[str, Any]) -> str:
    from hivemind.core.quantization import compute_state_dict_sha256

    return compute_state_dict_sha256(state)


def train(
    spec: dict[str, Any],
    ckpt: Path,
    crash_after_segment: int | None = None,
    lazy: bool = False,
    start_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``start_state`` chains workunits: training continues from the adapter the previous validated workunit produced."""
    import torch

    from hivemind.core.diloco_inner import replay_inner_segment
    from hivemind.core.quantization import extract_lora_state_dict, load_lora_state_dict

    raw_train_tokens: list[list[int]] = spec["train_tokens"]
    if spec.get("replay_pools_by_domain") and float(spec.get("replay_ratio", 0.0)) > 0.0:
        raw_train_tokens = build_mixed_domain_train_tokens(
            raw_train_tokens,
            spec.get("replay_pools_by_domain"),
            target_domain=str(spec.get("domain", "general")),
            replay_ratio=float(spec.get("replay_ratio", 0.2)),
            rng_seed=int(spec.get("rng_seed", 42)),
        )
    train_tokens = torch.tensor(raw_train_tokens, dtype=torch.long)
    model = load_peft_model(spec)
    if start_state is not None:
        load_lora_state_dict(model, start_state)
    seg, steps = int(spec["segment_steps"]), int(spec["inner_steps"])
    k_total = steps // seg
    states: list[dict[str, Any]] = [extract_lora_state_dict(model)]
    eval0 = None
    resumed = 0
    if ckpt.exists():
        c = torch.load(ckpt, map_location="cpu", weights_only=True)
        states, eval0, resumed = c["states"], c["eval0"], len(c["states"]) - 1
    if eval0 is None:
        eval0 = eval_by_domain(model, spec)
    t0 = time.time()
    use_cosine_lr = str(spec.get("lr_schedule", "constant")).lower() == "cosine"
    round_idx = int(spec.get("round_idx", 0))
    total_rounds = int(spec.get("total_rounds", 100))
    for k in range(len(states) - 1, k_total):
        seg_lr = (
            cosine_segment_lr(
                float(spec["lr"]),
                k,
                k_total,
                round_idx=round_idx,
                total_rounds=total_rounds,
            )
            if use_cosine_lr
            else float(spec["lr"])
        )
        if lazy:  # red-team control: claim progress without computing it
            states.append({n: t.clone() for n, t in states[-1].items()})
        else:
            states.append(
                replay_inner_segment(
                    model, states[k], train_tokens, k * seg, (k + 1) * seg,
                    lr=seg_lr, weight_decay=float(spec["weight_decay"]),
                    rng_seed=int(spec["rng_seed"]), micro_batch_size=int(spec["micro_batch"]),
                )
            )
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"states": states, "eval0": eval0}, tmp)
        os.replace(tmp, ckpt)
        if crash_after_segment is not None and k + 1 >= crash_after_segment:
            raise SystemExit(75)
    from hivemind.core.quantization import load_lora_state_dict

    load_lora_state_dict(model, states[-1])
    eval1 = eval_by_domain(model, spec)
    wall = time.time() - t0
    tokens_seen = steps * int(spec["micro_batch"]) * len(spec["train_tokens"][0])
    meta = {
        "app": spec.get("app", TRAIN_APP), "app_version": APP_VERSION, "wu_id": spec["wu_id"], "model_id": spec["model_id"],
        "device_name": torch.cuda.get_device_name(0), "fingerprint": class_fingerprint("cuda", "fp16"),
        "segments": k_total, "segment_steps": seg, "resumed_from_segment": resumed,
        "state_sha256": [state_sha(s) for s in states], "domain": spec.get("domain", "general"),
        "eval_by_domain_before": eval0, "eval_by_domain_after": eval1,
        "eval_loss_before": sum(eval0.values()) / len(eval0), "eval_loss_after": sum(eval1.values()) / len(eval1),  # mean over domains
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_vram_mib": torch.cuda.max_memory_allocated() // 2**20, "wall_seconds": wall,
        # forward+backward on ~0.4 B active parameters per token (granite-3.1-1b-a400m): an estimate, capped by the validator
        "flops_claimed": 6.0 * 4.0e8 * tokens_seen, "lazy": lazy,
    }
    return meta, states


def infer(spec: dict[str, Any], adapter_state: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    model = load_peft_model(spec, adapter_state)
    model.eval()
    tok = AutoTokenizer.from_pretrained(spec["model_id"], cache_dir=HF_CACHE)
    outs = []
    t0 = time.time()
    for prompt in spec["prompts"]:
        ids = tok(prompt, return_tensors="pt").input_ids.to(0)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=int(spec["max_new_tokens"]), do_sample=False, pad_token_id=tok.eos_token_id)
        new = gen[0, ids.size(1) :].tolist()
        outs.append({"prompt": prompt, "token_ids": new, "text": tok.decode(new, skip_special_tokens=True)})
    return {
        "app": INFER_APP, "app_version": APP_VERSION, "wu_id": spec["wu_id"], "model_id": spec["model_id"],
        "device_name": torch.cuda.get_device_name(0), "fingerprint": class_fingerprint("cuda", "fp16"),
        "adapter_sha256": state_sha(adapter_state), "completions": outs, "wall_seconds": time.time() - t0,
        "flops_claimed": 2.0 * 4.0e8 * sum(len(o["token_ids"]) for o in outs),
    }


def gpu_gate(spec: dict[str, Any], mode: str, have_mb: int) -> str | None:
    """Why this host must not run the workunit, or None. The scheduler should never send a big workunit to a small GPU, but a
    mis-registered host must fail loudly rather than run out of memory halfway through."""
    if mode != "train":
        return None
    if spec.get("app", TRAIN_APP) not in TRAIN_APPS:
        return f"unknown training app {spec.get('app')!r}"
    need_mb = int(spec.get("min_gpu_ram_mb", 0))
    if have_mb < need_mb:
        return f"workunit needs a GPU with {need_mb} MiB, this one has {have_mb} MiB"
    return None


def main(argv: list[str] | None = None) -> int:
    import torch

    args = argv if argv is not None else sys.argv[1:]
    mode = args[args.index("--mode") + 1] if "--mode" in args else "train"
    crash = next((int(a.split("=")[1]) for a in args if a.startswith("--crash-after-segment=")), None)
    lazy = "--lazy" in args
    if not torch.cuda.is_available():
        print("no CUDA device visible to the app", file=sys.stderr)
        return 2
    spec = json.loads(resolve_filename("in.json").read_text())
    refusal = gpu_gate(spec, mode, torch.cuda.get_device_properties(0).total_memory // 2**20)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    out_json = resolve_filename("out.json")
    if mode == "train":
        start_state = None
        if spec.get("start_adapter_sha256"):
            start_state = torch.load(resolve_filename("adapter.pt"), map_location="cpu", weights_only=True)["state"]
            if state_sha(start_state) != spec["start_adapter_sha256"]:
                raise RuntimeError("the start adapter does not match the workunit's start_adapter_sha256")
        meta, states = train(spec, Path("hivemind_moe_ckpt.pt"), crash, lazy, start_state)
        out_pt = resolve_filename("out.pt")
        torch.save({"states": states}, out_pt)
        out_json.write_text(json.dumps(meta, sort_keys=True))
        Path("hivemind_moe_ckpt.pt").unlink(missing_ok=True)
    else:
        adapter = torch.load(resolve_filename("adapter.pt"), map_location="cpu", weights_only=True)["state"]
        out_json.write_text(json.dumps(infer(spec, adapter), sort_keys=True))
    boinc_finish(0)
    return 0


def _entry() -> int:
    try:
        return main()
    except SystemExit:
        raise
    except BaseException as exc:
        import traceback

        traceback.print_exc()
        print(f"[diag] HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} HOME={os.environ.get('HOME')} cwd={os.getcwd()}", file=sys.stderr)
        print(f"app error: {exc!r}", file=sys.stderr)
        boinc_finish(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(_entry())

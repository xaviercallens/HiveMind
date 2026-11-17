#!/usr/bin/env python3
"""E17 stage 'train': QLoRA (nf4 base, LoRA r=16 a=32 on all linear layers) on the gate-accepted pairs from scripts/e17_sample.py.

Runs on the GPU VM (needs torch, transformers, peft, bitsandbytes). Plain PyTorch loop, no Trainer: assistant-token loss only,
fp16 autocast + GradScaler (the T4 has no bf16), cosine schedule with warmup, gradient checkpointing, one seed.
Preregistered in artifacts/experiments/E17_router_7b_rft_prereg.json.
Usage: python e17_train.py --base DIR_OR_HF_ID --pairs pairs.jsonl --out adapter_dir [--epochs 3 --lr 1e-4 --accum 16 --max-len 768]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import sys

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17_ckpt as ckpt  # noqa: E402  (resumable checkpoints; see that module)

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, rec: dict, max_len: int) -> dict | None:
    prompt_ids = tok.apply_chat_template(rec["messages"], add_generation_prompt=True, tokenize=True)
    if hasattr(prompt_ids, "input_ids"):  # newer transformers return a BatchEncoding
        prompt_ids = prompt_ids["input_ids"]
    completion_ids = tok(rec["completion"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = list(prompt_ids) + completion_ids
    if len(ids) > max_len:
        return None
    labels = [-100] * len(prompt_ids) + completion_ids
    return {"input_ids": ids, "labels": labels}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-hours", type=float, default=4.0)
    ap.add_argument("--ckpt-every", type=int, default=0, help="save a resumable checkpoint every N optimizer steps and resume from the newest one; 0 = off (the original behaviour)")
    ap.add_argument("--ckpt-dir", default="", help="checkpoint directory (default: OUT.ckpt)")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base)
    recs = [json.loads(line) for line in Path(args.pairs).read_text().splitlines() if line.strip()]
    data = [e for e in (encode(tok, r, args.max_len) for r in recs) if e is not None]
    print(f"pairs {len(recs)}, kept {len(data)} within {args.max_len} tokens", flush=True)
    if len(data) < 50:
        print("too little data to train", flush=True)
        return 3
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # fp32 adapter weights and optimizer state
    params = [p for p in model.parameters() if p.requires_grad]
    print("trainable params", sum(p.numel() for p in params), flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    steps_per_epoch = math.ceil(len(data) / args.accum)
    total = steps_per_epoch * args.epochs
    log = Path(args.out + ".train_log.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(args.ckpt_dir or args.out + ".ckpt")
    if args.ckpt_every > 0 and ckpt.is_complete(Path(args.out)):
        print("adapter already trained to completion (TRAIN_COMPLETE present): nothing to do", flush=True)
        return 0
    fp = ckpt.fingerprint(base=args.base, pairs_file=Path(args.pairs), n_data=len(data), epochs=args.epochs, lr=args.lr,
                          accum=args.accum, warmup=args.warmup, max_len=args.max_len, seed=args.seed)
    step, start_epoch, start_s, elapsed0 = 0, 0, 0, 0.0
    found = ckpt.latest_checkpoint(ckpt_root) if args.ckpt_every > 0 else None
    if found is not None:
        saved_step, st = found
        if st["fingerprint"] != fp:
            print(f"checkpoint at step {saved_step} ignored: it was written for a different configuration", flush=True)
        else:
            set_peft_model_state_dict(model, st["adapter"])
            opt.load_state_dict(st["opt"])
            scaler.load_state_dict(st["scaler"])
            step, start_epoch, start_s, elapsed0 = st["step"], st["next_epoch"], st["next_s"], float(st["elapsed_s"])
            random.setstate(st["epoch_rng"])  # python RNG as it was before the shuffle of next_epoch
            torch.set_rng_state(st["rng"]["torch_cpu"])
            if "torch_cuda" in st["rng"]:
                torch.cuda.set_rng_state_all(st["rng"]["torch_cuda"])
            print(f"resumed from checkpoint: step {step}, epoch {start_epoch}, step-in-epoch {start_s}, {elapsed0:.0f}s already spent", flush=True)
            log.open("a").write(json.dumps({"event": "resumed", "step": step, "epoch": start_epoch, "step_in_epoch": start_s, "elapsed_s": round(elapsed0)}) + "\n")
    t0 = time.time() - elapsed0
    model.train()
    epoch_rng = random.getstate()
    for epoch in range(start_epoch, args.epochs):
        epoch_rng = random.getstate()  # python RNG before this epoch's shuffle: a resume replays the same order
        order = list(range(len(data)))
        random.shuffle(order)
        for s in range(steps_per_epoch):
            if epoch == start_epoch and s < start_s:
                continue
            batch = [data[i] for i in order[s * args.accum:(s + 1) * args.accum]]
            lr = args.lr * (min(1.0, (step + 1) / args.warmup) if step < args.warmup else 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, total - args.warmup))))
            for g in opt.param_groups:
                g["lr"] = lr
            ntok = sum(sum(1 for t in e["labels"] if t != -100) for e in batch)
            loss_sum = 0.0
            for e in batch:  # micro-batch of one sequence: the T4 has 15 GB and the 7B nf4 base already takes ~5.5 GB
                ids = torch.tensor([e["input_ids"]], device="cuda")
                lab = torch.tensor([e["labels"]], device="cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids=ids).logits
                loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), lab[0, 1:], ignore_index=-100, reduction="sum") / ntok
                scaler.scale(loss).backward()
                loss_sum += float(loss)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": epoch, "loss": round(loss_sum, 5), "lr": lr, "elapsed_s": round(time.time() - t0), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            log.open("a").write(json.dumps(rec) + "\n")
            print(rec, flush=True)
            ne, ns = ckpt.next_position(epoch, s, steps_per_epoch)
            capped = (time.time() - t0) / 3600 > args.max_hours
            if args.ckpt_every > 0 and step < total and (step % args.ckpt_every == 0 or capped):
                ckpt.save_checkpoint(ckpt_root, step, {
                    "fingerprint": fp, "step": step, "next_epoch": ne, "next_s": ns, "elapsed_s": time.time() - t0,
                    "epoch_rng": random.getstate() if ns == 0 else epoch_rng, "rng": ckpt.capture_rng(),
                    "adapter": {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
                    "opt": opt.state_dict(), "scaler": scaler.state_dict()})
            if capped:
                print("max-hours reached: stopping early, adapter saved as is", flush=True)
                model.save_pretrained(args.out)
                return 4
    model.save_pretrained(args.out)
    if args.ckpt_every > 0:
        ckpt.mark_complete(Path(args.out), {"steps": step, "epochs": args.epochs})
    print("saved", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

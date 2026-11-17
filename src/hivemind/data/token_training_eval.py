# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Offline, measured token-training/adapter-transfer smoke test, not a capability benchmark.

A tiny random-initialized Qwen2 model consumes real UTF-8 bytes, never random
regression targets. Its architecture, seed, tokenizer, data and serialized base
weights are recorded. LoRA training and a fresh recipient run on CPU in one
process. No model downloads, remote code, QLoRA, WAN or GPU claims are made.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load, save
from transformers import Qwen2Config, Qwen2ForCausalLM

from hivemind.swarm.dht_storage import ContentAddressedSwarmStore

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "rust_text_smoke_v1.json"
FIXTURE_SHA256 = "c23620e24a3dda16d83b6d0e2bbaa053bef8ae0966d8950f9214d0e9459a0444"
MODEL_ID = "hivemind/tiny-qwen2-byte-random-v1"
TOKENIZER = {"id": "utf8-bytes-v1", "byte_ids": [0, 255], "bos_id": 256,
             "eos_id": 257, "pad_id": 258, "vocab_size": 259}
MODEL_CONFIG = {
    "vocab_size": 259, "hidden_size": 32, "intermediate_size": 64,
    "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
    "max_position_embeddings": 128, "rms_norm_eps": 1e-6, "use_cache": False,
    "tie_word_embeddings": False, "attention_dropout": 0.0,
    "bos_token_id": 256, "eos_token_id": 257, "pad_token_id": 258,
    "initializer_range": 0.02,
}
LORA_CONFIG = {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
               "target_modules": ["q_proj", "v_proj", "lm_head"], "bias": "none"}


def sha256(data: bytes) -> str:
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("data must be bytes or bytearray")
    return hashlib.sha256(data).hexdigest()


def tensor_bytes(state: dict[str, torch.Tensor]) -> bytes:
    """Safe, full-precision serialization; header overhead is included in len()."""
    if not isinstance(state, dict) or not state:
        raise ValueError("state must be a non-empty dict of tensors")
    for k, v in state.items():
        if (
            not isinstance(k, str)
            or not k.strip()
            or not isinstance(v, torch.Tensor)
            or v.dtype == torch.bool
            or v.numel() == 0
        ):
            raise ValueError("state must map non-empty string keys to non-empty numeric torch.Tensor values")
    return save({k: v.detach().cpu().contiguous() for k, v in sorted(state.items())})


def validate_splits(fixture: dict[str, Any]) -> None:
    """Reject empty/duplicate examples and exact normalized cross-split leakage."""
    if not isinstance(fixture, dict):
        raise ValueError("fixture must be a dict")
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for split in ("train", "holdout"):
        rows = fixture.get(split)
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or len(rows) == 0:
            raise ValueError(f"{split} must not be empty")
        for row in rows:
            if not isinstance(row, Mapping) or "id" not in row or "text" not in row:
                raise ValueError("each split row must be a mapping containing 'id' and 'text'")
            sample_id, text = row["id"], row["text"]
            if not isinstance(sample_id, str) or not isinstance(text, str):
                raise ValueError("sample IDs and text must be strings")
            sample_id_clean = sample_id.strip()
            normalized = " ".join(text.split())
            if not sample_id_clean or not normalized or sample_id_clean in seen_ids or normalized in seen_text:
                raise ValueError("empty or duplicate document/ID across fixed splits")
            seen_ids.add(sample_id_clean)
            seen_text.add(normalized)


def load_fixture(path: Path = FIXTURE_PATH) -> tuple[dict[str, Any], str]:
    if not isinstance(path, (Path, str)) or not str(path).strip():
        raise ValueError("path must be a non-empty Path or string")
    path = Path(path)
    raw = path.read_bytes()
    if path.resolve() == FIXTURE_PATH.resolve() and sha256(raw) != FIXTURE_SHA256:
        raise ValueError("pinned smoke fixture digest does not match")
    fixture = json.loads(raw)
    validate_splits(fixture)
    return fixture, sha256(raw)


def tokenize_documents(samples: Sequence[str], sequence_length: int = 128) -> dict[str, torch.Tensor]:
    """Fixed byte tokenizer; no learned vocabulary, truncation or cross-doc joins.

    Consecutive windows overlap by one token, scoring each target exactly once.
    Context resets at each window. BOS is context only, EOS is scored, padding
    is never scored. There is no text-dependent fitting on the holdout.
    """
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise ValueError("samples must be a sequence of strings")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or not 2 <= sequence_length <= MODEL_CONFIG["max_position_embeddings"]:
        raise ValueError("sequence_length must be between 2 and 128")
    windows = []
    for text in samples:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("documents must be nonempty strings")
        tokens = [TOKENIZER["bos_id"], *text.encode("utf-8"), TOKENIZER["eos_id"]]
        windows.extend(tokens[start:start + sequence_length]
                       for start in range(0, len(tokens) - 1, sequence_length - 1))
    if not windows:
        raise ValueError("at least one document is required")
    width = max(map(len, windows))
    ids = torch.full((len(windows), width), TOKENIZER["pad_id"], dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, window in enumerate(windows):
        ids[index, :len(window)] = torch.tensor(window)
        mask[index, :len(window)] = 1
    labels = ids.clone().masked_fill(mask == 0, -100)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def token_nll(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Sum next-token cross-entropy in nats and count scored non-padding tokens."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.dtype == torch.bool:
        raise ValueError("logits must be a 3D numeric torch.Tensor")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2 or labels.dtype == torch.bool:
        raise ValueError("labels must be a 2D integer torch.Tensor")
    if logits.shape[:2] != labels.shape or logits.size(1) < 2 or logits.size(-1) < 1:
        raise ValueError("logits batch/sequence dimensions must match labels shape with sequence_length >= 2")
    targets = labels[:, 1:].contiguous()
    count = int((targets != -100).sum().item())
    if not count:
        raise ValueError("no next-token targets to score")
    total = F.cross_entropy(logits[:, :-1].contiguous().reshape(-1, logits.size(-1)),
                            targets.reshape(-1), ignore_index=-100, reduction="sum")
    if not bool(torch.isfinite(total)):
        raise ValueError("non-finite token loss")
    return total, count


def evaluate_tokens(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    if not callable(model) or not callable(getattr(model, "eval", None)):
        raise ValueError("model must be a callable torch.nn.Module")
    if not isinstance(batch, Mapping) or not all(k in batch for k in ("input_ids", "attention_mask", "labels")):
        raise ValueError("batch must be a mapping containing 'input_ids', 'attention_mask', and 'labels'")
    ids, mask, lbls = batch["input_ids"], batch["attention_mask"], batch["labels"]
    if (
        not isinstance(ids, torch.Tensor)
        or not isinstance(mask, torch.Tensor)
        or not isinstance(lbls, torch.Tensor)
        or ids.ndim != 2
        or mask.shape != ids.shape
        or lbls.shape != ids.shape
        or ids.numel() == 0
    ):
        raise ValueError("batch tensors 'input_ids', 'attention_mask', and 'labels' must be non-empty 2D tensors of matching shape")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        total, count = token_nll(outputs.logits, batch["labels"])
    mean_nll = float(total.item()) / count
    try:
        perplexity = math.exp(mean_nll)
    except OverflowError:
        perplexity = None
    return {"status": "measured", "metric": "next_token_cross_entropy_nats",
            "total_nll_nats": float(total.item()), "scored_tokens": count,
            "mean_nll_nats": mean_nll, "perplexity": perplexity,
            "perplexity_status": "measured" if perplexity is not None else "overflow"}


def build_model(seed: int, base_bytes: bytes | None = None) -> tuple[torch.nn.Module, bytes]:
    """Pin the base via saved tensor bytes; no from_pretrained/network path exists."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if base_bytes is not None and (not isinstance(base_bytes, (bytes, bytearray)) or len(base_bytes) == 0):
        raise ValueError("base_bytes must be None or non-empty bytes")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        config = Qwen2Config(**MODEL_CONFIG)
        config._attn_implementation = "eager"
        base = Qwen2ForCausalLM(config).float().cpu()
        if base_bytes is not None:
            base.load_state_dict(load(base_bytes), strict=True)
        serialized_base = tensor_bytes(base.state_dict())
        model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, **LORA_CONFIG))
    return model, serialized_base


def adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    if not callable(getattr(model, "named_parameters", None)):
        raise ValueError("model must provide a callable named_parameters method")
    return {name: param.detach().clone() for name, param in model.named_parameters()
            if "lora_" in name}


def frozen_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    if not callable(getattr(model, "named_parameters", None)):
        raise ValueError("model must provide a callable named_parameters method")
    return {name: param.detach() for name, param in model.named_parameters() if not param.requires_grad}


def import_adapter(model: torch.nn.Module, payload: bytes, expected_sha256: str) -> None:
    """Verify the serialized bytes before safe decoding, then require exact shapes."""
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be a torch.nn.Module")
    if not isinstance(payload, (bytes, bytearray)) or len(payload) == 0:
        raise ValueError("payload must be non-empty bytes")
    if not isinstance(expected_sha256, str) or not expected_sha256.strip():
        raise ValueError("expected_sha256 must be a non-empty string")
    if sha256(payload) != expected_sha256:
        raise ValueError("adapter SHA-256 mismatch")
    incoming = load(payload)
    expected = adapter_state(model)
    if incoming.keys() != expected.keys():
        raise ValueError("adapter parameter names do not match the pinned model")
    for name, value in incoming.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"adapter shape/dtype mismatch: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite adapter tensor: {name}")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in incoming:
                param.copy_(incoming[name])


def run_token_training_evaluation(output_dir: Path, *, steps: int = 24, seed: int = 42,
                                  learning_rate: float = 0.01) -> dict[str, Any]:
    """Measure baseline/trained/imported on identical holdout; report regressions too."""
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 0
        or isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("steps must be nonnegative and learning_rate finite and positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be empty; use a new directory for each run")
    fixture, fixture_digest = load_fixture()
    train = tokenize_documents([row["text"] for row in fixture["train"]])
    holdout = tokenize_documents([row["text"] for row in fixture["holdout"]])
    output_dir.mkdir(parents=True, exist_ok=True)
    model, base_bytes = build_model(seed)
    (output_dir / "base_model.safetensors").write_bytes(base_bytes)
    frozen_before = sha256(tensor_bytes(frozen_state(model)))
    initial_adapter = sha256(tensor_bytes(adapter_state(model)))
    baseline = evaluate_tokens(model, holdout)
    initial_train = evaluate_tokens(model, train)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)
    losses = []
    started = time.perf_counter()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=train["input_ids"], attention_mask=train["attention_mask"])
        total, count = token_nll(outputs.logits, train["labels"])
        loss = total / count
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    training_seconds = time.perf_counter() - started
    trained = evaluate_tokens(model, holdout)
    final_train = evaluate_tokens(model, train)
    frozen_after = sha256(tensor_bytes(frozen_state(model)))
    if frozen_after != frozen_before:
        raise RuntimeError("frozen base parameters changed during adapter training")

    payload = tensor_bytes(adapter_state(model))
    digest = sha256(payload)
    (output_dir / "adapter.safetensors").write_bytes(payload)
    sender = ContentAddressedSwarmStore(output_dir / "sender_store")
    receiver = ContentAddressedSwarmStore(output_dir / "receiver_store")
    cid = sender.put_chunk(payload)
    # This is deliberately local byte-copy transfer, not WAN/remote-peer evidence.
    received_bytes = sender.read_chunk(cid)
    if sha256(received_bytes) != digest:
        raise ValueError("sender CAS payload SHA-256 mismatch")
    received_cid = receiver.put_chunk(received_bytes)
    peer, reloaded_base = build_model(seed, (output_dir / "base_model.safetensors").read_bytes())
    if sha256(reloaded_base) != sha256(base_bytes):
        raise ValueError("fresh recipient base model SHA-256 mismatch")
    recipient_baseline = evaluate_tokens(peer, holdout)
    import_adapter(peer, receiver.read_chunk(received_cid), digest)
    imported = evaluate_tokens(peer, holdout)
    imported_digest = sha256(tensor_bytes(adapter_state(peer)))
    improvement = baseline["mean_nll_nats"] - trained["mean_nll_nats"]
    parity = abs(trained["mean_nll_nats"] - imported["mean_nll_nats"])
    report = {
        "schema_version": 1, "measurement_kind": "measured_token_training_smoke",
        "scope": "single_process_cpu_local_cas_transfer", "model_id": MODEL_ID,
        "pretrained": False, "quantized": False, "seed": seed,
        "implementation_sha256": sha256(Path(__file__).read_bytes()),
        "model_config": MODEL_CONFIG, "attention_implementation": "eager",
        "lora_config": LORA_CONFIG, "tokenizer": TOKENIZER,
        "base_model_sha256": sha256(base_bytes), "base_model_serialized_bytes": len(base_bytes),
        "frozen_parameter_sha256_before": frozen_before,
        "frozen_parameter_sha256_after": frozen_after,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in params),
        "data": {"dataset_id": fixture["dataset_id"], "fixture_sha256": fixture_digest,
                 "source_revision": fixture["source_revision"], "sequence_length": 128,
                 "split_policy": "fixed_disjoint_documents; no truncation; context reset per window",
                 "train_ids": [r["id"] for r in fixture["train"]],
                 "holdout_ids": [r["id"] for r in fixture["holdout"]]},
        "training": {"optimizer": "AdamW", "learning_rate": learning_rate, "weight_decay": 0.0,
                     "steps_completed": steps, "batch_policy": "all_training_windows_each_step",
                     "initial_train": initial_train, "final_train": final_train,
                     "step_loss_nats": losses, "wall_seconds": training_seconds},
        "holdout": {"baseline": baseline, "trained_adapter": trained,
                    "recipient_baseline": recipient_baseline, "imported_adapter": imported,
                    "nll_improvement_nats": improvement, "improved": improvement > 0},
        "transfer": {"status": "measured", "serialization": "safetensors_float32_no_compression",
                     "actual_serialized_bytes": len(payload), "adapter_sha256": digest,
                     "initial_adapter_sha256": initial_adapter, "imported_adapter_sha256": imported_digest,
                     "sender_chunk_sha256": cid, "receiver_chunk_sha256": received_cid,
                     "recipient_training_steps": 0, "holdout_nll_absolute_difference": parity,
                     "exact_adapter_match": digest == imported_digest,
                     "holdout_matches_within_1e_7": parity <= 1e-7},
        "hardware": {"device": "cpu", "platform": platform.platform(),
                     "processor": platform.processor() or "unknown", "python": platform.python_version(),
                     "torch_threads": torch.get_num_threads()},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "peft", "safetensors", "numpy")},
        "unmeasured": {"parallel_speedup": {"status": "unavailable", "value": None,
                                               "reason": "no concurrent equivalent-work comparison"},
                       "gpu_validation": {"status": "not_run", "value": None},
                       "wan_transfer": {"status": "not_run", "value": None},
                       "language_capability": {"status": "not_established", "value": None}},
        "limitations": ["Random-initialized tiny model, not pretrained or QLoRA.",
                        "Six curated snippets; not a representative or external benchmark.",
                        "Same-process fresh recipient; no multi-host or WAN test.",
                        "No compression, throughput speedup, GPU or domain-mastery claim.",
                        "Perplexity is UTF-8-byte-token perplexity, not comparable across tokenizers."],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True,
                                                      allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_token_training_evaluation(args.output_dir, steps=args.steps, seed=args.seed)
    print(json.dumps({"report": str(args.output_dir / "report.json"),
                      "holdout": report["holdout"], "transfer": report["transfer"]}, indent=2))


if __name__ == "__main__":
    main()

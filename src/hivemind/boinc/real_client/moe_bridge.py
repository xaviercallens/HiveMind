"""Validators and the assimilator for the QLoRA MoE apps (needs torch, peft, bitsandbytes and a GPU).

Training results are checked by a **random-segment replay** (the spot-check of ``validator.py``, with the segment
chosen by the validator instead of always the first): the validator replays one segment from the volunteer's claimed
start state with :func:`replay_inner_segment` and requires the claimed end state within ``tolerance_eps`` (1e-4;
same execution class gives 0.0). It also requires the claimed chain to start at the deterministic LoRA init, the
claimed hashes to match the uploaded states, and the shard signature to verify. Inference results are checked by
re-running the greedy decode: token ids must be identical.

One model copy is kept resident (loading takes seconds once cached) and shared by all validations.
"""

from __future__ import annotations

import json
import random
import threading
from pathlib import Path
from typing import Any

from hivemind.boinc.real_client.bridge import Verdict
from hivemind.boinc.validator import DeterministicSpotCheckValidator
from hivemind.boinc.workunit import sign_shard_bytes
from hivemind.economy.compute_coin import CreditClaim, verified_credit

T4_PEAK_GFLOPS = 8141.0  # what the BOINC client itself reported for this card
EPS = DeterministicSpotCheckValidator().tolerance_eps


def _reject(wu: str, host: str, reason: str, detail: dict[str, Any] | None = None) -> Verdict:
    g = verified_credit(CreditClaim(host, wu, 0.0, 1.0, 1.0, False, False)).to_dict()
    return Verdict(wu, host, False, reason, None, None, g, detail or {})


class MoEValidator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any = None
        self._init_state: dict[str, Any] | None = None
        self._model_key: tuple | None = None
        self._rng = random.SystemRandom()

    def _ensure(self, spec: dict[str, Any]) -> None:
        from hivemind.boinc.real_client.moe_app import load_peft_model
        from hivemind.core.quantization import extract_lora_state_dict

        key = (spec["model_id"], spec["rank"], spec["alpha"], spec["init_seed"])
        if self._model is None or key != self._model_key:
            self._model = load_peft_model(spec)
            self._init_state = extract_lora_state_dict(self._model)
            self._model_key = key

    def validate_train(self, spec: dict[str, Any], meta_path: Path, states_path: Path, host: str) -> Verdict:
        import torch

        from hivemind.boinc.real_client.moe_app import state_sha
        from hivemind.core.diloco_inner import replay_inner_segment

        wu = spec["wu_id"]
        try:
            meta = json.loads(meta_path.read_text())
            states = torch.load(states_path, map_location="cpu")["states"]
        except Exception as exc:
            return _reject(wu, host, f"unreadable upload: {exc}")
        if meta.get("wu_id") != wu or meta.get("app") != "hivemind_qlora_moe":
            return _reject(wu, host, "result is not for this workunit/app")
        k_total = int(spec["inner_steps"]) // int(spec["segment_steps"])
        if len(states) != k_total + 1 or [state_sha(s) for s in states] != meta.get("state_sha256"):
            return _reject(wu, host, "claimed state hashes do not match the uploaded states")
        sha, sig = sign_shard_bytes(json.dumps(spec["train_tokens"], separators=(",", ":")).encode(), int(spec["rng_seed"]))
        if (sha, sig) != (spec["shard_sha256"], spec["shard_hmac_signature"]):
            return _reject(wu, host, "shard signature does not verify")
        with self._lock:
            self._ensure(spec)
            assert self._init_state is not None
            # a chained workunit starts from the previous validated adapter, a first one from the deterministic init
            expected_start = spec.get("start_adapter_sha256") or state_sha(self._init_state)
            if expected_start != meta["state_sha256"][0]:
                return _reject(wu, host, "chain does not start at the expected adapter (previous validated adapter or deterministic init)")
            k = self._rng.randrange(k_total)
            tokens = torch.tensor(spec["train_tokens"], dtype=torch.long)
            seg = int(spec["segment_steps"])
            expect = replay_inner_segment(
                self._model, states[k], tokens, k * seg, (k + 1) * seg, lr=float(spec["lr"]),
                weight_decay=float(spec["weight_decay"]), rng_seed=int(spec["rng_seed"]), micro_batch_size=int(spec["micro_batch"]),
            )
        diff = max(float((expect[n].float() - states[k + 1][n].float()).abs().max()) for n in expect)
        detail = {"replayed_segment": k, "of_segments": k_total, "max_abs_divergence": diff, "tolerance_eps": EPS,
                  "overhead_pct": round(100.0 / k_total, 1), "eval_loss_before": meta.get("eval_loss_before"), "eval_loss_after": meta.get("eval_loss_after")}
        if not diff <= EPS:
            return _reject(wu, host, f"REJECTED segment {k}: divergence {diff:.3e} exceeds eps={EPS:.0e}", detail)
        claim = CreditClaim(host, wu, float(meta["flops_claimed"]), max(float(meta["wall_seconds"]), 1e-6), T4_PEAK_GFLOPS, True, True)
        return Verdict(wu, host, True, f"verified segment {k}/{k_total}: max_diff={diff:.2e}", diff == 0.0, None, verified_credit(claim).to_dict(), detail)

    def validate_infer(self, spec: dict[str, Any], meta_path: Path, adapter: dict[str, Any], host: str) -> Verdict:
        from hivemind.boinc.real_client.moe_app import infer

        wu = spec["wu_id"]
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as exc:
            return _reject(wu, host, f"unreadable upload: {exc}")
        if meta.get("wu_id") != wu or meta.get("app") != "hivemind_moe_infer":
            return _reject(wu, host, "result is not for this workunit/app")
        with self._lock:  # one GPU consumer at a time on the shared card
            ref = infer(spec, adapter)  # loads its own model copy; inference WUs are rare and small
        claimed = [c["token_ids"] for c in meta["completions"]]
        expect = [c["token_ids"] for c in ref["completions"]]
        if claimed != expect or meta.get("adapter_sha256") != ref["adapter_sha256"]:
            return _reject(wu, host, "greedy completions differ from the validator's replay", {"claimed": claimed, "expected": expect})
        claim = CreditClaim(host, wu, float(meta["flops_claimed"]), max(float(meta["wall_seconds"]), 1e-6), T4_PEAK_GFLOPS, True, True)
        return Verdict(wu, host, True, "verified: identical greedy token ids", True, None, verified_credit(claim).to_dict(), {"n_prompts": len(claimed)})

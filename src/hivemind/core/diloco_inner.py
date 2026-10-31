# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Phase B (Worker): DiLoCo-QLoRA Inner Loop with Deterministic Micro-Checkpointing & Hardware Safety Circuit-Breaker.

Executes H local AdamW optimization steps on frozen 4-bit NF4 base weights + trainable LoRA
adapters with zero network communication, recording cryptographic SHA-256 micro-checkpoints
every s steps for BOINC spot-check verification while enforcing Phase S-4 GPU hardware safety
invariants (thermal circuit-breaker, WDDM TDR step watchdog, gradient norm clipping, and
NaN/diverged loss early abort).
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

from hivemind.config import configure_deterministic_runtime
from hivemind.core.compression import (
    CompressedDeltaPackage,
    compress_lora_delta,
    compute_pseudo_gradient_delta,
)
from hivemind.core.quantization import (
    compute_state_dict_sha256,
    extract_lora_state_dict,
    load_lora_state_dict,
)
from hivemind.security.gpu_execution_circuit_breaker import (
    DEFAULT_MAX_GRAD_NORM,
    GPUExecutionCircuitBreaker,
)

if TYPE_CHECKING:
    from hivemind.swarm.strategy_optimizer import EF21DiLoCoErrorFeedbackBuffer


@dataclass
class MicroCheckpointRecord:
    """Cryptographic record of intermediate LoRA weights at step s."""

    step: int
    lora_sha256: str
    loss_value: float
    state_snapshot: dict[str, torch.Tensor] = field(repr=False)


@dataclass
class InnerLoopResult:
    """Output produced by a worker node after completing (or preempting) H inner steps."""

    worker_id: str
    branch_name: str
    outer_step: int
    steps_completed: int
    initial_loss: float
    final_loss: float
    elapsed_seconds: float
    theta_init: dict[str, torch.Tensor] = field(repr=False)
    theta_final: dict[str, torch.Tensor] = field(repr=False)
    raw_delta: dict[str, torch.Tensor] = field(repr=False)
    compressed_delta: CompressedDeltaPackage
    micro_checkpoints: dict[int, MicroCheckpointRecord]
    preempted_by_daemon: bool = False
    emergency_dump_path: Path | None = None
    emergency_dump_latency_sec: float = 0.0
    error_feedback_residual_norm: float | None = None
    circuit_breaker_reason: str | None = None
    thermal_throttled_steps: int = 0
    max_step_latency_sec: float = 0.0
    muon_preconditioned_steps: int = 0
    use_muon_lora_preconditioner: bool = False
    muon_ns_steps: int = 5
    use_paged_8bit_adamw: bool = False


def quantize_tensor_blockwise_8bit(
    tensor: torch.Tensor,
    block_size: int = 64,
    signed: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """Quantize a tensor into 8-bit blocks (`int8` if signed else `uint8`) with per-block `absmax`."""
    orig_shape = tensor.shape
    flat = tensor.detach().to(torch.float32).reshape(-1)
    numel = flat.numel()
    pad = (block_size - (numel % block_size)) % block_size
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(-1, block_size)
    if signed:
        absmax = torch.clamp(blocks.abs().amax(dim=1), min=1e-12)
        q = torch.clamp(torch.round((blocks / absmax[:, None]) * 127.0), -127.0, 127.0).to(
            torch.int8
        )
    else:
        pos_blocks = torch.clamp(blocks, min=0.0)
        sqrt_blocks = torch.sqrt(pos_blocks)
        absmax = torch.clamp(sqrt_blocks.amax(dim=1), min=1e-12)
        raw_q = torch.round((sqrt_blocks / absmax[:, None]) * 255.0)
        raw_q = torch.where(pos_blocks > 0.0, torch.clamp(raw_q, min=1.0), raw_q)
        q = torch.clamp(raw_q, 0.0, 255.0).to(torch.uint8)
    return q, absmax, orig_shape


def dequantize_tensor_blockwise_8bit(
    q: torch.Tensor,
    absmax: torch.Tensor,
    orig_shape: torch.Size,
    signed: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize blockwise 8-bit tensor back to `orig_shape`."""
    if signed:
        deq_blocks = (q.to(torch.float32) / 127.0) * absmax[:, None]
    else:
        deq_blocks = ((q.to(torch.float32) / 255.0) * absmax[:, None]).square()
    numel = 1
    for d in orig_shape:
        numel *= d
    return deq_blocks.reshape(-1)[:numel].reshape(orig_shape).to(dtype=dtype)


class Blockwise8BitAdamW(torch.optim.AdamW):
    """Deterministic blockwise 8-bit AdamW optimizer for DiLoCo-QLoRA inner loops.

    Quantizes first (`exp_avg`, signed int8) and second (`exp_avg_sq`, unsigned uint8)
    moment states into 64-element blocks with fp32 `absmax` scales after every step,
    reducing optimizer state footprint by ~3.76x while remaining 100% deterministic across
    CPU and GPU for BOINC spot-check replay verification.
    """

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        block_size: int = 64,
    ) -> None:
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.is_paged_8bit: bool = True
        self.block_size: int = max(8, int(block_size))

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                if "exp_avg_q8" in st and "exp_avg" in st:
                    st["exp_avg"].copy_(
                        dequantize_tensor_blockwise_8bit(
                            st["exp_avg_q8"],
                            st["exp_avg_absmax"],
                            st["exp_avg"].shape,
                            signed=True,
                            dtype=st["exp_avg"].dtype,
                        )
                    )
                if "exp_avg_sq_q8" in st and "exp_avg_sq" in st:
                    st["exp_avg_sq"].copy_(
                        dequantize_tensor_blockwise_8bit(
                            st["exp_avg_sq_q8"],
                            st["exp_avg_sq_absmax"],
                            st["exp_avg_sq"].shape,
                            signed=False,
                            dtype=st["exp_avg_sq"].dtype,
                        )
                    )

        loss = super().step(closure)

        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                if "exp_avg" in st:
                    q1, abs1, sh1 = quantize_tensor_blockwise_8bit(
                        st["exp_avg"], block_size=self.block_size, signed=True
                    )
                    st["exp_avg_q8"] = q1
                    st["exp_avg_absmax"] = abs1
                    st["exp_avg"].copy_(
                        dequantize_tensor_blockwise_8bit(
                            q1, abs1, sh1, signed=True, dtype=st["exp_avg"].dtype
                        )
                    )
                if "exp_avg_sq" in st:
                    q2, abs2, sh2 = quantize_tensor_blockwise_8bit(
                        st["exp_avg_sq"], block_size=self.block_size, signed=False
                    )
                    st["exp_avg_sq_q8"] = q2
                    st["exp_avg_sq_absmax"] = abs2
                    st["exp_avg_sq"].copy_(
                        dequantize_tensor_blockwise_8bit(
                            q2, abs2, sh2, signed=False, dtype=st["exp_avg_sq"].dtype
                        )
                    )
        return loss

    def quantized_state_bytes(self) -> int:
        """Return total byte footprint of the 8-bit quantized moment buffers + block scales."""
        total = 0
        for st in self.state.values():
            for k in ("exp_avg_q8", "exp_avg_absmax", "exp_avg_sq_q8", "exp_avg_sq_absmax"):
                t = st.get(k)
                if isinstance(t, torch.Tensor):
                    total += t.numel() * t.element_size()
        return total

    def fp32_state_bytes(self) -> int:
        """Return equivalent unquantized fp32 moment byte footprint."""
        total = 0
        for st in self.state.values():
            for k in ("exp_avg", "exp_avg_sq"):
                t = st.get(k)
                if isinstance(t, torch.Tensor):
                    total += t.numel() * 4
        return total


def build_inner_optimizer(
    trainable_params: list[nn.Parameter],
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    use_paged_8bit_adamw: bool = False,
) -> torch.optim.Optimizer:
    """Instantiate deterministic AdamW (or 8-bit blockwise AdamW) optimizer for DiLoCo inner loop."""
    if use_paged_8bit_adamw:
        return Blockwise8BitAdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    opt = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    opt.is_paged_8bit = False  # type: ignore[attr-defined]
    return opt


def get_deterministic_batch(
    token_sequences: torch.Tensor,
    step: int,
    rng_seed: int,
    micro_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically index a micro-batch for step `step` given `rng_seed`."""
    num_samples = token_sequences.size(0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(rng_seed) + int(step) * 10007)
    idx = torch.randint(0, num_samples, (micro_batch_size,), generator=gen)
    batch = token_sequences[idx].to(device, dtype=torch.long)
    return batch, batch.clone()


@torch.no_grad()
def _apply_muon_lora_preconditioning(model: nn.Module, preconditioner: object) -> int:
    """Orthogonalize every 2D `lora_` gradient in place; return the number of tensors updated."""
    preconditioned = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None or "lora_" not in name or parameter.grad.ndim != 2:
            continue
        parameter.grad = preconditioner.precondition_gradient(parameter.grad)  # type: ignore[attr-defined]
        preconditioned += 1
    return preconditioned


def run_diloco_inner_loop(
    model: nn.Module,
    token_sequences: torch.Tensor,
    worker_id: str,
    branch_name: str,
    outer_step: int = 1,
    inner_steps_H: int = 500,
    checkpoint_interval_s: int = 50,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    rng_seed: int = 42,
    micro_batch_size: int = 1,
    topk_ratio: float = 0.20,
    use_paged_8bit_adamw: bool = False,
    should_preempt_fn: Callable[[], bool] | None = None,
    emergency_checkpoint_dir: Path | None = None,
    error_feedback: EF21DiLoCoErrorFeedbackBuffer | None = None,
    max_grad_norm: float | None = DEFAULT_MAX_GRAD_NORM,
    circuit_breaker: GPUExecutionCircuitBreaker | None = None,
    enforce_battery_guard: bool = False,
    enable_thermal_sleep: bool = False,
    use_muon_lora_preconditioner: bool = False,
    muon_ns_steps: int = 5,
) -> InnerLoopResult:
    """Execute H local AdamW steps on LoRA parameters with micro-checkpointing every s steps.

    With `error_feedback`, the uploaded package compresses `raw_delta` plus the residual the
    buffer carried from earlier outer steps, and the buffer keeps what this upload dropped.
    `raw_delta`, `theta_final` and the micro-checkpoints are unaffected, so spot-check replay
    sees the same trajectory either way.
    """
    if error_feedback is not None and error_feedback.topk_ratio != topk_ratio:
        raise ValueError(
            f"topk_ratio={topk_ratio} differs from the error-feedback buffer's "
            f"topk_ratio={error_feedback.topk_ratio}"
        )
    if muon_ns_steps < 1:
        raise ValueError(f"muon_ns_steps must be >= 1, got {muon_ns_steps}")
    muon_preconditioner: object | None = None
    if use_muon_lora_preconditioner:
        from hivemind.swarm.diloco_nextgen import MuonNewtonSchulzLoRAPreconditioner

        muon_preconditioner = MuonNewtonSchulzLoRAPreconditioner(ns_steps=muon_ns_steps)
    configure_deterministic_runtime(rng_seed)
    device = next(model.parameters()).device
    model.train()

    cb = circuit_breaker or GPUExecutionCircuitBreaker(
        allow_battery_execution=not enforce_battery_guard
    )
    seq_len = int(token_sequences.size(1)) if token_sequences.ndim >= 2 else int(token_sequences.numel())
    cb.verify_microbatch_tdr_safety(
        micro_batch_size=micro_batch_size,
        seq_len=seq_len,
        raise_on_violation=True,
    )

    theta_init = extract_lora_state_dict(model)
    init_sha256 = compute_state_dict_sha256(theta_init)

    trainable_params = [p for name, p in model.named_parameters() if "lora_" in name or p.requires_grad]
    optimizer = build_inner_optimizer(
        trainable_params=trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        use_paged_8bit_adamw=use_paged_8bit_adamw,
    )

    micro_checkpoints: dict[int, MicroCheckpointRecord] = {
        0: MicroCheckpointRecord(
            step=0,
            lora_sha256=init_sha256,
            loss_value=0.0,
            state_snapshot=theta_init,
        )
    }

    start_time = time.perf_counter()
    initial_loss = 0.0
    final_loss = 0.0
    steps_completed = 0
    preempted = False
    dump_path: Path | None = None
    dump_latency = 0.0
    cb_reason: str | None = None
    thermal_throttled_steps = 0
    max_step_latency_sec = 0.0
    muon_preconditioned_steps = 0

    for step in range(1, inner_steps_H + 1):
        # 1. Hardware Circuit-Breaker & Zero-Intrusion Daemon preemption check
        should_poll_hw = (
            step == 1
            or step % max(1, checkpoint_interval_s) == 0
            or cb.simulated_telemetry is not None
            or cb.is_tripped
        )
        hw_preempt = False
        if should_poll_hw:
            cb_status = cb.evaluate(last_step_latency_sec=max_step_latency_sec)
            if cb_status.should_preempt:
                hw_preempt = True
                cb_reason = f"[{cb_status.rule_id}] {cb_status.reason}"
            elif cb_status.should_throttle:
                thermal_throttled_steps += 1
                if enable_thermal_sleep and cb_status.recommended_sleep_sec > 0.0:
                    time.sleep(cb_status.recommended_sleep_sec)

        daemon_preempt = bool(should_preempt_fn is not None and should_preempt_fn())
        if daemon_preempt or hw_preempt:
            preempted = True
            t_dump = time.perf_counter()
            if emergency_checkpoint_dir is not None:
                emergency_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                dump_path = emergency_checkpoint_dir / f"emergency_{worker_id}_step{step - 1}.pt"
                torch.save(
                    {
                        "worker_id": worker_id,
                        "branch_name": branch_name,
                        "outer_step": outer_step,
                        "step": step - 1,
                        "lora_state": extract_lora_state_dict(model),
                        "optimizer_state": optimizer.state_dict(),
                        "circuit_breaker_reason": cb_reason,
                    },
                    dump_path,
                )
            # Yield VRAM immediately
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            dump_latency = round(time.perf_counter() - t_dump, 4)
            break

        step_t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        input_ids, labels = get_deterministic_batch(
            token_sequences=token_sequences,
            step=step,
            rng_seed=rng_seed,
            micro_batch_size=micro_batch_size,
            device=device,
        )
        outputs = model(input_ids=input_ids, labels=labels)
        loss: torch.Tensor = outputs.loss
        loss_val = float(loss.detach().cpu().item())

        # Verify loss finiteness and workunit wall-clock deadline before backward/step
        elapsed_wall_clock = time.perf_counter() - start_time
        cb.verify_loss_and_wall_clock(
            loss_value=loss_val,
            step=step,
            elapsed_wall_clock_sec=elapsed_wall_clock,
            raise_on_violation=True,
        )

        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
        if muon_preconditioner is not None:
            if _apply_muon_lora_preconditioning(model, muon_preconditioner) > 0:
                muon_preconditioned_steps += 1

        del outputs, loss, input_ids, labels
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step_latency = time.perf_counter() - step_t0
        max_step_latency_sec = max(max_step_latency_sec, step_latency)

        if step == 1:
            initial_loss = loss_val
        final_loss = loss_val
        steps_completed = step

        if step % checkpoint_interval_s == 0 or step == inner_steps_H:
            snap = extract_lora_state_dict(model)
            sha = compute_state_dict_sha256(snap)
            micro_checkpoints[step] = MicroCheckpointRecord(
                step=step,
                lora_sha256=sha,
                loss_value=loss_val,
                state_snapshot=snap,
            )

    optimizer.zero_grad(set_to_none=True)
    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = round(time.perf_counter() - start_time, 4)
    if steps_completed in micro_checkpoints:
        theta_final = micro_checkpoints[steps_completed].state_snapshot
    else:
        theta_final = extract_lora_state_dict(model)
    raw_delta = compute_pseudo_gradient_delta(theta_init, theta_final)
    residual_norm: float | None = None
    if error_feedback is None:
        compressed_delta = compress_lora_delta(
            delta_dict=raw_delta,
            worker_id=worker_id,
            branch_name=branch_name,
            outer_step=outer_step,
            inner_steps_completed=steps_completed,
            topk_ratio=topk_ratio,
        )
    else:
        compressed_delta = error_feedback.compress(
            raw_delta,
            worker_id=worker_id,
            branch_name=branch_name,
            outer_step=outer_step,
            inner_steps_completed=steps_completed,
        )
        residual_norm = error_feedback.residual_norm()

    return InnerLoopResult(
        worker_id=worker_id,
        branch_name=branch_name,
        outer_step=outer_step,
        steps_completed=steps_completed,
        initial_loss=round(initial_loss, 6),
        final_loss=round(final_loss, 6),
        elapsed_seconds=elapsed,
        theta_init=theta_init,
        theta_final=theta_final,
        raw_delta=raw_delta,
        compressed_delta=compressed_delta,
        micro_checkpoints=micro_checkpoints,
        preempted_by_daemon=preempted,
        emergency_dump_path=dump_path,
        emergency_dump_latency_sec=dump_latency,
        error_feedback_residual_norm=residual_norm,
        circuit_breaker_reason=cb_reason,
        thermal_throttled_steps=thermal_throttled_steps,
        max_step_latency_sec=round(max_step_latency_sec, 6),
        muon_preconditioned_steps=muon_preconditioned_steps,
        use_muon_lora_preconditioner=use_muon_lora_preconditioner,
        muon_ns_steps=muon_ns_steps,
        use_paged_8bit_adamw=use_paged_8bit_adamw,
    )


def replay_inner_segment(
    model: nn.Module,
    start_lora_state: dict[str, torch.Tensor],
    token_sequences: torch.Tensor,
    start_step: int,
    end_step: int,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    rng_seed: int = 42,
    micro_batch_size: int = 1,
    initial_step_state: dict[str, torch.Tensor] | None = None,
    max_grad_norm: float | None = DEFAULT_MAX_GRAD_NORM,
    use_muon_lora_preconditioner: bool = False,
    muon_ns_steps: int = 5,
    use_paged_8bit_adamw: bool = False,
) -> dict[str, torch.Tensor]:
    """Deterministically replay a segment of inner steps for BOINC Validator spot-checking."""
    if muon_ns_steps < 1:
        raise ValueError(f"muon_ns_steps must be >= 1, got {muon_ns_steps}")
    muon_preconditioner: object | None = None
    if use_muon_lora_preconditioner:
        from hivemind.swarm.diloco_nextgen import MuonNewtonSchulzLoRAPreconditioner

        muon_preconditioner = MuonNewtonSchulzLoRAPreconditioner(ns_steps=muon_ns_steps)
    configure_deterministic_runtime(rng_seed)
    device = next(model.parameters()).device
    model.train()

    # If verifying from step 0 or replaying from step 0 through end_step for exact AdamW moments
    replay_from = 0 if initial_step_state is not None else start_step
    base_state = initial_step_state if initial_step_state is not None else start_lora_state
    load_lora_state_dict(model, base_state)

    trainable_params = [p for name, p in model.named_parameters() if "lora_" in name or p.requires_grad]
    optimizer = build_inner_optimizer(
        trainable_params=trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        use_paged_8bit_adamw=use_paged_8bit_adamw,
    )

    for step in range(replay_from + 1, end_step + 1):
        optimizer.zero_grad(set_to_none=True)
        input_ids, labels = get_deterministic_batch(
            token_sequences=token_sequences,
            step=step,
            rng_seed=rng_seed,
            micro_batch_size=micro_batch_size,
            device=device,
        )
        outputs = model(input_ids=input_ids, labels=labels)
        loss: torch.Tensor = outputs.loss
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
        if muon_preconditioner is not None:
            _apply_muon_lora_preconditioning(model, muon_preconditioner)
        del outputs, loss, input_ids, labels
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return extract_lora_state_dict(model)

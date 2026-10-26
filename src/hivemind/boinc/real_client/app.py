"""HiveMind GPU science app run by a real BOINC client (anonymous platform).

The BOINC client starts this program inside a slot directory. The workunit input and the result output are
reached through the logical names ``in.json`` / ``out.json``; on Linux the client makes them symlinks, on other
platforms a ``<soft_link>`` text file, and :func:`resolve_filename` handles both (``boinc_resolve_filename``).

The work is a seeded, deterministic mini DiLoCo inner loop (a low-rank adapter on a frozen linear layer): the
same workunit on the same execution class reproduces bit-for-bit, which is what the HiveMind spot-check
validator relies on. The app checkpoints atomically so a preempted client resumes instead of restarting.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

APP_NAME = "hivemind_inner_loop"
APP_VERSION = 100


def resolve_filename(logical: str, base: Path | None = None) -> Path:
    """Equivalent of ``boinc_resolve_filename``: follow a symlink or a ``<soft_link>`` file."""
    p = (base or Path.cwd()) / logical
    if p.is_symlink():
        return Path(os.path.realpath(p))
    if p.is_file() and p.stat().st_size < 4096:
        m = re.search(r"<soft_link>(.*?)</soft_link>", p.read_text(errors="ignore"), re.S)
        if m:
            return Path(m.group(1).strip())
    return p


def class_fingerprint(device: str, dtype: str = "fp32") -> dict[str, Any]:
    import torch

    name = torch.cuda.get_device_name(0).replace(" ", "_") if device == "cuda" else "cpu_x86_64"
    return {
        "accelerator_class": name,
        "torch_build": torch.__version__,
        "dtype": dtype,
        "thread_count": 1 if device == "cuda" else torch.get_num_threads(),
        "deterministic_flags": True,
    }


def _build(spec: dict[str, Any], device: str):
    import torch

    g = torch.Generator(device="cpu").manual_seed(int(spec["seed"]))
    d_in, d_out, rank = int(spec["d_in"]), int(spec["d_out"]), int(spec["rank"])
    w0 = torch.randn(d_out, d_in, generator=g) / d_in**0.5
    a = torch.randn(rank, d_in, generator=g) * 0.02
    b = torch.zeros(d_out, rank)
    x = torch.randn(int(spec["batch"]) * int(spec["steps"]), d_in, generator=g)
    u = torch.randn(d_out, rank, generator=g) / rank**0.5
    v = torch.randn(rank, d_in, generator=g) / d_in**0.5
    y = x @ (w0.cpu() + u @ v).T  # the adapter can fit this exactly: a real, checkable learning signal
    return (w0.to(device), a.to(device).requires_grad_(), b.to(device).requires_grad_(), x.to(device), y.to(device))


def q16_digest(state: dict[str, Any]) -> str:
    from hivemind.boinc.validator import quantize_fixed_point_q16_state

    h = hashlib.sha256()
    for k, t in quantize_fixed_point_q16_state(state).items():
        h.update(k.encode())
        h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()


def run(
    spec: dict[str, Any],
    device: str,
    ckpt: Path,
    crash_after_step: int | None = None,
    inject_fault: float = 0.0,
) -> dict[str, Any]:
    """Run (or resume) the inner loop; returns the result document.

    ``crash_after_step`` simulates preemption; ``inject_fault`` adds that much noise to the reported adapter, a
    red-team control standing in for a cheating volunteer (the server must refuse such a result).
    """
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    w0, a, b, x, y = _build(spec, device)
    batch, steps, lr = int(spec["batch"]), int(spec["steps"]), float(spec["lr"])
    every = int(spec["spot_check_interval"])
    micro: dict[str, str] = {}
    start, losses = 0, []
    if ckpt.exists():
        c = torch.load(ckpt, map_location=device)
        a.data.copy_(c["a"])
        b.data.copy_(c["b"])
        start, losses, micro = int(c["step"]), list(c["losses"]), dict(c["micro"])
    t0 = time.time()
    for step in range(start, steps):
        xb, yb = x[step * batch : (step + 1) * batch], y[step * batch : (step + 1) * batch]
        pred = xb @ (w0 + b @ a).T
        loss = ((pred - yb) ** 2).mean()
        ga, gb = torch.autograd.grad(loss, [a, b])
        with torch.no_grad():
            a -= lr * ga
            b -= lr * gb
        losses.append(float(loss))
        if (step + 1) % every == 0 or step + 1 == steps:
            micro[str(step + 1)] = q16_digest({"a": a.detach().cpu(), "b": b.detach().cpu()})
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"step": step + 1, "a": a.detach(), "b": b.detach(), "losses": losses, "micro": micro}, tmp)
            os.replace(tmp, ckpt)
            if crash_after_step is not None and step + 1 >= crash_after_step:
                raise SystemExit(75)  # simulated preemption, restarted by the caller / client
    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    if inject_fault:
        with torch.no_grad():
            b += inject_fault * torch.ones_like(b)
        micro[str(steps)] = q16_digest({"a": a.detach().cpu(), "b": b.detach().cpu()})
    d_in, d_out, rank = int(spec["d_in"]), int(spec["d_out"]), int(spec["rank"])
    flops = 6.0 * batch * (d_in * d_out + rank * (d_in + d_out)) * steps  # fwd+bwd of the adapted layer
    return {
        "app": APP_NAME,
        "app_version": APP_VERSION,
        "wu_id": spec["wu_id"],
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "fingerprint": class_fingerprint(device),
        "resumed_from_step": start,
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "micro_digests": micro,
        "state_b64": {k: base64.b64encode(t.detach().cpu().contiguous().numpy().tobytes()).decode() for k, t in (("a", a), ("b", b))},
        "inject_fault": inject_fault,
        "final_digest": micro[str(steps)],
        "flops_claimed": flops,
        "wall_seconds": wall,
    }


def boinc_finish(status: int, cwd: Path | None = None) -> None:
    """What ``boinc_finish()`` in the BOINC API does (``api/boinc_api.cpp``): write ``boinc_finish_called``.

    Without this file the client treats an exit as premature and restarts the task, so every BOINC app, API
    linked or not, must leave it behind.
    """
    (cwd or Path.cwd()).joinpath("boinc_finish_called").write_text(f"{status}\n")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    device = "cuda" if "--cpu" not in args else "cpu"
    fault = next((float(a.split("=")[1]) for a in args if a.startswith("--inject-fault=")), 0.0)
    crash = next((int(a.split("=")[1]) for a in args if a.startswith("--crash-after=")), None)
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print("no CUDA device visible to the app", file=sys.stderr)
        return 2
    spec = json.loads(resolve_filename("in.json").read_text())
    out = resolve_filename("out.json")
    ckpt = Path("hivemind_ckpt.pt")
    result = run(spec, device, ckpt, crash_after_step=crash, inject_fault=fault)
    out.write_text(json.dumps(result, sort_keys=True))
    ckpt.unlink(missing_ok=True)
    boinc_finish(0)
    return 0


def _entry() -> int:
    try:
        return main()
    except SystemExit:
        raise  # simulated preemption (75): no finish file, so the client restarts the task
    except BaseException as exc:  # a real failure must be reported, not retried forever
        print(f"app error: {exc!r}", file=sys.stderr)
        boinc_finish(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(_entry())

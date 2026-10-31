#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HiveMind BOINC & Steam Idle-GPU Worker (`scripts/hivemind_boinc_idle_worker.py`).

Cross-platform (Linux / Windows / macOS) citizen-science and night-training worker that:
1. Monitors user input inactivity and NVIDIA GPU utilization/temperature (`nvidia-smi`).
2. Automatically pauses BOINC GPU compute (`boinccmd --set_gpu_mode never`, 0% GPU overhead, 0 CUDA allocations)
   while the user is actively playing a game (`idle_seconds < --idle-threshold-seconds` or `gpu_utilization > --max-gpu-load`),
   guaranteeing a locked 60 FPS gaming experience.
3. When the GPU is idle it launches an NVIDIA CUDA Driver API PTX kernel (`bls_fold_kernel` via `nvcuda.dll` / `libcuda.so.1`)
   on the same GPU the idle gate measured, as a local self-check fold over a synthetic light curve, and only then resumes BOINC
   (`boinccmd --set_gpu_mode auto`). If the fold falls back to the CPU (no libcuda, PTX load failure) the cycle is BLOCKED
   (`cuda_fold_unavailable`) and BOINC is not touched.

It does NOT lease workunits, train anything or submit anything: QLoRA training is BLOCKED (`no_trainer_wired`), so no HMT points
or cobblestones are earned by running it. When no GPU can be read the whole cycle is BLOCKED (`gpu_probe`).
"""

from __future__ import annotations

import argparse
import array
import ctypes
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import Any, Callable, Dict, Final, Literal, NamedTuple

DEFAULT_COORDINATOR_URL = "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app"

# Target transit depths (ppm) from live NASA Exoplanet Archive TAP API (mcp-servers/exoplanet_mcp.py)
TARGET_TRANSIT_PPM: Dict[str, tuple[float, float]] = {
    "K2-18 b": (2789.0, 32.93962),
    "TRAPPIST-1 e": (4996.0, 6.10101),
    "LHS 1140 b": (5385.0, 24.73723),
    "TOI-700 d": (653.0, 37.42346),
    "HD 80606 b": (10179.0, 111.43676),
    "WASP-39 b": (19549.0, 4.05529),
    "Kepler-11 f": (458.0, 46.6888),
    "Kepler-186 f": (420.0, 129.9441),
}

# Portable NVIDIA PTX 6.0 / sm_50+ kernel compiled by the NVIDIA driver JIT on any Maxwell/Pascal/Turing/Ampere/Ada GPU
_BLS_CUDA_PTX = b"""
.version 6.0
.target sm_50
.address_size 64

.visible .entry bls_fold_kernel(
    .param .u64 flux_ptr,
    .param .u64 out_ptr,
    .param .u32 n_pts
)
{
    .reg .pred %p1;
    .reg .b32 %r<5>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [flux_ptr];
    ld.param.u64 %rd2, [out_ptr];
    ld.param.u32 %r1, [n_pts];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra DONE;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];
    mov.f32 %f2, 0f3F800000;
    sub.f32 %f3, %f2, %f1;
    mul.f32 %f3, %f3, 0f49742400;
    add.s64 %rd6, %rd2, %rd4;
    st.global.f32 [%rd6], %f3;
DONE:
    ret;
}
"""


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


def get_user_idle_seconds(assume_idle_when_unknown: bool = False) -> float:
    """Return seconds since last keyboard/mouse input on Windows, Linux (xprintidle), or macOS (IOHIDSystem).

    When the idle time cannot be measured (no xprintidle, Wayland, a headless box, a failed call) the safe answer for someone who is
    playing is "the user is active": 0.0, which the idle gate never treats as idle. A dedicated headless machine that really is never
    used can opt back in with ``assume_idle_when_unknown=True`` (``--assume-idle-when-unmeasurable``), which answers 10.0.
    """
    if sys.platform == "win32":
        try:
            lii = _LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(_LASTINPUTINFO)
            if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):  # type: ignore[attr-defined]
                millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime  # type: ignore[attr-defined]
                return max(0.0, float(millis) / 1000.0)
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        xprintidle = shutil.which("xprintidle")
        if xprintidle:
            try:
                res = subprocess.run([xprintidle], capture_output=True, text=True, timeout=1.5, check=False)
                if res.returncode == 0 and res.stdout.strip().isdigit():
                    return max(0.0, float(res.stdout.strip()) / 1000.0)
            except Exception:
                pass
    elif sys.platform == "darwin":
        ioreg = shutil.which("ioreg")
        if ioreg:
            try:
                res = subprocess.run([ioreg, "-c", "IOHIDSystem"], capture_output=True, text=True, timeout=2.0, check=False)
                m = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', res.stdout)
                if m:
                    return max(0.0, float(m.group(1)) / 1_000_000_000.0)
            except Exception:
                pass
    return 10.0 if assume_idle_when_unknown else 0.0


NVIDIA_SMI_QUERY: Final[tuple[str, ...]] = (
    "nvidia-smi",
    "--query-gpu=name,utilization.gpu,temperature.gpu,memory.total,memory.used,power.draw,driver_version,pci.bus_id",
    "--format=csv,noheader,nounits",
)

_DETAIL_MAX_CHARS: Final[int] = 200


# NamedTuple (not @dataclass): the gateway execs this file per request in a threadpool without registering it in sys.modules,
# and dataclass needs a sys.modules entry for string annotations. NamedTuple needs none, so nothing global is mutated at import.
class GpuReading(NamedTuple):
    """One GPU as reported by the driver. Optional fields are None when the driver says N/A."""

    name: str
    total_mib: float
    used_mib: float | None
    utilization_pct: float | None
    temp_c: float | None
    power_w: float | None
    driver_version: str | None
    pci_bus_id: str | None = None


class GpuProbe(NamedTuple):
    """Result of a truthful GPU capability probe. BLOCKED means no GPU could be read; nothing is guessed."""

    status: Literal["READY", "BLOCKED"]
    reason: str  # 'ok' | 'nvidia_smi_missing' | 'nvidia_smi_failed' | 'nvidia_smi_timeout' | 'no_gpu_rows' | 'unparseable_output'
    gpus: tuple[GpuReading, ...]
    detail: str  # truncated stderr/stdout, <= 200 chars, '' when READY

    def best(self) -> GpuReading | None:
        """Return the GPU with the largest total VRAM, or None when the probe is BLOCKED."""
        if self.status != "READY" or not self.gpus:
            return None
        return max(self.gpus, key=lambda g: g.total_mib)


def _optional_float(token: str) -> float | None:
    """Parse an optional numeric nvidia-smi field; '[N/A]', 'N/A', '' and anything non-finite give None."""
    text = token.strip()
    if not text or text.upper() in ("N/A", "[N/A]") or text.startswith("[Not"):
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _text_field(parts: list[str], index: int) -> str | None:
    """Return a non-empty, non-N/A text column or None."""
    if len(parts) <= index:
        return None
    value = parts[index].strip()
    return None if not value or value.upper() in ("N/A", "[N/A]") else value


def parse_nvidia_smi_csv(stdout: str) -> tuple[GpuReading, ...]:
    """Parse `nvidia-smi --format=csv,noheader,nounits` output. Malformed lines are dropped, never defaulted."""
    readings: list[GpuReading] = []
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[0]:
            continue
        total = _optional_float(parts[3])
        if total is None or total <= 0.0:
            continue
        driver = parts[6] if len(parts) > 6 and parts[6] and parts[6].upper() not in ("N/A", "[N/A]") else None
        readings.append(
            GpuReading(
                name=parts[0],
                total_mib=total,
                used_mib=_optional_float(parts[4]) if len(parts) > 4 else None,
                utilization_pct=_optional_float(parts[1]),
                temp_c=_optional_float(parts[2]),
                power_w=_optional_float(parts[5]) if len(parts) > 5 else None,
                driver_version=driver,
                pci_bus_id=_text_field(parts, 7),
            )
        )
    return tuple(readings)


def _blocked(reason: str, detail: str = "") -> GpuProbe:
    return GpuProbe(status="BLOCKED", reason=reason, gpus=(), detail=detail.strip()[:_DETAIL_MAX_CHARS])


def probe_gpu(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    timeout_s: float = 5.0,
) -> GpuProbe:
    """Read the real GPU list via `nvidia-smi`. Never raises and never invents a GPU: failure is BLOCKED."""
    if which(NVIDIA_SMI_QUERY[0]) is None:
        return _blocked("nvidia_smi_missing")
    try:
        res = run(NVIDIA_SMI_QUERY, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        return _blocked("nvidia_smi_timeout", str(exc))
    except FileNotFoundError as exc:
        return _blocked("nvidia_smi_missing", str(exc))
    except (OSError, subprocess.SubprocessError) as exc:
        return _blocked("nvidia_smi_failed", f"{type(exc).__name__}: {exc}")
    stdout = res.stdout if isinstance(res.stdout, str) else ""
    stderr = res.stderr if isinstance(res.stderr, str) else ""
    if res.returncode != 0:
        return _blocked("nvidia_smi_failed", stderr.strip() or stdout.strip() or f"returncode={res.returncode}")
    if not stdout.strip():
        return _blocked("no_gpu_rows", stderr)
    gpus = parse_nvidia_smi_csv(stdout)
    if not gpus:
        return _blocked("unparseable_output", stdout)
    return GpuProbe(status="READY", reason="ok", gpus=gpus, detail="")


def _reading_to_dict(g: GpuReading) -> Dict[str, Any]:
    return {
        "gpu_model": g.name,
        "utilization_pct": g.utilization_pct,
        "temp_c": g.temp_c,
        "total_mib": g.total_mib,
        "used_mib": g.used_mib,
        "power_w": g.power_w,
        "driver_version": g.driver_version,
        "pci_bus_id": g.pci_bus_id,
    }


def query_nvidia_gpu_status() -> Dict[str, Any]:
    """Return the real status of the largest GPU as a dict with ``available: True``.

    If the card cannot be read the answer is ``available: False`` with the card treated as BUSY and HOT (100 %, 99 C), so a caller that
    forgets to check ``available`` still does not start. It never invents a card (the old fallback reported a made-up RTX 2070 at 8 % load).
    """
    best = probe_gpu(run=subprocess.run, which=shutil.which).best()
    if best is None:
        return {"available": False, "gpu_model": "unknown", "utilization_pct": 100.0, "temp_c": 99.0, "total_mib": 0.0, "used_mib": 0.0,
                "power_w": None, "driver_version": "unknown", "pci_bus_id": None}
    return {"available": True, **_reading_to_dict(best)}


def gpu_is_idle_for_boinc(
    idle_s: float,
    gpu: GpuReading,
    idle_threshold_seconds: float,
    max_gpu_load: float,
    max_temp_c: float = 78.0,
) -> bool:
    """True only when the user is idle and the GPU is measurably cool and unloaded. Unknown is not idle."""
    if gpu.utilization_pct is None or gpu.temp_c is None:
        return False
    return bool(idle_s >= idle_threshold_seconds and gpu.utilization_pct < max_gpu_load and gpu.temp_c < max_temp_c)


QLORA_BLOCKED: Final[Dict[str, str]] = {"status": "BLOCKED", "reason": "no_trainer_wired"}


def blocked_report(commander: str, coordinator_url: str, domain: str, probe: GpuProbe, boinc_driver: str | None = None) -> Dict[str, Any]:
    """Cycle report for a node whose GPU cannot be read: no fold, no lease, no submit, no network.

    ``boinc_driver`` is the result of the only BOINC call allowed here, a PAUSE (an unreadable GPU must not leave BOINC running next to a game).
    """
    return {
        "status": "BLOCKED",
        "stage": "gpu_probe",
        "reason": probe.reason,
        "detail": probe.detail,
        "handle": commander,
        "coordinator_url": coordinator_url,
        "domain": domain,
        "boinc_active": False,
        "boinc_driver": boinc_driver,
        "boinc_touched": bool(boinc_driver and boinc_driver.startswith("boinccmd_gpu_")),
        "lease_attempted": False,
        "submit_attempted": False,
        "qlora": dict(QLORA_BLOCKED),
    }


CUDA_BACKEND: Final[str] = "nvidia_cuda_driver_ptx_sm50"


def _select_cuda_device(cuda_lib: Any, gpu: GpuReading | None) -> ctypes.c_int | None:
    """Return the CUDA device that is the same physical GPU as ``gpu`` (the one the idle gate measured), or None.

    nvidia-smi indexes and CUDA ordinals can differ (CUDA defaults to fastest-first), so the device is matched by PCI bus id, never by
    position. With no PCI id it is only accepted when the machine has exactly one CUDA device. ``gpu=None`` keeps the legacy ordinal 0.
    """
    dev = ctypes.c_int()
    if gpu is None:
        return dev if cuda_lib.cuDeviceGet(ctypes.byref(dev), 0) == 0 else None
    if gpu.pci_bus_id:
        if cuda_lib.cuDeviceGetByPCIBusId(ctypes.byref(dev), gpu.pci_bus_id.encode("ascii", "replace")) != 0:
            return None
    else:
        count = ctypes.c_int()
        if cuda_lib.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value != 1 or cuda_lib.cuDeviceGet(ctypes.byref(dev), 0) != 0:
            return None
    name_buf = ctypes.create_string_buffer(100)
    if cuda_lib.cuDeviceGetName(name_buf, 100, dev) != 0 or name_buf.value.decode("utf-8", "replace") != gpu.name:
        return None
    return dev


def cuda_fold_ran_on_gpu(cuda_fold: Dict[str, Any]) -> bool:
    """True only when the fold result says the kernel ran on the CUDA backend (not cpu_fallback, not paused)."""
    return cuda_fold.get("backend") == CUDA_BACKEND


def blocked_fold_report(commander: str, coordinator_url: str, domain: str, cuda_fold: Dict[str, Any]) -> Dict[str, Any]:
    """Cycle report when the GPU was readable and idle but the fold did not run on CUDA: BOINC untouched, no lease, no submit."""
    return {
        "status": "BLOCKED",
        "stage": "cuda_fold",
        "reason": "cuda_fold_unavailable",
        "detail": f"backend={cuda_fold.get('backend')}",
        "handle": commander,
        "coordinator_url": coordinator_url,
        "domain": domain,
        "boinc_active": False,
        "boinc_touched": False,
        "lease_attempted": False,
        "submit_attempted": False,
        "cuda_fold": cuda_fold,
        "qlora": dict(QLORA_BLOCKED),
    }


def execute_cuda_ptx_bls_fold(
    target_planet: str = "K2-18 b",
    n_pts: int = 16384,
    gpu: GpuReading | None = None,
    cuda_lib: Any = None,
) -> Dict[str, Any]:
    """Execute the `bls_fold_kernel` PTX kernel on the NVIDIA GPU via `nvcuda.dll` / `libcuda.so.1`.

    ``gpu`` is the device the idle gate measured: the fold runs on that same card (matched by PCI bus id) or not on CUDA at all.
    The result's ``backend`` is ``nvidia_cuda_driver_ptx_sm50`` only when the kernel really ran on the GPU; any failure (no libcuda,
    device mismatch, PTX load, launch or copy error) returns ``cpu_fallback``, which callers must not treat as a GPU result.
    ``cuda_lib`` injects the driver library (tests); by default the real one is loaded.
    """
    expected_ppm, period_days = TARGET_TRANSIT_PPM.get(target_planet, (2789.0, 32.93962))
    dip_fraction = expected_ppm / 1_000_000.0
    host_flux = array.array("f", (1.0 - dip_fraction if 2048 <= i < 2304 else 1.0 for i in range(n_pts)))
    host_out = array.array("f", [0.0] * n_pts)
    byte_len = n_pts * 4

    for lib_name in (("nvcuda.dll", "libcuda.so.1", "libcuda.so") if cuda_lib is None else ()):
        try:
            cuda_lib = ctypes.WinDLL(lib_name) if sys.platform == "win32" else ctypes.CDLL(lib_name)
            break
        except Exception:
            continue

    if cuda_lib is not None:
        ctx = ctypes.c_void_p()
        try:
            if cuda_lib.cuInit(0) == 0:
                selected = _select_cuda_device(cuda_lib, gpu)
                if selected is None:
                    raise RuntimeError("cuda_device_mismatch")
                dev = selected
                dev_name_buf = ctypes.create_string_buffer(100)
                cuda_lib.cuDeviceGetName(dev_name_buf, 100, dev)
                total_vram = ctypes.c_size_t()
                cuda_lib.cuDeviceTotalMem_v2(ctypes.byref(total_vram), dev)
                if cuda_lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) == 0:
                    mod = ctypes.c_void_p()
                    fn = ctypes.c_void_p()
                    rc_mod = cuda_lib.cuModuleLoadData(ctypes.byref(mod), _BLS_CUDA_PTX)
                    rc_fn = cuda_lib.cuModuleGetFunction(ctypes.byref(fn), mod, b"bls_fold_kernel")
                    if rc_mod == 0 and rc_fn == 0:
                        d_flux = ctypes.c_uint64()
                        d_out = ctypes.c_uint64()
                        rc_alloc = cuda_lib.cuMemAlloc_v2(ctypes.byref(d_flux), byte_len) | cuda_lib.cuMemAlloc_v2(ctypes.byref(d_out), byte_len)
                        flux_addr, _ = host_flux.buffer_info()
                        out_addr, _ = host_out.buffer_info()
                        rc_in = cuda_lib.cuMemcpyHtoD_v2(d_flux, ctypes.c_void_p(flux_addr), byte_len)
                        n_u32 = ctypes.c_uint32(n_pts)
                        kernel_args = (ctypes.c_void_p * 3)(
                            ctypes.cast(ctypes.byref(d_flux), ctypes.c_void_p),
                            ctypes.cast(ctypes.byref(d_out), ctypes.c_void_p),
                            ctypes.cast(ctypes.byref(n_u32), ctypes.c_void_p),
                        )
                        grid_x = (n_pts + 255) // 256
                        t0 = time.perf_counter()
                        rc_launch = cuda_lib.cuLaunchKernel(fn, grid_x, 1, 1, 256, 1, 1, 0, None, kernel_args, None)
                        rc_sync = cuda_lib.cuCtxSynchronize()
                        kernel_ms = (time.perf_counter() - t0) * 1000.0
                        rc_out = cuda_lib.cuMemcpyDtoH_v2(ctypes.c_void_p(out_addr), d_out, byte_len)
                        cuda_lib.cuMemFree_v2(d_flux)
                        cuda_lib.cuMemFree_v2(d_out)
                        cuda_lib.cuCtxDestroy_v2(ctx)
                        ctx = ctypes.c_void_p()
                        if rc_alloc == 0 and rc_in == 0 and rc_launch == 0 and rc_sync == 0 and rc_out == 0:
                            peak_ppm = round(float(max(host_out)), 1)
                            in_transit_sum = round(float(sum(host_out[2048:2304])), 2)
                            digest = hashlib.sha256(
                                f"CUDA_PTX|{target_planet}|{n_pts}|{peak_ppm}|{in_transit_sum}".encode("utf-8")
                            ).hexdigest()
                            return {
                                "backend": CUDA_BACKEND,
                                "cuda_device": dev_name_buf.value.decode("utf-8", "replace"),
                                "cuda_total_vram_mib": int(total_vram.value // (1024 * 1024)),
                                "n_points_folded": n_pts,
                                "vram_allocated_bytes": byte_len * 2,
                                "kernel_exec_ms": round(kernel_ms, 3),
                                "recovered_transit_depth_ppm": peak_ppm,
                                "orbital_period_days": period_days,
                                "result_sha256": digest,
                            }
        except Exception:
            pass
        finally:
            if ctx.value:
                try:
                    cuda_lib.cuCtxDestroy_v2(ctx)
                except Exception:
                    pass

    t0 = time.perf_counter()
    for i in range(n_pts):
        host_out[i] = (1.0 - host_flux[i]) * 1_000_000.0
    cpu_ms = (time.perf_counter() - t0) * 1000.0
    peak_ppm = round(float(max(host_out)), 1)
    digest = hashlib.sha256(f"CPU_FALLBACK|{target_planet}|{n_pts}|{peak_ppm}".encode("utf-8")).hexdigest()
    return {
        "backend": "cpu_fallback",
        "cuda_device": "CPU",
        "cuda_total_vram_mib": 0,
        "n_points_folded": n_pts,
        "vram_allocated_bytes": 0,
        "kernel_exec_ms": round(cpu_ms, 3),
        "recovered_transit_depth_ppm": peak_ppm,
        "orbital_period_days": period_days,
        "result_sha256": digest,
    }


def set_boinc_client_gpu_mode(enable_gpu: bool) -> str:
    """Pause (`never`) or resume (`auto`) the standard BOINC client GPU queue via `boinccmd` if installed."""
    boinccmd = shutil.which("boinccmd")
    if not boinccmd:
        return "boinccmd_not_found"  # nothing was paused or resumed: say so
    mode = "auto" if enable_gpu else "never"
    try:
        done = subprocess.run([boinccmd, "--set_gpu_mode", mode], capture_output=True, timeout=2.5, check=False)
    except Exception:
        return "boinccmd_failed"
    return f"boinccmd_gpu_{mode}" if done.returncode == 0 else f"boinccmd_failed_rc{done.returncode}"


def run_one_idle_gpu_boinc_cycle(
    commander: str = "Cmdr_JeanPaul_NCC31337",
    coordinator_url: str = DEFAULT_COORDINATOR_URL,
    domain: str = "physics_sim",
    idle_threshold_seconds: float = 4.0,
    max_gpu_load: float = 45.0,
    target_planet: str = "K2-18 b",
    force_fold: bool = False,
    dry_run: bool = False,
    assume_idle_when_unknown: bool = False,
) -> Dict[str, Any]:
    """Run one idle-GPU check, pause/resume BOINC accordingly and, when idle, run the CUDA fold.

    Returns a BLOCKED report (BOINC is only ever paused, never resumed; no fold, no network) when the GPU cannot be read. QLoRA training is
    reported as BLOCKED (`no_trainer_wired`): no workunit is leased, no delta is submitted.
    `force_fold` and `dry_run` only bypass the user-idle timer, never a BLOCKED probe or a busy/hot GPU.
    The idle gate is evaluated on the GPU with the most VRAM and the fold runs on that same card. The cycle is READY (and BOINC is
    resumed) only if the fold really ran on CUDA; a CPU fallback gives BLOCKED `cuda_fold_unavailable` and boinccmd is not called.
    """
    probe = probe_gpu()
    best = probe.best()
    if best is None:
        return blocked_report(commander, coordinator_url, domain, probe, set_boinc_client_gpu_mode(False))

    idle_s = get_user_idle_seconds(assume_idle_when_unknown)
    gpu = _reading_to_dict(best)
    timer_s = float("inf") if (force_fold or dry_run) else idle_s
    is_gpu_idle_for_boinc = gpu_is_idle_for_boinc(timer_s, best, idle_threshold_seconds, max_gpu_load)

    if is_gpu_idle_for_boinc:
        cuda_fold = execute_cuda_ptx_bls_fold(target_planet=target_planet, n_pts=16384, gpu=best)
        if not cuda_fold_ran_on_gpu(cuda_fold):
            return blocked_fold_report(commander, coordinator_url, domain, cuda_fold)
        boinc_driver = set_boinc_client_gpu_mode(True)
    else:
        boinc_driver = set_boinc_client_gpu_mode(False)
        cuda_fold = {
            "backend": "paused_for_active_gameplay",
            "cuda_device": best.name,
            "cuda_total_vram_mib": int(best.total_mib),
            "n_points_folded": 0,
            "vram_allocated_bytes": 0,
            "kernel_exec_ms": 0.0,
            "recovered_transit_depth_ppm": 0.0,
            "orbital_period_days": TARGET_TRANSIT_PPM.get(target_planet, (2789.0, 32.93962))[1],
            "result_sha256": hashlib.sha256(f"PAUSED|{commander}|{time.time():.3f}".encode("utf-8")).hexdigest(),
        }
    digest = cuda_fold["result_sha256"]

    sync_result: Dict[str, Any] = {"local_hub": "not_attempted"}
    if is_gpu_idle_for_boinc:
        # Dev-only convenience: tell a locally running hub about the fold. No federated workunit is
        # leased or submitted because no real trainer exists; the coordinator is never contacted.
        payload = {
            "commander": commander,
            "handle": commander,
            "project_id": "hivemind_tess_at_home",
            "gpu_idle_seconds": round(idle_s, 2),
            "gpu_utilization_pct": best.utilization_pct,
            "gpu_model": best.name,
            "target_planet": target_planet,
            "boinc_driver": boinc_driver,
            "cuda_fold": cuda_fold,
            "digest": digest,
        }
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8095/api/v1/community/boinc-sync",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                sync_result["local_hub"] = json.loads(resp.read().decode("utf-8"))
        except Exception:
            sync_result["local_hub"] = "not_running_locally"

    out: Dict[str, Any] = {
        "status": "READY",
        "handle": commander,
        "coordinator_url": coordinator_url,
        "domain": domain,
        "target_planet": target_planet,
        "idle_seconds": round(idle_s, 2),
        "idle_threshold_seconds": idle_threshold_seconds,
        "gpu": gpu,
        "boinc_active": is_gpu_idle_for_boinc,
        "boinc_driver": boinc_driver,
        "cuda_fold": cuda_fold,
        "qlora": dict(QLORA_BLOCKED),
        "lease_attempted": False,
        "submit_attempted": False,
        "sha256": digest,
        "sync": sync_result,
    }
    if dry_run:
        out["dry_run"] = True
        out["dry_run_passed"] = bool(cuda_fold["n_points_folded"] == 16384 and cuda_fold_ran_on_gpu(cuda_fold))
    return out


def _transition_key(report: Dict[str, Any]) -> tuple[Any, ...]:
    return (report.get("status"), report.get("reason"), report.get("boinc_active"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="HiveMind BOINC & Steam Idle-GPU Worker: pauses GPU work while you play and folds on CUDA when idle. Reports BLOCKED when no GPU can be read."
    )
    parser.add_argument("--coordinator", default=DEFAULT_COORDINATOR_URL, help="HTTPS URL of the HiveMind BOINC / La Ruche coordinator")
    parser.add_argument("--handle", default="volunteer-node", help="Public contributor handle credited with HMT points")
    parser.add_argument("--domain", default="physics_sim", help="Preferred workunit domain (physics_sim, code_generation, math_reasoning)")
    parser.add_argument("--target-planet", default="K2-18 b", help="Exoplanet light curve target to fold via CUDA PTX BLS kernel")
    parser.add_argument("--idle-threshold-seconds", type=float, default=4.0, help="Seconds of user inactivity before GPU folding resumes")
    parser.add_argument("--max-gpu-load", type=float, default=45.0, help="Pause BOINC if GPU load exceeds this percentage")
    parser.add_argument("--assume-idle-when-unmeasurable", action="store_true",
                        help="Treat the PC as idle when keyboard/mouse idle time cannot be measured (no xprintidle, Wayland, headless). Only for a machine nobody plays on: "
                             "by default an unmeasurable idle time counts as 'the user is active' so a game is never disturbed")
    parser.add_argument("--force-fold", action="store_true", help="Bypass the mouse/keyboard idle timer (never a BLOCKED GPU probe or a busy GPU)")
    parser.add_argument("--dry-run", action="store_true", help="Run a single local verification cycle and exit (exit code 2 when BLOCKED)")
    parser.add_argument("--pause-now", action="store_true", help="Immediately pause BOINC GPU compute and exit")
    parser.add_argument("--resume-now", action="store_true", help="Immediately resume BOINC GPU compute and exit")
    parser.add_argument("--once", action="store_true", help="Run one idle check cycle and print JSON result (exit code 2 when BLOCKED)")
    args = parser.parse_args(argv)

    if args.pause_now:
        driver = set_boinc_client_gpu_mode(False)
        print(json.dumps({"status": "paused", "boinc_driver": driver}))
        return 0
    if args.resume_now:
        driver = set_boinc_client_gpu_mode(True)
        print(json.dumps({"status": "resumed", "boinc_driver": driver}))
        return 0

    single_shot = bool(args.once or args.dry_run)
    last_key: tuple[Any, ...] | None = None
    while True:
        report = run_one_idle_gpu_boinc_cycle(
            commander=args.handle,
            coordinator_url=args.coordinator,
            domain=args.domain,
            idle_threshold_seconds=args.idle_threshold_seconds,
            max_gpu_load=args.max_gpu_load,
            target_planet=args.target_planet,
            force_fold=args.force_fold or args.dry_run,
            dry_run=args.dry_run,
            assume_idle_when_unknown=args.assume_idle_when_unmeasurable,
        )
        blocked = report.get("status") == "BLOCKED"
        if single_shot:
            print(json.dumps(report, indent=2))
            return 2 if blocked else 0
        key = _transition_key(report)
        if key != last_key:
            print(json.dumps(report))
            last_key = key
        time.sleep(60.0 if blocked else 5.0)


if __name__ == "__main__":
    raise SystemExit(main())

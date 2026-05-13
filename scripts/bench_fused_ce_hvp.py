"""Microbenchmark: eager vs fused CE HVP peak memory and wall time.

Reports peak memory and wall time across vocabulary scales, dtypes, and
batch (N = B*T) scales.

Modes:
  * ``--quick`` -- single shape (B=8, T=64, V=4096) for smoke validation.
  * default (no flag) -- LM-scale single shape (B=64, T=256, V=50304).
  * ``--full`` -- comprehensive sweep:
      - vocab scales V ∈ {50304, 100000, 128000} at fixed N
      - dtype: fp32 and bf16
      - N scaling at fixed V: N ∈ {64, 256, 1024, 4096, 16384, 65536}

Memory accounting:
  - CUDA: ``torch.cuda.max_memory_allocated``.
  - CPU/MPS: ``torch.profiler`` with ``profile_memory=True``. We sum the
    "Self CPU Mem" of all ops between a ``record_function`` marker. This
    captures every aten kernel allocation along the call chain, which is
    exactly what we want for an "intermediates ratio".

The expected story at B=64, T=256, V=50304, fp32:
  - eager allocates ~5x the output buffer (softmax temp + p + p*u + p*dot + sub)
    plus the softmax intermediate, ~6 (N, V) tensors = ~19.6 GB
  - compile fuses to ~1 (N, V) tensor (the output) = ~3.3 GB
  - target ratio: >=3x; expected ~6x

Run::

    python scripts/bench_fused_ce_hvp.py [--quick | --full] [--dtype {fp32,bf16}]
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from hessian_eigenthings.loss_fns._fused_ce_hvp import (
    _ce_hvp_reference,
    compiled_ce_hvp,
)


def _profiler_self_mem_bytes(prof: profile, region: str) -> int:
    """Sum of ``self_cpu_memory_usage`` for ops nested under a ``record_function``
    region. Returns positive bytes for the *total allocated* in that region
    (we filter out negative entries which are deallocations of held tensors).

    Note: we use ``self_cpu_memory_usage`` rather than ``cpu_memory_usage``
    because the latter double-counts when a parent op's bucket contains its
    children. Self-memory sums to "total bytes this op directly allocated".
    """
    total = 0
    events = prof.events()
    region_evs = [e for e in events if e.name == region]
    if not region_evs:
        return 0
    rev = region_evs[0]
    t0, t1 = rev.time_range.start, rev.time_range.end
    for e in events:
        if e.name == region:
            continue
        if e.time_range.start >= t0 and e.time_range.end <= t1:
            self_mem = getattr(e, "self_cpu_memory_usage", 0) or 0
            if self_mem > 0:
                total += self_mem
    return total


def measure_cpu(fn, args_tuple: tuple, label: str) -> tuple[float, int]:
    """Returns (wall_seconds, peak_bytes) on CPU/MPS via torch.profiler."""
    for _ in range(2):
        out = fn(*args_tuple)
        del out
    gc.collect()

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        with record_function(label):
            out = fn(*args_tuple)
        del out

    peak_bytes = _profiler_self_mem_bytes(prof, label)

    n_iter = 5
    for _ in range(2):
        fn(*args_tuple)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args_tuple)
        del out
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter, peak_bytes


def measure_cuda(fn, args_tuple: tuple, label: str) -> tuple[float, int]:
    device = args_tuple[0].device
    for _ in range(3):
        fn(*args_tuple)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    out = fn(*args_tuple)
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated(device))
    del out

    n_iter = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(*args_tuple)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    del out
    return (t1 - t0) / n_iter, peak


def _make_inputs(
    N: int, V: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(0)
    flat_logits = torch.randn(N, V, dtype=dtype, device=device, generator=g)
    flat_u = torch.randn(N, V, dtype=dtype, device=device, generator=g)
    valid = (torch.rand(N, device=device, generator=g) > 0.1).to(dtype)
    n = valid.sum().clamp_min(1.0)
    return flat_logits, flat_u, valid, n


def _run_cell(
    label: str, N: int, V: int, dtype: torch.dtype, device: torch.device, measure
) -> None:
    """Run eager / compile / triton on one (N, V, dtype) cell and print results."""
    nv_bytes = N * V * dtype.itemsize
    print(f"--- {label}: N={N} V={V} dtype={dtype} ---")
    print(f"  (N, V) tensor = {N} x {V} = {nv_bytes / 1e9:.3f} GB per copy")

    targs = _make_inputs(N, V, dtype, device)

    eager_t, eager_mem = measure(_ce_hvp_reference, targs, f"eager-{label}")
    print(
        f"  eager:   wall {eager_t*1000:8.2f} ms   peak {eager_mem/1e9:7.3f} GB  "
        f"({eager_mem/max(nv_bytes,1):4.1f}x (N,V))"
    )

    compile_t, compile_mem = measure(compiled_ce_hvp, targs, f"compile-{label}")
    print(
        f"  compile: wall {compile_t*1000:8.2f} ms   peak {compile_mem/1e9:7.3f} GB  "
        f"({compile_mem/max(nv_bytes,1):4.1f}x (N,V))   "
        f"speedup={eager_t/max(compile_t,1e-9):4.2f}x  mem={eager_mem/max(compile_mem,1):4.2f}x"
    )

    if torch.cuda.is_available():
        from hessian_eigenthings.loss_fns._fused_ce_hvp import triton_ce_hvp

        triton_t, triton_mem = measure(triton_ce_hvp, targs, f"triton-{label}")
        print(
            f"  triton:  wall {triton_t*1000:8.2f} ms   peak {triton_mem/1e9:7.3f} GB  "
            f"({triton_mem/max(nv_bytes,1):4.1f}x (N,V))   "
            f"speedup={eager_t/max(triton_t,1e-9):4.2f}x  mem={eager_mem/max(triton_mem,1):4.2f}x"
        )

    # Free between cells to keep peak-memory stats clean on CUDA.
    del targs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="single small shape (B=8,T=64,V=4096)")
    g.add_argument("--full", action="store_true", help="comprehensive sweep across V, dtype, and N")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        measure = measure_cuda
    else:
        device = torch.device("cpu")
        measure = measure_cpu

    print(f"device={device}")
    if device.type == "cuda":
        print(f"  gpu={torch.cuda.get_device_name(0)}")

    if args.quick:
        B, T, V = 8, 64, 4096
        dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
        _run_cell(f"quick-{args.dtype}", B * T, V, dtype, device, measure)
        return

    if args.full:
        # 1) Vocab scaling at fixed N=16384 (matches B=64, T=256) -- this is the
        #    headline "is the kernel still fast at Mistral-Large scale?" sweep.
        print("\n=== vocab scaling at N=16384, fp32 ===")
        for V in (50304, 100000, 128000):
            _run_cell(f"V={V}", 16384, V, torch.float32, device, measure)

        # 2) Dtype comparison at the headline shape: fp32 vs bf16.
        print("\n=== dtype comparison at N=16384, V=50304 ===")
        for label, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            _run_cell(f"dtype-{label}", 16384, 50304, dt, device, measure)

        # 3) N scaling at fixed V=50304, fp32 -- shows how the kernels scale
        #    from short sequences (tiny LM eval) up to long-context training.
        print("\n=== N scaling at V=50304, fp32 ===")
        for N in (64, 256, 1024, 4096, 16384, 65536):
            _run_cell(f"N={N}", N, 50304, torch.float32, device, measure)
        return

    # Default: single LM-scale shape.
    B, T, V = 64, 256, 50304
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    _run_cell(f"default-{args.dtype}", B * T, V, dtype, device, measure)


if __name__ == "__main__":
    main()

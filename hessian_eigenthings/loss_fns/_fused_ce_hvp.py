"""Fused cross-entropy loss-Hessian-vector product.

The math (per non-ignored position, with `p = softmax(logits)`):

    H_loss @ u = (p * u - p * <p, u>) * mask / n_valid

In the eager implementation this expression allocates ~5 `(N, V)` intermediates
(`p`, `p*u`, `p*dot`, the difference, the masked/scaled result) on top of the
softmax temporaries themselves. At LM-scale vocabularies (V=50k+) these
dominate the peak memory of an HVP call.

This module provides two fused implementations of the same expression:

1. ``compiled_ce_hvp`` -- ``torch.compile``-wrapped reference. Inductor fuses
   the softmax + elementwise + reduction chain into ~1-2 kernels, which
   eliminates most of the (N, V) intermediates. This is the default fused path
   and works on CPU/CUDA/MPS (any backend that ``torch.compile`` supports).

2. ``triton_ce_hvp`` -- hand-written Triton kernel. One program per (N,) row,
   two passes over the V axis in BLOCK_V tiles: first to compute the softmax
   normalizer and `<p, u>` dot product (numerically-stable, online), then to
   write the output. Materializes zero (N, V) intermediates -- output buffer
   only. Requires CUDA + Triton.

Both produce ``out_flat = (p * u - p * <p, u>) * mask / n_valid`` with
shape ``(N, V)``, given ``logits, u: (N, V)``, ``mask: (N,)``, and a scalar
``n_valid``. The caller is responsible for reshaping back to ``(B, T, V)``
and zeroing the slices that have no corresponding label (the last time step
under causal shift).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch


def _ce_hvp_reference(
    flat_logits: torch.Tensor,
    flat_u: torch.Tensor,
    valid: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    """Eager reference implementation. Same math as ``compiled_ce_hvp``."""
    p = torch.softmax(flat_logits, dim=-1)
    dot = (p * flat_u).sum(dim=-1, keepdim=True)
    return (p * flat_u - p * dot) * valid.unsqueeze(-1) / n


# ``torch.compile`` over the reference. Inductor fuses softmax + the two
# elementwise passes (with the row reduction in between) into a small number
# of kernels. We use ``dynamic=True`` so we don't recompile per shape.
_compiled_ce_hvp_impl: Callable[..., torch.Tensor] | None = None


def _get_compiled_impl() -> Callable[..., torch.Tensor]:
    """Lazy-construct the compiled kernel. Compiling at import time would force
    every importer to pay the compilation cost even if they never use it."""
    global _compiled_ce_hvp_impl
    if _compiled_ce_hvp_impl is None:
        _compiled_ce_hvp_impl = cast(
            Callable[..., torch.Tensor],
            torch.compile(_ce_hvp_reference, mode="default", dynamic=True, fullgraph=True),
        )
    return _compiled_ce_hvp_impl


def compiled_ce_hvp(
    flat_logits: torch.Tensor,
    flat_u: torch.Tensor,
    valid: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    """``torch.compile``-fused CE HVP. See module docstring for the math."""
    return _get_compiled_impl()(flat_logits, flat_u, valid, n)


# --- Triton kernel (CUDA only) ------------------------------------------------
# Importing triton is optional; we guard on availability.

_TRITON_AVAILABLE: bool | None = None


def _triton_available() -> bool:
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        try:
            import triton  # noqa: F401
            import triton.language as tl  # noqa: F401

            _TRITON_AVAILABLE = torch.cuda.is_available()
        except ImportError:
            _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE


def _build_triton_kernel() -> Callable[..., torch.Tensor]:
    """Compile the Triton kernel on first use. Returns a launcher closure."""
    import triton
    import triton.language as tl

    @triton.jit  # type: ignore[untyped-decorator]
    def _ce_hvp_kernel(  # type: ignore[no-untyped-def]
        logits_ptr,
        u_ptr,
        out_ptr,
        valid_ptr,  # (N,) float
        n_valid,  # scalar python float
        N,
        V,
        stride_n,
        stride_v,
        BLOCK_V: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return

        row_off = pid * stride_n

        # Pass 1: online max + sum for softmax normalizer.
        running_max = tl.full([1], float("-inf"), dtype=tl.float32)
        running_sum = tl.zeros([1], dtype=tl.float32)
        for v_start in range(0, V, BLOCK_V):
            offs = v_start + tl.arange(0, BLOCK_V)
            mask = offs < V
            x = tl.load(
                logits_ptr + row_off + offs * stride_v,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            block_max = tl.max(x, axis=0)
            new_max = tl.maximum(running_max, block_max)
            # rescale prior running_sum to the new max.
            running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
                tl.exp(x - new_max), axis=0
            )
            running_max = new_max

        log_z = running_max + tl.log(running_sum)

        # Pass 2: compute dot = sum_v p[v] * u[v] = sum_v exp(x[v] - log_z) * u[v]
        dot = tl.zeros([1], dtype=tl.float32)
        for v_start in range(0, V, BLOCK_V):
            offs = v_start + tl.arange(0, BLOCK_V)
            mask = offs < V
            x = tl.load(logits_ptr + row_off + offs * stride_v, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(u_ptr + row_off + offs * stride_v, mask=mask, other=0.0).to(tl.float32)
            p = tl.exp(x - log_z)
            dot += tl.sum(p * u * mask.to(tl.float32), axis=0)

        valid = tl.load(valid_ptr + pid).to(tl.float32)
        scale = valid / n_valid

        # Pass 3: write out[v] = (p[v]*u[v] - p[v]*dot) * scale
        for v_start in range(0, V, BLOCK_V):
            offs = v_start + tl.arange(0, BLOCK_V)
            mask = offs < V
            x = tl.load(logits_ptr + row_off + offs * stride_v, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(u_ptr + row_off + offs * stride_v, mask=mask, other=0.0).to(tl.float32)
            p = tl.exp(x - log_z)
            y = (p * u - p * dot) * scale
            tl.store(out_ptr + row_off + offs * stride_v, y, mask=mask)

    def launcher(
        flat_logits: torch.Tensor,
        flat_u: torch.Tensor,
        valid: torch.Tensor,
        n: torch.Tensor,
    ) -> torch.Tensor:
        assert flat_logits.is_cuda, "triton_ce_hvp requires CUDA tensors"
        assert flat_logits.shape == flat_u.shape
        assert flat_logits.dim() == 2
        flat_logits = flat_logits.contiguous()
        flat_u = flat_u.contiguous()
        valid = valid.contiguous().to(torch.float32)
        N, V = flat_logits.shape
        out = torch.empty_like(flat_logits)
        BLOCK_V = 1024 if V >= 1024 else triton.next_power_of_2(V)
        n_scalar = float(n.item()) if isinstance(n, torch.Tensor) else float(n)
        _ce_hvp_kernel[(N,)](
            flat_logits,
            flat_u,
            out,
            valid,
            n_scalar,
            N,
            V,
            flat_logits.stride(0),
            flat_logits.stride(1),
            BLOCK_V=BLOCK_V,
        )
        return out

    return launcher


_triton_launcher = None


def triton_ce_hvp(
    flat_logits: torch.Tensor,
    flat_u: torch.Tensor,
    valid: torch.Tensor,
    n: torch.Tensor,
) -> torch.Tensor:
    """Triton-fused CE HVP. Requires CUDA + Triton. Materializes only the output
    buffer; reads the (N, V) logits/u tensors twice from HBM (no intermediates)."""
    global _triton_launcher
    if not _triton_available():
        raise RuntimeError(
            "triton_ce_hvp requires CUDA + Triton; install triton and run on a CUDA device"
        )
    if _triton_launcher is None:
        _triton_launcher = _build_triton_kernel()
    return _triton_launcher(flat_logits, flat_u, valid, n)

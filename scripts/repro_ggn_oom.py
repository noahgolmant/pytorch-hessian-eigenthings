"""Minimal CPU repro for the GGNOperator memory blowup.

Status: **fixed in `feature/ggn-oom-fix` branch** — the new default
`loss_hvp="analytical"` path on `GGNOperator` implements the
`fd_jvp_single_vjp` variant below. See `tests/test_ggn_matvec.py` for the
CPU-side memory-regression test that locks in the fix, and
`tests/test_ggn_matvec_a100.py` for the A100 validation suite. This script
is kept as documentation of the four variants and the live-tensor probe
methodology used during diagnosis.

Hypothesis: in `_matvec_one_batch`, the combination of

    torch.func.jvp(model_call, (param_dict,), (v_dict,))
    ...
    grad_loss  = autograd.grad(loss, output_leaf, create_graph=True)[0]
    h_loss_jvp = autograd.grad(grad_loss, output_leaf, grad_outputs=jvp_out)[0]
    ...
    _, vjp_fn  = torch.func.vjp(model_call, param_dict)
    gv_dict    = vjp_fn(h_loss_jvp)[0]

allocates intermediate tensors whose total size grows with `vocab_size` (and
model param count), not just batch tokens. This script measures peak Python
heap allocation while varying:

    A. seq_len (and batch_size) at fixed vocab_size   -> should grow linearly in B*T
    B. vocab_size at fixed seq_len, batch_size        -> should grow ~ vocab if the
                                                         workload is "intended", but in
                                                         the bug-path grows faster.

We use a *tiny* transformer-style backbone with parameter count comparable to
the vocab head (so the "head" tensor dominates everything else), running on
CPU with bfloat32 to keep absolute numbers small enough to measure under
tracemalloc.

The script also runs three variants of the matvec for comparison:

  1. `current`         : exact code path from `hessian_eigenthings/operators/ggn.py:97-125`.
  2. `analytical_ce`   : same JVP+VJP, but the H_loss·Jv step uses the closed-form
                         categorical-CE Hessian-vector product
                            H_psi @ Jv = p * Jv - p * (p . Jv) / B  (per row)
                         removing the `create_graph=True` autograd path entirely.
  3. `fd_jvp`          : finite-difference Jv (two forwards, no autograd graph)
                         and finite-difference J^T u (one extra backward).
                         Fully torch.func-free.

Output: a per-variant table of peak-allocated bytes and elapsed time. Cross-
referencing the rows tells us whether the bloat is in `torch.func.jvp`, in
the `create_graph=True` double-backward, or in the `torch.func.vjp` step.
"""

from __future__ import annotations

import argparse
import gc
import threading
from collections.abc import Callable
from time import perf_counter, sleep
from typing import Any, cast

import psutil
import torch
import torch.nn.functional as F
from torch import nn


class RSSPeakMonitor:
    """Background thread that polls process RSS and records the peak.

    We poll at 1 ms intervals which is generally finer than the matvec
    duration but coarser than a single tensor allocation. Good enough to
    distinguish 60 MB from 600 MB.
    """

    def __init__(self, interval_s: float = 0.001) -> None:
        self.interval_s = interval_s
        self._proc = psutil.Process()
        self._peak = 0
        self._baseline = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self._peak:
                self._peak = rss
            sleep(self.interval_s)

    def __enter__(self) -> RSSPeakMonitor:
        gc.collect()
        self._baseline = self._proc.memory_info().rss
        self._peak = self._baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak_bytes_above_baseline(self) -> int:
        return max(0, self._peak - self._baseline)


# ---------------------------------------------------------------------------
# Tiny LM-like model: a single Linear "backbone" then a Linear "lm_head".
# Parameter counts are dominated by `lm_head.weight` of shape (vocab, hidden).
# ---------------------------------------------------------------------------


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        h = torch.tanh(self.proj(h))
        return self.lm_head(h)


def make_batch(
    batch_size: int, seq_len: int, vocab_size: int, *, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
    return input_ids, labels


def forward_fn(model: nn.Module, batch: Any) -> torch.Tensor:
    input_ids, _ = batch
    return model(input_ids)


def loss_of_output_fn(output: torch.Tensor, batch: Any) -> torch.Tensor:
    _, labels = batch
    return F.cross_entropy(output.reshape(-1, output.size(-1)), labels.reshape(-1))


# ---------------------------------------------------------------------------
# Functional-call adapter (mirrors `_FunctionalModel` in ggn.py).
# ---------------------------------------------------------------------------


class _FunctionalModel:
    def __init__(self, model: nn.Module, params_and_buffers: dict[str, torch.Tensor]) -> None:
        self._model = model
        self._params_and_buffers = params_and_buffers

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return torch.func.functional_call(self._model, self._params_and_buffers, args, kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


# ---------------------------------------------------------------------------
# Three matvec variants.
# ---------------------------------------------------------------------------


def build_param_dict(model: nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    return params, buffers


def random_v(params: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {n: torch.randn(p.shape, generator=g, dtype=p.dtype) for n, p in params.items()}


def _live_tensor_bytes() -> int:
    """Sum of `untyped_storage().nbytes()` over all currently-live torch.Tensor
    objects reachable from the Python GC. Counts each *storage* once."""
    gc.collect()
    seen: set[int] = set()
    total = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor):
                st = obj.untyped_storage()
                ptr = st.data_ptr()
                if ptr not in seen:
                    seen.add(ptr)
                    total += st.nbytes()
        except Exception:
            continue
    return total


def matvec_current(
    model: nn.Module,
    batch: Any,
    v_dict: dict[str, torch.Tensor],
    *,
    probe: list[tuple[str, int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Reproduces `_matvec_one_batch` exactly (ggn.py:97-125)."""
    params, buffers = build_param_dict(model)

    def model_call(p_subset: dict[str, torch.Tensor]) -> torch.Tensor:
        full = {**p_subset, **buffers}
        adapter = cast(nn.Module, _FunctionalModel(model, full))
        return forward_fn(adapter, batch)

    if probe is not None:
        probe.append(("start", _live_tensor_bytes()))

    output, jvp_out = torch.func.jvp(model_call, (params,), (v_dict,))
    if probe is not None:
        probe.append(("after_jvp", _live_tensor_bytes()))

    output_leaf = output.detach().requires_grad_(True)
    loss = loss_of_output_fn(output_leaf, batch)
    grad_loss = torch.autograd.grad(loss, output_leaf, create_graph=True)[0]
    if probe is not None:
        probe.append(("after_grad_loss_create_graph", _live_tensor_bytes()))

    h_loss_jvp = torch.autograd.grad(grad_loss, output_leaf, grad_outputs=jvp_out)[0]
    if probe is not None:
        probe.append(("after_h_loss_jvp", _live_tensor_bytes()))

    _, vjp_fn = torch.func.vjp(model_call, params)
    if probe is not None:
        probe.append(("after_vjp_setup", _live_tensor_bytes()))

    out = vjp_fn(h_loss_jvp)[0]
    if probe is not None:
        probe.append(("after_vjp_call", _live_tensor_bytes()))
    return out


def analytical_ce_hvp(logits: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Closed-form Hessian-vec product for cross-entropy + softmax.

    Loss = mean over (B*T) of -log p_{label}, where p = softmax(logits) along dim=-1.
    H_loss is block-diagonal in (B*T) with each block being diag(p) - p p^T.
    Total scale = 1 / (B*T) because loss is averaged. Returns the same shape as `u`.
    """
    p = torch.softmax(logits, dim=-1)
    dot = (p * u).sum(dim=-1, keepdim=True)
    n = logits.numel() // logits.size(-1)
    return (p * u - p * dot) / n


def matvec_analytical_ce(
    model: nn.Module,
    batch: Any,
    v_dict: dict[str, torch.Tensor],
    *,
    probe: list[tuple[str, int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Replace the double-backward H_loss step with the closed-form CE HVP."""
    params, buffers = build_param_dict(model)

    def model_call(p_subset: dict[str, torch.Tensor]) -> torch.Tensor:
        full = {**p_subset, **buffers}
        adapter = cast(nn.Module, _FunctionalModel(model, full))
        return forward_fn(adapter, batch)

    if probe is not None:
        probe.append(("start", _live_tensor_bytes()))

    output, jvp_out = torch.func.jvp(model_call, (params,), (v_dict,))
    if probe is not None:
        probe.append(("after_jvp", _live_tensor_bytes()))

    h_loss_jvp = analytical_ce_hvp(output.detach(), jvp_out)
    if probe is not None:
        probe.append(("after_analytic_hvp", _live_tensor_bytes()))

    _, vjp_fn = torch.func.vjp(model_call, params)
    if probe is not None:
        probe.append(("after_vjp_setup", _live_tensor_bytes()))

    out = vjp_fn(h_loss_jvp)[0]
    if probe is not None:
        probe.append(("after_vjp_call", _live_tensor_bytes()))
    return out


def matvec_fd_jvp(
    model: nn.Module, batch: Any, v_dict: dict[str, torch.Tensor], eps: float = 1e-3
) -> dict[str, torch.Tensor]:
    """Fully torch.func-free path: finite-difference JVP, analytical loss-HVP,
    then a normal autograd VJP to map H·Jv back through parameters."""
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
    try:
        # Forward at theta + eps*v
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.add_(v_dict[n], alpha=eps)
        out_plus = forward_fn(model, batch).detach()

        # Forward at theta - eps*v
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.add_(v_dict[n], alpha=-2.0 * eps)
        out_minus = forward_fn(model, batch).detach()
    finally:
        # restore parameters
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(snapshot[n])

    jvp_out = (out_plus - out_minus) / (2.0 * eps)

    # Need logits at theta for the analytical CE Hessian
    with torch.no_grad():
        logits = forward_fn(model, batch)
    h_loss_jvp = analytical_ce_hvp(logits, jvp_out)

    # J^T u: a normal forward + backward with grad_outputs=h_loss_jvp
    out = forward_fn(model, batch)
    grads = torch.autograd.grad(out, list(model.parameters()), grad_outputs=h_loss_jvp)
    return {n: g for (n, _), g in zip(model.named_parameters(), grads, strict=True)}


# ---------------------------------------------------------------------------
# Measurement harness
# ---------------------------------------------------------------------------


def matvec_fd_jvp_single_vjp(
    model: nn.Module,
    batch: Any,
    v_dict: dict[str, torch.Tensor],
    eps: float = 1e-3,
    *,
    probe: list[tuple[str, int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Option C: 2 forwards (FD JVP, no-grad) + analytical CE H + 1 regular
    forward+backward to apply J^T to H_loss·Jv. No torch.func at all.

    Memory: one autograd graph at a time. Peak is the size of *one* normal
    backward pass — same as a vanilla training step.
    """
    if probe is not None:
        probe.append(("start", _live_tensor_bytes()))

    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
    try:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.add_(v_dict[n], alpha=eps)
            out_plus = forward_fn(model, batch).clone()
            for n, p in model.named_parameters():
                p.add_(v_dict[n], alpha=-2.0 * eps)
            out_minus = forward_fn(model, batch).clone()
    finally:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(snapshot[n])

    jvp_out = (out_plus - out_minus) / (2.0 * eps)
    del out_plus, out_minus, snapshot
    if probe is not None:
        probe.append(("after_fd_jvp", _live_tensor_bytes()))

    # Single grad-enabled forward; we'll use the same logits for both the
    # analytical CE HVP and the backward.
    logits = forward_fn(model, batch)
    h_loss_jvp = analytical_ce_hvp(logits.detach(), jvp_out)
    if probe is not None:
        probe.append(("after_analytic_hvp", _live_tensor_bytes()))

    grads = torch.autograd.grad(logits, list(model.parameters()), grad_outputs=h_loss_jvp)
    out = {n: g for (n, _), g in zip(model.named_parameters(), grads, strict=True)}
    if probe is not None:
        probe.append(("after_backward", _live_tensor_bytes()))
    return out


VARIANTS: dict[str, Callable[..., Any]] = {
    "current": matvec_current,
    "analytical_ce": matvec_analytical_ce,
    "fd_jvp": matvec_fd_jvp,
    "fd_jvp_single_vjp": matvec_fd_jvp_single_vjp,
}


def measure_one(
    variant: str, vocab_size: int, batch_size: int, seq_len: int, *, hidden: int = 64
) -> dict[str, float]:
    torch.manual_seed(0)
    model = TinyLM(vocab_size=vocab_size, hidden=hidden)
    batch = make_batch(batch_size, seq_len, vocab_size)
    v_dict = random_v(dict(model.named_parameters()), seed=42)

    fn = VARIANTS[variant]
    n_params = sum(p.numel() for p in model.parameters())

    gc.collect()
    with RSSPeakMonitor(interval_s=0.0001) as mon:
        t0 = perf_counter()
        out = fn(model, batch, v_dict)
        elapsed = perf_counter() - t0
    peak = mon.peak_bytes_above_baseline

    # Touch `out` so the optimizer can't elide it
    flat = sum(o.float().pow(2).sum().item() for o in out.values())
    del out
    gc.collect()

    return {
        "variant": variant,
        "vocab": vocab_size,
        "batch": batch_size,
        "seq": seq_len,
        "n_params": n_params,
        "peak_MB": peak / 1e6,
        "elapsed_s": elapsed,
        "checksum": flat,
    }


def print_row(r: dict[str, float]) -> None:
    print(
        f"{r['variant']:>14}  vocab={int(r['vocab']):>6}  "
        f"B={int(r['batch']):>2}  T={int(r['seq']):>3}  "
        f"params={int(r['n_params']):>8}  "
        f"peak={r['peak_MB']:>8.1f} MB  t={r['elapsed_s']:>5.2f}s  "
        f"chk={r['checksum']:.3e}"
    )


def sweep_vocab(
    variant: str, vocabs: list[int], *, batch_size: int = 4, seq_len: int = 64, hidden: int = 128
) -> None:
    print(f"\n--- {variant}: sweep vocab at B={batch_size}, T={seq_len}, hidden={hidden} ---")
    rows = []
    for v in vocabs:
        r = measure_one(variant, v, batch_size, seq_len, hidden=hidden)
        rows.append(r)
        print_row(r)
    # Ratios: peak / n_params
    print("  scaling: peak_MB / n_params * 1e6 (constant => bytes/param):")
    for r in rows:
        print(f"    vocab={int(r['vocab']):>6}: {r['peak_MB'] / r['n_params'] * 1e6:8.2f}")


def sweep_batch(
    variant: str,
    batch_sizes: list[int],
    *,
    vocab: int = 16384,
    seq_len: int = 64,
    hidden: int = 128,
) -> None:
    print(f"\n--- {variant}: sweep batch at vocab={vocab}, T={seq_len}, hidden={hidden} ---")
    rows = []
    for b in batch_sizes:
        r = measure_one(variant, vocab, b, seq_len, hidden=hidden)
        rows.append(r)
        print_row(r)
    print("  scaling: peak_MB / (B*T):")
    for r in rows:
        print(
            f"    B={int(r['batch']):>2}, T={int(r['seq']):>3}: "
            f"{r['peak_MB'] / (r['batch'] * r['seq']):8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="compare",
        choices=("compare", "vocab_sweep", "batch_sweep", "all"),
    )
    args = parser.parse_args()

    if args.mode in ("compare", "all"):
        print("=" * 80)
        print("Variant comparison at a fixed problem size")
        print("=" * 80)
        for variant in VARIANTS:
            r = measure_one(variant, vocab_size=4096, batch_size=2, seq_len=16, hidden=32)
            print_row(r)

        print()
        print("=" * 80)
        print("Per-step live-tensor-bytes probe (current variant), bigger problem")
        print("=" * 80)
        torch.manual_seed(0)
        vocab, B, T, H = 32768, 4, 64, 128
        model = TinyLM(vocab_size=vocab, hidden=H)
        batch = make_batch(B, T, vocab)
        v_dict = random_v(dict(model.named_parameters()), seed=42)
        baseline = _live_tensor_bytes()
        print(f"  vocab={vocab}, B={B}, T={T}, hidden={H}")
        print(f"  n_params         = {sum(p.numel() for p in model.parameters()):>10}")
        print(f"  baseline_bytes   = {baseline:>10} ({baseline / 1e6:.1f} MB)")
        print(f"  logits tensor    = {B * T * vocab * 4:>10} ({B * T * vocab * 4 / 1e6:.1f} MB)")
        for fn_name, fn in (("current", matvec_current), ("analytical_ce", matvec_analytical_ce)):
            print(f"\n  -- {fn_name} --")
            probe: list[tuple[str, int]] = []
            gc.collect()
            with RSSPeakMonitor(interval_s=0.0001) as mon:
                out = fn(model, batch, v_dict, probe=probe)
            print(
                f"  RSS peak above baseline (transient): {mon.peak_bytes_above_baseline / 1e6:.1f} MB"
            )
            prev = baseline
            for label, b in probe:
                delta = b - prev
                print(
                    f"  {label:>30}: live={b:>10} ({b / 1e6:7.1f} MB)  "
                    f"delta={delta:+10d} ({delta / 1e6:+7.1f} MB)"
                )
                prev = b
            del out
            gc.collect()

    if args.mode in ("vocab_sweep", "all"):
        print("=" * 80)
        print("Vocab sweep (peak should scale with model params; batch tokens fixed)")
        print("=" * 80)
        for variant in VARIANTS:
            sweep_vocab(variant, vocabs=[2048, 4096, 8192, 16384, 32768])

    if args.mode in ("batch_sweep", "all"):
        print("=" * 80)
        print("Batch sweep (peak should scale with B*T; vocab fixed)")
        print("=" * 80)
        for variant in VARIANTS:
            sweep_batch(variant, batch_sizes=[1, 2, 4, 8, 16])


if __name__ == "__main__":
    main()

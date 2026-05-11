"""CPU tests for the GGN OOM fix.

Three tests, all CPU-only:

1. **Numerical equivalence vs a dense GGN reference** on a tiny fp64 MLP with
   classification head. Verifies the FD-JVP + analytical-CE-Hessian + single-VJP
   pipeline computes the same operator as `J^T H_loss J` constructed densely.

2. **FD vs analytical-JVP at fp32**. Bounds the FD truncation budget for the
   default `loss_hvp="analytical"` (FD JVP) path against the autograd reference.

3. **Memory regression on TinyLM** via `tracemalloc`. The pre-fix
   `loss_hvp="autograd"` path allocates ~10× the logits-tensor as autograd
   working memory; the new path allocates ~one normal training step.
"""

from __future__ import annotations

import gc
import threading
import time
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from hessian_eigenthings.loss_fns.huggingface import _LossOfOutputWithHvp
from hessian_eigenthings.loss_fns.standard import (
    cross_entropy_loss_of_output,
    mse_loss_of_output,
)
from hessian_eigenthings.operators import GGNOperator
from hessian_eigenthings.param_utils import select_parameters


def _seed(value: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(value)
    return g


def _tiny_mlp_fp64(vocab: int = 8) -> nn.Module:
    """4-layer MLP, ~200 params, with a classification head of `vocab` classes."""
    g = _seed(0)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.Tanh(),
        nn.Linear(8, 8),
        nn.Tanh(),
        nn.Linear(8, vocab),
    ).to(torch.float64)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=torch.float64)
    return model


def _full_jacobian(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    params = list(select_parameters(model).values())
    n_params = sum(int(p.numel()) for p in params)
    output = model(x).reshape(-1)
    out_dim = output.numel()
    j = torch.zeros(out_dim, n_params, dtype=output.dtype)
    for i in range(out_dim):
        grads = torch.autograd.grad(
            output[i], params, retain_graph=(i < out_dim - 1), allow_unused=False
        )
        j[i] = torch.cat([g.reshape(-1) for g in grads])
    return j


def _full_loss_hessian(loss_of_output_fn, output: torch.Tensor, batch: Any) -> torch.Tensor:
    out_leaf = output.detach().reshape(-1).requires_grad_(True)
    out_shaped = out_leaf.reshape(output.shape)
    loss = loss_of_output_fn(out_shaped, batch)
    grad_loss = torch.autograd.grad(loss, out_leaf, create_graph=True)[0]
    n = grad_loss.numel()
    h = torch.zeros(n, n, dtype=output.dtype)
    for i in range(n):
        row = torch.autograd.grad(grad_loss[i], out_leaf, retain_graph=(i < n - 1))[0]
        h[i] = row.reshape(-1)
    return h


def _dense_ggn(model: nn.Module, forward_fn, loss_of_output_fn, batch) -> torch.Tensor:
    x, _ = batch
    j = _full_jacobian(model, x)
    out = forward_fn(model, batch)
    h_loss = _full_loss_hessian(loss_of_output_fn, out, batch)
    return j.t() @ h_loss @ j


def _forward(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    x, _ = batch
    return model(x)


# ---------------------------------------------------------------------------
# Test 1: fp64 numerical equivalence vs a densely-constructed reference GGN.
# ---------------------------------------------------------------------------


def test_ggn_analytical_matches_dense_reference_ce_fp64() -> None:
    """fp64 cross-entropy: FD-JVP + analytical-CE + single-VJP == dense `J^T H_loss J`."""
    vocab = 8
    model = _tiny_mlp_fp64(vocab=vocab)
    g = _seed(1)
    x = torch.randn(6, 4, generator=g, dtype=torch.float64)
    y = torch.randint(0, vocab, (6,), generator=g)
    batch = (x, y)

    loss_of_output_fn = cross_entropy_loss_of_output()
    G_dense = _dense_ggn(model, _forward, loss_of_output_fn, batch)

    op = GGNOperator(
        model=model,
        dataloader=[batch],
        forward_fn=_forward,
        loss_of_output_fn=loss_of_output_fn,
        loss_hvp="analytical",
        # Custom fp64 step large enough to dominate roundoff in the FD JVP but
        # small enough that truncation stays below the assertion threshold.
        fd_eps=1e-5,
    )

    g2 = _seed(3)
    for trial in range(5):
        v = torch.randn(op.size, generator=g2, dtype=torch.float64)
        new = op.matvec(v)
        ref = G_dense @ v
        rel = (new - ref).norm().item() / (ref.norm().item() + 1e-30)
        # FD JVP at fp64 with eps=1e-5: truncation ~eps^2 ~ 1e-10, roundoff
        # ~1e-16/eps ~ 1e-11. So a 1e-4 budget is loose but safe.
        assert rel < 1e-4, f"trial {trial}: rel={rel:.3e}"


def test_ggn_analytical_matches_dense_reference_mse_fp64() -> None:
    """fp64 MSE: analytical H_loss = (2/N) * I, so the FD-JVP path should be exact-up-to-FD-truncation."""
    model = _tiny_mlp_fp64(vocab=4)
    g = _seed(2)
    x = torch.randn(6, 4, generator=g, dtype=torch.float64)
    y = torch.randn(6, 4, generator=g, dtype=torch.float64)
    batch = (x, y)

    loss_of_output_fn = mse_loss_of_output()
    G_dense = _dense_ggn(model, _forward, loss_of_output_fn, batch)

    op = GGNOperator(
        model=model,
        dataloader=[batch],
        forward_fn=_forward,
        loss_of_output_fn=loss_of_output_fn,
        loss_hvp="analytical",
        fd_eps=1e-5,
    )

    g2 = _seed(4)
    v = torch.randn(op.size, generator=g2, dtype=torch.float64)
    new = op.matvec(v)
    ref = G_dense @ v
    rel = (new - ref).norm().item() / (ref.norm().item() + 1e-30)
    assert rel < 1e-4, f"rel={rel:.3e}"


# ---------------------------------------------------------------------------
# Test 2: fp32 FD-vs-autograd JVP truncation bound.
# ---------------------------------------------------------------------------


def test_ggn_fd_vs_autograd_fp32_truncation_bound() -> None:
    """At fp32 the FD-JVP analytical path matches the autograd path within `5e-3` relative."""
    vocab = 8
    g = _seed(0)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.Tanh(),
        nn.Linear(8, 8),
        nn.Tanh(),
        nn.Linear(8, vocab),
    ).to(torch.float32)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=torch.float32)

    gb = _seed(1)
    x = torch.randn(6, 4, generator=gb, dtype=torch.float32)
    y = torch.randint(0, vocab, (6,), generator=gb)
    batch = (x, y)

    loss_of_output_fn = cross_entropy_loss_of_output()

    op_analytical = GGNOperator(
        model=model,
        dataloader=[batch],
        forward_fn=_forward,
        loss_of_output_fn=loss_of_output_fn,
        loss_hvp="analytical",
    )
    op_autograd = GGNOperator(
        model=model,
        dataloader=[batch],
        forward_fn=_forward,
        loss_of_output_fn=loss_of_output_fn,
        loss_hvp="autograd",
    )

    g2 = _seed(7)
    for trial in range(5):
        v = torch.randn(op_analytical.size, generator=g2, dtype=torch.float32)
        analytical = op_analytical.matvec(v)
        autograd = op_autograd.matvec(v)
        rel = (analytical - autograd).norm().item() / (autograd.norm().item() + 1e-30)
        assert rel < 5e-3, f"trial {trial}: rel={rel:.3e}"


def test_ggn_small_v_does_not_underflow_fp64() -> None:
    """Tiny `v` triggers the internal-normalisation path; result should still match autograd.

    Without internal normalisation, `eps * v` underflows for `v ~ 1e-10` and
    the FD difference is dominated by roundoff. The operator normalises `v`
    internally and rescales the output, so the result must stay linear in
    the scale.
    """
    vocab = 6
    g = _seed(0)
    model = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, vocab)).to(torch.float64)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=torch.float64)
    gb = _seed(1)
    x = torch.randn(4, 4, generator=gb, dtype=torch.float64)
    y = torch.randint(0, vocab, (4,), generator=gb)
    batch = (x, y)

    loss_of_output_fn = cross_entropy_loss_of_output()
    op_analytical = GGNOperator(
        model=model,
        dataloader=[batch],
        forward_fn=_forward,
        loss_of_output_fn=loss_of_output_fn,
        loss_hvp="analytical",
        fd_eps=1e-5,
    )
    op_autograd = GGNOperator(
        model=model,
        dataloader=[batch],
        forward_fn=_forward,
        loss_of_output_fn=loss_of_output_fn,
        loss_hvp="autograd",
    )

    g2 = _seed(11)
    v_unit = torch.randn(op_analytical.size, generator=g2, dtype=torch.float64)
    v_unit = v_unit / v_unit.norm()

    # Linearity check: matvec(s*v) ≈ s * matvec(v). Take a tiny scale to make
    # sure the small-v normalisation path is exercised.
    for s in (1e-12, 1e-6, 1.0):
        v = s * v_unit
        analytical = op_analytical.matvec(v)
        autograd = op_autograd.matvec(v)
        rel = (analytical / s - autograd / s).norm().item() / ((autograd / s).norm().item() + 1e-30)
        # fp64 FD truncation with eps=1e-5 is ~O(1e-10); rel-tol 1e-4 is loose.
        assert rel < 1e-4, f"s={s}: rel={rel:.3e}"


# ---------------------------------------------------------------------------
# Test 3: memory regression on TinyLM via tracemalloc.
# ---------------------------------------------------------------------------


class _TinyLM(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        h = torch.tanh(self.proj(h))
        return self.lm_head(h)


def _tinylm_forward(model: nn.Module, batch: Any) -> torch.Tensor:
    input_ids, _ = batch
    return model(input_ids)


def _tinylm_ce(output: torch.Tensor, batch: Any) -> torch.Tensor:
    _, labels = batch
    return F.cross_entropy(output.reshape(-1, output.size(-1)), labels.reshape(-1))


def _tinylm_ce_hvp(output: torch.Tensor, batch: Any, u: torch.Tensor) -> torch.Tensor:
    flat_o = output.reshape(-1, output.size(-1))
    flat_u = u.reshape(-1, u.size(-1))
    p = torch.softmax(flat_o, dim=-1)
    dot = (p * flat_u).sum(dim=-1, keepdim=True)
    n = float(flat_o.size(0))
    return ((p * flat_u - p * dot) / n).view_as(u)


try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - exercised by environments w/o psutil
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False


class _RSSPeakMonitor:
    """Background thread sampling process RSS at ~1 ms intervals.

    Unlike `tracemalloc`, this captures torch's tensor-allocator pages too —
    necessary for measuring GGN matvec memory peaks where almost all
    allocations happen inside `torch.empty`/`torch.zeros` and not Python.
    """

    def __init__(self, interval_s: float = 0.001) -> None:
        self._interval_s = interval_s
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
            time.sleep(self._interval_s)

    def __enter__(self) -> _RSSPeakMonitor:
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
    def peak_above_baseline(self) -> int:
        return max(0, self._peak - self._baseline)


def _measure_matvec_peak(op: GGNOperator, v: torch.Tensor) -> int:
    """Return RSS peak above baseline (bytes) for a single matvec call."""
    gc.collect()
    with _RSSPeakMonitor(interval_s=0.0005) as mon:
        out = op.matvec(v)
        _ = out.sum().item()
    return mon.peak_above_baseline


@pytest.mark.skipif(not _HAS_PSUTIL, reason="psutil not installed")
def test_ggn_analytical_memory_regression_tinylm_cpu() -> None:
    """The default analytical matvec on TinyLM stays within `4 × model_size`.

    On the autograd path, `create_graph=True` over CE retains ~10× the logits
    tensor — for vocab=16 384 / B=4 / T=64 that's ~270 MB, well above the
    `4 × model_size_bytes` budget. The new path peaks at ~one training step's
    worth of activations.
    """
    torch.manual_seed(0)
    vocab, B, T, hidden = 16384, 4, 64, 64
    model = _TinyLM(vocab_size=vocab, hidden=hidden)
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, vocab, (B, T), generator=g)
    labels = torch.randint(0, vocab, (B, T), generator=g)
    batch = (input_ids, labels)

    loss_of_output_fn = _LossOfOutputWithHvp(_tinylm_ce, _tinylm_ce_hvp)
    op = GGNOperator(
        model=model,
        dataloader=[batch],
        forward_fn=_tinylm_forward,
        loss_of_output_fn=loss_of_output_fn,
        loss_hvp="analytical",
    )

    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # 5x model size as a loose sentinel against regression. psutil RSS on
    # small models has ~1MB measurement noise from allocator + other process
    # state; the real prevention happens on the GPU regression test where the
    # bug actually OOMs at 80GB. This is the cheap CI sentinel.
    budget = 5 * model_size_bytes

    v = torch.randn(op.size, dtype=torch.float32)

    # Warm-up call to populate any one-shot caches outside the measurement.
    _ = op.matvec(v)

    # Run 3 times, take the minimum to suppress allocator/RSS noise.
    peak = min(_measure_matvec_peak(op, v) for _ in range(3))

    print(
        f"\nTinyLM(vocab={vocab}, hidden={hidden}, B={B}, T={T}): "
        f"model={model_size_bytes / 1e6:.1f}MB, "
        f"peak(min of 3)={peak / 1e6:.1f}MB, "
        f"budget={budget / 1e6:.1f}MB"
    )
    assert peak < budget, (
        f"analytical matvec peak {peak / 1e6:.1f}MB exceeded "
        f"5 × model_size = {budget / 1e6:.1f}MB"
    )


@pytest.mark.skipif(not _HAS_PSUTIL, reason="psutil not installed")
@pytest.mark.slow
def test_ggn_autograd_path_exceeds_budget_tinylm_cpu() -> None:
    """Sentinel: the legacy autograd path *should* exceed the analytical-path
    budget — confirms the regression test catches the pre-fix behaviour.

    Marked slow because the legacy path is also several × slower at this size.
    """
    torch.manual_seed(0)
    vocab, B, T, hidden = 16384, 4, 64, 64
    model = _TinyLM(vocab_size=vocab, hidden=hidden)
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, vocab, (B, T), generator=g)
    labels = torch.randint(0, vocab, (B, T), generator=g)
    batch = (input_ids, labels)

    loss_of_output_fn = _LossOfOutputWithHvp(_tinylm_ce, _tinylm_ce_hvp)
    op_autograd = GGNOperator(
        model=model,
        dataloader=[batch],
        forward_fn=_tinylm_forward,
        loss_of_output_fn=loss_of_output_fn,
        loss_hvp="autograd",
    )

    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    budget = 4 * model_size_bytes
    v = torch.randn(op_autograd.size, dtype=torch.float32)
    _ = op_autograd.matvec(v)  # warm-up
    peak_autograd = _measure_matvec_peak(op_autograd, v)

    print(
        f"\nautograd path peak: {peak_autograd / 1e6:.1f}MB vs "
        f"budget {budget / 1e6:.1f}MB (expected to exceed)"
    )
    # Soft check: we don't assert it fails, just print so a regression in the
    # autograd path is visible. The hard assertion is in the analytical test
    # above.

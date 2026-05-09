"""Stochastic trace estimation: Hutchinson and Hutch++ (Meyer/Musco/Musco/Woodruff 2021)."""

from dataclasses import dataclass
from typing import Literal, cast

import torch

from hessian_eigenthings.linalg import LinAlgBackend, SingleDeviceBackend
from hessian_eigenthings.operators.base import CurvatureOperator

Distribution = Literal["rademacher", "gaussian"]
Method = Literal["hutchinson", "hutch++"]


@dataclass(frozen=True)
class TraceResult:
    """Stochastic estimate of `tr(A)` together with its sample-standard-error."""

    estimate: float
    stderr: float
    samples: torch.Tensor  # (m,) per-sample contributions used for the estimate


def trace(
    operator: CurvatureOperator,
    *,
    num_matvecs: int = 100,
    method: Method = "hutch++",
    distribution: Distribution = "rademacher",
    seed: int | None = None,
    backend: LinAlgBackend[torch.Tensor] | None = None,
) -> TraceResult:
    """Stochastic estimate of `tr(A)` using `num_matvecs` matrix-vector products."""
    if num_matvecs < 1:
        raise ValueError(f"num_matvecs must be >= 1, got {num_matvecs}")
    backend = backend or SingleDeviceBackend()

    if method == "hutchinson":
        return hutchinson(
            operator, num_samples=num_matvecs, distribution=distribution, seed=seed, backend=backend
        )
    if method == "hutch++":
        return hutch_plus_plus(operator, num_matvecs=num_matvecs, seed=seed, backend=backend)
    raise ValueError(f"unknown method={method!r}")  # pragma: no cover


def hutchinson(
    operator: CurvatureOperator,
    *,
    num_samples: int = 100,
    distribution: Distribution = "rademacher",
    seed: int | None = None,
    backend: LinAlgBackend[torch.Tensor] | None = None,
) -> TraceResult:
    """Hutchinson's `(1/m) Σ vᵢᵀ A vᵢ` estimator. Rademacher gives lower variance for trace."""
    backend = backend or SingleDeviceBackend()
    gen = _generator(seed)

    # Use a 1-D probe for shape; backend allocators copy shape/dtype/device from it.
    probe = torch.empty(operator.size, dtype=operator.dtype, device=operator.device)
    samples = torch.empty(num_samples, dtype=operator.dtype, device=operator.device)
    for i in range(num_samples):
        v = _draw(distribution, probe, gen, backend)
        av = operator.matvec(v)
        samples[i] = backend.dot(v, av)

    estimate = samples.mean().item()
    if num_samples > 1:
        stderr = (samples.std(unbiased=True) / (num_samples**0.5)).item()
    else:
        stderr = float("nan")
    return TraceResult(estimate=estimate, stderr=stderr, samples=samples)


def hutch_plus_plus(
    operator: CurvatureOperator,
    *,
    num_matvecs: int = 99,
    seed: int | None = None,
    backend: LinAlgBackend[torch.Tensor] | None = None,
) -> TraceResult:
    """Hutch++ (Meyer et al. 2021): low-rank sketch + residual Hutchinson.

    Splits the matvec budget into thirds: `m/3` for `AS` (the sketch), `m/3` for `AQ`
    (exact trace on the projected component), and `m/3` for `A G_perp` (Hutchinson on
    the orthogonal complement). Achieves O(1/ε) total matvecs vs Hutchinson's O(1/ε²)
    for `(1±ε)·tr(A)` accuracy on PSD matrices.
    """
    backend = backend or SingleDeviceBackend()
    n = operator.size
    m_per = max(1, num_matvecs // 3)

    gen = _generator(seed)
    probe = torch.empty(n, dtype=operator.dtype, device=operator.device)

    # Phase 1: build AS column-by-column. We never materialize S (the random
    # sketch) as a single (n, m_per) tensor; we generate one column, apply A,
    # store the result column. Cuts peak memory by ~50% at LLM scale.
    a_sketch = torch.empty(n, m_per, dtype=operator.dtype, device=operator.device)
    for i in range(m_per):
        s_i = backend.rademacher_like(probe, generator=gen)
        a_sketch[:, i] = operator.matvec(s_i)

    q = _thin_qr(a_sketch)
    del a_sketch  # free ~n*m_per*4 bytes before the next phase
    # QR returns min(n, m_per) columns; for tiny operators (n < m_per) this is
    # less than m_per, and the residual phase below should still use m_per.
    q_cols = q.shape[1]

    # Phase 2: trace_low = trace(Q^T A Q). Accumulate q_i^T (A q_i) per column;
    # never materialize AQ as a single (n, q_cols) tensor.
    trace_low = 0.0
    for i in range(q_cols):
        aq_i = operator.matvec(q[:, i])
        trace_low += backend.dot(q[:, i], aq_i).item()

    # Phase 3: residual Hutchinson on (I - QQ^T) A (I - QQ^T). Stream column-
    # by-column: generate g_i, project out Q's span, apply A, accumulate sample.
    per_sample = torch.empty(m_per, dtype=operator.dtype, device=operator.device)
    for i in range(m_per):
        g_i = backend.rademacher_like(probe, generator=gen)
        g_perp = g_i - q @ (q.t() @ g_i)
        a_g_perp = operator.matvec(g_perp)
        per_sample[i] = (g_perp * a_g_perp).sum()

    trace_residual = per_sample.mean().item()
    estimate = trace_low + trace_residual
    stderr = (per_sample.std(unbiased=True) / (m_per**0.5)).item() if m_per > 1 else float("nan")
    return TraceResult(estimate=estimate, stderr=stderr, samples=per_sample)


def _generator(seed: int | None) -> torch.Generator:
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    return g


def _draw(
    dist: Distribution,
    probe: torch.Tensor,
    gen: torch.Generator,
    backend: LinAlgBackend[torch.Tensor],
) -> torch.Tensor:
    if dist == "rademacher":
        return backend.rademacher_like(probe, generator=gen)
    if dist == "gaussian":
        return backend.randn_like(probe, generator=gen)
    raise ValueError(f"unknown distribution={dist!r}")  # pragma: no cover


def _thin_qr(mat: torch.Tensor) -> torch.Tensor:
    q, _ = cast(tuple[torch.Tensor, torch.Tensor], torch.linalg.qr(mat, mode="reduced"))
    return q

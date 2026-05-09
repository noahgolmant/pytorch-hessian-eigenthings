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

    sketch = _rademacher_matrix(probe, m_per, gen, backend)
    a_sketch = _apply_columnwise(operator, sketch)

    q = _thin_qr(a_sketch)

    g = _rademacher_matrix(probe, m_per, gen, backend)
    g_perp = g - q @ (q.t() @ g)

    a_q = _apply_columnwise(operator, q)
    trace_low = (q.t() @ a_q).diagonal().sum().item()

    a_g_perp = _apply_columnwise(operator, g_perp)
    per_sample = (g_perp * a_g_perp).sum(dim=0)
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


def _rademacher_matrix(
    probe: torch.Tensor,
    cols: int,
    gen: torch.Generator,
    backend: LinAlgBackend[torch.Tensor],
) -> torch.Tensor:
    return torch.stack([backend.rademacher_like(probe, generator=gen) for _ in range(cols)], dim=1)


def _apply_columnwise(operator: CurvatureOperator, mat: torch.Tensor) -> torch.Tensor:
    return torch.stack([operator.matvec(mat[:, i]) for i in range(mat.shape[1])], dim=1)


def _thin_qr(mat: torch.Tensor) -> torch.Tensor:
    q, _ = cast(tuple[torch.Tensor, torch.Tensor], torch.linalg.qr(mat, mode="reduced"))
    return q

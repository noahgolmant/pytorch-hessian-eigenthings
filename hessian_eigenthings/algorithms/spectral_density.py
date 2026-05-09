"""Stochastic Lanczos Quadrature for eigenvalue spectral density (Ubaru/Chen/Saad 2017).

Per run: draw a Rademacher init vector v, run m Lanczos steps from v, eigendecompose
the tridiagonal T_m to get nodes θ_k and weights τ_k² = (e₁ᵀ y_k)². The estimator
for the density φ(t) = (1/n) Σᵢ δ(t − λᵢ) at point t, smoothed by a Gaussian of
bandwidth σ, is:

    φ(t) ≈ (1 / n_v) Σ_l Σ_k (τ_k^(l))² · g_σ(t − θ_k^(l))

where g_σ is the unit-mass Gaussian kernel. Each run's quadrature weights sum to 1
(they are squared first components of an orthogonal matrix), so the estimator
integrates to ≈ 1 — i.e. it is a probability density of eigenvalues.
"""

import math
from dataclasses import dataclass

import torch

from hessian_eigenthings.algorithms.lanczos import lanczos_tridiagonal
from hessian_eigenthings.linalg import LinAlgBackend, SingleDeviceBackend
from hessian_eigenthings.operators.base import CurvatureOperator


@dataclass(frozen=True)
class SpectralDensityResult:
    """Smoothed eigenvalue density φ(t) on a regular grid, plus the raw quadrature data."""

    grid: torch.Tensor  # (G,) abscissae
    density: torch.Tensor  # (G,) probability density, ∫ density(t) dt ≈ 1
    raw_eigenvalues: torch.Tensor  # (n_runs, m) tridiagonal eigenvalues per run
    raw_weights: torch.Tensor  # (n_runs, m) squared first components per run
    sigma: float  # Gaussian smoothing bandwidth used


def spectral_density(
    operator: CurvatureOperator,
    *,
    num_runs: int = 10,
    lanczos_steps: int = 80,
    num_grid_points: int = 10000,
    sigma: float | None = None,
    grid_padding: float = 0.1,
    seed: int | None = None,
    backend: LinAlgBackend[torch.Tensor] | None = None,
) -> SpectralDensityResult:
    """Estimate the eigenvalue density of a symmetric operator via Stochastic Lanczos Quadrature.

    Args:
        operator: symmetric curvature operator.
        num_runs: number of independent Lanczos runs (n_v in the paper). 10 is a
            reasonable default; larger reduces Monte-Carlo noise.
        lanczos_steps: Lanczos steps per run (m in the paper). 80-100 is typical;
            larger improves spectral resolution but costs more matvecs per run.
        num_grid_points: density evaluated on a regular grid this size.
        sigma: Gaussian smoothing bandwidth. If None, defaults to
            `(spectrum_range / lanczos_steps)`, a standard rule-of-thumb relating
            kernel width to quadrature resolution.
        grid_padding: fraction of spectrum range padded on each side of the grid.
        seed: base seed; per-run init vectors use seed + run_idx.
        backend: vector-arithmetic backend.
    """
    if num_runs < 1:
        raise ValueError(f"num_runs must be >= 1, got {num_runs}")
    if lanczos_steps < 1:
        raise ValueError(f"lanczos_steps must be >= 1, got {lanczos_steps}")
    if lanczos_steps > operator.size:
        raise ValueError(f"lanczos_steps={lanczos_steps} exceeds operator size {operator.size}")

    backend = backend or SingleDeviceBackend()
    n = operator.size

    raw_nodes_runs: list[torch.Tensor] = []
    raw_weights_runs: list[torch.Tensor] = []

    for run in range(num_runs):
        gen = torch.Generator(device="cpu")
        gen.manual_seed((seed if seed is not None else 0) + run)
        probe = torch.empty(n, dtype=operator.dtype, device=operator.device)
        v0 = backend.rademacher_like(probe, generator=gen)

        td = lanczos_tridiagonal(operator, v0, lanczos_steps, reorthogonalize=True, backend=backend)
        m = td.alphas.shape[0]

        tridiag = torch.zeros(m, m, dtype=operator.dtype, device=operator.device)
        tridiag.diagonal().copy_(td.alphas)
        if m > 1 and td.betas.numel() > 0:
            off = td.betas[: m - 1]
            tridiag.diagonal(1).copy_(off)
            tridiag.diagonal(-1).copy_(off)

        theta, y = torch.linalg.eigh(tridiag)
        weights = y[0, :].pow(2)

        raw_nodes_runs.append(theta)
        raw_weights_runs.append(weights)

    max_m = max(t.numel() for t in raw_nodes_runs)
    raw_eigenvalues = _pad_to(raw_nodes_runs, max_m, fill=float("nan"))
    raw_weights = _pad_to(raw_weights_runs, max_m, fill=0.0)

    nodes_flat = torch.cat(raw_nodes_runs)
    lo, hi = float(nodes_flat.min()), float(nodes_flat.max())
    spread = max(hi - lo, 1e-9)
    if sigma is None:
        sigma = spread / lanczos_steps

    pad = grid_padding * spread
    grid = torch.linspace(
        lo - pad, hi + pad, num_grid_points, dtype=operator.dtype, device=operator.device
    )

    density = torch.zeros_like(grid)
    norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    for nodes, weights in zip(raw_nodes_runs, raw_weights_runs, strict=True):
        diffs = grid.unsqueeze(1) - nodes.unsqueeze(0)
        gauss = torch.exp(-(diffs.pow(2)) * inv_two_sigma_sq) * norm
        density += (gauss * weights.unsqueeze(0)).sum(dim=1)
    density /= num_runs

    return SpectralDensityResult(
        grid=grid,
        density=density,
        raw_eigenvalues=raw_eigenvalues,
        raw_weights=raw_weights,
        sigma=sigma,
    )


def _pad_to(tensors: list[torch.Tensor], width: int, *, fill: float) -> torch.Tensor:
    out = torch.full((len(tensors), width), fill, dtype=tensors[0].dtype, device=tensors[0].device)
    for i, t in enumerate(tensors):
        out[i, : t.numel()] = t
    return out

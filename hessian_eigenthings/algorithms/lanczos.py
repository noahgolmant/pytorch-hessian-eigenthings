"""In-house symmetric Lanczos with optional full reorthogonalization.

Reorthogonalization defaults on for `max_iter <= 50` to suppress the ghost
eigenvalues and loss-of-orthogonality that classical Lanczos is famous for
(Paige 1976). For larger Krylov dimensions reorth becomes O(m^2 n) and we
default it off; users analyzing near-degenerate spectra should turn it back on.
"""

from typing import Literal

import torch

from hessian_eigenthings.algorithms.result import EigenResult
from hessian_eigenthings.linalg import LinAlgBackend, SingleDeviceBackend
from hessian_eigenthings.operators.base import CurvatureOperator

_EPS = 1e-12

Which = Literal["LM", "LA", "SA"]


def lanczos(
    operator: CurvatureOperator,
    *,
    k: int = 10,
    max_iter: int | None = None,
    tol: float = 1e-4,
    reorthogonalize: bool | None = None,
    which: Which = "LM",
    seed: int | None = None,
    backend: LinAlgBackend[torch.Tensor] | None = None,
) -> EigenResult:
    """Compute top-k eigenpairs by symmetric Lanczos + tridiagonal eigendecomposition.

    Args:
        operator: symmetric curvature operator providing matvec.
        k: number of Ritz pairs to return.
        max_iter: number of Lanczos steps. Defaults to `min(2 * k, n - 1)`.
        tol: convergence tolerance for Ritz residuals (`|β_m s_m,i| < tol * |λ_i|`).
        reorthogonalize: if True, full Gram-Schmidt against all prior basis vectors
            after each step. None (default) selects True for `max_iter <= 50`.
        which: 'LM' largest magnitude, 'LA' largest algebraic, 'SA' smallest algebraic.
        seed: seed for the initial random vector.
        backend: vector-arithmetic backend.
    """
    n = operator.size
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > n:
        raise ValueError(f"k={k} exceeds operator size {n}")

    if max_iter is None:
        max_iter = min(2 * k, n - 1)
    if max_iter < k:
        raise ValueError(f"max_iter={max_iter} must be at least k={k}")
    if max_iter > n:
        raise ValueError(f"max_iter={max_iter} exceeds operator size {n}")

    if reorthogonalize is None:
        reorthogonalize = max_iter <= 50

    backend = backend or SingleDeviceBackend()

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    v = torch.randn(n, dtype=operator.dtype, generator=gen).to(operator.device)
    v = backend.scale(1.0 / backend.norm(v), v)

    basis: list[torch.Tensor] = [v]
    alphas: list[float] = []
    betas: list[float] = []
    prev_v = backend.zeros_like(v)
    beta = 0.0
    last_beta = 0.0
    iterations = 0

    for j in range(max_iter):
        iterations = j + 1
        av = operator.matvec(basis[j])
        if j > 0:
            av = backend.axpy(-beta, prev_v, av)

        alpha = backend.dot(basis[j], av).item()
        alphas.append(alpha)
        av = backend.axpy(-alpha, basis[j], av)

        if reorthogonalize:
            for vi in basis:
                av = backend.axpy(-backend.dot(vi, av).item(), vi, av)

        beta_next = backend.norm(av).item()
        last_beta = beta_next
        if beta_next < _EPS:
            break
        if j < max_iter - 1:
            betas.append(beta_next)
            prev_v = basis[j]
            basis.append(backend.scale(1.0 / beta_next, av))
            beta = beta_next

    m = len(alphas)
    tridiag = torch.zeros(m, m, dtype=operator.dtype, device=operator.device)
    tridiag.diagonal().copy_(torch.tensor(alphas, dtype=operator.dtype, device=operator.device))
    if m > 1 and betas:
        off = torch.tensor(betas[: m - 1], dtype=operator.dtype, device=operator.device)
        tridiag.diagonal(1).copy_(off)
        tridiag.diagonal(-1).copy_(off)

    theta, s = torch.linalg.eigh(tridiag)

    if which == "LM":
        order = torch.argsort(theta.abs(), descending=True)
    elif which == "LA":
        order = torch.argsort(theta, descending=True)
    elif which == "SA":
        order = torch.argsort(theta, descending=False)
    else:  # pragma: no cover
        raise ValueError(f"unknown which={which!r}")

    sel = order[:k]
    eigenvalues = theta[sel]

    basis_mat = torch.stack(basis, dim=1)
    s_sel = s[:, sel]
    eigenvectors = (basis_mat @ s_sel).t().contiguous()

    last_components = s[-1, sel]
    residuals = last_components.abs() * last_beta

    converged = residuals < tol * eigenvalues.abs().clamp(min=_EPS)

    return EigenResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        residuals=residuals,
        iterations=iterations,
        converged=converged,
    )

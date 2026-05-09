"""In-house symmetric Lanczos with optional full reorthogonalization.

Reorthogonalization defaults on for `max_iter <= 50` to suppress the ghost
eigenvalues and loss-of-orthogonality that classical Lanczos is famous for
(Paige 1976). For larger Krylov dimensions reorth becomes O(m^2 n) and we
default it off; users analyzing near-degenerate spectra should turn it back on.
"""

from dataclasses import dataclass
from typing import Literal

import torch

from hessian_eigenthings.algorithms.result import EigenResult
from hessian_eigenthings.linalg import LinAlgBackend, SingleDeviceBackend
from hessian_eigenthings.operators.base import CurvatureOperator

_EPS = 1e-12

Which = Literal["LM", "LA", "SA"]


@dataclass(frozen=True)
class LanczosTridiag:
    """Output of one Lanczos run: tridiagonal coefficients + the basis used to build them."""

    alphas: torch.Tensor  # (m,) diagonal
    betas: torch.Tensor  # (m-1,) off-diagonal
    basis: list[torch.Tensor]  # length m, each (n,)
    last_beta: float  # ||r_m|| residual norm at termination
    iterations: int  # m, the actual number of Lanczos steps completed


def lanczos_tridiagonal(
    operator: CurvatureOperator,
    v0: torch.Tensor,
    max_iter: int,
    *,
    reorthogonalize: bool = True,
    backend: LinAlgBackend[torch.Tensor] | None = None,
) -> LanczosTridiag:
    """Run `max_iter` Lanczos steps from `v0` and return the tridiagonal + basis.

    Public so SLQ and other quadrature consumers can reuse the same Lanczos kernel.
    """
    backend = backend or SingleDeviceBackend()

    nrm0 = backend.norm(v0)
    if nrm0.item() < _EPS:
        raise ValueError("v0 has near-zero norm")
    v = backend.scale(1.0 / nrm0, v0)

    basis: list[torch.Tensor] = [v]
    alphas_list: list[float] = []
    betas_list: list[float] = []
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
        alphas_list.append(alpha)
        av = backend.axpy(-alpha, basis[j], av)

        if reorthogonalize:
            for vi in basis:
                av = backend.axpy(-backend.dot(vi, av).item(), vi, av)

        beta_next = backend.norm(av).item()
        last_beta = beta_next
        if beta_next < _EPS:
            break
        if j < max_iter - 1:
            betas_list.append(beta_next)
            prev_v = basis[j]
            basis.append(backend.scale(1.0 / beta_next, av))
            beta = beta_next

    alphas = torch.tensor(alphas_list, dtype=operator.dtype, device=operator.device)
    betas = torch.tensor(betas_list, dtype=operator.dtype, device=operator.device)
    return LanczosTridiag(
        alphas=alphas, betas=betas, basis=basis, last_beta=last_beta, iterations=iterations
    )


def _build_tridiag_matrix(td: LanczosTridiag) -> torch.Tensor:
    m = td.alphas.shape[0]
    out = torch.zeros(m, m, dtype=td.alphas.dtype, device=td.alphas.device)
    out.diagonal().copy_(td.alphas)
    if m > 1 and td.betas.numel() > 0:
        off = td.betas[: m - 1]
        out.diagonal(1).copy_(off)
        out.diagonal(-1).copy_(off)
    return out


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
    v0 = torch.randn(n, dtype=operator.dtype, generator=gen).to(operator.device)

    td = lanczos_tridiagonal(
        operator, v0, max_iter, reorthogonalize=reorthogonalize, backend=backend
    )

    tridiag = _build_tridiag_matrix(td)
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

    # Accumulate Ritz vectors column-by-column instead of materializing the full
    # (n, m) basis matrix. The basis takes O(n * m) memory; for an LLM-scale n
    # and m ~= 30 that's tens of GB and a single torch.stack OOMs the GPU.
    # This loop touches each basis vector once and only allocates O(n * k).
    s_sel = s[:, sel]
    n = td.basis[0].shape[0]
    eigenvectors = torch.zeros(n, sel.shape[0], dtype=operator.dtype, device=operator.device)
    for j, basis_vec in enumerate(td.basis):
        eigenvectors.add_(basis_vec.unsqueeze(1) * s_sel[j].unsqueeze(0))
    eigenvectors = eigenvectors.t().contiguous()

    last_components = s[-1, sel]
    residuals = last_components.abs() * td.last_beta

    converged = residuals < tol * eigenvalues.abs().clamp(min=_EPS)

    return EigenResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        residuals=residuals,
        iterations=td.iterations,
        converged=converged,
    )

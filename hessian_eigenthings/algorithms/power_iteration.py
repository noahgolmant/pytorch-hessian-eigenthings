"""Deflated power iteration for top-k eigenpairs of a symmetric operator."""

import torch

from hessian_eigenthings.algorithms.result import EigenResult
from hessian_eigenthings.linalg import LinAlgBackend, SingleDeviceBackend
from hessian_eigenthings.operators.base import CurvatureOperator, LambdaOperator

_EPS = 1e-12


def power_iteration_one(
    operator: CurvatureOperator,
    *,
    max_iter: int = 100,
    tol: float = 1e-4,
    momentum: float = 0.0,
    init: torch.Tensor | None = None,
    seed: int | None = None,
    backend: LinAlgBackend[torch.Tensor] | None = None,
) -> tuple[float, torch.Tensor, float, int, bool]:
    """Compute the dominant eigenpair via power iteration with optional Polyak momentum.

    Returns (eigenvalue, eigenvector, residual_norm, iterations, converged).
    """
    backend = backend or SingleDeviceBackend()

    if init is None:
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        v = torch.randn(operator.size, dtype=operator.dtype, generator=gen).to(operator.device)
    else:
        v = init.clone()

    nrm0 = backend.norm(v)
    if nrm0.item() < _EPS:
        raise ValueError("init vector has near-zero norm")
    v = backend.scale(1.0 / nrm0, v)

    prev_v = backend.zeros_like(v)
    lambda_prev = 0.0
    lambda_est = 0.0
    residual = float("inf")
    converged = False
    last_iter = 0

    for it in range(max_iter):
        last_iter = it + 1
        av = operator.matvec(v)
        if momentum != 0.0:
            av = backend.axpy(-momentum, prev_v, av)

        lambda_est = backend.dot(v, av).item()
        residual = backend.norm(backend.axpy(-lambda_est, v, av)).item()

        nrm = backend.norm(av).item()
        if nrm < _EPS:
            return 0.0, v, 0.0, last_iter, True

        denom = max(abs(lambda_est), _EPS)
        rel_change = abs(lambda_est - lambda_prev) / denom
        rel_residual = residual / denom

        if it > 0 and rel_change < tol and rel_residual < tol:
            converged = True
            break

        prev_v = v
        v = backend.scale(1.0 / nrm, av)
        lambda_prev = lambda_est

    return lambda_est, v, residual, last_iter, converged


def deflated_power_iteration(
    operator: CurvatureOperator,
    *,
    k: int = 10,
    max_iter: int = 100,
    tol: float = 1e-4,
    momentum: float = 0.0,
    seed: int | None = None,
    backend: LinAlgBackend[torch.Tensor] | None = None,
) -> EigenResult:
    """Top-k eigenpairs by repeatedly computing the dominant eigenpair and deflating it out.

    After finding `(λ_i, v_i)`, the operator is replaced with `A - sum_i λ_i v_i v_i^T`
    so the next iteration can find the next-largest eigenpair.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > operator.size:
        raise ValueError(f"k={k} exceeds operator size {operator.size}")

    backend = backend or SingleDeviceBackend()

    eigenvalues: list[float] = []
    eigenvectors: list[torch.Tensor] = []
    residuals: list[float] = []
    converged_flags: list[bool] = []
    total_iters = 0

    current_op: CurvatureOperator = operator

    for i in range(k):
        sub_seed = None if seed is None else seed + i
        lam, vec, res, iters, conv = power_iteration_one(
            current_op,
            max_iter=max_iter,
            tol=tol,
            momentum=momentum,
            seed=sub_seed,
            backend=backend,
        )
        eigenvalues.append(lam)
        eigenvectors.append(vec)
        residuals.append(res)
        converged_flags.append(conv)
        total_iters += iters

        if i < k - 1:
            current_op = _deflate(current_op, lam, vec, backend)

    vals = torch.tensor(eigenvalues, dtype=operator.dtype, device=operator.device)
    vecs = torch.stack(eigenvectors, dim=0)
    res_t = torch.tensor(residuals, dtype=operator.dtype, device=operator.device)
    conv_t = torch.tensor(converged_flags, dtype=torch.bool, device=operator.device)

    order = torch.argsort(vals.abs(), descending=True)
    return EigenResult(
        eigenvalues=vals[order],
        eigenvectors=vecs[order],
        residuals=res_t[order],
        iterations=total_iters,
        converged=conv_t[order],
    )


def _deflate(
    base: CurvatureOperator,
    lam: float,
    vec: torch.Tensor,
    backend: LinAlgBackend[torch.Tensor],
) -> CurvatureOperator:
    def _matvec(v: torch.Tensor) -> torch.Tensor:
        out = base.matvec(v)
        coeff = lam * backend.dot(vec, v)
        return backend.axpy(-coeff, vec, out)

    return LambdaOperator(_matvec, size=base.size, device=base.device, dtype=base.dtype)

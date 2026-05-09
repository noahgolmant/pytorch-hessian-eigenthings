import numpy as np
import pytest
import torch

from hessian_eigenthings.algorithms.power_iteration import (
    deflated_power_iteration,
    power_iteration_one,
)
from hessian_eigenthings.operators.base import LambdaOperator


def _wishart(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64)
    return (a.T @ a) / n


def _operator_from_matrix(M: torch.Tensor) -> LambdaOperator:
    return LambdaOperator(
        lambda v: M @ v,
        size=M.shape[0],
        device=M.device,
        dtype=M.dtype,
    )


def test_top_eigenpair_matches_numpy_on_wishart() -> None:
    M = _wishart(40, seed=0)
    op = _operator_from_matrix(M)
    lam, vec, res, _, conv = power_iteration_one(op, max_iter=300, tol=1e-8, seed=0)

    expected = np.linalg.eigvalsh(M.numpy())[-1]
    assert conv
    assert lam == pytest.approx(expected, rel=1e-4)
    assert res < 1e-3 * abs(lam)


def test_top_eigenvector_satisfies_eigenequation() -> None:
    M = _wishart(40, seed=1)
    op = _operator_from_matrix(M)
    lam, vec, _, _, _ = power_iteration_one(op, max_iter=500, tol=1e-9, seed=1)

    residual = (M @ vec - lam * vec).norm()
    assert residual < 1e-4 * abs(lam)


def test_deflated_recovers_top_k() -> None:
    M = _wishart(50, seed=2)
    op = _operator_from_matrix(M)
    result = deflated_power_iteration(op, k=5, max_iter=400, tol=1e-8, seed=2)

    expected = torch.from_numpy(np.linalg.eigvalsh(M.numpy()))
    expected_top = torch.sort(expected.abs(), descending=True).values[:5]

    torch.testing.assert_close(result.eigenvalues.abs(), expected_top, rtol=5e-3, atol=1e-5)


def test_deflated_returns_orthogonal_eigvecs() -> None:
    M = _wishart(30, seed=3)
    op = _operator_from_matrix(M)
    result = deflated_power_iteration(op, k=4, max_iter=400, tol=1e-9, seed=3)

    gram = result.eigenvectors @ result.eigenvectors.t()
    off_diag = gram - torch.diag(torch.diag(gram))
    assert off_diag.abs().max().item() < 1e-2


def test_residuals_reported() -> None:
    M = _wishart(20, seed=4)
    op = _operator_from_matrix(M)
    result = deflated_power_iteration(op, k=3, max_iter=400, tol=1e-9, seed=4)
    assert result.residuals.shape == (3,)
    assert torch.all(result.residuals >= 0)


def test_k_must_be_positive() -> None:
    M = _wishart(10)
    op = _operator_from_matrix(M)
    with pytest.raises(ValueError, match="k must be"):
        deflated_power_iteration(op, k=0)


def test_k_cannot_exceed_size() -> None:
    M = _wishart(10)
    op = _operator_from_matrix(M)
    with pytest.raises(ValueError, match="exceeds operator size"):
        deflated_power_iteration(op, k=11)


def test_seeded_results_reproducible() -> None:
    M = _wishart(30, seed=5)
    op = _operator_from_matrix(M)
    r1 = deflated_power_iteration(op, k=3, max_iter=200, seed=42)
    r2 = deflated_power_iteration(op, k=3, max_iter=200, seed=42)
    torch.testing.assert_close(r1.eigenvalues, r2.eigenvalues)

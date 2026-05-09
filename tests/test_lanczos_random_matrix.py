import numpy as np
import pytest
import torch

from hessian_eigenthings.algorithms.lanczos import lanczos
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


def _expected_top_k(M: torch.Tensor, k: int, which: str) -> torch.Tensor:
    eig = np.linalg.eigvalsh(M.numpy())
    if which == "LM":
        idx = np.argsort(np.abs(eig))[::-1][:k]
    elif which == "LA":
        idx = np.argsort(eig)[::-1][:k]
    elif which == "SA":
        idx = np.argsort(eig)[:k]
    else:  # pragma: no cover
        raise ValueError(which)
    return torch.from_numpy(eig[idx])


def test_top_k_largest_magnitude() -> None:
    M = _wishart(40, seed=0)
    op = _operator_from_matrix(M)
    result = lanczos(op, k=5, max_iter=30, tol=1e-9, which="LM", seed=0)
    expected = _expected_top_k(M, k=5, which="LM")
    torch.testing.assert_close(result.eigenvalues, expected, rtol=1e-5, atol=1e-7)


def test_top_k_largest_algebraic() -> None:
    M = _wishart(40, seed=1) - 0.5 * torch.eye(40, dtype=torch.float64)
    op = _operator_from_matrix(M)
    result = lanczos(op, k=4, max_iter=40, tol=1e-9, which="LA", seed=1)
    expected = _expected_top_k(M, k=4, which="LA")
    torch.testing.assert_close(result.eigenvalues, expected, rtol=1e-5, atol=1e-7)


def test_smallest_algebraic() -> None:
    M = _wishart(60, seed=2) - 0.5 * torch.eye(60, dtype=torch.float64)
    op = _operator_from_matrix(M)
    result = lanczos(op, k=3, max_iter=60, tol=1e-9, which="SA", seed=2, reorthogonalize=True)
    expected = _expected_top_k(M, k=3, which="SA")
    torch.testing.assert_close(result.eigenvalues, expected, rtol=1e-3, atol=1e-5)


def test_eigenvectors_satisfy_eigenequation() -> None:
    M = _wishart(40, seed=3)
    op = _operator_from_matrix(M)
    result = lanczos(op, k=3, max_iter=40, tol=1e-9, which="LM", seed=3)

    for i in range(3):
        v = result.eigenvectors[i]
        lam = result.eigenvalues[i]
        residual = (M @ v - lam * v).norm()
        assert residual < 1e-4 * abs(lam.item())


def test_eigenvectors_orthonormal_with_reorth() -> None:
    M = _wishart(40, seed=4)
    op = _operator_from_matrix(M)
    result = lanczos(op, k=5, max_iter=20, tol=1e-9, which="LM", seed=4, reorthogonalize=True)
    gram = result.eigenvectors @ result.eigenvectors.t()
    torch.testing.assert_close(gram, torch.eye(5, dtype=torch.float64), rtol=1e-4, atol=1e-6)


def test_residuals_reported_and_finite() -> None:
    M = _wishart(40, seed=5)
    op = _operator_from_matrix(M)
    result = lanczos(op, k=4, max_iter=20, tol=1e-9, seed=5)
    assert result.residuals.shape == (4,)
    assert torch.all(torch.isfinite(result.residuals))
    assert torch.all(result.residuals >= 0)


def test_iterations_at_most_max_iter() -> None:
    M = _wishart(40, seed=6)
    op = _operator_from_matrix(M)
    result = lanczos(op, k=3, max_iter=15, tol=1e-9, seed=6)
    assert result.iterations <= 15


def test_seeded_results_reproducible() -> None:
    M = _wishart(30, seed=7)
    op = _operator_from_matrix(M)
    r1 = lanczos(op, k=3, max_iter=20, seed=42)
    r2 = lanczos(op, k=3, max_iter=20, seed=42)
    torch.testing.assert_close(r1.eigenvalues, r2.eigenvalues)


def test_k_must_be_positive() -> None:
    M = _wishart(10)
    op = _operator_from_matrix(M)
    with pytest.raises(ValueError, match="k must be"):
        lanczos(op, k=0)


def test_k_cannot_exceed_size() -> None:
    M = _wishart(10)
    op = _operator_from_matrix(M)
    with pytest.raises(ValueError, match="exceeds operator size"):
        lanczos(op, k=11)


def test_max_iter_cannot_exceed_size() -> None:
    M = _wishart(10)
    op = _operator_from_matrix(M)
    with pytest.raises(ValueError, match="exceeds operator size"):
        lanczos(op, k=2, max_iter=11)

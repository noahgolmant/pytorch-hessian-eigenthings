import numpy as np
import pytest
import torch
from torch import nn

from hessian_eigenthings.algorithms.trace import hutch_plus_plus, hutchinson, trace
from hessian_eigenthings.operators import HessianOperator
from hessian_eigenthings.operators.base import LambdaOperator
from hessian_eigenthings.param_utils import select_parameters


def _diag_operator(diag: torch.Tensor) -> LambdaOperator:
    return LambdaOperator(
        lambda v: diag * v, size=diag.numel(), device=diag.device, dtype=diag.dtype
    )


def _matrix_operator(M: torch.Tensor) -> LambdaOperator:
    return LambdaOperator(lambda v: M @ v, size=M.shape[0], device=M.device, dtype=M.dtype)


def _wishart(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64)
    return (a.T @ a) / n


def _decaying_spectrum_matrix(n: int, seed: int = 0) -> torch.Tensor:
    """Random orthogonal matrix U times diag(1/(i+1)) times U^T — keeps the spectrum
    decaying but breaks the diagonal `Rademacher v: vᵀ D v ≡ tr(D)` pathology."""
    g = torch.Generator()
    g.manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(a)
    eigvals = torch.tensor([1.0 / (i + 1) for i in range(n)], dtype=torch.float64)
    return q @ torch.diag(eigvals) @ q.T


def test_hutchinson_recovers_known_trace_within_3_stderr() -> None:
    M = _decaying_spectrum_matrix(80, seed=0)
    op = _matrix_operator(M)
    truth = M.diagonal().sum().item()

    result = hutchinson(op, num_samples=400, seed=0)
    assert abs(result.estimate - truth) < 3 * result.stderr
    assert result.samples.shape == (400,)


def test_hutchinson_gaussian_distribution_works() -> None:
    M = _decaying_spectrum_matrix(80, seed=1)
    op = _matrix_operator(M)
    truth = M.diagonal().sum().item()
    result = hutchinson(op, num_samples=400, distribution="gaussian", seed=1)
    assert abs(result.estimate - truth) < 3 * result.stderr


def test_hutch_plus_plus_recovers_known_trace() -> None:
    M = _decaying_spectrum_matrix(80, seed=2)
    op = _matrix_operator(M)
    truth = M.diagonal().sum().item()
    result = hutch_plus_plus(op, num_matvecs=99, seed=0)
    assert result.estimate == pytest.approx(truth, rel=5e-2)


def test_hutch_plus_plus_lower_variance_than_hutchinson_at_equal_budget() -> None:
    """Hutch++ should beat vanilla Hutchinson on a decaying-spectrum matrix at equal matvec budget."""
    M = _decaying_spectrum_matrix(80, seed=3)
    op = _matrix_operator(M)
    truth = M.diagonal().sum().item()
    budget = 60

    n_trials = 30
    hutch_errors = []
    hutchpp_errors = []
    for trial in range(n_trials):
        h = hutchinson(op, num_samples=budget, seed=1000 + trial)
        hpp = hutch_plus_plus(op, num_matvecs=budget, seed=1000 + trial)
        hutch_errors.append(abs(h.estimate - truth))
        hutchpp_errors.append(abs(hpp.estimate - truth))

    assert float(np.mean(hutchpp_errors)) < float(np.mean(hutch_errors))


def test_trace_dispatches_by_method() -> None:
    diag = torch.linspace(1.0, 5.0, 50, dtype=torch.float64)
    op = _diag_operator(diag)
    truth = diag.sum().item()

    r1 = trace(op, num_matvecs=99, method="hutchinson", seed=0)
    r2 = trace(op, num_matvecs=99, method="hutch++", seed=0)
    assert r1.estimate == pytest.approx(truth, rel=0.2)
    assert r2.estimate == pytest.approx(truth, rel=0.2)


def test_num_matvecs_must_be_positive() -> None:
    op = _diag_operator(torch.ones(10, dtype=torch.float64))
    with pytest.raises(ValueError, match="num_matvecs"):
        trace(op, num_matvecs=0)


def _supervised_loss(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    x, y = batch
    return nn.functional.mse_loss(model(x), y)


def _full_hessian(model: nn.Module, loss_fn, batch) -> torch.Tensor:
    params = list(select_parameters(model).values())
    n = sum(int(p.numel()) for p in params)
    loss = loss_fn(model, batch)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_vec = torch.cat([g.reshape(-1) for g in grads])
    cols = []
    for i in range(n):
        row = torch.autograd.grad(grad_vec[i], params, retain_graph=(i < n - 1))
        cols.append(torch.cat([r.reshape(-1) for r in row]))
    return torch.stack(cols, dim=1)


def test_trace_on_real_hessian_matches_numpy() -> None:
    g = torch.Generator()
    g.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2)).to(torch.float64)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=torch.float64)

    g2 = torch.Generator()
    g2.manual_seed(1)
    x = torch.randn(8, 3, generator=g2, dtype=torch.float64)
    y = torch.randn(8, 2, generator=g2, dtype=torch.float64)

    H = _full_hessian(model, _supervised_loss, (x, y))
    truth = H.diagonal().sum().item()

    op = HessianOperator(model=model, dataloader=[(x, y)], loss_fn=_supervised_loss)

    h = hutchinson(op, num_samples=300, seed=0)
    hpp = hutch_plus_plus(op, num_matvecs=99, seed=0)

    assert h.estimate == pytest.approx(truth, abs=3 * h.stderr + 0.05 * abs(truth))
    assert hpp.estimate == pytest.approx(truth, rel=0.05)

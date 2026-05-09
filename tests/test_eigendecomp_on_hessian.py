"""Lanczos and deflated power iteration on a real HessianOperator, verified against the full numerical Hessian."""

import numpy as np
import pytest
import torch
from torch import nn

from hessian_eigenthings.algorithms.lanczos import lanczos
from hessian_eigenthings.algorithms.power_iteration import deflated_power_iteration
from hessian_eigenthings.operators import HessianOperator
from hessian_eigenthings.param_utils import select_parameters


def _seed(value: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(value)
    return g


def _tiny_mlp() -> nn.Module:
    g = _seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2)).to(torch.float64)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=torch.float64)
    return model


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


@pytest.fixture
def hessian_setup():
    model = _tiny_mlp()
    g = _seed(1)
    x = torch.randn(8, 3, generator=g, dtype=torch.float64)
    y = torch.randn(8, 2, generator=g, dtype=torch.float64)
    H = _full_hessian(model, _supervised_loss, (x, y))
    op = HessianOperator(model=model, dataloader=[(x, y)], loss_fn=_supervised_loss)
    return H, op


def test_lanczos_recovers_top_k_of_real_hessian(hessian_setup) -> None:
    H, op = hessian_setup
    k = 3
    result = lanczos(op, k=k, max_iter=op.size, tol=1e-9, which="LM", seed=0)

    expected = np.linalg.eigvalsh(H.numpy())
    expected_top = torch.from_numpy(expected[np.argsort(np.abs(expected))[::-1][:k]].copy())

    torch.testing.assert_close(result.eigenvalues, expected_top, rtol=1e-5, atol=1e-7)


def test_lanczos_eigenvectors_solve_real_hessian_eigeneq(hessian_setup) -> None:
    H, op = hessian_setup
    result = lanczos(op, k=3, max_iter=op.size, tol=1e-9, which="LM", seed=1)
    for i in range(3):
        v = result.eigenvectors[i]
        lam = result.eigenvalues[i]
        residual = (H @ v - lam * v).norm()
        assert residual < 1e-4 * max(abs(lam.item()), 1e-6)


def test_deflated_power_iter_recovers_top_k_of_real_hessian(hessian_setup) -> None:
    H, op = hessian_setup
    k = 3
    result = deflated_power_iteration(op, k=k, max_iter=500, tol=1e-9, seed=0)

    expected = np.linalg.eigvalsh(H.numpy())
    expected_top = torch.from_numpy(expected[np.argsort(np.abs(expected))[::-1][:k]].copy())

    torch.testing.assert_close(result.eigenvalues.abs(), expected_top.abs(), rtol=5e-3, atol=1e-4)

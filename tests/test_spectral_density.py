import numpy as np
import pytest
import torch
from torch import nn

from hessian_eigenthings.algorithms.spectral_density import spectral_density
from hessian_eigenthings.operators import HessianOperator
from hessian_eigenthings.operators.base import LambdaOperator
from hessian_eigenthings.param_utils import select_parameters


def _diag_operator(diag: torch.Tensor) -> LambdaOperator:
    return LambdaOperator(
        lambda v: diag * v, size=diag.numel(), device=diag.device, dtype=diag.dtype
    )


def _trapz(y: torch.Tensor, x: torch.Tensor) -> float:
    return float(torch.trapz(y, x))


def test_density_integrates_to_approximately_one() -> None:
    diag = torch.linspace(1.0, 5.0, 50, dtype=torch.float64)
    op = _diag_operator(diag)
    result = spectral_density(op, num_runs=8, lanczos_steps=40, num_grid_points=4000, seed=0)
    integral = _trapz(result.density, result.grid)
    assert integral == pytest.approx(1.0, rel=0.05)


def test_density_concentrates_near_two_clusters() -> None:
    """Two well-separated eigenvalue clusters: density should peak near both."""
    g = torch.Generator()
    g.manual_seed(0)
    a = torch.randn(60, 60, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(a)
    eigvals = torch.cat(
        [
            torch.full((30,), 1.0, dtype=torch.float64),
            torch.full((30,), 5.0, dtype=torch.float64),
        ]
    )
    M = q @ torch.diag(eigvals) @ q.T
    op = LambdaOperator(lambda v: M @ v, size=60, device=M.device, dtype=M.dtype)

    result = spectral_density(
        op, num_runs=20, lanczos_steps=50, num_grid_points=4000, sigma=0.1, seed=0
    )

    grid_np = result.grid.numpy()
    density_np = result.density.numpy()

    mass_low = float(np.trapezoid(density_np[grid_np < 3.0], grid_np[grid_np < 3.0]))
    mass_high = float(np.trapezoid(density_np[grid_np > 3.0], grid_np[grid_np > 3.0]))
    assert mass_low == pytest.approx(0.5, abs=0.15)
    assert mass_high == pytest.approx(0.5, abs=0.15)

    peak_low_idx = density_np[grid_np < 3.0].argmax()
    peak_low_loc = grid_np[grid_np < 3.0][peak_low_idx]
    peak_high_idx = density_np[grid_np > 3.0].argmax()
    peak_high_loc = grid_np[grid_np > 3.0][peak_high_idx]
    assert abs(peak_low_loc - 1.0) < 0.3
    assert abs(peak_high_loc - 5.0) < 0.3


def test_density_recovers_uniform_spectrum_shape() -> None:
    """Uniform spectrum should give roughly uniform density across the support."""
    diag = torch.linspace(0.0, 10.0, 200, dtype=torch.float64)
    op = _diag_operator(diag)
    result = spectral_density(
        op, num_runs=12, lanczos_steps=80, num_grid_points=2000, sigma=0.3, seed=0
    )

    grid_np = result.grid.numpy()
    density_np = result.density.numpy()
    interior = (grid_np > 1.0) & (grid_np < 9.0)
    interior_density = density_np[interior]
    assert interior_density.min() > 0.5 * interior_density.max()


def test_raw_eigenvalues_and_weights_shapes() -> None:
    diag = torch.linspace(1.0, 5.0, 50, dtype=torch.float64)
    op = _diag_operator(diag)
    result = spectral_density(op, num_runs=5, lanczos_steps=30, seed=0)
    assert result.raw_eigenvalues.shape == (5, 30)
    assert result.raw_weights.shape == (5, 30)
    weight_sums = result.raw_weights.sum(dim=1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)


def test_seeded_results_reproducible() -> None:
    diag = torch.linspace(1.0, 5.0, 50, dtype=torch.float64)
    op = _diag_operator(diag)
    r1 = spectral_density(op, num_runs=4, lanczos_steps=30, seed=42)
    r2 = spectral_density(op, num_runs=4, lanczos_steps=30, seed=42)
    torch.testing.assert_close(r1.density, r2.density)
    torch.testing.assert_close(r1.raw_eigenvalues, r2.raw_eigenvalues)


def test_lanczos_steps_must_not_exceed_operator_size() -> None:
    op = _diag_operator(torch.ones(10, dtype=torch.float64))
    with pytest.raises(ValueError, match="exceeds operator size"):
        spectral_density(op, lanczos_steps=11)


def test_num_runs_must_be_positive() -> None:
    op = _diag_operator(torch.ones(10, dtype=torch.float64))
    with pytest.raises(ValueError, match="num_runs"):
        spectral_density(op, num_runs=0)


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


def test_density_on_real_hessian_brackets_full_spectrum() -> None:
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
    true_eigvals = np.linalg.eigvalsh(H.numpy())

    op = HessianOperator(model=model, dataloader=[(x, y)], loss_fn=_supervised_loss)
    result = spectral_density(op, num_runs=8, lanczos_steps=op.size, seed=0)
    grid_np = result.grid.numpy()
    assert grid_np.min() <= true_eigvals.min() + 1e-6
    assert grid_np.max() >= true_eigvals.max() - 1e-6

    integral = _trapz(result.density, result.grid)
    assert integral == pytest.approx(1.0, rel=0.1)

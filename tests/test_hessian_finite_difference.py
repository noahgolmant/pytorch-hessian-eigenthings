"""Verify the finite-difference HVP path matches autograd within the predicted O(eps^2) bound."""

import pytest
import torch
from torch import nn

from hessian_eigenthings.operators import HessianOperator


def _seed(value: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(value)
    return g


def _tiny_mlp(dtype: torch.dtype = torch.float64) -> nn.Module:
    g = _seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2)).to(dtype)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=dtype)
    return model


def _supervised_loss(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    x, y = batch
    return nn.functional.mse_loss(model(x), y)


@pytest.fixture
def setup():
    model = _tiny_mlp()
    g = _seed(1)
    x = torch.randn(8, 3, generator=g, dtype=torch.float64)
    y = torch.randn(8, 2, generator=g, dtype=torch.float64)
    return model, (x, y)


def test_finite_difference_matches_autograd_fp64(setup) -> None:
    model, batch = setup
    op_ag = HessianOperator(
        model=model, dataloader=[batch], loss_fn=_supervised_loss, method="autograd"
    )
    op_fd = HessianOperator(
        model=model,
        dataloader=[batch],
        loss_fn=_supervised_loss,
        method="finite_difference",
        fd_eps=1e-5,
    )

    g = _seed(2)
    for _ in range(5):
        v = torch.randn(op_ag.size, generator=g, dtype=torch.float64)
        ag = op_ag.matvec(v)
        fd = op_fd.matvec(v)
        rel_err = (fd - ag).norm() / ag.norm().clamp(min=1e-12)
        # Predicted bound at eps=1e-5 fp64: O(eps^2) ~ 1e-10 truncation,
        # plus O(machine_eps / eps) ~ 1e-11 roundoff. We allow 1e-6 slack.
        assert rel_err.item() < 1e-6


def test_finite_difference_restores_parameters(setup) -> None:
    """If FD HVP raises mid-call, params should still be restored to original values."""
    model, batch = setup
    snapshot = [p.detach().clone() for p in model.parameters()]
    op = HessianOperator(
        model=model, dataloader=[batch], loss_fn=_supervised_loss, method="finite_difference"
    )
    v = torch.randn(op.size, dtype=torch.float64)
    op.matvec(v)
    for p, snap in zip(model.parameters(), snapshot, strict=True):
        torch.testing.assert_close(p.data, snap)


def test_finite_difference_restores_parameters_on_exception() -> None:
    g = _seed(0)
    model = nn.Linear(3, 2).to(torch.float64)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=torch.float64)
    snapshot = [p.detach().clone() for p in model.parameters()]

    def bad_loss(_model: nn.Module, _batch) -> torch.Tensor:
        raise RuntimeError("simulated forward failure")

    op = HessianOperator(
        model=model,
        dataloader=[
            (torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64))
        ],
        loss_fn=bad_loss,
        method="finite_difference",
    )
    v = torch.randn(op.size, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="simulated"):
        op.matvec(v)
    for p, snap in zip(model.parameters(), snapshot, strict=True):
        torch.testing.assert_close(p.data, snap)


def test_finite_difference_default_eps_per_dtype() -> None:
    g = _seed(0)
    model_fp32 = nn.Linear(3, 2).to(torch.float32)
    for p in model_fp32.parameters():
        p.data = torch.randn(p.shape, generator=g)
    model_fp64 = nn.Linear(3, 2).to(torch.float64)
    for p in model_fp64.parameters():
        p.data = torch.randn(p.shape, generator=_seed(1), dtype=torch.float64)

    batch_fp32 = (torch.zeros(1, 3), torch.zeros(1, 2))
    batch_fp64 = (torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64))

    op32 = HessianOperator(
        model=model_fp32,
        dataloader=[batch_fp32],
        loss_fn=_supervised_loss,
        method="finite_difference",
    )
    op64 = HessianOperator(
        model=model_fp64,
        dataloader=[batch_fp64],
        loss_fn=_supervised_loss,
        method="finite_difference",
    )
    assert op32.fd_eps > op64.fd_eps  # higher precision -> smaller eps


def test_unknown_method_raises(setup) -> None:
    model, batch = setup
    op = HessianOperator(
        model=model, dataloader=[batch], loss_fn=_supervised_loss, method="bogus"  # type: ignore[arg-type]
    )
    v = torch.randn(op.size, dtype=torch.float64)
    with pytest.raises(ValueError, match="unknown method"):
        op.matvec(v)

import pytest
import torch
from torch import nn

from hessian_eigenthings.operators import GGNOperator
from hessian_eigenthings.param_utils import select_parameters


def _seed(value: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(value)
    return g


def _tiny_mlp() -> nn.Module:
    g = _seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 4)).to(torch.float64)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=torch.float64)
    return model


def _forward(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    x, _ = batch
    return model(x)


def _ce_loss(output: torch.Tensor, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    _, y = batch
    return nn.functional.cross_entropy(output, y)


def _mse_loss(output: torch.Tensor, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    _, y = batch
    return nn.functional.mse_loss(output, y)


def _full_jacobian(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return J = ∂model(x)/∂θ as a (batch*output_dim, n_params) matrix."""
    params = list(select_parameters(model).values())
    n_params = sum(int(p.numel()) for p in params)
    output = model(x).reshape(-1)
    out_dim = output.numel()
    j = torch.zeros(out_dim, n_params, dtype=output.dtype)
    for i in range(out_dim):
        grads = torch.autograd.grad(
            output[i], params, retain_graph=(i < out_dim - 1), allow_unused=False
        )
        j[i] = torch.cat([g.reshape(-1) for g in grads])
    return j


def _full_loss_hessian(loss_of_output_fn, output: torch.Tensor, batch) -> torch.Tensor:
    """H_loss = ∂²loss/∂output²."""
    out_leaf = output.detach().reshape(-1).requires_grad_(True)
    out_shaped = out_leaf.reshape(output.shape)
    loss = loss_of_output_fn(out_shaped, batch)
    grad_loss = torch.autograd.grad(loss, out_leaf, create_graph=True)[0]
    n = grad_loss.numel()
    h = torch.zeros(n, n, dtype=output.dtype)
    for i in range(n):
        row = torch.autograd.grad(grad_loss[i], out_leaf, retain_graph=(i < n - 1))[0]
        h[i] = row.reshape(-1)
    return h


def _full_ggn(model, forward_fn, loss_of_output_fn, batch) -> torch.Tensor:
    x, _ = batch
    j = _full_jacobian(model, x)
    out = forward_fn(model, batch)
    h_loss = _full_loss_hessian(loss_of_output_fn, out, batch)
    return j.t() @ h_loss @ j


@pytest.fixture
def ce_setup():
    model = _tiny_mlp()
    g = _seed(1)
    x = torch.randn(6, 3, generator=g, dtype=torch.float64)
    y = torch.randint(0, 4, (6,), generator=g)
    return model, (x, y)


@pytest.fixture
def mse_setup():
    model = _tiny_mlp()
    g = _seed(2)
    x = torch.randn(6, 3, generator=g, dtype=torch.float64)
    y = torch.randn(6, 4, generator=g, dtype=torch.float64)
    return model, (x, y)


def test_ggn_matvec_matches_full_construction_cross_entropy(ce_setup) -> None:
    """Autograd path (legacy fallback) on CE."""
    model, batch = ce_setup
    G_full = _full_ggn(model, _forward, _ce_loss, batch)
    op = GGNOperator(
        model=model,
        dataloader=[batch],
        forward_fn=_forward,
        loss_of_output_fn=_ce_loss,
        loss_hvp="autograd",
    )
    g = _seed(3)
    for _ in range(5):
        v = torch.randn(op.size, generator=g, dtype=torch.float64)
        torch.testing.assert_close(op.matvec(v), G_full @ v, rtol=1e-7, atol=1e-9)


def test_ggn_matvec_matches_full_construction_mse(mse_setup) -> None:
    """Autograd path on MSE."""
    model, batch = mse_setup
    G_full = _full_ggn(model, _forward, _mse_loss, batch)
    op = GGNOperator(
        model=model,
        dataloader=[batch],
        forward_fn=_forward,
        loss_of_output_fn=_mse_loss,
        loss_hvp="autograd",
    )
    g = _seed(4)
    v = torch.randn(op.size, generator=g, dtype=torch.float64)
    torch.testing.assert_close(op.matvec(v), G_full @ v, rtol=1e-7, atol=1e-9)


def test_ggn_full_dataset_averages(ce_setup) -> None:
    model, (x, y) = ce_setup
    b1 = (x[:3], y[:3])
    b2 = (x[3:], y[3:])
    op = GGNOperator(
        model=model,
        dataloader=[b1, b2],
        forward_fn=_forward,
        loss_of_output_fn=_ce_loss,
        loss_hvp="autograd",
    )
    G1 = _full_ggn(model, _forward, _ce_loss, b1)
    G2 = _full_ggn(model, _forward, _ce_loss, b2)
    G_avg = 0.5 * (G1 + G2)
    g = _seed(5)
    v = torch.randn(op.size, generator=g, dtype=torch.float64)
    torch.testing.assert_close(op.matvec(v), G_avg @ v, rtol=1e-7, atol=1e-9)


def test_ggn_is_psd(ce_setup) -> None:
    """For convex losses (CE+softmax) GGN must be PSD: vᵀGv ≥ 0 for all v."""
    model, batch = ce_setup
    op = GGNOperator(
        model=model,
        dataloader=[batch],
        forward_fn=_forward,
        loss_of_output_fn=_ce_loss,
        loss_hvp="autograd",
    )
    g = _seed(6)
    for _ in range(20):
        v = torch.randn(op.size, generator=g, dtype=torch.float64)
        gv = op.matvec(v)
        assert torch.dot(v, gv).item() >= -1e-10


def test_ggn_rejects_wrong_shape(ce_setup) -> None:
    model, batch = ce_setup
    op = GGNOperator(
        model=model,
        dataloader=[batch],
        forward_fn=_forward,
        loss_of_output_fn=_ce_loss,
        loss_hvp="autograd",
    )
    with pytest.raises(ValueError, match="shape"):
        op.matvec(torch.zeros(op.size + 1, dtype=torch.float64))


def test_ggn_analytical_requires_hvp(ce_setup) -> None:
    """The default analytical path needs `loss_of_output_fn.hvp` to be defined."""
    model, batch = ce_setup
    with pytest.raises(ValueError, match="hvp"):
        GGNOperator(
            model=model,
            dataloader=[batch],
            forward_fn=_forward,
            loss_of_output_fn=_ce_loss,  # no .hvp attribute
        )

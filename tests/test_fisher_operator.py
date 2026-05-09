import pytest
import torch
from torch import nn

from hessian_eigenthings.operators import EmpiricalFisherOperator
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


def _per_sample_mse(model: nn.Module, sample: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    x, y = sample
    return ((model(x) - y) ** 2).mean()


def _full_empirical_fisher(model: nn.Module, batch) -> torch.Tensor:
    """F = (1/N) Σ_i g_i g_i^T computed by looping over samples."""
    x, y = batch
    n_samples = x.shape[0]
    params = list(select_parameters(model).values())

    grads = []
    for i in range(n_samples):
        sample = (x[i], y[i])
        loss = _per_sample_mse(model, sample)
        g = torch.autograd.grad(loss, params)
        grads.append(torch.cat([gi.reshape(-1) for gi in g]))
    g_mat = torch.stack(grads, dim=0)
    return (g_mat.t() @ g_mat) / n_samples


@pytest.fixture
def fisher_setup():
    model = _tiny_mlp()
    g = _seed(1)
    x = torch.randn(8, 3, generator=g, dtype=torch.float64)
    y = torch.randn(8, 2, generator=g, dtype=torch.float64)
    return model, (x, y)


def test_fisher_matvec_matches_full_construction(fisher_setup) -> None:
    model, batch = fisher_setup
    F_full = _full_empirical_fisher(model, batch)
    op = EmpiricalFisherOperator(
        model=model,
        dataloader=[batch],
        per_sample_loss_fn=_per_sample_mse,
    )
    g = _seed(2)
    for _ in range(5):
        v = torch.randn(op.size, generator=g, dtype=torch.float64)
        torch.testing.assert_close(op.matvec(v), F_full @ v, rtol=1e-7, atol=1e-9)


def test_fisher_is_psd(fisher_setup) -> None:
    """Empirical Fisher is a sum of outer products → always PSD."""
    model, batch = fisher_setup
    op = EmpiricalFisherOperator(
        model=model, dataloader=[batch], per_sample_loss_fn=_per_sample_mse
    )
    g = _seed(3)
    for _ in range(20):
        v = torch.randn(op.size, generator=g, dtype=torch.float64)
        fv = op.matvec(v)
        assert torch.dot(v, fv).item() >= -1e-10


def test_fisher_full_dataset_averages_correctly(fisher_setup) -> None:
    model, (x, y) = fisher_setup
    b1 = (x[:4], y[:4])
    b2 = (x[4:], y[4:])
    op = EmpiricalFisherOperator(
        model=model, dataloader=[b1, b2], per_sample_loss_fn=_per_sample_mse
    )
    F1 = _full_empirical_fisher(model, b1)
    F2 = _full_empirical_fisher(model, b2)
    F_avg = 0.5 * (F1 + F2)
    g = _seed(4)
    v = torch.randn(op.size, generator=g, dtype=torch.float64)
    torch.testing.assert_close(op.matvec(v), F_avg @ v, rtol=1e-7, atol=1e-9)


def test_fisher_rejects_wrong_shape(fisher_setup) -> None:
    model, batch = fisher_setup
    op = EmpiricalFisherOperator(
        model=model, dataloader=[batch], per_sample_loss_fn=_per_sample_mse
    )
    with pytest.raises(ValueError, match="shape"):
        op.matvec(torch.zeros(op.size + 1, dtype=torch.float64))

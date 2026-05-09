import pytest
import torch
from torch import nn

from hessian_eigenthings.operators import HessianOperator
from hessian_eigenthings.param_utils import match_names, select_parameters


def _seed(value: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(value)
    return g


def _tiny_mlp() -> nn.Module:
    g = _seed(0)
    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.Tanh(),
        nn.Linear(5, 2),
    ).to(torch.float64)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=torch.float64)
    return model


def _supervised_loss(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    x, y = batch
    return nn.functional.mse_loss(model(x), y)


def _full_hessian(model: nn.Module, loss_fn, batch) -> torch.Tensor:
    """Materialize the full Hessian by stacking per-parameter HVPs against basis vectors."""
    params = list(select_parameters(model).values())
    n = sum(int(p.numel()) for p in params)

    loss = loss_fn(model, batch)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_vec = torch.cat([g.reshape(-1) for g in grads])

    cols = []
    for i in range(n):
        row = torch.autograd.grad(grad_vec[i], params, retain_graph=(i < n - 1), allow_unused=False)
        cols.append(torch.cat([r.reshape(-1) for r in row]))
    return torch.stack(cols, dim=1)


@pytest.fixture
def model_and_data() -> tuple[nn.Module, tuple[torch.Tensor, torch.Tensor]]:
    model = _tiny_mlp()
    g = _seed(1)
    x = torch.randn(8, 3, generator=g, dtype=torch.float64)
    y = torch.randn(8, 2, generator=g, dtype=torch.float64)
    return model, (x, y)


def test_hessian_matvec_matches_full_hessian(model_and_data) -> None:
    model, batch = model_and_data
    H = _full_hessian(model, _supervised_loss, batch)
    op = HessianOperator(
        model=model,
        dataloader=[batch],
        loss_fn=_supervised_loss,
        full_dataset=True,
    )

    g = _seed(2)
    for _ in range(5):
        v = torch.randn(op.size, generator=g, dtype=torch.float64)
        torch.testing.assert_close(op.matvec(v), H @ v, rtol=1e-7, atol=1e-9)


def test_full_dataset_averages_over_batches(model_and_data) -> None:
    model, (x, y) = model_and_data
    b1 = (x[:4], y[:4])
    b2 = (x[4:], y[4:])
    op = HessianOperator(
        model=model,
        dataloader=[b1, b2],
        loss_fn=_supervised_loss,
        full_dataset=True,
    )
    H1 = _full_hessian(model, _supervised_loss, b1)
    H2 = _full_hessian(model, _supervised_loss, b2)
    H_avg = 0.5 * (H1 + H2)

    g = _seed(3)
    v = torch.randn(op.size, generator=g, dtype=torch.float64)
    torch.testing.assert_close(op.matvec(v), H_avg @ v, rtol=1e-7, atol=1e-9)


def test_param_filter_narrows_operator_size(model_and_data) -> None:
    model, batch = model_and_data
    op = HessianOperator(
        model=model,
        dataloader=[batch],
        loss_fn=_supervised_loss,
        param_filter=match_names("2.weight"),
    )
    assert op.size == model[2].weight.numel()


def test_param_filter_matches_corresponding_hessian_block(model_and_data) -> None:
    model, batch = model_and_data
    selected = select_parameters(model, match_names("2.*"))
    op = HessianOperator(
        model=model,
        dataloader=[batch],
        loss_fn=_supervised_loss,
        param_filter=match_names("2.*"),
    )

    full_H = _full_hessian(model, _supervised_loss, batch)
    full_params = select_parameters(model)
    full_keys = list(full_params)
    sub_keys = list(selected)

    offsets = {}
    cur = 0
    for k in full_keys:
        offsets[k] = (cur, cur + full_params[k].numel())
        cur += full_params[k].numel()
    indices = torch.cat([torch.arange(*offsets[k]) for k in sub_keys])
    sub_H = full_H[indices][:, indices]

    g = _seed(4)
    v = torch.randn(op.size, generator=g, dtype=torch.float64)
    torch.testing.assert_close(op.matvec(v), sub_H @ v, rtol=1e-7, atol=1e-9)


def test_matvec_rejects_wrong_shape(model_and_data) -> None:
    model, batch = model_and_data
    op = HessianOperator(model=model, dataloader=[batch], loss_fn=_supervised_loss)
    with pytest.raises(ValueError, match="shape"):
        op.matvec(torch.zeros(op.size + 1, dtype=torch.float64))


def test_microbatch_guard_raises_for_bn() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2)).to(torch.float64)
    batch = (torch.randn(8, 4, dtype=torch.float64), torch.randn(8, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="BatchNorm"):
        HessianOperator(
            model=model,
            dataloader=[batch],
            loss_fn=_supervised_loss,
            microbatch_size=4,
        )


def test_microbatch_unsafe_bypass_allows_bn() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2)).to(torch.float64)
    model.eval()
    batch = (torch.randn(8, 4, dtype=torch.float64), torch.randn(8, 2, dtype=torch.float64))
    op = HessianOperator(
        model=model,
        dataloader=[batch],
        loss_fn=_supervised_loss,
        microbatch_size=4,
        microbatch_unsafe=True,
    )
    v = torch.randn(op.size, dtype=torch.float64)
    out = op.matvec(v)
    assert out.shape == (op.size,)
    assert torch.all(torch.isfinite(out))


def test_full_dataset_false_uses_one_batch_per_call(model_and_data) -> None:
    model, (x, y) = model_and_data
    b1 = (x[:4], y[:4])
    b2 = (x[4:], y[4:])
    op = HessianOperator(
        model=model,
        dataloader=[b1, b2],
        loss_fn=_supervised_loss,
        full_dataset=False,
    )
    g = _seed(5)
    v = torch.randn(op.size, generator=g, dtype=torch.float64)
    H1v = _full_hessian(model, _supervised_loss, b1) @ v
    H2v = _full_hessian(model, _supervised_loss, b2) @ v
    torch.testing.assert_close(op.matvec(v), H1v, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(op.matvec(v), H2v, rtol=1e-7, atol=1e-9)

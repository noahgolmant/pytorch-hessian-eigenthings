"""Closed-form Hessian / GGN checks on linear and logistic regression.

The existing operator tests (`test_hessian_operator.py`, `test_ggn_operator.py`)
materialize the full curvature matrix via autograd over basis vectors and check
`op.matvec(v) ≈ H_autograd @ v`. That validates the HVP path against autograd,
but a bug in autograd-double-backward would be invisible — both sides share the
same channel.

These tests use an independent analytic baseline:

- **Linear regression, mean-reduced MSE.** For a bias-free linear model with
  scalar output, `H = (2/N) X^T X`.
- **Binary logistic regression, mean-reduced BCEWithLogits.** For a bias-free
  linear model with scalar logit, `H = (1/N) X^T diag(σ(1-σ)) X`.
- **Multiclass softmax cross-entropy.** For a bias-free linear model with `C`
  logits, the Hessian over the flattened `(C, d)` weight is the average of
  Kronecker products `(diag(p_k) - p_k p_k^T) ⊗ (x_k x_k^T)`.

For these (model linear in params, convex loss) cases the GGN equals the
Hessian, so we check both operators against the same baseline.
"""

import pytest
import torch
from torch import nn

from hessian_eigenthings.loss_fns import (
    cross_entropy_loss_of_output,
    mse_loss_of_output,
    supervised_forward,
    supervised_loss,
    supervised_loss_of_output,
)
from hessian_eigenthings.operators import GGNOperator, HessianOperator


def _seed(value: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(value)
    return g


def _linear(in_features: int, out_features: int, seed: int = 0) -> nn.Linear:
    g = _seed(seed)
    layer = nn.Linear(in_features, out_features, bias=False).to(torch.float64)
    layer.weight.data = torch.randn(layer.weight.shape, generator=g, dtype=torch.float64)
    return layer


def _check_matvec(op, H_analytic: torch.Tensor, n_probes: int = 5, seed: int = 99) -> None:
    g = _seed(seed)
    for _ in range(n_probes):
        v = torch.randn(op.size, generator=g, dtype=torch.float64)
        torch.testing.assert_close(op.matvec(v), H_analytic @ v, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# Linear regression: H = (2/N) X^T X
# ---------------------------------------------------------------------------


def _linear_regression_setup(n: int = 16, d: int = 5):
    model = _linear(d, 1, seed=0)
    g = _seed(1)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randn(n, 1, generator=g, dtype=torch.float64)
    return model, x, y


def _analytic_linear_regression_hessian(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    return (2.0 / n) * (x.T @ x)


def test_linear_regression_hessian_matches_closed_form() -> None:
    model, x, y = _linear_regression_setup()
    H = _analytic_linear_regression_hessian(x)
    op = HessianOperator(
        model=model,
        dataloader=[(x, y)],
        loss_fn=supervised_loss(nn.functional.mse_loss),
        full_dataset=True,
    )
    _check_matvec(op, H)


def test_linear_regression_ggn_matches_closed_form() -> None:
    """For a model linear in params with MSE, GGN = Hessian. Uses the
    analytical `mse_loss_of_output()` HVP path."""
    model, x, y = _linear_regression_setup()
    H = _analytic_linear_regression_hessian(x)
    op = GGNOperator(
        model=model,
        dataloader=[(x, y)],
        forward_fn=supervised_forward,
        loss_of_output_fn=mse_loss_of_output(),
    )
    _check_matvec(op, H)


# ---------------------------------------------------------------------------
# Binary logistic regression: H = (1/N) X^T diag(σ(1-σ)) X
# ---------------------------------------------------------------------------


def _binary_logistic_setup(n: int = 16, d: int = 5):
    model = _linear(d, 1, seed=2)
    g = _seed(3)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = (torch.rand(n, 1, generator=g, dtype=torch.float64) > 0.5).to(torch.float64)
    return model, x, y


def _analytic_binary_logistic_hessian(model: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    with torch.no_grad():
        logits = model(x).reshape(-1)
        p = torch.sigmoid(logits)
        d_diag = p * (1.0 - p)
    return (x.T * d_diag) @ x / n


def test_binary_logistic_hessian_matches_closed_form() -> None:
    model, x, y = _binary_logistic_setup()
    H = _analytic_binary_logistic_hessian(model, x)
    op = HessianOperator(
        model=model,
        dataloader=[(x, y)],
        loss_fn=supervised_loss(nn.functional.binary_cross_entropy_with_logits),
        full_dataset=True,
    )
    _check_matvec(op, H)


def test_binary_logistic_ggn_matches_closed_form() -> None:
    """For a linear-in-params model with BCE, GGN = Hessian. No analytical
    loss-HVP shipped for BCE, so this exercises the autograd fallback path."""
    model, x, y = _binary_logistic_setup()
    H = _analytic_binary_logistic_hessian(model, x)
    op = GGNOperator(
        model=model,
        dataloader=[(x, y)],
        forward_fn=supervised_forward,
        loss_of_output_fn=supervised_loss_of_output(
            nn.functional.binary_cross_entropy_with_logits
        ),
        loss_hvp="autograd",
    )
    _check_matvec(op, H)


# ---------------------------------------------------------------------------
# Multiclass softmax cross-entropy:
#   H = (1/N) Σ_k (diag(p_k) - p_k p_k^T) ⊗ (x_k x_k^T)
# Row-major flattening of W ∈ R^{C×d} makes the Kronecker order
# (class-block) ⊗ (feature-block) match PyTorch's parameter layout.
# ---------------------------------------------------------------------------


def _multiclass_setup(n: int = 12, d: int = 4, c: int = 3):
    model = _linear(d, c, seed=4)
    g = _seed(5)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randint(0, c, (n,), generator=g)
    return model, x, y


def _analytic_multiclass_ce_hessian(model: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    c = model.out_features
    with torch.no_grad():
        logits = model(x)
        p = torch.softmax(logits, dim=-1)  # (N, C)

    h = torch.zeros(c * d, c * d, dtype=torch.float64)
    for k in range(n):
        pk = p[k]
        xk = x[k]
        class_block = torch.diag(pk) - torch.outer(pk, pk)
        feat_block = torch.outer(xk, xk)
        h += torch.kron(class_block, feat_block)
    return h / n


def test_multiclass_softmax_ce_hessian_matches_closed_form() -> None:
    model, x, y = _multiclass_setup()
    H = _analytic_multiclass_ce_hessian(model, x)
    op = HessianOperator(
        model=model,
        dataloader=[(x, y)],
        loss_fn=supervised_loss(nn.functional.cross_entropy),
        full_dataset=True,
    )
    _check_matvec(op, H)


def test_multiclass_softmax_ce_ggn_matches_closed_form() -> None:
    """Linear model + softmax-CE: GGN = Hessian. Also exercises the closed-form
    `cross_entropy_loss_of_output` HVP path that bypasses double-backward."""
    model, x, y = _multiclass_setup()
    H = _analytic_multiclass_ce_hessian(model, x)
    op = GGNOperator(
        model=model,
        dataloader=[(x, y)],
        forward_fn=supervised_forward,
        loss_of_output_fn=cross_entropy_loss_of_output(),
    )
    _check_matvec(op, H)


# ---------------------------------------------------------------------------
# Sanity: GGN agrees with the Hessian on these convex linear-in-params cases.
# Catches regressions where one operator drifts but not the other.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "setup, loss_fn, loss_of_output_fn, loss_hvp",
    [
        (
            _linear_regression_setup,
            supervised_loss(nn.functional.mse_loss),
            mse_loss_of_output(),
            "analytical",
        ),
        (
            _binary_logistic_setup,
            supervised_loss(nn.functional.binary_cross_entropy_with_logits),
            supervised_loss_of_output(nn.functional.binary_cross_entropy_with_logits),
            "autograd",
        ),
        (
            _multiclass_setup,
            supervised_loss(nn.functional.cross_entropy),
            cross_entropy_loss_of_output(),
            "analytical",
        ),
    ],
    ids=["linear_mse", "binary_logistic", "multiclass_ce"],
)
def test_hessian_and_ggn_agree_on_linear_in_params_models(
    setup, loss_fn, loss_of_output_fn, loss_hvp
) -> None:
    model, x, y = setup()
    h_op = HessianOperator(
        model=model, dataloader=[(x, y)], loss_fn=loss_fn, full_dataset=True
    )
    g_op = GGNOperator(
        model=model,
        dataloader=[(x, y)],
        forward_fn=supervised_forward,
        loss_of_output_fn=loss_of_output_fn,
        loss_hvp=loss_hvp,
    )
    g = _seed(7)
    for _ in range(5):
        v = torch.randn(h_op.size, generator=g, dtype=torch.float64)
        torch.testing.assert_close(h_op.matvec(v), g_op.matvec(v), rtol=1e-8, atol=1e-10)

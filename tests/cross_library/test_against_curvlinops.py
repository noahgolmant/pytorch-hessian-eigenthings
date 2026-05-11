"""Verify our operators agree with curvlinops on the same problem.

curvlinops uses a different implementation strategy (BackPack-style hooks + manual
graph traversal in places). Two independent implementations agreeing is much
stronger evidence of correctness than either alone — this catches bugs that would
escape closed-form tests because they'd be present in both. See the validation
philosophy in the project plan.

All tests in this file are marked `@pytest.mark.curvlinops` and skip if curvlinops
is not installed.
"""

import numpy as np
import pytest
import torch
from torch import nn

from hessian_eigenthings.algorithms.lanczos import lanczos
from hessian_eigenthings.algorithms.trace import hutchinson
from hessian_eigenthings.operators import (
    EmpiricalFisherOperator,
    GGNOperator,
    HessianOperator,
)

pytestmark = pytest.mark.curvlinops

curvlinops = pytest.importorskip("curvlinops")
from curvlinops import (  # noqa: E402
    EFLinearOperator,
    GGNLinearOperator,
    HessianLinearOperator,
)


def _seed(value: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(value)
    return g


def _classification_setup() -> tuple[nn.Module, list, list[nn.Parameter]]:
    g = _seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 4)).to(torch.float64)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=torch.float64)
    g2 = _seed(1)
    x = torch.randn(8, 3, generator=g2, dtype=torch.float64)
    y = torch.randint(0, 4, (8,), generator=g2)
    data = [(x, y)]
    params = list(model.parameters())
    return model, data, params


def _regression_setup() -> tuple[nn.Module, list, list[nn.Parameter]]:
    g = _seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2)).to(torch.float64)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=torch.float64)
    g2 = _seed(2)
    x = torch.randn(8, 3, generator=g2, dtype=torch.float64)
    y = torch.randn(8, 2, generator=g2, dtype=torch.float64)
    data = [(x, y)]
    params = list(model.parameters())
    return model, data, params


def _ours_hessian_mse(model, batch_list) -> HessianOperator:
    return HessianOperator(
        model=model,
        dataloader=batch_list,
        loss_fn=lambda m, b: nn.functional.mse_loss(m(b[0]), b[1]),
    )


def _ours_hessian_ce(model, batch_list) -> HessianOperator:
    return HessianOperator(
        model=model,
        dataloader=batch_list,
        loss_fn=lambda m, b: nn.functional.cross_entropy(m(b[0]), b[1]),
    )


def _ours_ggn_ce(model, batch_list) -> GGNOperator:
    return GGNOperator(
        model=model,
        dataloader=batch_list,
        forward_fn=lambda m, b: m(b[0]),
        loss_of_output_fn=lambda out, b: nn.functional.cross_entropy(out, b[1]),
        loss_hvp="autograd",
    )


def _ours_ef_ce(model, batch_list) -> EmpiricalFisherOperator:
    return EmpiricalFisherOperator(
        model=model,
        dataloader=batch_list,
        per_sample_loss_fn=lambda m, s: nn.functional.cross_entropy(
            m(s[0].unsqueeze(0)), s[1].unsqueeze(0)
        ),
    )


def test_hessian_matches_curvlinops_mse() -> None:
    model, data, params = _regression_setup()
    ours = _ours_hessian_mse(model, data)
    theirs = HessianLinearOperator(
        model, nn.MSELoss(reduction="mean"), params, data, check_deterministic=False
    )
    g = _seed(10)
    for _ in range(5):
        v = torch.randn(ours.size, generator=g, dtype=torch.float64)
        torch.testing.assert_close(ours.matvec(v), theirs @ v, rtol=1e-6, atol=1e-9)


def test_hessian_matches_curvlinops_cross_entropy() -> None:
    model, data, params = _classification_setup()
    ours = _ours_hessian_ce(model, data)
    theirs = HessianLinearOperator(
        model,
        nn.CrossEntropyLoss(reduction="mean"),
        params,
        data,
        check_deterministic=False,
    )
    g = _seed(11)
    for _ in range(5):
        v = torch.randn(ours.size, generator=g, dtype=torch.float64)
        torch.testing.assert_close(ours.matvec(v), theirs @ v, rtol=1e-6, atol=1e-9)


def test_ggn_matches_curvlinops_cross_entropy() -> None:
    model, data, params = _classification_setup()
    ours = _ours_ggn_ce(model, data)
    theirs = GGNLinearOperator(
        model,
        nn.CrossEntropyLoss(reduction="mean"),
        params,
        data,
        check_deterministic=False,
    )
    g = _seed(12)
    for _ in range(5):
        v = torch.randn(ours.size, generator=g, dtype=torch.float64)
        torch.testing.assert_close(ours.matvec(v), theirs @ v, rtol=1e-6, atol=1e-9)


def test_empirical_fisher_matches_curvlinops_cross_entropy() -> None:
    model, data, params = _classification_setup()
    ours = _ours_ef_ce(model, data)
    theirs = EFLinearOperator(
        model,
        nn.CrossEntropyLoss(reduction="mean"),
        params,
        data,
        check_deterministic=False,
    )
    g = _seed(13)
    for _ in range(5):
        v = torch.randn(ours.size, generator=g, dtype=torch.float64)
        torch.testing.assert_close(ours.matvec(v), theirs @ v, rtol=1e-6, atol=1e-9)


def test_lanczos_top_k_matches_curvlinops_via_scipy() -> None:
    """Our Lanczos top-k vs scipy.sparse.linalg.eigsh on curvlinops's HessianLinearOperator."""
    from scipy.sparse.linalg import eigsh

    model, data, params = _classification_setup()
    ours_op = _ours_hessian_ce(model, data)
    theirs_op = HessianLinearOperator(
        model,
        nn.CrossEntropyLoss(reduction="mean"),
        params,
        data,
        check_deterministic=False,
    )
    scipy_op = theirs_op.to_scipy()

    k = 4
    expected_vals, _ = eigsh(scipy_op, k=k, which="LM", tol=1e-9)
    expected_vals = np.sort(np.abs(expected_vals))[::-1]

    result = lanczos(ours_op, k=k, max_iter=ours_op.size, tol=1e-9, which="LM", seed=0)
    ours_vals = np.sort(np.abs(result.eigenvalues.numpy()))[::-1]

    np.testing.assert_allclose(ours_vals, expected_vals, rtol=1e-5, atol=1e-7)


def test_hutchinson_trace_matches_curvlinops_explicit_trace() -> None:
    """Our Hutchinson estimator on curvlinops's operator should converge to its true trace."""
    model, data, params = _classification_setup()
    theirs_op = HessianLinearOperator(
        model,
        nn.CrossEntropyLoss(reduction="mean"),
        params,
        data,
        check_deterministic=False,
    )

    n = sum(p.numel() for p in params)
    h_full = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        h_full[:, i] = (theirs_op @ torch.from_numpy(e).to(torch.float64)).numpy()
    truth = float(np.trace(h_full))

    from hessian_eigenthings.operators.base import LambdaOperator

    op_for_us = LambdaOperator(
        lambda v: theirs_op @ v, size=n, device=torch.device("cpu"), dtype=torch.float64
    )
    result = hutchinson(op_for_us, num_samples=400, seed=0)
    assert abs(result.estimate - truth) < 3 * result.stderr + 1e-2 * abs(truth)

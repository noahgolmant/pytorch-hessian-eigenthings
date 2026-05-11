"""Loss-function helpers for the standard supervised setup `(input, target)` batches."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

# See `huggingface.py:_LossOfOutputWithHvp` — same pattern, reused here.
from hessian_eigenthings.loss_fns.huggingface import _LossOfOutputWithHvp


def supervised_loss(
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[nn.Module, Any], torch.Tensor]:
    """Make a `loss_fn(model, batch)` for `HessianOperator` from a (input, target) criterion."""

    def _fn(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        return criterion(model(x), y)

    return _fn


def supervised_forward(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """The `forward_fn` for `GGNOperator` on a (input, target) batch: returns model(input)."""
    x, _ = batch
    out: torch.Tensor = model(x)
    return out


def supervised_loss_of_output(
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Make a `loss_of_output_fn` for `GGNOperator` from a (output, target) criterion."""

    def _fn(output: torch.Tensor, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        _, y = batch
        return criterion(output, y)

    return _fn


def _ce_loss(output: torch.Tensor, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    _, y = batch
    return torch.nn.functional.cross_entropy(output, y)


def _ce_hvp(
    output: torch.Tensor, batch: tuple[torch.Tensor, torch.Tensor], u: torch.Tensor
) -> torch.Tensor:
    """Closed-form H @ u for mean-reduced softmax + cross-entropy.

    `output` has shape `(N, C)` (logits). For each row,
    `H_row = (diag(p) - p p^T) / N` where `p = softmax(output)`.
    """
    flat_output = output.reshape(-1, output.size(-1))
    flat_u = u.reshape(-1, u.size(-1))
    n = float(flat_output.size(0))
    p = torch.softmax(flat_output, dim=-1)
    dot = (p * flat_u).sum(dim=-1, keepdim=True)
    return ((p * flat_u - p * dot) / n).view_as(u)


def cross_entropy_loss_of_output() -> (
    Callable[[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]
):
    """`loss_of_output_fn` for supervised classification with mean-reduced cross-entropy.

    Carries a `.hvp(output, batch, u)` method holding the closed-form
    loss-Hessian-vector product, which `GGNOperator` picks up to bypass the
    autograd double-backward.
    """
    return _LossOfOutputWithHvp(_ce_loss, _ce_hvp)


def _mse_loss(output: torch.Tensor, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    _, y = batch
    return torch.nn.functional.mse_loss(output, y)


def _mse_hvp(
    output: torch.Tensor, batch: tuple[torch.Tensor, torch.Tensor], u: torch.Tensor
) -> torch.Tensor:
    """Closed-form H @ u for mean-reduced MSE.

    `mse_loss = mean((output - target)^2)`. The loss-Hessian w.r.t. output is
    constant `(2/N) * I` where `N = output.numel()`. So `H @ u = (2/N) * u`.
    """
    n = float(output.numel())
    return (2.0 / n) * u


def mse_loss_of_output() -> (
    Callable[[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]
):
    """`loss_of_output_fn` for supervised regression with mean-reduced MSE.

    Carries a `.hvp(output, batch, u)` method holding the closed-form
    loss-Hessian-vector product `(2/N) * u`.
    """
    return _LossOfOutputWithHvp(_mse_loss, _mse_hvp)


def supervised_per_sample_loss(
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[nn.Module, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Make a `per_sample_loss_fn` for `EmpiricalFisherOperator`. The criterion is called on
    a single un-batched sample after `vmap` strips the batch dimension."""

    def _fn(model: nn.Module, sample: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = sample
        return criterion(model(x.unsqueeze(0)).squeeze(0), y)

    return _fn

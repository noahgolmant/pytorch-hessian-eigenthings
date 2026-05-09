"""Loss-function helpers for the standard supervised setup `(input, target)` batches."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


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


def supervised_per_sample_loss(
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[nn.Module, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Make a `per_sample_loss_fn` for `EmpiricalFisherOperator`. The criterion is called on
    a single un-batched sample after `vmap` strips the batch dimension."""

    def _fn(model: nn.Module, sample: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = sample
        return criterion(model(x.unsqueeze(0)).squeeze(0), y)

    return _fn

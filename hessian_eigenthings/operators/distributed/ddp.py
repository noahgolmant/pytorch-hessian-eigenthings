"""DDP-aware Hessian operator: averages HVPs across data-parallel ranks.

DDP normally fires its all-reduce inside the autograd graph during `loss.backward()`,
so a regular HessianOperator using `torch.autograd.grad` does *not* trigger it (the
hooks fire on `.grad` accumulation, not on autograd.grad's return value). We add an
explicit autograd-aware all-reduce after each grad call so the resulting HVP equals
the single-process HVP computed on the union of all per-rank batches.
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch import nn

from hessian_eigenthings.linalg import LinAlgBackend
from hessian_eigenthings.operators.hessian import HessianOperator, HvpMethod, LossFn
from hessian_eigenthings.param_utils import ParamFilter


class DDPHessianOperator(HessianOperator):
    """HessianOperator that all-reduces the HVP across `torch.distributed` ranks.

    The model passed in may already be wrapped with
    `torch.nn.parallel.DistributedDataParallel`; we read params from it directly.
    Each rank should be receiving its own shard of the dataset (typical pattern: a
    `torch.utils.data.distributed.DistributedSampler`).
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        loss_fn: LossFn,
        *,
        param_filter: ParamFilter | None = None,
        full_dataset: bool = True,
        num_batches: int | None = None,
        method: HvpMethod = "autograd",
        fd_eps: float | None = None,
        backend: LinAlgBackend[torch.Tensor] | None = None,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            param_filter=param_filter,
            full_dataset=full_dataset,
            num_batches=num_batches,
            method=method,
            fd_eps=fd_eps,
            backend=backend,
        )
        self.process_group = process_group
        if dist.is_available() and dist.is_initialized():
            self._world_size = dist.get_world_size(group=process_group)
        else:
            self._world_size = 1

    def _hvp_autograd(self, v_split: list[torch.Tensor], batch: Any) -> torch.Tensor:
        loss = self.loss_fn(self.model, batch)
        grads = torch.autograd.grad(loss, self._param_list, create_graph=True)
        if self._world_size > 1:
            grads = tuple(self._all_reduce_mean(g) for g in grads)
        hvp = torch.autograd.grad(grads, self._param_list, grad_outputs=v_split)
        if self._world_size > 1:
            hvp = tuple(self._all_reduce_mean(h) for h in hvp)
        return torch.cat([h.reshape(-1) for h in hvp])

    def _hvp_finite_difference(self, v_split: list[torch.Tensor], batch: Any) -> torch.Tensor:
        # Each rank's _compute_grad_flat already returns its local gradient; we
        # all-reduce both g+ and g- before the difference.
        hvp_local = super()._hvp_finite_difference(v_split, batch)
        if self._world_size > 1:
            hvp_local = self._all_reduce_mean(hvp_local)
        return hvp_local

    def _all_reduce_mean(self, t: torch.Tensor) -> torch.Tensor:
        reduced: torch.Tensor = dist_nn.all_reduce(  # type: ignore[no-untyped-call]
            t, op=dist.ReduceOp.SUM, group=self.process_group
        )
        return reduced / self._world_size

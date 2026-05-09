"""HessianOperator: matrix-free Hessian-vector product over (a subset of) model parameters."""

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch
from torch import nn

from hessian_eigenthings.batching import (
    assert_microbatch_safe,
    iterate_batches,
    move_batch_to_device,
)
from hessian_eigenthings.linalg import LinAlgBackend, SingleDeviceBackend
from hessian_eigenthings.operators.base import CurvatureOperator
from hessian_eigenthings.param_utils import (
    ParamFilter,
    select_parameters,
    total_size,
)

LossFn = Callable[[nn.Module, Any], torch.Tensor]


class HessianOperator(CurvatureOperator):
    """Hessian of `loss_fn(model, batch)` averaged over batches in `dataloader`."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        loss_fn: LossFn,
        *,
        param_filter: ParamFilter | None = None,
        full_dataset: bool = True,
        num_batches: int | None = None,
        microbatch_size: int | None = None,
        microbatch_unsafe: bool = False,
        backend: LinAlgBackend[torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.full_dataset = full_dataset
        self.num_batches = num_batches
        self.microbatch_size = microbatch_size
        self.backend: LinAlgBackend[torch.Tensor] = backend or SingleDeviceBackend()

        if microbatch_size is not None and not microbatch_unsafe:
            assert_microbatch_safe(model)

        self._params = select_parameters(model, param_filter)
        self._param_list = list(self._params.values())
        self._sizes = [int(p.numel()) for p in self._param_list]
        self._size = total_size(self._params)

        first = self._param_list[0]
        self._device = first.device
        self._dtype = first.dtype

        self._batch_iter: Iterator[Any] | None = None

    @property
    def size(self) -> int:
        return self._size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        if v.shape != (self._size,):
            raise ValueError(f"expected vector of shape ({self._size},), got {tuple(v.shape)}")

        if self.full_dataset:
            return self._matvec_full(v)
        return self._matvec_one_batch(v, self._next_batch())

    def _matvec_full(self, v: torch.Tensor) -> torch.Tensor:
        total = self.backend.zeros_like(v)
        n = 0
        for batch in iterate_batches(self.dataloader, self.num_batches):
            chunk = self._matvec_one_batch(v, batch)
            total = self.backend.axpy(1.0, chunk, total)
            n += 1
        if n == 0:
            raise RuntimeError("dataloader yielded no batches")
        return self.backend.scale(1.0 / n, total)

    def _matvec_one_batch(self, v: torch.Tensor, batch: Any) -> torch.Tensor:
        batch = move_batch_to_device(batch, self._device)
        v_split = self._split(v)

        if self.microbatch_size is None:
            return self._hvp(v_split, batch)
        return self._hvp_microbatched(v_split, batch)

    def _hvp(self, v_split: list[torch.Tensor], batch: Any) -> torch.Tensor:
        loss = self.loss_fn(self.model, batch)
        grads = torch.autograd.grad(loss, self._param_list, create_graph=True)
        hvp = torch.autograd.grad(grads, self._param_list, grad_outputs=v_split)
        return torch.cat([h.reshape(-1) for h in hvp])

    def _hvp_microbatched(self, v_split: list[torch.Tensor], batch: Any) -> torch.Tensor:
        assert self.microbatch_size is not None
        chunks = _split_batch(batch, self.microbatch_size)
        if not chunks:
            raise RuntimeError("microbatching produced no chunks")
        total: torch.Tensor | None = None
        for chunk in chunks:
            hvp = self._hvp(v_split, chunk)
            total = hvp if total is None else total + hvp
        assert total is not None
        return total / len(chunks)

    def _split(self, v: torch.Tensor) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        offset = 0
        for n, p in zip(self._sizes, self._param_list, strict=True):
            out.append(v[offset : offset + n].reshape_as(p))
            offset += n
        return out

    def _next_batch(self) -> Any:
        if self._batch_iter is None:
            self._batch_iter = iter(self.dataloader)
        try:
            return next(self._batch_iter)
        except StopIteration:
            self._batch_iter = iter(self.dataloader)
            return next(self._batch_iter)


def _split_field(field: Any, microbatch_size: int) -> list[Any]:
    if isinstance(field, torch.Tensor):
        return [field[i : i + microbatch_size] for i in range(0, field.shape[0], microbatch_size)]
    return [field]


def _split_batch(batch: Any, microbatch_size: int) -> list[Any]:
    """Split a batch (tensor / tuple / list / dict of tensors) into microbatches along dim 0."""
    if isinstance(batch, torch.Tensor):
        return _split_field(batch, microbatch_size)
    if isinstance(batch, (list, tuple)):
        per_field_seq = [_split_field(f, microbatch_size) for f in batch]
        n_chunks = max(len(c) for c in per_field_seq)
        ctor = list if isinstance(batch, list) else tuple
        return [ctor(c[i] if len(c) > 1 else c[0] for c in per_field_seq) for i in range(n_chunks)]
    if isinstance(batch, dict):
        per_field_map = {k: _split_field(v, microbatch_size) for k, v in batch.items()}
        n_chunks = max(len(c) for c in per_field_map.values())
        return [
            {k: (c[i] if len(c) > 1 else c[0]) for k, c in per_field_map.items()}
            for i in range(n_chunks)
        ]
    raise TypeError(f"don't know how to microbatch {type(batch).__name__}")

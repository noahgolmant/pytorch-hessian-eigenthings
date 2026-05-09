"""Empirical Fisher operator: F = (1/N) Σ_i g_i g_i^T using per-sample gradients."""

from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import torch
from torch import nn

from hessian_eigenthings.batching import iterate_batches, move_batch_to_device
from hessian_eigenthings.linalg import LinAlgBackend, SingleDeviceBackend
from hessian_eigenthings.operators.base import CurvatureOperator
from hessian_eigenthings.param_utils import ParamFilter, select_parameters, total_size

# (model, single_sample) -> scalar loss for that one sample.
PerSampleLossFn = Callable[[nn.Module, Any], torch.Tensor]


class EmpiricalFisherOperator(CurvatureOperator):
    """Empirical Fisher `F = (1/N) Σ_i g_i g_i^T` where `g_i = ∂loss_i/∂θ` are per-sample grads.

    Empirical Fisher uses the *true* labels in the loss (unlike the MC Fisher which
    samples labels from the model's predictive distribution), and is therefore a
    biased estimator of the actual Fisher information. Conflating the two is the
    classic GGN-vs-Fisher-vs-empirical-Fisher pitfall — see Martens 2014.

    Per-sample gradients are computed in one pass via `torch.func.vmap(grad(...))`,
    so the cost is one forward+backward per batch, not per sample.

    The `per_sample_loss_fn(model, sample) -> Tensor` takes a single (un-batched)
    sample. The `sample_dim` argument tells the operator which axis to vmap over
    when receiving a batch from the dataloader.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        per_sample_loss_fn: PerSampleLossFn,
        *,
        param_filter: ParamFilter | None = None,
        full_dataset: bool = True,
        num_batches: int | None = None,
        sample_dim: int = 0,
        backend: LinAlgBackend[torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.per_sample_loss_fn = per_sample_loss_fn
        self.full_dataset = full_dataset
        self.num_batches = num_batches
        self.sample_dim = sample_dim
        self.backend: LinAlgBackend[torch.Tensor] = backend or SingleDeviceBackend()

        self._params = select_parameters(model, param_filter)
        self._param_names = list(self._params)
        self._param_list = list(self._params.values())
        self._sizes = [int(p.numel()) for p in self._param_list]
        self._size = total_size(self._params)
        self._fixed_params = {n: p for n, p in model.named_parameters() if n not in self._params}
        self._buffers = dict(model.named_buffers())

        first = self._param_list[0]
        self._device = first.device
        self._dtype = first.dtype

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
        return self._matvec_one_batch(v, next(iter(self.dataloader)))

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
        per_sample_grads = self._per_sample_grads(batch)
        # Stack into (n_samples, n_params).
        grads_mat = torch.cat(
            [
                per_sample_grads[n].reshape(per_sample_grads[n].shape[0], -1)
                for n in self._param_names
            ],
            dim=1,
        )
        n_samples = grads_mat.shape[0]
        # F v = (1/N) G^T (G v).
        return (grads_mat.t() @ (grads_mat @ v)) / n_samples

    def _per_sample_grads(self, batch: Any) -> Mapping[str, torch.Tensor]:
        param_dict: dict[str, torch.Tensor] = dict(
            zip(self._param_names, self._param_list, strict=True)
        )

        def loss_at(p_subset: dict[str, torch.Tensor], single: Any) -> torch.Tensor:
            full = {**self._fixed_params, **p_subset, **self._buffers}
            adapter = cast(nn.Module, _FunctionalModel(self.model, full))
            return self.per_sample_loss_fn(adapter, single)

        grad_fn = torch.func.grad(loss_at, argnums=0)
        in_dims = (None, _broadcast_dim(batch, self.sample_dim))
        result = torch.func.vmap(grad_fn, in_dims=in_dims)(param_dict, batch)
        return cast(Mapping[str, torch.Tensor], result)


class _FunctionalModel:
    """Adapter making `torch.func.functional_call` look like a regular nn.Module."""

    def __init__(self, model: nn.Module, params_and_buffers: dict[str, torch.Tensor]) -> None:
        self._model = model
        self._params_and_buffers = params_and_buffers

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return torch.func.functional_call(self._model, self._params_and_buffers, args, kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def _broadcast_dim(batch: Any, sample_dim: int) -> Any:
    """Build the in_dims spec for vmap matching the structure of `batch`."""
    if isinstance(batch, torch.Tensor):
        return sample_dim
    if isinstance(batch, (list, tuple)):
        return type(batch)(_broadcast_dim(field, sample_dim) for field in batch)
    if isinstance(batch, dict):
        return {k: _broadcast_dim(v, sample_dim) for k, v in batch.items()}
    return None

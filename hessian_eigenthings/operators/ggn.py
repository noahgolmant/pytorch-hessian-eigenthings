"""Generalized Gauss-Newton operator: G = J^T H_loss J."""

from collections.abc import Callable, Iterable
from typing import Any, cast

import torch
from torch import nn

from hessian_eigenthings.batching import iterate_batches, move_batch_to_device
from hessian_eigenthings.linalg import LinAlgBackend, SingleDeviceBackend
from hessian_eigenthings.operators.base import CurvatureOperator
from hessian_eigenthings.param_utils import ParamFilter, select_parameters, total_size

# (model, batch) -> model_output. The user calls model(...) here as they normally would.
ForwardFn = Callable[[nn.Module, Any], torch.Tensor]
# (model_output, batch) -> scalar loss. Called separately from the forward so the
# loss-Hessian-vec product can be computed cheaply on the loss-only graph.
LossOfOutputFn = Callable[[torch.Tensor, Any], torch.Tensor]


class GGNOperator(CurvatureOperator):
    """Generalized Gauss-Newton matrix `G = J^T H_loss J`.

    For convex per-sample losses (cross-entropy + softmax, MSE) `H_loss` is PSD so
    `G` is PSD by construction. For cross-entropy + softmax classification, `G`
    equals the Fisher information matrix.

    The two-function API (`forward_fn` returns the model output, `loss_of_output_fn`
    converts that output + batch into a scalar loss) lets us compute `J v` (JVP
    through the model alone), `H_loss · (Jv)` (a small autograd HVP on the loss-only
    graph), and `J^T · (H_loss · Jv)` (VJP back through the model) without either
    redoing the forward or coupling to the loss internals.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        forward_fn: ForwardFn,
        loss_of_output_fn: LossOfOutputFn,
        *,
        param_filter: ParamFilter | None = None,
        full_dataset: bool = True,
        num_batches: int | None = None,
        backend: LinAlgBackend[torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.forward_fn = forward_fn
        self.loss_of_output_fn = loss_of_output_fn
        self.full_dataset = full_dataset
        self.num_batches = num_batches
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
        v_dict = self._unflatten(v)
        param_dict = dict(zip(self._param_names, self._param_list, strict=True))

        def model_call(p_subset: dict[str, torch.Tensor]) -> torch.Tensor:
            full = {**self._fixed_params, **p_subset, **self._buffers}
            adapter = cast(nn.Module, _FunctionalModel(self.model, full))
            return self.forward_fn(adapter, batch)

        jvp_result = cast(
            tuple[torch.Tensor, torch.Tensor],
            torch.func.jvp(model_call, (param_dict,), (v_dict,)),
        )
        output, jvp_out = jvp_result

        output_leaf = output.detach().requires_grad_(True)
        loss = self.loss_of_output_fn(output_leaf, batch)
        grad_loss = torch.autograd.grad(loss, output_leaf, create_graph=True)[0]
        h_loss_jvp = torch.autograd.grad(grad_loss, output_leaf, grad_outputs=jvp_out)[0]

        vjp_result = cast(
            tuple[torch.Tensor, Callable[[torch.Tensor], tuple[dict[str, torch.Tensor]]]],
            torch.func.vjp(model_call, param_dict),
        )
        _, vjp_fn = vjp_result
        gv_dict = vjp_fn(h_loss_jvp)[0]

        return torch.cat([gv_dict[n].reshape(-1) for n in self._param_names])

    def _unflatten(self, v: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        offset = 0
        for n, p, sz in zip(self._param_names, self._param_list, self._sizes, strict=True):
            out[n] = v[offset : offset + sz].reshape_as(p)
            offset += sz
        return out


class _FunctionalModel:
    """Adapter making `torch.func.functional_call` look like a regular nn.Module to the user's forward_fn."""

    def __init__(self, model: nn.Module, params_and_buffers: dict[str, torch.Tensor]) -> None:
        self._model = model
        self._params_and_buffers = params_and_buffers

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return torch.func.functional_call(self._model, self._params_and_buffers, args, kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

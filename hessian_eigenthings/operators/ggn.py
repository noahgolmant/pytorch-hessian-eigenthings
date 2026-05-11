"""Generalized Gauss-Newton operator: G = J^T H_loss J."""

from collections.abc import Callable, Iterable
from typing import Any, Literal, cast

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
#
# Optionally the callable may also expose `.hvp(output, batch, u) -> H_loss @ u`
# returning the closed-form loss-Hessian-vector product (same shape as `output`).
# When present, `GGNOperator` uses it instead of an autograd double-backward —
# see e.g. `cross_entropy_loss_of_output()` and `mse_loss_of_output()` in
# `hessian_eigenthings.loss_fns.standard`.
LossOfOutputFn = Callable[[torch.Tensor, Any], torch.Tensor]

LossHvpMethod = Literal["analytical", "autograd"]

# Roughly machine_eps^(1/3) per dtype, matching `HessianOperator._FD_EPS_BY_DTYPE`.
# Balances O(eps^2) truncation against O(eps_machine / eps) roundoff for central
# differences of `forward(theta + eps*v) - forward(theta - eps*v)`.
_FD_EPS_BY_DTYPE = {
    torch.float64: 6e-6,
    torch.float32: 5e-3,
    torch.bfloat16: 0.2,
    torch.float16: 5e-2,
}

# Floor for `||v||` when computing the finite-difference perturbation.
# When v is tiny we normalise it internally and rescale the result; otherwise
# `eps * v` can underflow and `(out_plus - out_minus)` is dominated by roundoff.
_V_NORM_FLOOR = 1e-8


class GGNOperator(CurvatureOperator):
    """Generalized Gauss-Newton matrix `G = J^T H_loss J`.

    For convex per-sample losses (cross-entropy + softmax, MSE) `H_loss` is PSD so
    `G` is PSD by construction. For cross-entropy + softmax classification, `G`
    equals the Fisher information matrix.

    The two-function API (`forward_fn` returns the model output, `loss_of_output_fn`
    converts that output + batch into a scalar loss) lets us compute `J v`, the
    loss-Hessian-vector product `H_loss · (Jv)`, and `J^T · (H_loss · Jv)` without
    coupling to the loss internals.

    Two implementations of the matvec are available via `loss_hvp=`:

    * ``"analytical"`` (default): finite-difference JVP + analytical loss-Hessian-vec
      product (read from `loss_of_output_fn.hvp`, which must be present) + a single
      normal backward to apply `J^T`. Memory footprint matches one normal training
      step. Required for LM-scale use; see the OOM diagnostic in
      `scripts/repro_ggn_oom.py`.

    * ``"autograd"``: the original `torch.func.jvp` + autograd double-backward +
      `torch.func.vjp` path. Numerically exact and supports any loss, but memory
      scales badly with output size — for cross-entropy heads with large vocab the
      `create_graph=True` step alone can dominate. Kept as a fallback for losses
      without an analytical `.hvp`.
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
        loss_hvp: LossHvpMethod = "analytical",
        fd_eps: float | None = None,
        backend: LinAlgBackend[torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.forward_fn = forward_fn
        self.loss_of_output_fn = loss_of_output_fn
        self.full_dataset = full_dataset
        self.num_batches = num_batches
        self.loss_hvp: LossHvpMethod = loss_hvp
        self.backend: LinAlgBackend[torch.Tensor] = backend or SingleDeviceBackend()

        if loss_hvp not in ("analytical", "autograd"):
            raise ValueError(f"loss_hvp={loss_hvp!r} not in ('analytical', 'autograd')")
        if loss_hvp == "analytical" and not hasattr(loss_of_output_fn, "hvp"):
            raise ValueError(
                "loss_hvp='analytical' requires `loss_of_output_fn.hvp(output, "
                "batch, u)` to be defined (use `cross_entropy_loss_of_output()` "
                "or `mse_loss_of_output()` from `hessian_eigenthings.loss_fns`, "
                "or wrap your callable with `_LossOfOutputWithHvp`). Pass "
                "loss_hvp='autograd' to fall back to the double-backward path."
            )

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

        self.fd_eps = fd_eps if fd_eps is not None else _FD_EPS_BY_DTYPE.get(self._dtype, 1e-3)

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
        if self.loss_hvp == "analytical":
            return self._matvec_fd_jvp(v, batch)
        return self._matvec_autograd(v, batch)

    # --- analytical (FD JVP + analytical H_loss + single backward) ----------

    def _matvec_fd_jvp(self, v: torch.Tensor, batch: Any) -> torch.Tensor:
        """`fd_jvp_single_vjp`: 2 no-grad forwards (FD JVP) + analytical loss-HVP +
        1 grad-enabled forward+backward to apply `J^T`. Memory peaks at one
        normal training step.
        """
        v_split = self._split(v)

        # Normalise v internally so eps * ||v|| can't underflow on tiny v.
        # We compute matvec(v / s) and then multiply by s — `G` is linear in v.
        v_norm = float(torch.linalg.vector_norm(v).item())
        scale = max(v_norm, _V_NORM_FLOOR)
        if scale != 1.0:
            v_split = [vs / scale for vs in v_split]

        eps = self.fd_eps
        snapshot = [p.detach().clone() for p in self._param_list]
        try:
            with torch.no_grad():
                self._add_inplace(v_split, +eps)
                out_plus = self.forward_fn(self.model, batch).detach().clone()
                self._add_inplace(v_split, -2.0 * eps)
                out_minus = self.forward_fn(self.model, batch).detach().clone()
        finally:
            with torch.no_grad():
                for p, snap in zip(self._param_list, snapshot, strict=True):
                    p.copy_(snap)
        del snapshot

        jvp_out = (out_plus - out_minus) / (2.0 * eps)
        del out_plus, out_minus

        # Single grad-enabled forward; we'll reuse `logits` both for the
        # analytical loss-HVP and as the source for `J^T h_loss_jvp`.
        logits = self.forward_fn(self.model, batch)
        # `.hvp` is guaranteed to exist for loss_hvp=="analytical" — checked in __init__.
        h_loss_jvp = self.loss_of_output_fn.hvp(logits.detach(), batch, jvp_out)  # type: ignore[attr-defined]
        del jvp_out

        grads = torch.autograd.grad(logits, self._param_list, grad_outputs=h_loss_jvp)
        result = torch.cat([g.reshape(-1) for g in grads])
        if scale != 1.0:
            result = result * scale
        return result

    def _add_inplace(self, v_split: list[torch.Tensor], alpha: float) -> None:
        for p, dv in zip(self._param_list, v_split, strict=True):
            p.add_(dv, alpha=alpha)

    def _split(self, v: torch.Tensor) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        offset = 0
        for n, p in zip(self._sizes, self._param_list, strict=True):
            out.append(v[offset : offset + n].reshape_as(p))
            offset += n
        return out

    # --- autograd fallback (original implementation) ------------------------

    def _matvec_autograd(self, v: torch.Tensor, batch: Any) -> torch.Tensor:
        v_dict = self._unflatten(v)
        param_dict = dict(zip(self._param_names, self._param_list, strict=True))

        def model_call(p_subset: dict[str, torch.Tensor]) -> torch.Tensor:
            full = {**self._fixed_params, **p_subset, **self._buffers}
            adapter = _FunctionalModel(self.model, full)
            return self.forward_fn(adapter, batch)  # type: ignore[arg-type]

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
    """Adapter making `torch.func.functional_call` look like a regular nn.Module to the user's forward_fn.

    Only used by the `loss_hvp="autograd"` fallback path; the default analytical
    path mutates params in-place under `no_grad` instead (mirroring
    `HessianOperator._hvp_finite_difference`).
    """

    def __init__(self, model: nn.Module, params_and_buffers: dict[str, torch.Tensor]) -> None:
        self._model = model
        self._params_and_buffers = params_and_buffers

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return torch.func.functional_call(self._model, self._params_and_buffers, args, kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

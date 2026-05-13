"""HuggingFace Transformers loss-function helpers.

HF models (subclasses of `transformers.PreTrainedModel`) compute their loss
internally when `labels` are present in the batch dict, returning a `ModelOutput`
with `.loss` and `.logits`. These wrappers shape that into the callable signatures
the operators expect.
"""

from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import nn

from hessian_eigenthings.loss_fns._fused_ce_hvp import (
    _triton_available,
    compiled_ce_hvp,
    triton_ce_hvp,
)

FusedCEHvpBackend = Literal["auto", "eager", "compile", "triton"]


def _resolve_backend(
    backend: FusedCEHvpBackend, logits: torch.Tensor | None = None
) -> FusedCEHvpBackend:
    """Resolve ``"auto"`` to the fastest available concrete backend.

    Preference order: triton (if CUDA + triton import succeed *and* the input
    tensor is on CUDA) → compile → eager. ``"compile"`` is essentially always
    available on torch >= 2.0 and is a strict improvement over eager;
    ``"triton"`` adds another ~30% over compile by streaming the online softmax
    instead of materializing it.

    Tensor device matters: the Triton kernel asserts ``logits.is_cuda``, so
    on a CUDA-equipped host running CPU inputs (mixed setups, CI smoke tests,
    the GGN matvec with CPU params on a GPU box) we must fall back to compile.
    """
    if backend != "auto":
        return backend
    # On a CUDA-capable host, Triton is preferred — but only if the actual
    # inputs are on CUDA. Falling through to compile on CPU inputs keeps
    # ``fused="auto"`` safe in mixed-device setups.
    if _triton_available() and (logits is None or logits.is_cuda):
        return "triton"
    return "compile"


# Optional `.hvp` member on `loss_of_output_fn` callables: closed-form
# (output, batch, u) -> H_loss @ u, where `u` has the same shape as `output`.
# `GGNOperator` looks for this attribute and skips the autograd double-backward
# when it's present. See `_LossOfOutputWithHvp`.
#
# `batch` is intentionally typed as `Any` here: this protocol is shared across
# HF-style dict batches, tuple-of-tensor batches (supervised), and whatever
# else user code provides. Strict typing per-callsite is the user's choice.
LossHvpFn = Callable[[torch.Tensor, Any, torch.Tensor], torch.Tensor]


class _LossOfOutputWithHvp:
    """Loss-of-output callable that also carries an analytical `.hvp` method.

    Wraps a plain `(output, batch) -> loss` function and a `(output, batch, u)
    -> H_loss @ u` function in a single callable. `GGNOperator` checks for the
    presence of `.hvp` and uses it as the loss-Hessian-vector product, skipping
    the autograd `create_graph=True` double-backward path entirely.
    """

    def __init__(
        self,
        loss_fn: Callable[[torch.Tensor, Any], torch.Tensor],
        hvp_fn: LossHvpFn,
    ) -> None:
        self._loss_fn = loss_fn
        self.hvp = hvp_fn

    def __call__(self, output: torch.Tensor, batch: Any) -> torch.Tensor:
        return self._loss_fn(output, batch)


def hf_lm_loss() -> Callable[[nn.Module, dict[str, Any]], torch.Tensor]:
    """For autoregressive LMs: `loss_fn(model, batch)` calls `model(**batch).loss`.

    The batch must include `labels` so HF computes the loss internally; for causal LMs
    that's typically `labels=input_ids` (with the standard internal shift).
    """

    def _fn(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
        out = model(**batch)
        return out.loss  # type: ignore[no-any-return]

    return _fn


def hf_seq2seq_loss() -> Callable[[nn.Module, dict[str, Any]], torch.Tensor]:
    """For seq2seq models (e.g. T5/BART) that compute the decoder cross-entropy internally."""
    return hf_lm_loss()


def hf_lm_forward() -> Callable[[nn.Module, dict[str, Any]], torch.Tensor]:
    """`forward_fn` for `GGNOperator` on an HF causal LM: returns `logits` (no loss)."""

    def _fn(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
        batch_no_labels = {k: v for k, v in batch.items() if k != "labels"}
        out = model(**batch_no_labels)
        return out.logits  # type: ignore[no-any-return]

    return _fn


def _hf_lm_shifted_ce(logits: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    labels = batch["labels"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _hf_lm_ce_hvp_eager(
    flat_logits: torch.Tensor, flat_u: torch.Tensor, valid: torch.Tensor, n: torch.Tensor
) -> torch.Tensor:
    """Eager `(p*u - p*<p,u>) * mask / n` implementation. Materializes ~5 (N, V)
    intermediates -- this is the OOM path at LM scale."""
    p = torch.softmax(flat_logits, dim=-1)
    dot = (p * flat_u).sum(dim=-1, keepdim=True)
    return (p * flat_u - p * dot) * valid.unsqueeze(-1) / n


def _hf_lm_ce_hvp(
    logits: torch.Tensor,
    batch: dict[str, Any],
    u: torch.Tensor,
    *,
    backend: FusedCEHvpBackend = "auto",
) -> torch.Tensor:
    """Closed-form H_loss @ u for the shifted-CE loss above.

    For mean-reduced cross-entropy with softmax: H_loss is block-diagonal in
    `(B, T-1)` with each block (over the vocab axis) being
    `(diag(p_t) - p_t p_t^T) / n` where `p_t = softmax(shift_logits_t)` and
    `n` is the count of non-ignored positions (matches the mean reduction
    `cross_entropy` performs over un-ignored positions).

    `u` has the *unshifted* logits shape `(B, T, V)`. We zero out the last
    time slice (which has no corresponding label after the shift) and the
    rows for ignored labels, mirroring what `cross_entropy(ignore_index=-100)`
    does in its gradient.

    ``backend`` controls how the core (N, V) computation is fused:
      - ``"auto"`` (default): triton if CUDA + triton available, else compile.
      - ``"eager"``: plain PyTorch, easy to debug, OOM-prone at V>=50k.
      - ``"compile"``: ``torch.compile``-fused; Inductor folds the softmax +
        elementwise + reduction into a small number of kernels and eliminates
        most of the (N, V) intermediates. Works on CPU/CUDA/MPS.
      - ``"triton"``: hand-written CUDA Triton kernel. Zero (N, V) intermediates;
        output buffer only. Falls back to ``"compile"`` if Triton/CUDA is
        unavailable.
    """
    backend = _resolve_backend(backend, logits)
    labels = batch["labels"]
    # Match the shift used in the loss.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_u = u[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_u = shift_u.view(-1, shift_u.size(-1))
    flat_labels = shift_labels.view(-1)

    # mask: 1.0 for valid positions, 0.0 for ignored ones.
    valid = (flat_labels != -100).to(flat_logits.dtype)
    n = valid.sum().clamp_min(1.0)

    if backend == "triton" and _triton_available():
        hvp_flat = triton_ce_hvp(flat_logits, flat_u, valid, n)
    elif backend == "compile" or (backend == "triton" and not _triton_available()):
        hvp_flat = compiled_ce_hvp(flat_logits, flat_u, valid, n)
    else:
        hvp_flat = _hf_lm_ce_hvp_eager(flat_logits, flat_u, valid, n)

    # Re-embed into the original (B, T, V) shape with zeros for the last
    # time slice (no label after shift).
    out = torch.zeros_like(u)
    out[..., :-1, :] = hvp_flat.view_as(shift_u)
    return out


def hf_lm_loss_of_output(
    *, fused: FusedCEHvpBackend = "auto"
) -> Callable[[torch.Tensor, dict[str, Any]], torch.Tensor]:
    """`loss_of_output_fn` for `GGNOperator` on an HF causal LM: standard shifted CE on `logits` and `labels`.

    The returned callable carries a `.hvp(output, batch, u)` method holding the
    closed-form loss-Hessian-vector product for mean-reduced cross-entropy with
    softmax: `H @ u = (p * u - p * (p · u)) / n` per non-ignored position,
    where `p = softmax(logits)` and `n` is the count of non-ignored positions.
    `GGNOperator` picks this up automatically and skips the autograd
    double-backward.

    ``fused`` selects the kernel for the core CE HVP. Default is ``"auto"``,
    which picks the fastest available backend at first call: Triton on CUDA
    (~3.4x speedup, 2x peak-memory reduction over eager), else ``torch.compile``
    (~2.6x speedup, 2x peak-memory reduction). Pass ``"eager"`` explicitly to
    force the unfused reference implementation, useful for debugging or when
    a graph break would otherwise trigger.
    """

    def _hvp(logits: torch.Tensor, batch: dict[str, Any], u: torch.Tensor) -> torch.Tensor:
        return _hf_lm_ce_hvp(logits, batch, u, backend=fused)

    return _LossOfOutputWithHvp(_hf_lm_shifted_ce, _hvp)

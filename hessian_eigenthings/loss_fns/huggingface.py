"""HuggingFace Transformers loss-function helpers.

HF models (subclasses of `transformers.PreTrainedModel`) compute their loss
internally when `labels` are present in the batch dict, returning a `ModelOutput`
with `.loss` and `.logits`. These wrappers shape that into the callable signatures
the operators expect.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


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


def hf_lm_loss_of_output() -> Callable[[torch.Tensor, dict[str, Any]], torch.Tensor]:
    """`loss_of_output_fn` for `GGNOperator` on an HF causal LM: standard shifted CE on `logits` and `labels`."""

    def _fn(logits: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        labels = batch["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    return _fn

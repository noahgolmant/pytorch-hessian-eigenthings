"""TransformerLens loss-function helpers.

TLens `HookedTransformer` accepts `return_type='loss'` to compute the standard
shifted-cross-entropy LM loss internally. Batches may be either a tensor of token
ids or a dict containing a 'tokens' key.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


def tlens_loss() -> Callable[[nn.Module, Any], torch.Tensor]:
    """For TLens HookedTransformer: `loss_fn(model, tokens) = model(tokens, return_type='loss')`."""

    def _fn(model: nn.Module, batch: Any) -> torch.Tensor:
        tokens = batch["tokens"] if isinstance(batch, dict) else batch
        return model(tokens, return_type="loss")  # type: ignore[no-any-return]

    return _fn


def tlens_forward() -> Callable[[nn.Module, Any], torch.Tensor]:
    """`forward_fn` for `GGNOperator`: returns the model's logits."""

    def _fn(model: nn.Module, batch: Any) -> torch.Tensor:
        tokens = batch["tokens"] if isinstance(batch, dict) else batch
        return model(tokens, return_type="logits")  # type: ignore[no-any-return]

    return _fn


def tlens_loss_of_output() -> Callable[[torch.Tensor, Any], torch.Tensor]:
    """`loss_of_output_fn` for `GGNOperator`: shifted CE on the TLens logits/tokens."""

    def _fn(logits: torch.Tensor, batch: Any) -> torch.Tensor:
        tokens = batch["tokens"] if isinstance(batch, dict) else batch
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_tokens.view(-1),
        )

    return _fn

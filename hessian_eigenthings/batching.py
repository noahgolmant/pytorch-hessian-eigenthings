"""Dataloader iteration helpers and the microbatching safety guard.

Microbatching (splitting a dataloader batch into smaller chunks for the HVP) is
*not* used in the default path. It silently produces the Hessian of a different
loss when BatchNorm is present (per-chunk statistics ≠ per-batch statistics) and
also breaks for cross-sample losses (contrastive, in-batch negatives, etc.).
Operators expose `microbatch_size` as opt-in with the BN guard below; users who
want lower memory should prefer activation checkpointing or shrinking the
dataloader's batch size.
"""

from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch import nn

_BN_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


def assert_microbatch_safe(model: nn.Module) -> None:
    """Raise if the model contains BatchNorm layers that microbatching would corrupt."""
    bn = [name for name, m in model.named_modules() if isinstance(m, _BN_TYPES)]
    if not bn:
        return
    head = ", ".join(bn[:5])
    extra = ", ..." if len(bn) > 5 else ""
    raise ValueError(
        f"microbatch_size is set, but the model contains BatchNorm layers "
        f"({head}{extra}). Microbatching changes per-chunk statistics and silently "
        f"produces the Hessian of a different loss. Either remove BN, switch to "
        f"eval mode and pass microbatch_unsafe=True, or compute over the full batch."
    )


def iterate_batches(source: Iterable[Any], num_batches: int | None = None) -> Iterator[Any]:
    """Yield batches from `source`, optionally capping at `num_batches`."""
    if num_batches is None:
        yield from source
        return
    for i, batch in enumerate(source):
        if i >= num_batches:
            return
        yield batch


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors in a batch to `device`. Leaves non-tensors as-is."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [move_batch_to_device(v, device) for v in batch]
        return type(batch)(moved)
    return batch

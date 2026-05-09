"""Parameter selection (glob/regex) and flat-vector ↔ per-param dict conversion."""

import fnmatch
import re
from collections.abc import Callable, Iterable, Mapping

import torch
from torch import nn

ParamFilter = Callable[[str, nn.Parameter], bool]


def select_parameters(
    model: nn.Module,
    param_filter: ParamFilter | None = None,
) -> dict[str, nn.Parameter]:
    """Return parameters matching `param_filter`, in `named_parameters` order."""
    out: dict[str, nn.Parameter] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if param_filter is not None and not param_filter(name, p):
            continue
        out[name] = p
    if not out:
        raise ValueError("param_filter selected zero parameters")
    return out


def match_names(*patterns: str) -> ParamFilter:
    """Glob-style name match. Matches if *any* pattern matches the parameter name."""

    def _filter(name: str, _: nn.Parameter) -> bool:
        return any(fnmatch.fnmatchcase(name, pat) for pat in patterns)

    return _filter


def match_regex(*patterns: str) -> ParamFilter:
    """Regex name match. Matches if *any* compiled pattern matches the parameter name."""
    compiled = [re.compile(pat) for pat in patterns]

    def _filter(name: str, _: nn.Parameter) -> bool:
        return any(p.search(name) is not None for p in compiled)

    return _filter


def total_size(params: Mapping[str, torch.Tensor] | Iterable[torch.Tensor]) -> int:
    """Total element count across the parameter collection."""
    items = params.values() if isinstance(params, Mapping) else params
    return sum(int(p.numel()) for p in items)


def params_to_vector(params: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate per-param tensors into a single flat vector. Order follows iteration order."""
    return torch.cat([p.reshape(-1) for p in params.values()])


def vector_to_params(
    vec: torch.Tensor, reference: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Split a flat vector into a dict of param-shaped tensors matching `reference`."""
    if vec.dim() != 1:
        raise ValueError(f"expected 1-D vector, got shape {tuple(vec.shape)}")
    expected = total_size(reference)
    if vec.numel() != expected:
        raise ValueError(f"vector has {vec.numel()} elements, reference expects {expected}")
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for name, ref in reference.items():
        n = ref.numel()
        out[name] = vec[offset : offset + n].reshape_as(ref)
        offset += n
    return out

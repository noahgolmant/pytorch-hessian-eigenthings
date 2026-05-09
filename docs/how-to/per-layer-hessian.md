# Per-layer Hessian

Restrict the curvature operator to a subset of parameters using `param_filter`. Operator size shrinks to match, and matvec only differentiates through the selected parameters.

## Filtering by name

`match_names` accepts glob patterns:

```python
from hessian_eigenthings.param_utils import match_names
from hessian_eigenthings.operators import HessianOperator

# All attention parameters across all transformer blocks
attn_op = HessianOperator(
    model=model,
    dataloader=loader,
    loss_fn=loss_fn,
    param_filter=match_names("blocks.*.attn.*"),
)

# Just block 3's MLP weights
block3_mlp_w = HessianOperator(
    model=model,
    dataloader=loader,
    loss_fn=loss_fn,
    param_filter=match_names("blocks.3.mlp.W_*"),
)

# Multiple patterns: union (any match)
embed_or_unembed = HessianOperator(
    model=model,
    dataloader=loader,
    loss_fn=loss_fn,
    param_filter=match_names("embed.*", "unembed.*"),
)
```

## Filtering by regex

For more complex selection, use `match_regex`:

```python
from hessian_eigenthings.param_utils import match_regex

# All Q/K/V matrices in any block
qkv_op = HessianOperator(
    model=model,
    dataloader=loader,
    loss_fn=loss_fn,
    param_filter=match_regex(r"blocks\.\d+\.attn\.W_[QKV]"),
)
```

## Custom filter

`param_filter` is just a `Callable[[str, nn.Parameter], bool]`. You can write any predicate:

```python
def big_layers_only(name: str, param) -> bool:
    return param.numel() > 1_000_000

op = HessianOperator(
    model=model, dataloader=loader, loss_fn=loss_fn,
    param_filter=big_layers_only,
)
```

## What gets fixed

Parameters not selected by `param_filter` are treated as **constants** for the purpose of the operator — the Jacobian-vector product flows through them but their derivatives are not part of the matvec output. Buffers (BN running stats, etc.) are similarly fixed.

If you also want to *freeze* the gradients of unselected params at training time, set `param.requires_grad_(False)` on the model — the operator skips frozen params automatically.

## Why this is useful

- Per-block sharpness studies (attention vs MLP). Liu et al. 2025.
- Curvature of just the LM head, or just the embedding matrix.
- Splitting a giant Hessian into smaller diagonal blocks for an approximate analysis.

## Limitations

- The filter must select at least one parameter — empty selection raises `ValueError`.
- Cross-parameter cross-derivatives between selected and unselected params are not exposed. The operator gives the Hessian *block* corresponding to the selected params, not the full Hessian projected onto a subspace.

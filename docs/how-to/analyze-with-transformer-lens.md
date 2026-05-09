# Analyze with TransformerLens

Compute curvature on a [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) `HookedTransformer`. Useful for mechanistic-interpretability work where you want to compute Hessian-related quantities while using TLens hooks.

## Install

```bash
uv add "hessian-eigenthings[transformer-lens]"
# or
pip install "hessian-eigenthings[transformer-lens]"
```

## Pattern

```python
import torch
from transformer_lens import HookedTransformer

from hessian_eigenthings.algorithms import lanczos
from hessian_eigenthings.loss_fns import tlens_loss
from hessian_eigenthings.operators import HessianOperator

model = HookedTransformer.from_pretrained("solu-1l")
model.eval()

tokens = torch.randint(0, model.cfg.d_vocab, (2, 32))
dataloader = [tokens]

op = HessianOperator(model=model, dataloader=dataloader, loss_fn=tlens_loss())
result = lanczos(op, k=3, max_iter=20, tol=1e-3, seed=0)
print(result.eigenvalues)
```

`tlens_loss()` calls `model(batch, return_type="loss")` under the hood, the standard TLens shifted-cross-entropy LM loss.

## Per-block analysis

TLens parameter names follow the `blocks.{i}.attn.{...}` and `blocks.{i}.mlp.{...}` pattern, which makes per-block filtering straightforward:

```python
from hessian_eigenthings.param_utils import match_names

attn_op = HessianOperator(
    model=model,
    dataloader=dataloader,
    loss_fn=tlens_loss(),
    param_filter=match_names("blocks.*.attn.*"),
)

mlp_op = HessianOperator(
    model=model,
    dataloader=dataloader,
    loss_fn=tlens_loss(),
    param_filter=match_names("blocks.*.mlp.*"),
)
```

Per-block sharpness disparities — the attention vs MLP eigenvalue gap — are the setup Liu et al. (2025) studied.

## Full runnable example

See [`examples/transformer_lens_attention_only.py`](https://github.com/noahgolmant/pytorch-hessian-eigenthings/blob/v1/examples/transformer_lens_attention_only.py).

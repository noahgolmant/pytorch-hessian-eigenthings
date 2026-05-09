# Custom loss functions

Each operator takes a loss-function callable. The exact signature differs slightly by operator because GGN and empirical Fisher need to know more than just "the loss".

## HessianOperator: `loss_fn(model, batch) -> Tensor`

The simplest case. Whatever closure you'd use to compute the loss in your training loop.

```python
from hessian_eigenthings.operators import HessianOperator

def loss_fn(model, batch):
    inputs, targets = batch
    logits = model(inputs)
    return torch.nn.functional.cross_entropy(logits, targets)

op = HessianOperator(model=model, dataloader=loader, loss_fn=loss_fn)
```

For common patterns we ship helpers:

```python
from hessian_eigenthings.loss_fns import supervised_loss, hf_lm_loss, tlens_loss

# Equivalent to the loss_fn above
op = HessianOperator(model=model, dataloader=loader,
                    loss_fn=supervised_loss(torch.nn.functional.cross_entropy))
```

## GGNOperator: split into `forward_fn` + `loss_of_output_fn`

GGN computes $G v = J^\top H_\ell J v$, which needs the model output and the loss-of-output separately. We can't extract them from a single closure efficiently.

```python
from hessian_eigenthings.operators import GGNOperator

def forward_fn(model, batch):
    inputs, _ = batch
    return model(inputs)

def loss_of_output_fn(output, batch):
    _, targets = batch
    return torch.nn.functional.cross_entropy(output, targets)

op = GGNOperator(
    model=model,
    dataloader=loader,
    forward_fn=forward_fn,
    loss_of_output_fn=loss_of_output_fn,
)
```

Or use the helpers:

```python
from hessian_eigenthings.loss_fns import supervised_forward, supervised_loss_of_output

op = GGNOperator(
    model=model,
    dataloader=loader,
    forward_fn=supervised_forward,
    loss_of_output_fn=supervised_loss_of_output(torch.nn.functional.cross_entropy),
)
```

## EmpiricalFisherOperator: `per_sample_loss_fn(model, sample) -> Tensor`

Empirical Fisher needs **per-sample** gradients. Provide a function that takes a single un-batched sample:

```python
from hessian_eigenthings.operators import EmpiricalFisherOperator

def per_sample_loss(model, sample):
    x, y = sample
    return torch.nn.functional.cross_entropy(
        model(x.unsqueeze(0)), y.unsqueeze(0)
    )

op = EmpiricalFisherOperator(
    model=model,
    dataloader=loader,
    per_sample_loss_fn=per_sample_loss,
    sample_dim=0,    # which axis of the batch tensors is the sample axis
)
```

The operator uses `torch.func.vmap(grad(...))` to vectorize over the batch, so per-sample grads are computed in one efficient pass — not a Python loop.

## HuggingFace and TransformerLens

For HuggingFace causal LMs:

```python
from hessian_eigenthings.loss_fns import hf_lm_loss, hf_lm_forward, hf_lm_loss_of_output

# Hessian
HessianOperator(model=hf_model, dataloader=batches, loss_fn=hf_lm_loss())

# GGN
GGNOperator(
    model=hf_model, dataloader=batches,
    forward_fn=hf_lm_forward(),
    loss_of_output_fn=hf_lm_loss_of_output(),
)
```

For TransformerLens:

```python
from hessian_eigenthings.loss_fns import tlens_loss, tlens_forward, tlens_loss_of_output

HessianOperator(model=tlens_model, dataloader=tokens, loss_fn=tlens_loss())
GGNOperator(model=tlens_model, dataloader=tokens,
            forward_fn=tlens_forward(),
            loss_of_output_fn=tlens_loss_of_output())
```

## Notes

- The loss should be a scalar (`tensor.dim() == 0`).
- Reduction matters for normalization: `reduction='mean'` produces the per-sample-mean Hessian; `reduction='sum'` is `batch_size` times larger. Stay consistent across batches in the same dataloader.
- For losses that depend on multiple samples interacting (contrastive, in-batch negatives), the per-sample-loss API for empirical Fisher is undefined — use Hessian or GGN instead.

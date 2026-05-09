# Analyze a HuggingFace model

Compute the Hessian spectrum of any HuggingFace causal LM (e.g. GPT-2, Llama, Qwen). Requires the optional `transformers` extra.

## Install

```bash
uv add "hessian-eigenthings[transformers]"
# or
pip install "hessian-eigenthings[transformers]"
```

## Pattern

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hessian_eigenthings.algorithms import lanczos, trace
from hessian_eigenthings.loss_fns import hf_lm_loss
from hessian_eigenthings.operators import HessianOperator

# Load with attn_implementation='eager': flash/SDPA attention has no
# CPU double-backward, which the autograd HVP needs.
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

texts = ["the quick brown fox", "stochastic gradient descent"]
enc = tokenizer(texts, padding=True, return_tensors="pt")
enc["labels"] = enc["input_ids"].clone()  # HF computes shifted CE internally

dataloader = [dict(enc)]

operator = HessianOperator(model=model, dataloader=dataloader, loss_fn=hf_lm_loss())
result = lanczos(operator, k=5, max_iter=20, tol=1e-3, seed=0)
print(result.eigenvalues)
```

## Tips

- **Use `attn_implementation="eager"`** for the autograd HVP path. Flash/SDPA backends don't expose a second-derivative kernel on CPU and may not on GPU either.
- **Or use the finite-difference HVP** (`HessianOperator(method="finite_difference")`), which doesn't need a second-backward graph and works with any attention implementation. Trade-off documented in [numerical stability](../concepts/numerical-stability.md).
- **Keep batches small** for HVP: each batch costs ~3× the memory of a normal forward+backward.
- **Set `model.eval()`** to disable dropout. With dropout active, repeated HVPs on the same vector aren't deterministic — though if the loss surface you're analyzing *is* the dropout-active training-time loss, leave it on.

## Restricting to a parameter subset

Use [`param_filter`](per-layer-hessian.md) to compute the Hessian of just the attention layers, the MLP layers, the LM head, etc.:

```python
from hessian_eigenthings.param_utils import match_names

attn_op = HessianOperator(
    model=model,
    dataloader=dataloader,
    loss_fn=hf_lm_loss(),
    param_filter=match_names("transformer.h.*.attn.*"),
)
```

## Full runnable example

See [`examples/huggingface_tiny_gpt2.py`](https://github.com/noahgolmant/pytorch-hessian-eigenthings/blob/v1/examples/huggingface_tiny_gpt2.py) — runs end-to-end on CPU in well under a minute on tiny-gpt2 (~100k params).

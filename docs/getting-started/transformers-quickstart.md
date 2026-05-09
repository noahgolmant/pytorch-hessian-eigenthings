# Transformers quickstart

End-to-end on a HuggingFace causal LM. Requires the optional `transformers` extra.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hessian_eigenthings.algorithms import lanczos, trace
from hessian_eigenthings.loss_fns import hf_lm_loss
from hessian_eigenthings.operators import HessianOperator
from hessian_eigenthings.param_utils import match_names

# Load with attn_implementation='eager': flash/SDPA attention has no
# CPU double-backward, which the autograd HVP needs.
name = "sshleifer/tiny-gpt2"
model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Encode a couple of sentences and use input_ids as labels (HF computes shifted CE internally).
texts = ["the quick brown fox", "stochastic gradient descent has converged"]
enc = tokenizer(texts, padding=True, return_tensors="pt")
enc["labels"] = enc["input_ids"].clone()
dataloader = [dict(enc)]

# Whole-model Hessian.
op = HessianOperator(model=model, dataloader=dataloader, loss_fn=hf_lm_loss())
print(f"operator size: {op.size} parameters")

result = lanczos(op, k=3, max_iter=20, tol=1e-3, seed=0)
print("top-3 eigenvalues (full model):", result.eigenvalues.tolist())

# Restrict to attention layers using a glob match against parameter names.
attn_op = HessianOperator(
    model=model, dataloader=dataloader, loss_fn=hf_lm_loss(),
    param_filter=match_names("transformer.h.*.attn.*"),
)
print(f"attention-only operator size: {attn_op.size}")

attn_result = lanczos(attn_op, k=3, max_iter=20, tol=1e-3, seed=0)
print("top-3 eigenvalues (attention only):", attn_result.eigenvalues.tolist())
```

## Notes

- For larger models (Llama 7B, GPT-2 XL): use `model.eval()`, drop the batch size, and consider `method="finite_difference"` to reduce peak memory.
- For TransformerLens, swap `hf_lm_loss()` → `tlens_loss()` and pass the `HookedTransformer` model directly. See the [TransformerLens recipe](../how-to/analyze-with-transformer-lens.md).
- For a runnable script: [`examples/huggingface_tiny_gpt2.py`](https://github.com/noahgolmant/pytorch-hessian-eigenthings/blob/v1/examples/huggingface_tiny_gpt2.py).

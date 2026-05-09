"""Top-k Hessian eigenvalues of a HuggingFace causal LM (sshleifer/tiny-gpt2).

Uses the `hf_lm_loss` helper. Runs on CPU in well under a minute.

    uv run --extra transformers python examples/huggingface_tiny_gpt2.py
"""

from __future__ import annotations

import torch

from hessian_eigenthings.algorithms import lanczos, trace
from hessian_eigenthings.loss_fns import hf_lm_loss
from hessian_eigenthings.operators import HessianOperator
from hessian_eigenthings.param_utils import match_names


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Eager attention: HF defaults to flash/SDPA which lacks a CPU
    # double-backward. The autograd HVP path needs the second backward to exist.
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")

    texts = [
        "the quick brown fox jumps over the lazy dog",
        "stochastic gradient descent has converged",
        "curvature analysis is useful for sharpness studies",
        "language models predict the next token",
    ]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
    encoded["labels"] = encoded["input_ids"].clone()
    dataloader = [dict(encoded)]

    full_op = HessianOperator(model=model, dataloader=dataloader, loss_fn=hf_lm_loss())
    print(f"Full Hessian operator size: {full_op.size} parameters")

    eig_full = lanczos(full_op, k=3, max_iter=20, tol=1e-3, seed=0)
    print("\nTop-3 eigenvalues (full model):")
    for i, val in enumerate(eig_full.eigenvalues):
        print(f"  λ_{i + 1} = {val.item(): .4e}")

    trace_full = trace(full_op, num_matvecs=30, method="hutch++", seed=0)
    print(f"Hutch++ trace estimate: {trace_full.estimate: .4e}")

    attn_op = HessianOperator(
        model=model,
        dataloader=dataloader,
        loss_fn=hf_lm_loss(),
        param_filter=match_names("transformer.h.*.attn.*"),
    )
    print(f"\nAttention-only Hessian size: {attn_op.size} parameters")

    eig_attn = lanczos(attn_op, k=3, max_iter=20, tol=1e-3, seed=0)
    print("Top-3 eigenvalues (attention only):")
    for i, val in enumerate(eig_attn.eigenvalues):
        print(f"  λ_{i + 1} = {val.item(): .4e}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()

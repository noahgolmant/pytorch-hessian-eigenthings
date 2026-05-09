"""Per-block Hessian analysis of a TransformerLens model: attention-only vs MLP-only.

Uses `param_filter` to narrow the operator to a subset of parameters. The two
restricted Hessians have different sizes and different top eigenvalues — these
correspond to the per-block sharpness disparities Liu et al. 2025 observed in
transformer training.

    uv run --extra transformer-lens python examples/transformer_lens_attention_only.py
"""

from __future__ import annotations

import torch

from hessian_eigenthings.algorithms import lanczos
from hessian_eigenthings.loss_fns import tlens_loss
from hessian_eigenthings.operators import HessianOperator
from hessian_eigenthings.param_utils import match_names


def main() -> None:
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained("solu-1l")
    model.eval()

    torch.manual_seed(0)
    seq_len = 32
    batch = torch.randint(0, model.cfg.d_vocab, (2, seq_len))
    dataloader = [batch]

    full_op = HessianOperator(model=model, dataloader=dataloader, loss_fn=tlens_loss())
    print(f"Full Hessian size: {full_op.size}")

    eig_full = lanczos(full_op, k=3, max_iter=20, tol=1e-3, seed=0)
    print("Top-3 eigenvalues (full model):")
    for i, val in enumerate(eig_full.eigenvalues):
        print(f"  λ_{i + 1} = {val.item(): .4e}")

    attn_op = HessianOperator(
        model=model,
        dataloader=dataloader,
        loss_fn=tlens_loss(),
        param_filter=match_names("blocks.*.attn.*"),
    )
    print(f"\nAttention-only Hessian size: {attn_op.size}")
    eig_attn = lanczos(attn_op, k=3, max_iter=20, tol=1e-3, seed=0)
    for i, val in enumerate(eig_attn.eigenvalues):
        print(f"  λ_{i + 1} = {val.item(): .4e}")

    mlp_op = HessianOperator(
        model=model,
        dataloader=dataloader,
        loss_fn=tlens_loss(),
        param_filter=match_names("blocks.*.mlp.*"),
    )
    print(f"\nMLP-only Hessian size: {mlp_op.size}")
    eig_mlp = lanczos(mlp_op, k=3, max_iter=20, tol=1e-3, seed=0)
    for i, val in enumerate(eig_mlp.eigenvalues):
        print(f"  λ_{i + 1} = {val.item(): .4e}")


if __name__ == "__main__":
    main()

# Quickstart

A complete top-k Hessian eigenvalue calculation on a small MLP, end to end.

```python
import torch
from torch import nn

from hessian_eigenthings.algorithms import lanczos, trace, spectral_density
from hessian_eigenthings.loss_fns import supervised_loss
from hessian_eigenthings.operators import HessianOperator

torch.manual_seed(0)

# 1. Any nn.Module and a dataloader-like iterable of (input, target) tuples.
model = nn.Sequential(nn.Linear(20, 32), nn.Tanh(), nn.Linear(32, 1)).to(torch.float64)
x = torch.randn(128, 20, dtype=torch.float64)
y = torch.randn(128, 1, dtype=torch.float64)
dataloader = [(x[i:i+32], y[i:i+32]) for i in range(0, 128, 32)]

# 2. Build a curvature operator.
op = HessianOperator(
    model=model,
    dataloader=dataloader,
    loss_fn=supervised_loss(nn.functional.mse_loss),
)

# 3. Run any algorithm against it.
top_k = lanczos(op, k=5, max_iter=40, seed=0)
print("top eigenvalues:", top_k.eigenvalues)

trace_est = trace(op, num_matvecs=99, method="hutch++", seed=0)
print(f"trace ≈ {trace_est.estimate:.3f} ± {trace_est.stderr:.3f}")

density = spectral_density(op, num_runs=8, lanczos_steps=40, seed=0)
print(f"density grid: {density.grid.numel()} points, integrates to "
      f"{torch.trapz(density.density, density.grid).item():.3f}")
```

## What just happened

1. **Operator construction** is matrix-free — we never form the full Hessian, only the means to compute Hessian-vector products on demand.
2. **`lanczos`** runs symmetric Lanczos with `torch.linalg.eigh` on the resulting tridiagonal. Returns the top-k Ritz pairs and their convergence residuals.
3. **`trace`** uses Hutch++ by default — about 3× lower variance than vanilla Hutchinson at the same matvec budget, with the same algorithmic interface.
4. **`spectral_density`** runs Stochastic Lanczos Quadrature: `num_runs` independent Lanczos chains from random Rademacher start vectors, then a Gaussian-smoothed quadrature density that integrates to ≈ 1.

## Next

- Choose between [Hessian / GGN / Fisher](../concepts/ggn-vs-fisher-vs-hessian.md) operators for your use case.
- Use [`param_filter`](../how-to/per-layer-hessian.md) for per-layer or per-block analysis.
- For HuggingFace and TransformerLens, see the [transformers quickstart](transformers-quickstart.md).

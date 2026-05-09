"""Top-k eigenvalues, trace, and spectral density of a small MLP's Hessian.

Synthetic regression data so the script is self-contained — no downloads, no GPU
required, runs in a few seconds on CPU.

    uv run python examples/supervised_mlp.py
"""

from __future__ import annotations

import torch
from torch import nn

from hessian_eigenthings.algorithms import lanczos, spectral_density, trace
from hessian_eigenthings.loss_fns import supervised_loss
from hessian_eigenthings.operators import HessianOperator


def main() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(20, 32),
        nn.Tanh(),
        nn.Linear(32, 16),
        nn.Tanh(),
        nn.Linear(16, 1),
    ).to(torch.float64)

    n_samples = 256
    x = torch.randn(n_samples, 20, dtype=torch.float64)
    true_w = torch.randn(20, 1, dtype=torch.float64)
    y = x @ true_w + 0.1 * torch.randn(n_samples, 1, dtype=torch.float64)

    batch_size = 32
    dataloader = [
        (x[i : i + batch_size], y[i : i + batch_size]) for i in range(0, n_samples, batch_size)
    ]

    operator = HessianOperator(
        model=model,
        dataloader=dataloader,
        loss_fn=supervised_loss(nn.functional.mse_loss),
    )

    print(f"Hessian operator size: {operator.size} parameters")

    eig_result = lanczos(operator, k=5, max_iter=40, tol=1e-7, seed=0)
    print("\nTop-5 eigenvalues (largest magnitude):")
    for i, (val, res) in enumerate(zip(eig_result.eigenvalues, eig_result.residuals, strict=True)):
        print(f"  λ_{i + 1} = {val.item(): .6e}    residual = {res.item():.2e}")

    trace_result = trace(operator, num_matvecs=99, method="hutch++", seed=0)
    print(f"\nHutch++ trace estimate: {trace_result.estimate: .6e}")
    print(f"Sample stderr:          {trace_result.stderr: .2e}")

    density_result = spectral_density(operator, num_runs=8, lanczos_steps=40, seed=0)
    grid = density_result.grid
    density = density_result.density
    integral = float(torch.trapz(density, grid))
    print(f"\nSpectral density grid: [{grid.min():.3e}, {grid.max():.3e}], {grid.numel()} points")
    print(f"∫ density(λ) dλ = {integral:.4f}  (should be ≈ 1)")
    print(f"Smoothing bandwidth σ = {density_result.sigma:.3e}")


if __name__ == "__main__":
    main()

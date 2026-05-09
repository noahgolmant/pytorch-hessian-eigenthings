# Numerical stability

A short guide to dtype choice, finite-difference $\varepsilon$ tuning, and reorthogonalization defaults.

## Dtype

| dtype  | Where it makes sense                                                  | Caveats                                          |
|--------|-----------------------------------------------------------------------|--------------------------------------------------|
| fp64   | Validation tests, small models, papers that quote eigenvalues to 6+ sig figs | Slow; doubles memory; CUDA kernels less optimized |
| fp32   | Most analysis on models up to ~1B params                              | The default                                       |
| bf16   | LLM scale (≥7B), where fp32 master copies are infeasible              | Per-matvec noise on the order of 1%               |
| fp16   | Generally avoid for HVP — narrow exponent range underflows            | Use bf16 instead at scale                         |

The library inherits the operator's dtype from the model parameters. Cast the model with `model.to(torch.float64)` for high-precision validation.

## Finite-difference HVP $\varepsilon$ selection

The central-difference approximation $Hv \approx (\nabla L(\theta + \varepsilon v) - \nabla L(\theta - \varepsilon v)) / 2\varepsilon$ has two competing error sources:

- **Truncation bias**: $O(\varepsilon^2 \cdot \|v\| \cdot \|\nabla^3 L\|)$
- **Roundoff (cancellation)**: $O(\varepsilon_\text{machine} \cdot \|\nabla L\| / \varepsilon)$

Optimal $\varepsilon$ minimizes their sum at $\varepsilon^* \approx \varepsilon_\text{machine}^{1/3}$:

| dtype  | Optimal $\varepsilon$ | Best-case relative error | Realistic relative error |
|--------|------------------------|--------------------------|--------------------------|
| fp64   | $\sim 10^{-5}$         | $\sim 10^{-10}$          | $\sim 10^{-9}$           |
| fp32   | $\sim 10^{-3}$         | $\sim 10^{-6}$           | $\sim 10^{-5}$ to $10^{-4}$ |
| bf16   | $\sim 0.1$             | $\sim 10^{-2}$           | $\sim 10^{-2}$ to $5 \cdot 10^{-2}$ |

`HessianOperator(method="finite_difference")` selects these defaults automatically. Override with `fd_eps=`.

For LLM-scale spectral analysis where you're already in bf16, the finite-difference noise is dominated by precision noise of the gradient itself — finite-difference is the right tool. For high-precision eigenvalue claims on small models, use the autograd path in fp32 or fp64.

## Reorthogonalization in Lanczos

Without reorthogonalization, classical Lanczos loses orthogonality after enough steps and produces *ghost* eigenvalues — duplicated copies of converged eigenvalues that pollute the result (Paige 1976). The fix is full Gram-Schmidt against all prior basis vectors after each step, at $O(m^2 n)$ cost.

`lanczos(operator, ..., reorthogonalize=None)` defaults to `True` for `max_iter <= 50` and `False` for larger Krylov dimensions. If you're analyzing a Hessian known to have many near-degenerate eigenvalues — common for transformers — keep reorthogonalization on regardless.

## Mixed-precision tradeoffs in iterative algorithms

For top eigenpairs in fp32 / bf16, expect the eigenvalue estimates to inherit roughly the per-matvec relative error compounded across Lanczos steps. Specifically:

- **Lanczos with reorthogonalization**: tridiagonal eigenvalues good to ~3-5x the per-matvec error.
- **Power iteration**: convergence is exponential in $|\lambda_2/\lambda_1|^t$; precision noise sets a floor on how close to the true eigenvalue you can get.
- **Hutchinson trace**: stochastic noise dominates over precision noise as long as you take enough samples.
- **SLQ density**: smoothing washes out per-matvec noise effectively.

In practice, for *qualitative* spectral analysis (sharpness trends, density shape, top-eigenvalue changes during training), fp32 or bf16 is fine. For *absolute precision* of specific eigenvalues, drop to fp64 on a tractable problem.

## References

- Paige, C. C. (1976). *Error analysis of the Lanczos algorithm for tridiagonalizing a symmetric matrix.*
- Granziol & Juarev (2026). *Hessian Spectral Analysis at Foundation Model Scale.* arXiv:2602.00816.
- Pearlmutter, B. A. (1994). *Fast Exact Multiplication by the Hessian.* Neural Computation 6(1).

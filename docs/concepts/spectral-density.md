# Spectral density

The eigenvalue density (or *density of states*) of an $n \times n$ symmetric operator $H$ is the probability density of its eigenvalues:

$$\phi(\lambda) \;=\; \frac{1}{n} \sum_{i=1}^n \delta(\lambda - \lambda_i)$$

For a trained neural network this density typically has a large bulk concentrated near zero plus a small number of outliers — see Ghorbani, Krishnan, Xiao (2019) and Pan et al. (ICML 2025) on the power-law structure in LLMs.

## Stochastic Lanczos Quadrature (SLQ)

Computing $\phi$ exactly requires the full eigendecomposition. Ubaru/Chen/Saad 2017 give a stochastic estimator that costs $O(n_v \cdot m)$ matvecs, where $n_v$ is the number of random init vectors and $m$ is the number of Lanczos steps per init.

For each of $n_v$ random Rademacher init vectors $v_l$:

1. Run $m$ Lanczos steps from $v_l$ to obtain a tridiagonal $T_m^{(l)}$.
2. Eigendecompose $T_m^{(l)}$. The nodes $\theta_k^{(l)}$ are the eigenvalues of $T_m^{(l)}$, and the weights $\tau_k^{(l)} = (e_1^\top y_k)^2$ are the squared first components of its eigenvectors $y_k$, where $e_1 = (1, 0, \ldots, 0)$.
3. The smoothed density is the average over runs of a sum of Gaussian-blurred contributions:

$$\phi(t) \;\approx\; \frac{1}{n_v} \sum_{l=1}^{n_v} \sum_{k=1}^m (\tau_k^{(l)})^2 \, g_\sigma(t - \theta_k^{(l)})$$

where $g_\sigma$ is a unit-mass Gaussian of bandwidth $\sigma$. Each run's quadrature weights sum to 1, so the result integrates to $\approx 1$ — it is a probability density of eigenvalues.

```python
from hessian_eigenthings.algorithms import spectral_density

result = spectral_density(
    operator,
    num_runs=10,         # n_v in the paper
    lanczos_steps=80,    # m in the paper
    num_grid_points=10000,
    sigma=None,          # default: spectrum_range / lanczos_steps
    seed=0,
)
```

`result.density` integrates to ≈ 1 over `result.grid`. `result.raw_eigenvalues` and `result.raw_weights` give you the per-run nodes and weights so you can do your own smoothing or plotting.

## Reading a spectral density

Trained-network Hessians are well-known to have:

- **A bulk** concentrated near zero — most directions have very little curvature.
- **A small number of outlier eigenvalues** rising sharply above the bulk — for classification, the count is roughly the number of classes minus one (Ghorbani et al. 2019).
- **A power-law tail** in the bulk for well-trained LLMs (Pan et al. ICML 2025).

Plot in log-log to see the power-law structure. The outliers are usually clear in linear-y plots.

## Choosing $m$ and $n_v$

The paper's bound: for $\varepsilon$-accurate spectral approximation,

$$m \gtrsim \sqrt{\kappa} \log(K/\varepsilon), \qquad n_v \gtrsim \frac{1}{\varepsilon^2}\log(2/\eta)$$

where $\kappa$ is the condition number and $\eta$ is failure probability. In practice:

| Goal                           | Reasonable defaults  |
|--------------------------------|----------------------|
| Quick visual check             | `lanczos_steps=50`, `num_runs=5`  |
| Publication-quality density    | `lanczos_steps=100`, `num_runs=20` |
| High-resolution outlier study  | `lanczos_steps=200`, `num_runs=10` |

Larger `lanczos_steps` improves spectral resolution; larger `num_runs` reduces Monte-Carlo noise.

## Bandwidth $\sigma$

The default $\sigma = (\lambda_\max - \lambda_\min) / m$ relates the kernel width to the quadrature resolution and works well across most spectra. For very narrow features (closely spaced outliers) try a smaller $\sigma$; for noisy estimates a larger $\sigma$ smooths the result.

## Limitations

- SLQ holds the entire Lanczos basis per run in memory while computing the tridiagonal — the basis-storage problem at LLM scale. v1.3 will support a basis-free SLQ variant.
- The bias depends on $m$ (Krylov truncation) and Gaussian smoothing; sharp delta-like features in the true spectrum get blurred. Nuance is recoverable from `raw_eigenvalues` / `raw_weights`.
- For indefinite operators (Hessians at saddle points), reorthogonalization in the Lanczos kernel matters more — we keep it on by default in SLQ.

## References

- Ubaru, Chen, Saad (2017). *Fast Estimation of $\mathrm{tr}(F(A))$ via Stochastic Lanczos Quadrature.* SIMAX 38(4).
- Lin, Saad, Yang (2016). *Approximating Spectral Densities of Large Matrices.* SIAM Review.
- Ghorbani, Krishnan, Xiao (2019). *An Investigation into Neural Net Optimization via Hessian Eigenvalue Density.* ICML / arXiv:1901.10159.

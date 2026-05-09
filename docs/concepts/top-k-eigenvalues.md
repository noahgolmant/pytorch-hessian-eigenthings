# Top-k eigenvalues

The two algorithms in this library for computing the top $k$ eigenpairs of a symmetric operator: **power iteration with deflation** and **Lanczos**. They share the same `EigenResult` return type but differ in how they search the spectrum.

## Power iteration with deflation

The basic recurrence:

$$v_{k+1} \;=\; \frac{H v_k}{\|H v_k\|}, \qquad \lambda_k \;=\; v_k^\top H v_k$$

Starting from a random vector, the iterates converge to the eigenvector of largest absolute eigenvalue. The convergence rate is $|\lambda_2 / \lambda_1|^t$ — fast if there's a clear gap between the top two eigenvalues, slow otherwise.

For top $k$, we **deflate**: after finding $(\lambda_1, v_1)$, we replace $H$ with $H - \lambda_1 v_1 v_1^\top$ and run power iteration again to find the next eigenpair, and so on.

```python
from hessian_eigenthings.algorithms import deflated_power_iteration

result = deflated_power_iteration(
    operator,
    k=5,
    max_iter=100,
    tol=1e-4,
    momentum=0.0,        # optional Polyak momentum (De Sa et al. 2017)
    seed=0,
)
print(result.eigenvalues)    # (5,)
print(result.eigenvectors)   # (5, n)
print(result.residuals)      # (5,) ||H v - λ v||
```

Stopping uses a combined criterion: relative change in the Rayleigh estimate $|\lambda_t - \lambda_{t-1}| / |\lambda_t|$ AND residual $\|Hv - \lambda v\| / |\lambda|$ both below tolerance.

## Lanczos

Lanczos builds a Krylov subspace $\mathrm{span}(v, Hv, H^2 v, \ldots, H^{m-1} v)$ in tridiagonal form via a three-term recurrence:

$$\beta_{j+1} v_{j+1} \;=\; H v_j \;-\; \alpha_j v_j \;-\; \beta_j v_{j-1}$$

with $\alpha_j = v_j^\top H v_j$ and $\beta_j = \|w_j\|$. After $m$ steps, the tridiagonal matrix $T_m$ has the same top eigenvalues as $H$ (to within Krylov-projection error). We solve $T_m$'s eigenproblem directly with `torch.linalg.eigh` — cheap because $T_m$ is $m \times m$ with $m \ll n$.

```python
from hessian_eigenthings.algorithms import lanczos

result = lanczos(
    operator,
    k=5,
    max_iter=20,             # Krylov dimension; defaults to min(2k, n-1)
    tol=1e-4,
    reorthogonalize=None,    # auto: True for max_iter <= 50
    which="LM",              # 'LM' largest-magnitude, 'LA'/'SA' largest/smallest algebraic
    seed=0,
)
```

### Reorthogonalization

In exact arithmetic the Lanczos basis is orthogonal. In floating point it isn't, and the loss of orthogonality produces *ghost* eigenvalues (Paige 1976) — duplicated copies of converged eigenvalues that pollute the result.

By default we run **full Gram-Schmidt reorthogonalization** for `max_iter <= 50` and disable it for larger Krylov dimensions where the cost ($O(m^2 n)$) becomes significant. If you're analyzing a Hessian with many near-degenerate eigenvalues — common for transformers — keep reorthogonalization on regardless.

### Ritz residuals

Lanczos reports residuals as $|\beta_m \cdot s_{m,i}|$ where $s_{m,i}$ is the last component of $T_m$'s $i$-th eigenvector. This is the standard cheap estimate of $\|H v_i - \lambda_i v_i\|$ without an extra matvec per Ritz pair.

## Choosing between the two

- **Lanczos is almost always preferable** for top $k$ on a symmetric operator. It converges much faster than power iteration on the outer eigenvalues.
- **Power iteration with deflation** is mostly useful as a baseline, or when you can't store a Krylov basis (very large $n$, no offload). v1.3 will provide block Lanczos and CPU-offload-of-basis to extend Lanczos's reach.
- For **smallest-magnitude** eigenvalues, raw Lanczos converges slowly. Future versions will add shift-invert. For now, `which="SA"` works but expects more Krylov steps to converge.

## Limitations

- Lanczos with reorthogonalization holds the entire Krylov basis in memory. For a model with $n$ parameters and $m$ Lanczos steps, that's $O(mn)$ — the basis-storage problem at LLM scale.
- Both algorithms are sensitive to the ordering of near-degenerate eigenvalues and may return them in slightly different orders across runs.
- The `seed` argument controls the initial random vector; results are reproducible given a fixed seed and the same random number generator on the same device.

## References

- Lanczos, C. (1950). *An iteration method for the solution of the eigenvalue problem of linear differential and integral operators.*
- Paige, C. C. (1976). *Error analysis of the Lanczos algorithm for tridiagonalizing a symmetric matrix.*
- De Sa, He, Mitliagkas, Ré, Xu (2017). *Accelerated Stochastic Power Iteration.* arXiv:1707.02670.

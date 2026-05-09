# Trace estimation

The trace of a matrix-free operator can't be read off directly — we don't have the diagonal. Stochastic estimators infer it from a small number of $Hv$ products.

## Hutchinson's estimator

For any random vector $v$ with $\mathbb{E}[v v^\top] = I$:

$$\mathbb{E}[v^\top H v] \;=\; \mathbb{E}[\mathrm{tr}(H v v^\top)] \;=\; \mathrm{tr}(H)$$

Average over $m$ independent draws:

$$\mathrm{tr}(H) \;\approx\; \frac{1}{m} \sum_{i=1}^m v_i^\top H v_i$$

```python
from hessian_eigenthings.algorithms import hutchinson

result = hutchinson(operator, num_samples=200, distribution="rademacher", seed=0)
print(result.estimate)   # the trace estimate
print(result.stderr)     # sample standard error
```

**Rademacher** (each entry $\pm 1$ with equal probability) gives lower variance than Gaussian for trace estimation, so it's our default.

The variance of Hutchinson at $m$ samples scales as $\|H\|_F^2 / m$. To get $\varepsilon$-relative accuracy on a PSD matrix you need $m \approx 1/\varepsilon^2$ matvecs.

## Hutch++

Meyer/Musco/Musco/Woodruff 2021 ([arXiv:2010.09649](https://arxiv.org/abs/2010.09649)) reduce that to $O(1/\varepsilon)$ matvecs by exploiting low-rank structure:

1. Draw a random sketch matrix $S \in \mathbb{R}^{n \times m/3}$ and compute $Q = \mathrm{orth}(HS)$ — a basis for the top-$m/3$-dimensional subspace of $H$'s range.
2. Compute the **exact** trace of $H$ projected onto that subspace: $\mathrm{tr}(Q^\top H Q)$.
3. Run **Hutchinson** on the orthogonal residual $H - Q Q^\top H Q Q^\top$ with $m/3$ probe vectors.
4. Sum the two pieces.

The total matvec budget is $m$ (split into thirds for $HS$, $HQ$, and $HG_\perp$). Variance is much lower than Hutchinson at the same budget whenever $H$ has decaying spectrum — which Hessians of trained networks always do.

```python
from hessian_eigenthings.algorithms import hutch_plus_plus

result = hutch_plus_plus(operator, num_matvecs=99, seed=0)
```

Or via the unified API:

```python
from hessian_eigenthings.algorithms import trace

result = trace(operator, num_matvecs=99, method="hutch++", seed=0)
```

## When does Hutch++ help, and when doesn't it?

- **Big win**: matrices with steep spectral decay. The low-rank projection captures most of the trace mass with $O(\sqrt{m})$ matvecs, leaving only a small residual for Hutchinson.
- **Small win**: matrices with flat spectra. The sketch has no structure to exploit; you might as well use Hutchinson.
- **Pathological case**: diagonal matrices with Rademacher probes. $v^\top D v = \sum_i v_i^2 D_{ii} = \sum_i D_{ii} = \mathrm{tr}(D)$ exactly because $v_i^2 = 1$. So Hutchinson is *deterministic* (zero variance) and Hutch++ adds overhead. Almost no real Hessian is purely diagonal.

## Limitations

- Hutch++ allocates $O(n \cdot m/3)$ memory for the sketch and $Q$. For very large models, the sketch dimension is the new memory bottleneck. v1.3 will add streaming variants.
- Both algorithms assume the operator is symmetric (which curvature operators always are).
- Variance bounds assume the operator is PSD. For indefinite Hessians (e.g. at saddle points), the estimators are still unbiased but the bounds loosen.

## References

- Hutchinson, M. F. (1989). *A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines.*
- Meyer, Musco, Musco, Woodruff (2021). *Hutch++: Optimal Stochastic Trace Estimation.* SOSA / arXiv:2010.09649.

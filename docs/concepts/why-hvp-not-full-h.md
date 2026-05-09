# Why HVP, not full $H$

Forming the full Hessian costs $O(n^2)$ memory. For a 7B-parameter model that's $\sim$200 PB. Iterative algorithms (power iteration, Lanczos, Hutchinson) only ever need to apply $H$ to a vector, never to materialize it — so they cost $O(n)$ memory.

## Two ways to compute $Hv$

### Autograd (exact)

Pearlmutter's $R$-operator trick (1994), reinvented as automatic double-backward:

$$Hv \;=\; \nabla_\theta \big( \nabla_\theta L(\theta) \cdot v \big)$$

In PyTorch:

```python
g = torch.autograd.grad(loss, params, create_graph=True)         # ∇L
g_dot_v = sum((gi * vi).sum() for gi, vi in zip(g, v_split))     # <∇L, v>
Hv = torch.autograd.grad(g_dot_v, params)                        # ∇(<∇L, v>) = Hv
```

Cost: roughly one extra backward pass per $Hv$ on top of the original forward+backward. Numerically exact (to floating-point rounding). This is the default in `HessianOperator(method="autograd")`.

### Finite difference (FSDP-friendly)

The classic central difference:

$$Hv \;\approx\; \frac{\nabla L(\theta + \varepsilon v) \;-\; \nabla L(\theta - \varepsilon v)}{2\varepsilon}$$

Two normal forward+backward passes, no second-backward graph anywhere. This is the technique Granziol & Juarev 2026 (arXiv:2602.00816) revive at LLM scale, because **Fully Sharded Data Parallel (FSDP)** — PyTorch's standard mechanism for training models too large to fit on a single device — detaches its gradient collectives from the autograd graph, breaking double-backward. Finite difference doesn't care: it only uses first-order gradients, which FSDP handles correctly out of the box.

Use via `HessianOperator(method="finite_difference")`. See [Numerical stability](numerical-stability.md) for how to pick $\varepsilon$.

## Why this is enough

For the things people actually want from the Hessian:

- **Top eigenpairs**: power iteration or Lanczos — both only need $Hv$.
- **Trace**: Hutchinson's $\frac{1}{m}\sum v_i^\top H v_i \approx \mathrm{tr}(H)$ — only $Hv$ products.
- **Spectral density**: Stochastic Lanczos Quadrature — same.

The full Hessian is never required. The library's job is to expose $Hv$ in a clean, distributed-ready way and run these algorithms on top.

## Reference

- Pearlmutter, B. A. (1994). *Fast Exact Multiplication by the Hessian.* Neural Computation 6(1), 147-160.
- Granziol & Juarev (2026). *Hessian Spectral Analysis at Foundation Model Scale.* arXiv:2602.00816.

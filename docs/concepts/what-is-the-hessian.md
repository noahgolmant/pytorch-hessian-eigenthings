# What is the Hessian

The Hessian of a scalar loss $L(\theta)$ with respect to model parameters $\theta \in \mathbb{R}^n$ is the matrix of second partial derivatives:

$$H_{ij} = \frac{\partial^2 L}{\partial \theta_i \, \partial \theta_j}$$

It describes the local curvature of the loss surface at the current parameter point. Eigenvalues of $H$ at a minimum tell you how steeply the loss rises in each direction; eigenvectors say which combinations of parameters are most affected. People look at it for:

- **Sharpness / generalization studies.** Top eigenvalues quantify how curved the basin is. Many empirical observations relate flat minima to better generalization.
- **Optimization analysis.** Second-order methods, learning-rate selection, and conditioning all depend on the spectrum.
- **Mode connectivity, loss-landscape geometry, and other diagnostics.** The structure of the spectrum (e.g. bulk + a few outlier eigenvalues) is itself informative about the model.

## Why we never form $H$ explicitly

A model with $n$ parameters has an $n \times n$ Hessian, which costs $O(n^2)$ memory. For a 7B-parameter model that's roughly **200 PB**. We only ever interact with $H$ through matrix-vector products $Hv$, computed via automatic differentiation in $O(n)$ memory.

See [Why HVP, not full H](why-hvp-not-full-h.md) for the math behind the matrix-vector trick.

## A first example

```python
import torch
from torch import nn
from hessian_eigenthings.loss_fns import supervised_loss
from hessian_eigenthings.operators import HessianOperator

model = nn.Sequential(nn.Linear(10, 16), nn.Tanh(), nn.Linear(16, 1))
data = [(torch.randn(32, 10), torch.randn(32, 1))]

H = HessianOperator(
    model=model,
    dataloader=data,
    loss_fn=supervised_loss(nn.functional.mse_loss),
)

v = torch.randn(H.size)
Hv = H @ v        # the Hessian-vector product
```

`H` is a [`CurvatureOperator`](../reference/api.md), and most of the algorithms in this library accept any `CurvatureOperator` — they don't care whether it's a Hessian, GGN, or empirical Fisher.

## Related curvature matrices

The Hessian isn't the only useful curvature matrix. See [GGN vs Fisher vs Hessian](ggn-vs-fisher-vs-hessian.md) for the three-way distinction (it's a common pitfall) and when to use which.

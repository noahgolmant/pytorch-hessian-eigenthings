# GGN vs Fisher vs Hessian

These three matrices are easy to conflate and often called "the curvature" interchangeably. They are not the same. Read this page before deciding which operator to instantiate; mistaking one for another is the most common pitfall in curvature analysis.

## The setup

A model $f_\theta(x)$ produces an output (logits of dimension $C$, regression mean of dimension $C$, etc.). A per-sample loss $\ell$ takes the output and the target and returns a scalar. The empirical risk is $L(\theta) = \frac{1}{N}\sum_i \ell(f_\theta(x_i), y_i)$.

Two derivatives appear repeatedly:

- $J = \partial f_\theta(x) / \partial \theta$ is the model Jacobian. Shape $C \times n$ where $n$ is the parameter count.
- $H_\ell = \partial^2 \ell / \partial f^2$ is the Hessian of the loss with respect to the *model output*. Shape $C \times C$. Cheap to compute since $C$ is small (e.g. number of classes).

These are not the same as the parameter-space Hessian $H$ defined below. $H$ is $n \times n$; $H_\ell$ is $C \times C$. Easy to conflate by name; very different by dimension.

## The three matrices

### 1. Hessian

$$H \;=\; \frac{\partial^2 L}{\partial \theta^2} \;=\; J^\top H_\ell J \;+\; \sum_{c=1}^{C} \frac{\partial \ell}{\partial f_c} \cdot \frac{\partial^2 f_c}{\partial \theta^2}$$

Direct second derivative of the loss with respect to parameters. Shape $n \times n$. The first term is the GGN; the second term — sum over output coordinates of the loss-residual times the model's own second derivative — is what makes the Hessian different from the GGN.

May be **indefinite** at saddle points. This is what you want for sharpness analysis at a critical point.

In code: [`HessianOperator`](../reference/api.md).

### 2. Generalized Gauss-Newton (GGN)

$$G \;=\; J^\top H_\ell J$$

Shape $n \times n$. Drops the model-curvature term, keeping only the part of $H$ that flows through the model Jacobian. For convex per-sample losses (cross-entropy + softmax, MSE), $H_\ell$ is PSD, so $G$ is **PSD by construction** — useful when you need a positive semidefinite preconditioner.

At a global minimum where the loss residuals $\partial \ell / \partial f_c$ vanish, the dropped term is zero and $H = G$.

For cross-entropy + softmax classification, $G$ equals the Fisher information matrix exactly.

In code: [`GGNOperator`](../reference/api.md).

### 3. Fisher (and empirical Fisher)

The **Fisher information matrix** is

$$F \;=\; \mathbb{E}_{y \sim p(y \mid x; \theta)} \big[ \nabla_\theta \log p \cdot \nabla_\theta \log p^\top \big]$$

The expectation is over labels sampled from the *model's* predictive distribution.

The **empirical Fisher** uses the true labels in the data:

$$F_{\text{emp}} \;=\; \frac{1}{N} \sum_i g_i g_i^\top, \qquad g_i = \nabla_\theta \ell(f_\theta(x_i), y_i)$$

For a well-trained model where the predictive distribution matches the data, $F \approx F_{\text{emp}}$. **Otherwise they're different objects.** The empirical Fisher is biased for the actual Fisher information.

In code: [`EmpiricalFisherOperator`](../reference/api.md). The MC Fisher is on the v1.x roadmap.

## When are they equal?

- **Cross-entropy + softmax**: GGN = Fisher (true, not empirical).
- **MSE with Gaussian likelihood**: GGN = Fisher up to a constant.
- **Far from a minimum**: Hessian, GGN, and Fisher can all differ substantially.
- **At a global minimum of a well-specified model**: $H = G = F$.

## Which to use

| You want…                                                       | Use                          |
|------------------------------------------------------------------|------------------------------|
| True local curvature, indefinite at saddles                     | `HessianOperator`            |
| PSD curvature for sharpness on classification/regression losses | `GGNOperator`                |
| Diagonal preconditioner / cheap natural-gradient approximation  | `EmpiricalFisherOperator`    |
| Statistically meaningful Fisher information                      | (MC Fisher, planned)         |

When a paper says "the Hessian", they often mean one of GGN/Fisher and just don't distinguish. If your loss is cross-entropy + softmax, GGN is usually what you want — it's PSD, equals the (true) Fisher, and matches the practical "sharpness" most authors describe.

## Common mistake

Treating empirical Fisher as if it were Fisher information, then claiming statistical conclusions about the parameters. Empirical Fisher is a *gradient-second-moment estimator*, not an information matrix in the statistical sense unless your model fits the data well.

For an extended discussion, read Martens 2014/2020 in full — it's the canonical reference and worth the time.

## References

- Schraudolph, N. N. (2002). *Fast Curvature Matrix-Vector Products for Second-Order Gradient Descent.* Neural Computation 14(7).
- Martens, J. (2014/2020). *New Insights and Perspectives on the Natural Gradient Method.* JMLR.

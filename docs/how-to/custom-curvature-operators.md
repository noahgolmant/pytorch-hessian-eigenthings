# Custom curvature operators

The algorithms in this library (Lanczos, power iteration, Hutchinson, Hutch++, SLQ) operate on a `CurvatureOperator` interface. Subclass it to wire in any matrix-free symmetric operator.

## The interface

```python
from hessian_eigenthings.operators.base import CurvatureOperator
import torch


class MyOperator(CurvatureOperator):
    @property
    def size(self) -> int:
        ...  # input/output vector length

    @property
    def device(self) -> torch.device:
        ...

    @property
    def dtype(self) -> torch.dtype:
        ...

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        ...  # symmetric: must return M @ v
```

That's the whole contract. The base class also provides `rmatvec(v) = matvec(v)` (symmetry) and `__call__(v) = matvec(v)` for ergonomics.

## Quick wrapper: LambdaOperator

For one-off operators (e.g. wrapping an existing matrix), use the built-in `LambdaOperator`:

```python
from hessian_eigenthings.operators.base import LambdaOperator

M = torch.randn(100, 100)
M = (M + M.T) / 2

op = LambdaOperator(
    matvec_fn=lambda v: M @ v,
    size=100,
    device=M.device,
    dtype=M.dtype,
)

from hessian_eigenthings.algorithms import lanczos
result = lanczos(op, k=5, seed=0)
```

## Example: damped curvature (Tikhonov regularization)

```python
class DampedHessian(CurvatureOperator):
    def __init__(self, base_op: CurvatureOperator, damping: float):
        self.base = base_op
        self.damping = damping

    @property
    def size(self): return self.base.size
    @property
    def device(self): return self.base.device
    @property
    def dtype(self): return self.base.dtype

    def matvec(self, v):
        return self.base.matvec(v) + self.damping * v


op = DampedHessian(HessianOperator(...), damping=1e-3)
```

## Example: chaining operators

```python
class SumOperator(CurvatureOperator):
    """A + B as a single operator."""
    def __init__(self, a, b):
        assert a.size == b.size
        self.a, self.b = a, b

    @property
    def size(self): return self.a.size
    @property
    def device(self): return self.a.device
    @property
    def dtype(self): return self.a.dtype

    def matvec(self, v):
        return self.a.matvec(v) + self.b.matvec(v)


# Hessian + lambda*GGN as a single operator for an iterative algorithm
combined = SumOperator(hessian_op, lambda_op)
```

## Use with the LinAlgBackend

If you want your operator to play nice with future distributed backends, keep all vector arithmetic going through `LinAlgBackend`:

```python
from hessian_eigenthings.linalg import LinAlgBackend, SingleDeviceBackend

class MyOp(CurvatureOperator):
    def __init__(self, ..., backend: LinAlgBackend | None = None):
        self.backend = backend or SingleDeviceBackend()

    def matvec(self, v):
        # Use self.backend.dot, .norm, .axpy, .scale
        # rather than raw torch ops, so distributed backends drop in later.
        ...
```

## Limitations

- `matvec` must be **symmetric** — the algorithms assume `<u, M v> = <M u, v>`. Asymmetric operators violate Lanczos correctness.
- The size of the operator is fixed at construction; resizing requires a new instance.
- For algorithm-specific diagnostics (residuals, eigenvalue convergence), the operator just needs `matvec`. Everything else is computed from it.

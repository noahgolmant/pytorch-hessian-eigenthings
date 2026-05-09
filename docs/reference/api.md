# API reference

The library's public surface is re-exported from the top-level `hessian_eigenthings` package, so `from hessian_eigenthings import HessianOperator, lanczos, ...` works for everything below.

The reference is split by area: **operators** are the curvature matrices you can multiply against (Hessian, GGN, Fisher, DDP-aware variants), **algorithms** are the iterative routines that consume them (Lanczos, power iteration, trace, spectral density), and **loss functions** are the small helpers for the most common loss-fn shapes.

- [Operators →](operators.md)
- [Algorithms →](algorithms.md)
- [Loss functions →](loss_fns.md)
- [Parameter selection →](param_utils.md)

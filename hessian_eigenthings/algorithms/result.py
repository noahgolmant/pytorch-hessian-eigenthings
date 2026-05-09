"""Result dataclasses returned by the iterative algorithms."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EigenResult:
    """Top-k eigenpairs of a symmetric operator, sorted in the order requested by the algorithm."""

    eigenvalues: torch.Tensor  # (k,)
    eigenvectors: torch.Tensor  # (k, n)
    residuals: torch.Tensor  # (k,) ||A v - λ v||
    iterations: int
    converged: torch.Tensor  # (k,) bool

"""Curvature operator base class. Concrete operators (Hessian, GGN, Fisher) inherit from this."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch


class CurvatureOperator(ABC):
    """Symmetric matrix-free linear operator over a flat parameter vector."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of elements in the operator's input/output vector."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device on which `matvec` produces its output."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Dtype of the output of `matvec`."""

    @abstractmethod
    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        """Compute `M @ v` where `M` is this operator. `v` is a flat 1-D tensor of length `self.size`."""

    def rmatvec(self, v: torch.Tensor) -> torch.Tensor:
        """Compute `v^T @ M`. Equal to `matvec` because curvature operators are symmetric."""
        return self.matvec(v)

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return self.matvec(v)

    def __matmul__(self, v: torch.Tensor) -> torch.Tensor:
        return self.matvec(v)


class LambdaOperator(CurvatureOperator):
    """Wrap a callable as a CurvatureOperator. Useful for tests and ad-hoc operators."""

    def __init__(
        self,
        matvec_fn: Callable[[torch.Tensor], torch.Tensor],
        size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._matvec_fn = matvec_fn
        self._size = size
        self._device = torch.device(device)
        self._dtype = dtype

    @property
    def size(self) -> int:
        return self._size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self._matvec_fn(v)

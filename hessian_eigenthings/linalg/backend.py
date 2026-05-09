"""Vector-arithmetic backends. Algorithms call these instead of raw torch ops so they remain agnostic to how vectors are stored (single tensor, list of shards, etc.)."""

from typing import Protocol, TypeVar, cast

import torch

V = TypeVar("V")


class LinAlgBackend(Protocol[V]):
    """Vector-arithmetic interface used by all iterative algorithms."""

    def dot(self, a: V, b: V) -> torch.Tensor:
        """Inner product `<a, b>` returned as a 0-dim tensor."""
        ...

    def norm(self, a: V) -> torch.Tensor:
        """Euclidean norm `||a||_2` returned as a 0-dim tensor."""
        ...

    def axpy(self, alpha: float | torch.Tensor, a: V, b: V) -> V:
        """Return `alpha * a + b` (BLAS axpy, functional form)."""
        ...

    def scale(self, alpha: float | torch.Tensor, a: V) -> V:
        """Return `alpha * a`."""
        ...

    def zeros_like(self, a: V) -> V:
        """Return a zero vector with the same shape, dtype, and device as `a`."""
        ...

    def randn_like(self, a: V, *, generator: torch.Generator | None = None) -> V:
        """Return a Gaussian-random vector with the same shape, dtype, and device as `a`."""
        ...

    def rademacher_like(self, a: V, *, generator: torch.Generator | None = None) -> V:
        """Return a Rademacher (±1 with equal probability) vector matching `a`."""
        ...


class SingleDeviceBackend:
    """Backend for non-distributed `torch.Tensor` vectors."""

    def dot(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.dot(a.reshape(-1), b.reshape(-1))

    def norm(self, a: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, torch.linalg.vector_norm(a))

    def axpy(self, alpha: float | torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return alpha * a + b

    def scale(self, alpha: float | torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return alpha * a

    def zeros_like(self, a: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(a)

    def randn_like(
        self, a: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        # torch.randn with `device` and `generator` requires both on the same
        # device. Callers commonly use a CPU generator (for portability) with a
        # CUDA tensor; sample on the generator's device and move to match.
        if generator is None or generator.device == a.device:
            return torch.randn(a.shape, dtype=a.dtype, device=a.device, generator=generator)
        out = torch.randn(a.shape, generator=generator)
        return out.to(device=a.device, dtype=a.dtype)

    def rademacher_like(
        self, a: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        if generator is None or generator.device == a.device:
            bits = torch.randint(
                0, 2, a.shape, dtype=torch.int64, device=a.device, generator=generator
            )
        else:
            bits = torch.randint(0, 2, a.shape, dtype=torch.int64, generator=generator).to(a.device)
        return bits.to(a.dtype) * 2.0 - 1.0

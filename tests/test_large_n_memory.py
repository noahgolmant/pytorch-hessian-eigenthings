"""CPU regression tests that catch O(n*m) tensor materializations at moderate n.

These tests build a synthetic operator with `n` in the millions (small enough to
fit in CI RAM, large enough that an O(n*m) intermediate would dwarf the
operator's own footprint). They watch peak Python tensor memory via
tracemalloc and assert the algorithms don't allocate giant transient tensors
of the form (n, m).

Without this, OOMs of the form "torch.stack(list_of_n_vectors) for m elements
allocates n*m bytes contiguously" only surface on GPU at LLM scale, where each
debug iteration costs cloud time.
"""

from __future__ import annotations

import gc
import tracemalloc
from collections.abc import Callable

import pytest
import torch

from hessian_eigenthings.algorithms.lanczos import lanczos
from hessian_eigenthings.algorithms.power_iteration import deflated_power_iteration
from hessian_eigenthings.algorithms.spectral_density import spectral_density
from hessian_eigenthings.algorithms.trace import hutch_plus_plus, hutchinson
from hessian_eigenthings.operators.base import LambdaOperator

# Operator size in millions of params. Big enough that O(n*m) intermediates
# blow past O(n) baseline; small enough to fit in CI RAM (4M floats = 16 MB
# per vector in fp32).
N = 4_000_000


def _diag_operator(n: int) -> LambdaOperator:
    """Cheap matvec that doesn't allocate beyond the input + output vectors."""
    diag = torch.linspace(0.5, 1.5, n, dtype=torch.float32)
    return LambdaOperator(
        lambda v: diag * v, size=n, device=torch.device("cpu"), dtype=torch.float32
    )


def _peak_alloc_bytes(fn: Callable[[], object]) -> int:
    """Run fn() and return peak resident memory observed by tracemalloc.

    Most CPU-tensor allocations route through Python's allocator and show up in
    tracemalloc. Not exact (some allocations bypass it) but catches the worst
    O(n*m) regressions which are 10-100x the baseline.
    """
    gc.collect()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def _bytes_for(n: int, multiplier: float) -> int:
    """fp32 tensor budget: n * multiplier * 4 bytes."""
    return int(n * multiplier * 4)


@pytest.mark.slow
def test_lanczos_no_n_times_m_intermediate() -> None:
    """Top-k Lanczos at moderate n: memory should be O(n*m) for the basis itself
    plus O(n*k) for output, NOT a transient O(n*k) doubling from .t().contiguous()."""
    op = _diag_operator(N)
    k = 5
    m = 15
    peak = _peak_alloc_bytes(
        lambda: lanczos(op, k=k, max_iter=m, tol=1e-3, seed=0, reorthogonalize=True)
    )
    # Loose ceiling: basis (n*m) + output (n*k) + bookkeeping. The old code's
    # transient .t().contiguous() would add another 2*n*k.
    ceiling = _bytes_for(N, m + 3 * k + 10)
    assert (
        peak < ceiling
    ), f"lanczos peak {peak / 1e9:.2f} GB exceeds ceiling {ceiling / 1e9:.2f} GB"


@pytest.mark.slow
def test_hutch_plus_plus_no_full_sketch_or_aq() -> None:
    """Hutch++ at moderate n: should NOT materialize S, AS-replicated, AQ, or
    AG_perp as full (n, m_per) matrices. Only Q (and one transient AS during phase 1)."""
    op = _diag_operator(N)
    num_matvecs = 30  # m_per = 10
    peak = _peak_alloc_bytes(lambda: hutch_plus_plus(op, num_matvecs=num_matvecs, seed=0))
    # m_per = 10 columns; the streamed implementation peaks at ~2*n*m_per (AS + Q
    # alive briefly), then drops to n*m_per (Q only) plus per-iteration transients.
    # Old code held S + AS + G + AG_perp + Q simultaneously = 5*n*m_per minimum.
    ceiling = _bytes_for(N, 3 * (num_matvecs // 3) + 10)
    assert peak < ceiling, (
        f"hutch++ peak {peak / 1e9:.2f} GB exceeds ceiling {ceiling / 1e9:.2f} GB; "
        "likely a stack-of-columns regression"
    )


@pytest.mark.slow
def test_hutchinson_is_streaming() -> None:
    """Vanilla Hutchinson should not pre-materialize the sample matrix."""
    op = _diag_operator(N)
    peak = _peak_alloc_bytes(lambda: hutchinson(op, num_samples=20, seed=0))
    # Should be O(n) per iteration plus tiny scalar accumulator, NOT n*m.
    ceiling = _bytes_for(N, 5)
    assert (
        peak < ceiling
    ), f"hutchinson peak {peak / 1e9:.2f} GB exceeds ceiling {ceiling / 1e9:.2f} GB"


@pytest.mark.slow
def test_deflated_power_iter_no_stack_of_eigenvectors() -> None:
    """deflated_power_iteration should not torch.stack the k eigenvectors at the
    end (an O(n*k) transient on top of the existing storage)."""
    op = _diag_operator(N)
    k = 4
    peak = _peak_alloc_bytes(
        lambda: deflated_power_iteration(op, k=k, max_iter=20, tol=1e-3, seed=0)
    )
    # Should be O(n*k) for the result + small per-iteration overhead.
    ceiling = _bytes_for(N, 3 * k + 10)
    assert (
        peak < ceiling
    ), f"deflated power iter peak {peak / 1e9:.2f} GB exceeds ceiling {ceiling / 1e9:.2f} GB"


@pytest.mark.slow
def test_spectral_density_basis_freed_between_runs() -> None:
    """SLQ creates a Lanczos basis per run; should not retain previous runs' bases.
    Peak memory should be O(n*m) for one run's basis, not O(n*m*num_runs)."""
    op = _diag_operator(N)
    m = 20
    num_runs = 4
    peak = _peak_alloc_bytes(
        lambda: spectral_density(op, num_runs=num_runs, lanczos_steps=m, seed=0)
    )
    # Single-run basis (n*m) plus modest accumulators. NOT num_runs * n*m.
    ceiling = _bytes_for(N, m + 30)
    assert peak < ceiling, (
        f"SLQ peak {peak / 1e9:.2f} GB exceeds ceiling {ceiling / 1e9:.2f} GB; "
        f"would need {num_runs * m * N * 4 / 1e9:.2f} GB if all bases retained"
    )

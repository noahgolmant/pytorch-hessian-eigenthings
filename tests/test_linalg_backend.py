import pytest
import torch

from hessian_eigenthings.linalg import SingleDeviceBackend


@pytest.fixture
def backend() -> SingleDeviceBackend:
    return SingleDeviceBackend()


@pytest.fixture
def gen() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


def _randn(gen: torch.Generator, shape: tuple[int, ...] = (32,)) -> torch.Tensor:
    return torch.randn(shape, generator=gen, dtype=torch.float64)


def test_dot_matches_torch(backend: SingleDeviceBackend, gen: torch.Generator) -> None:
    a, b = _randn(gen), _randn(gen)
    torch.testing.assert_close(backend.dot(a, b), torch.dot(a, b))


def test_dot_works_on_higher_rank(backend: SingleDeviceBackend, gen: torch.Generator) -> None:
    a, b = _randn(gen, (4, 8)), _randn(gen, (4, 8))
    expected = torch.dot(a.reshape(-1), b.reshape(-1))
    torch.testing.assert_close(backend.dot(a, b), expected)


def test_norm_consistency(backend: SingleDeviceBackend, gen: torch.Generator) -> None:
    a = _randn(gen)
    torch.testing.assert_close(backend.norm(a) ** 2, backend.dot(a, a))


def test_axpy_correctness(backend: SingleDeviceBackend, gen: torch.Generator) -> None:
    a, b = _randn(gen), _randn(gen)
    alpha = 1.7
    torch.testing.assert_close(backend.axpy(alpha, a, b), alpha * a + b)


def test_axpy_with_tensor_alpha(backend: SingleDeviceBackend, gen: torch.Generator) -> None:
    a, b = _randn(gen), _randn(gen)
    alpha = torch.tensor(2.5, dtype=a.dtype)
    torch.testing.assert_close(backend.axpy(alpha, a, b), alpha * a + b)


def test_scale_correctness(backend: SingleDeviceBackend, gen: torch.Generator) -> None:
    a = _randn(gen)
    torch.testing.assert_close(backend.scale(0.3, a), 0.3 * a)


def test_zeros_like_matches_input(backend: SingleDeviceBackend, gen: torch.Generator) -> None:
    a = _randn(gen, (3, 5))
    z = backend.zeros_like(a)
    assert z.shape == a.shape
    assert z.dtype == a.dtype
    assert z.device == a.device
    assert torch.all(z == 0)


def test_randn_like_seeded_is_reproducible(backend: SingleDeviceBackend) -> None:
    a = torch.empty(64, dtype=torch.float64)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    torch.testing.assert_close(
        backend.randn_like(a, generator=g1), backend.randn_like(a, generator=g2)
    )


def test_rademacher_only_emits_pm1(backend: SingleDeviceBackend, gen: torch.Generator) -> None:
    a = torch.empty(2048, dtype=torch.float64)
    r = backend.rademacher_like(a, generator=gen)
    unique = torch.unique(r)
    assert torch.equal(torch.sort(unique).values, torch.tensor([-1.0, 1.0], dtype=a.dtype))


def test_rademacher_mean_is_near_zero(backend: SingleDeviceBackend, gen: torch.Generator) -> None:
    a = torch.empty(100_000, dtype=torch.float64)
    r = backend.rademacher_like(a, generator=gen)
    assert abs(r.mean().item()) < 0.02


def test_backend_implements_protocol() -> None:
    from hessian_eigenthings.linalg import LinAlgBackend

    backend: LinAlgBackend[torch.Tensor] = SingleDeviceBackend()
    assert callable(backend.dot)

import pytest
import torch
from torch import nn

from hessian_eigenthings.batching import (
    assert_microbatch_safe,
    iterate_batches,
    move_batch_to_device,
)


def test_assert_microbatch_safe_passes_for_layernorm() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2))
    assert_microbatch_safe(model)


def test_assert_microbatch_safe_raises_for_batchnorm1d() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))
    with pytest.raises(ValueError, match="BatchNorm"):
        assert_microbatch_safe(model)


def test_assert_microbatch_safe_raises_for_batchnorm2d() -> None:
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8))
    with pytest.raises(ValueError, match="BatchNorm"):
        assert_microbatch_safe(model)


def test_assert_microbatch_safe_lists_module_names() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8))
    with pytest.raises(ValueError, match=r"\b1\b"):
        assert_microbatch_safe(model)


def test_iterate_batches_no_cap() -> None:
    src = [1, 2, 3]
    assert list(iterate_batches(src)) == [1, 2, 3]


def test_iterate_batches_with_cap() -> None:
    src = [1, 2, 3, 4, 5]
    assert list(iterate_batches(src, num_batches=2)) == [1, 2]


def test_move_batch_tensor() -> None:
    t = torch.zeros(4)
    moved = move_batch_to_device(t, torch.device("cpu"))
    assert moved.device == torch.device("cpu")


def test_move_batch_tuple() -> None:
    batch = (torch.zeros(4), torch.zeros(2))
    moved = move_batch_to_device(batch, torch.device("cpu"))
    assert isinstance(moved, tuple)
    assert all(t.device == torch.device("cpu") for t in moved)


def test_move_batch_dict() -> None:
    batch = {"x": torch.zeros(4), "y": torch.zeros(2), "label": "ignored"}
    moved = move_batch_to_device(batch, torch.device("cpu"))
    assert isinstance(moved, dict)
    assert moved["label"] == "ignored"
    assert moved["x"].device == torch.device("cpu")

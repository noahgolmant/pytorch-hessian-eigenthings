import pytest
import torch
from torch import nn

from hessian_eigenthings.param_utils import (
    match_names,
    match_regex,
    params_to_vector,
    select_parameters,
    total_size,
    vector_to_params,
)


def _toy_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )


def test_select_all_parameters_default() -> None:
    model = _toy_model()
    selected = select_parameters(model)
    assert list(selected) == [n for n, _ in model.named_parameters()]


def test_select_with_glob() -> None:
    model = _toy_model()
    selected = select_parameters(model, match_names("0.*"))
    assert set(selected) == {"0.weight", "0.bias"}


def test_select_with_multiple_globs() -> None:
    model = _toy_model()
    selected = select_parameters(model, match_names("*.bias", "0.weight"))
    assert set(selected) == {"0.weight", "0.bias", "2.bias"}


def test_select_with_regex() -> None:
    model = _toy_model()
    selected = select_parameters(model, match_regex(r"^2\."))
    assert set(selected) == {"2.weight", "2.bias"}


def test_select_skips_frozen_params() -> None:
    model = _toy_model()
    for p in model[0].parameters():
        p.requires_grad_(False)
    selected = select_parameters(model)
    assert "0.weight" not in selected
    assert "2.weight" in selected


def test_empty_selection_raises() -> None:
    model = _toy_model()
    with pytest.raises(ValueError, match="zero parameters"):
        select_parameters(model, match_names("nonexistent.*"))


def test_total_size() -> None:
    params = {
        "a": torch.zeros(3, 4),
        "b": torch.zeros(7),
    }
    assert total_size(params) == 19


def test_vector_round_trip() -> None:
    model = _toy_model()
    params = select_parameters(model)
    vec = params_to_vector(params)
    split = vector_to_params(vec, params)
    assert list(split) == list(params)
    for k in params:
        torch.testing.assert_close(split[k], params[k])


def test_vector_to_params_size_mismatch() -> None:
    model = _toy_model()
    params = select_parameters(model)
    with pytest.raises(ValueError, match="elements"):
        vector_to_params(torch.zeros(3), params)


def test_vector_to_params_rejects_non_1d() -> None:
    model = _toy_model()
    params = select_parameters(model)
    with pytest.raises(ValueError, match="1-D"):
        vector_to_params(torch.zeros(2, total_size(params) // 2), params)

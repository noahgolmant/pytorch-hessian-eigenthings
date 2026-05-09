"""Smoke tests for the loss-function helpers; HF and TLens variants are gated by markers."""

import pytest
import torch
from torch import nn

from hessian_eigenthings.loss_fns import (
    supervised_forward,
    supervised_loss,
    supervised_loss_of_output,
    supervised_per_sample_loss,
)
from hessian_eigenthings.operators import (
    EmpiricalFisherOperator,
    GGNOperator,
    HessianOperator,
)


def _model() -> nn.Module:
    g = torch.Generator()
    g.manual_seed(0)
    m = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 4)).to(torch.float64)
    for p in m.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=torch.float64)
    return m


def test_supervised_loss_returns_scalar() -> None:
    m = _model()
    loss_fn = supervised_loss(nn.functional.cross_entropy)
    x = torch.randn(4, 3, dtype=torch.float64)
    y = torch.randint(0, 4, (4,))
    loss = loss_fn(m, (x, y))
    assert loss.dim() == 0


def test_supervised_loss_powers_hessian_operator() -> None:
    m = _model()
    op = HessianOperator(
        model=m,
        dataloader=[(torch.randn(4, 3, dtype=torch.float64), torch.randint(0, 4, (4,)))],
        loss_fn=supervised_loss(nn.functional.cross_entropy),
    )
    v = torch.randn(op.size, dtype=torch.float64)
    out = op.matvec(v)
    assert out.shape == (op.size,)
    assert torch.all(torch.isfinite(out))


def test_supervised_forward_and_loss_of_output_power_ggn_operator() -> None:
    m = _model()
    batch = (torch.randn(4, 3, dtype=torch.float64), torch.randint(0, 4, (4,)))
    op = GGNOperator(
        model=m,
        dataloader=[batch],
        forward_fn=supervised_forward,
        loss_of_output_fn=supervised_loss_of_output(nn.functional.cross_entropy),
    )
    v = torch.randn(op.size, dtype=torch.float64)
    out = op.matvec(v)
    assert torch.all(torch.isfinite(out))
    assert torch.dot(v, out).item() >= -1e-9  # PSD


def test_supervised_per_sample_loss_powers_fisher_operator() -> None:
    m = _model()
    batch = (torch.randn(4, 3, dtype=torch.float64), torch.randint(0, 4, (4,)))
    op = EmpiricalFisherOperator(
        model=m,
        dataloader=[batch],
        per_sample_loss_fn=supervised_per_sample_loss(nn.functional.cross_entropy),
    )
    v = torch.randn(op.size, dtype=torch.float64)
    out = op.matvec(v)
    assert torch.all(torch.isfinite(out))


@pytest.mark.transformers
def test_hf_lm_loss_runs_on_tiny_gpt2() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

    from hessian_eigenthings.loss_fns import hf_lm_loss

    name = "sshleifer/tiny-gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    enc = tok(["hello world", "foo bar baz"], padding=True, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    loss = hf_lm_loss()(model, dict(enc))
    assert loss.dim() == 0
    assert torch.isfinite(loss)


@pytest.mark.transformer_lens
def test_tlens_loss_runs_on_solu_1l() -> None:
    from transformer_lens import HookedTransformer  # type: ignore[import-untyped]

    from hessian_eigenthings.loss_fns import tlens_loss

    model = HookedTransformer.from_pretrained("solu-1l")
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    loss = tlens_loss()(model, tokens)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

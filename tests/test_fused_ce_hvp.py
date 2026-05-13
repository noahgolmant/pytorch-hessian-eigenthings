"""Tests for the fused cross-entropy loss-Hessian-vector product.

The fused variants (``torch.compile`` and Triton) must agree numerically with
the eager reference. Triton requires CUDA + Triton; we skip if either is
missing. ``torch.compile`` is available on any backend supported by Inductor
(CPU/CUDA/MPS).
"""

from __future__ import annotations

import pytest
import torch

from hessian_eigenthings.loss_fns._fused_ce_hvp import (
    _ce_hvp_reference,
    _triton_available,
    compiled_ce_hvp,
    triton_ce_hvp,
)
from hessian_eigenthings.loss_fns.huggingface import (
    _hf_lm_ce_hvp,
    _hf_lm_ce_hvp_eager,
    hf_lm_loss_of_output,
)


def _make_inputs(
    B: int, T: int, V: int, dtype: torch.dtype, device: torch.device, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(flat_logits, flat_u, valid, n)`` for an (B, T, V) problem."""
    g = torch.Generator(device=device).manual_seed(seed)
    flat_logits = torch.randn(B * T, V, dtype=dtype, device=device, generator=g)
    flat_u = torch.randn(B * T, V, dtype=dtype, device=device, generator=g)
    # Mark ~10% of positions as ignored.
    valid_bool = torch.rand(B * T, generator=g, device=device) > 0.1
    valid = valid_bool.to(dtype)
    n = valid.sum().clamp_min(1.0)
    return flat_logits, flat_u, valid, n


def _check_close(a: torch.Tensor, b: torch.Tensor, *, rtol: float, atol: float) -> None:
    """``|a - b| <= atol + rtol * |b|`` elementwise, with a helpful failure msg.

    Plain ``|a-b|/|b|`` is uninformative on CE-HVP outputs because individual
    elements of ``(p*u - p*<p,u>) * mask / n`` are O(1/(N*V)) -- often
    ~1e-5 -- and fp32 reordering noise of ~1e-7 absolute trivially exceeds
    any pure-relative bound on those near-zero elements. We pair an absolute
    floor (``atol``) with a relative tolerance (``rtol``), matching the
    convention of ``torch.testing.assert_close``.
    """
    diff = (a.float() - b.float()).abs()
    bound = atol + rtol * b.float().abs()
    max_excess = float((diff - bound).max())
    if max_excess > 0:
        idx = (diff - bound).argmax()
        ai = a.flatten()[idx].item()
        bi = b.flatten()[idx].item()
        raise AssertionError(
            f"close-check failed: max excess {max_excess:.3e} "
            f"(rtol={rtol}, atol={atol}); a={ai}, b={bi}, |a-b|={abs(ai-bi):.3e}"
        )


# Tolerance budget per dtype: (rtol, atol). atol floor accounts for the fact
# that individual output elements are O(1/(N*V)) ~ 1e-5 here, so fp reordering
# noise of ~1e-7 abs would otherwise blow up any pure-relative bound.
_TOL = {
    torch.float32: (1e-5, 1e-6),
    torch.bfloat16: (1e-2, 1e-4),
    torch.float16: (2e-2, 1e-3),
}


def _backends_to_check(dtype: torch.dtype) -> list[tuple[str, object]]:
    """Return ``[(name, fn), ...]`` for every fused backend usable on this
    machine. ``triton`` is omitted when CUDA + triton aren't available; the
    eager reference is always included as the comparison anchor.
    """
    out: list[tuple[str, object]] = [
        ("eager", _ce_hvp_reference),
        ("compiled", compiled_ce_hvp),
    ]
    if _triton_available() and dtype != torch.float64:
        out.append(("triton", triton_ce_hvp))
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_matches_eager_reference(dtype: torch.dtype) -> None:
    """``compiled_ce_hvp`` must agree with the eager reference."""
    device = torch.device("cpu")
    flat_logits, flat_u, valid, n = _make_inputs(B=4, T=32, V=128, dtype=dtype, device=device)
    ref = _ce_hvp_reference(flat_logits, flat_u, valid, n)
    fused = compiled_ce_hvp(flat_logits, flat_u, valid, n)
    assert fused.shape == ref.shape
    assert fused.dtype == ref.dtype
    rtol, atol = _TOL[dtype]
    _check_close(fused, ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not _triton_available(), reason="Triton kernel requires CUDA + triton")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_matches_eager_reference(dtype: torch.dtype) -> None:
    """``triton_ce_hvp`` must agree with the eager reference."""
    device = torch.device("cuda")
    flat_logits, flat_u, valid, n = _make_inputs(B=4, T=32, V=128, dtype=dtype, device=device)
    ref = _ce_hvp_reference(flat_logits, flat_u, valid, n)
    fused = triton_ce_hvp(flat_logits, flat_u, valid, n)
    rtol, atol = _TOL[dtype]
    _check_close(fused.to(dtype), ref, rtol=rtol, atol=atol)


def test_hf_lm_ce_hvp_backends_match_on_real_shape() -> None:
    """End-to-end: ``_hf_lm_ce_hvp(..., backend="compile")`` matches ``"eager"``.

    Exercises the shift, mask, and re-embed paths in the public wrapper.
    """
    device = torch.device("cpu")
    B, T, V = 2, 16, 64
    g = torch.Generator(device=device).manual_seed(42)
    logits = torch.randn(B, T, V, dtype=torch.float32, device=device, generator=g)
    u = torch.randn(B, T, V, dtype=torch.float32, device=device, generator=g)
    labels = torch.randint(0, V, (B, T), generator=g, device=device)
    # Inject some ignored positions.
    labels[0, 0] = -100
    labels[1, 5] = -100
    batch = {"labels": labels}

    eager = _hf_lm_ce_hvp(logits, batch, u, backend="eager")
    compiled = _hf_lm_ce_hvp(logits, batch, u, backend="compile")

    assert eager.shape == compiled.shape == u.shape
    # The last time slice and ignored rows must be exactly zero in both.
    assert torch.all(eager[:, -1, :] == 0)
    assert torch.all(compiled[:, -1, :] == 0)
    _check_close(compiled, eager, rtol=1e-5, atol=1e-6)


def test_hf_lm_loss_of_output_fused_kwarg_threads_through() -> None:
    """The ``fused=`` kwarg on the factory must reach the underlying HVP."""
    device = torch.device("cpu")
    B, T, V = 2, 8, 32
    g = torch.Generator(device=device).manual_seed(0)
    logits = torch.randn(B, T, V, dtype=torch.float32, device=device, generator=g)
    u = torch.randn(B, T, V, dtype=torch.float32, device=device, generator=g)
    labels = torch.randint(0, V, (B, T), generator=g, device=device)
    batch = {"labels": labels}

    eager_fn = hf_lm_loss_of_output()  # default
    fused_fn = hf_lm_loss_of_output(fused="compile")

    eager_out = eager_fn.hvp(logits, batch, u)
    fused_out = fused_fn.hvp(logits, batch, u)

    _check_close(fused_out, eager_out, rtol=1e-5, atol=1e-6)


def test_eager_helper_matches_inline_math() -> None:
    """Sanity check on ``_hf_lm_ce_hvp_eager`` matching the inline formula.

    Guards against accidental refactoring breaking the reference path that
    every other test compares against.
    """
    device = torch.device("cpu")
    flat_logits, flat_u, valid, n = _make_inputs(B=2, T=8, V=16, dtype=torch.float32, device=device)
    out = _hf_lm_ce_hvp_eager(flat_logits, flat_u, valid, n)

    p = torch.softmax(flat_logits, dim=-1)
    dot = (p * flat_u).sum(dim=-1, keepdim=True)
    expected = (p * flat_u - p * dot) * valid.unsqueeze(-1) / n
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# 1. Analytical full-Hessian comparison
# ---------------------------------------------------------------------------


def _analytical_ce_hessian_matvec(
    flat_logits: torch.Tensor, flat_u: torch.Tensor, valid: torch.Tensor, n: torch.Tensor
) -> torch.Tensor:
    """Build ``H_loss`` for the mean-reduced CE and apply it via dense matmul.

    For each valid row ``t``, the per-token loss Hessian is
    ``(diag(p_t) - p_t p_t^T)``; the mean-reduced loss adds a ``/ n`` factor
    and ignored rows contribute zero. This is the brute-force reference that
    catches sign/scale/index bugs the fused kernels can't see.
    """
    N, V = flat_logits.shape
    p = torch.softmax(flat_logits.double(), dim=-1)  # double for precision
    u = flat_u.double()
    out = torch.zeros((N, V), dtype=torch.float64, device=flat_logits.device)
    for t in range(N):
        if float(valid[t]) == 0.0:
            continue
        # H_t = diag(p_t) - p_t p_t^T
        H_t = torch.diag(p[t]) - torch.outer(p[t], p[t])
        out[t] = H_t @ u[t]
    out = out / float(n.item())
    return out.to(flat_logits.dtype)


def test_analytical_full_hessian_matches_all_backends() -> None:
    """Brute-force ``H_loss @ u`` via diag-block dense matmul agrees with every
    fused implementation. Tiny shape (B=2, T=4, V=8) keeps the O(N V^2) full
    Hessian build cheap while still exercising softmax + masking + reduction.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    B, T, V = 2, 4, 8
    g = torch.Generator(device=device).manual_seed(7)
    flat_logits = torch.randn(B * T, V, dtype=torch.float32, device=device, generator=g)
    flat_u = torch.randn(B * T, V, dtype=torch.float32, device=device, generator=g)
    valid_bool = torch.rand(B * T, generator=g, device=device) > 0.2
    # Ensure at least one valid and at least one invalid row.
    valid_bool[0] = True
    valid_bool[-1] = False
    valid = valid_bool.to(torch.float32)
    n = valid.sum().clamp_min(1.0)

    expected = _analytical_ce_hessian_matvec(flat_logits, flat_u, valid, n)

    rtol, atol = _TOL[torch.float32]
    for name, fn in _backends_to_check(torch.float32):
        out = fn(flat_logits, flat_u, valid, n)
        assert out.shape == expected.shape, f"{name}: shape mismatch"
        try:
            _check_close(out, expected, rtol=rtol, atol=atol)
        except AssertionError as e:
            raise AssertionError(f"backend {name!r}: {e}") from e


# ---------------------------------------------------------------------------
# 2. Shape coverage: non-power-of-2, V < BLOCK_V, very large V, small V
# ---------------------------------------------------------------------------


# Shapes chosen to cover: V smaller than typical BLOCK_V=1024, very small V,
# small generic V, GPT-2 vocab (non-power-of-2), Llama scale, Mistral Large.
# Some are huge; we use a tiny N for those to keep memory bounded on CPU.
_SHAPE_CASES = [
    # (label, N, V) -- N kept tiny on the large-V cases to stay in CPU memory
    ("V=7_tiny", 8, 7),
    ("V=100", 16, 100),
    ("V=50257_gpt2", 4, 50257),
    ("V=130000_mistral_large", 2, 130000),
]


@pytest.mark.parametrize("label,N,V", _SHAPE_CASES, ids=[c[0] for c in _SHAPE_CASES])
def test_shape_coverage_compile(label: str, N: int, V: int) -> None:
    """``compiled_ce_hvp`` produces the same output as eager across V shapes,
    including non-power-of-2 (GPT-2 50257), V < BLOCK_V (7), and Mistral-scale
    (130k)."""
    device = torch.device("cpu")
    g = torch.Generator(device=device).manual_seed(hash(label) & 0xFFFF)
    flat_logits = torch.randn(N, V, dtype=torch.float32, device=device, generator=g)
    flat_u = torch.randn(N, V, dtype=torch.float32, device=device, generator=g)
    valid = (torch.rand(N, device=device, generator=g) > 0.1).to(torch.float32)
    valid[0] = 1.0  # at least one valid
    n = valid.sum().clamp_min(1.0)

    ref = _ce_hvp_reference(flat_logits, flat_u, valid, n)
    out = compiled_ce_hvp(flat_logits, flat_u, valid, n)
    rtol, atol = _TOL[torch.float32]
    _check_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not _triton_available(), reason="Triton kernel requires CUDA + triton")
@pytest.mark.parametrize("label,N,V", _SHAPE_CASES, ids=[c[0] for c in _SHAPE_CASES])
def test_shape_coverage_triton(label: str, N: int, V: int) -> None:
    """``triton_ce_hvp`` handles non-power-of-2 V (50257), V smaller than
    BLOCK_V=1024 (7), and very large V (130k) correctly."""
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(hash(label) & 0xFFFF)
    flat_logits = torch.randn(N, V, dtype=torch.float32, device=device, generator=g)
    flat_u = torch.randn(N, V, dtype=torch.float32, device=device, generator=g)
    valid = (torch.rand(N, device=device, generator=g) > 0.1).to(torch.float32)
    valid[0] = 1.0
    n = valid.sum().clamp_min(1.0)

    ref = _ce_hvp_reference(flat_logits, flat_u, valid, n)
    out = triton_ce_hvp(flat_logits, flat_u, valid, n)
    rtol, atol = _TOL[torch.float32]
    _check_close(out, ref, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------


def test_all_zero_mask_output_is_zero() -> None:
    """When ``valid`` is all-zero, the output should be exactly zero on every
    backend (``mask`` factor zeros every row; ``n_valid`` is clamped to 1)."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    N, V = 8, 64
    g = torch.Generator(device=device).manual_seed(0)
    flat_logits = torch.randn(N, V, dtype=torch.float32, device=device, generator=g)
    flat_u = torch.randn(N, V, dtype=torch.float32, device=device, generator=g)
    valid = torch.zeros(N, dtype=torch.float32, device=device)
    n = valid.sum().clamp_min(1.0)

    for name, fn in _backends_to_check(torch.float32):
        out = fn(flat_logits, flat_u, valid, n)
        assert torch.all(out == 0), f"backend {name!r} produced non-zero output for all-zero mask"


def test_single_valid_token_in_large_batch() -> None:
    """One valid token among many ignored: only that row should be non-zero."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    N, V = 64, 32
    g = torch.Generator(device=device).manual_seed(11)
    flat_logits = torch.randn(N, V, dtype=torch.float32, device=device, generator=g)
    flat_u = torch.randn(N, V, dtype=torch.float32, device=device, generator=g)
    valid = torch.zeros(N, dtype=torch.float32, device=device)
    valid[17] = 1.0
    n = valid.sum().clamp_min(1.0)

    ref = _ce_hvp_reference(flat_logits, flat_u, valid, n)
    # The eager reference is itself correct here (no zero-division because
    # valid * (anything) / 1.0 is fine), so compare backends to it.
    rtol, atol = _TOL[torch.float32]
    for name, fn in _backends_to_check(torch.float32):
        out = fn(flat_logits, flat_u, valid, n)
        try:
            _check_close(out, ref, rtol=rtol, atol=atol)
        except AssertionError as e:
            raise AssertionError(f"backend {name!r}: {e}") from e
        # All non-valid rows must be exactly zero.
        non_valid = torch.cat([out[:17], out[18:]])
        assert torch.all(non_valid == 0), f"backend {name!r}: ignored rows non-zero"


def test_concentrated_probability_softmax_stability() -> None:
    """One logit at +50, rest at 0: softmax must not overflow/underflow.

    The online-softmax in the Triton kernel computes ``exp(x - max)`` so this
    is the canonical numerical-stability test. Compares all backends to eager.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    N, V = 4, 32
    flat_logits = torch.zeros(N, V, dtype=torch.float32, device=device)
    flat_logits[:, 0] = 50.0  # heavy concentration on token 0
    g = torch.Generator(device=device).manual_seed(3)
    flat_u = torch.randn(N, V, dtype=torch.float32, device=device, generator=g)
    valid = torch.ones(N, dtype=torch.float32, device=device)
    n = valid.sum().clamp_min(1.0)

    ref = _ce_hvp_reference(flat_logits, flat_u, valid, n)
    assert torch.all(torch.isfinite(ref))
    rtol, atol = _TOL[torch.float32]
    for name, fn in _backends_to_check(torch.float32):
        out = fn(flat_logits, flat_u, valid, n)
        assert torch.all(torch.isfinite(out)), f"backend {name!r} produced non-finite output"
        try:
            _check_close(out, ref, rtol=rtol, atol=atol)
        except AssertionError as e:
            raise AssertionError(f"backend {name!r}: {e}") from e


def test_near_uniform_probability() -> None:
    """All logits equal: ``p = 1/V`` uniformly, ``<p, u> = mean(u)``.

    Tests the small-magnitude regime where each output element is
    ``(1/V * u_i - 1/V * mean(u)) / n_valid`` -- the limit where relative-only
    tolerances would fail.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    N, V = 8, 64
    flat_logits = torch.full((N, V), 0.5, dtype=torch.float32, device=device)
    g = torch.Generator(device=device).manual_seed(5)
    flat_u = torch.randn(N, V, dtype=torch.float32, device=device, generator=g)
    valid = torch.ones(N, dtype=torch.float32, device=device)
    n = valid.sum().clamp_min(1.0)

    # Closed-form expected: p = 1/V, dot = mean(u), out = (u/V - mean(u)/V) / N
    p_val = 1.0 / V
    expected = (flat_u * p_val - flat_u.mean(dim=-1, keepdim=True) * p_val) / float(n.item())

    rtol, atol = _TOL[torch.float32]
    for name, fn in _backends_to_check(torch.float32):
        out = fn(flat_logits, flat_u, valid, n)
        try:
            _check_close(out, expected, rtol=rtol, atol=atol)
        except AssertionError as e:
            raise AssertionError(f"backend {name!r}: {e}") from e


# ---------------------------------------------------------------------------
# 4. Mixed dtype tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_compiled_lowp_matches_eager(dtype: torch.dtype) -> None:
    """``compiled_ce_hvp`` agrees with the eager reference at fp16 and bf16
    within the dtype-appropriate tolerance budget. fp16 is more permissive
    (rtol=2e-2, atol=1e-3) than bf16 (rtol=1e-2, atol=1e-4) because fp16's
    5-bit exponent makes the softmax normalizer noisier.
    """
    if dtype == torch.float16 and not torch.cuda.is_available():
        # fp16 softmax on CPU is unsupported by torch in some builds.
        pytest.skip("fp16 inference on CPU is patchy; gating to CUDA")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    flat_logits, flat_u, valid, n = _make_inputs(B=4, T=32, V=128, dtype=dtype, device=device)
    ref = _ce_hvp_reference(flat_logits, flat_u, valid, n)
    out = compiled_ce_hvp(flat_logits, flat_u, valid, n)
    rtol, atol = _TOL[dtype]
    _check_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not _triton_available(), reason="Triton kernel requires CUDA + triton")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_lowp_matches_eager(dtype: torch.dtype) -> None:
    """``triton_ce_hvp`` agrees with the eager reference at fp16 and bf16.

    The kernel upcasts to fp32 internally for the online softmax + dot, so
    the dominant noise is the reduced-precision load/store of the inputs and
    output; the bf16 budget (rtol=1e-2, atol=1e-4) is the same as compile.
    """
    device = torch.device("cuda")
    flat_logits, flat_u, valid, n = _make_inputs(B=4, T=32, V=128, dtype=dtype, device=device)
    ref = _ce_hvp_reference(flat_logits, flat_u, valid, n)
    out = triton_ce_hvp(flat_logits, flat_u, valid, n)
    rtol, atol = _TOL[dtype]
    _check_close(out.to(dtype), ref, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# 5. End-to-end GGN matvec test with tiny HF model
# ---------------------------------------------------------------------------


@pytest.mark.transformers
def test_ggn_matvec_fused_matches_eager_on_tiny_gpt2() -> None:
    """``GGNOperator`` with ``fused="eager"`` vs the auto-selected fused path
    (triton/compile) must produce the same matvec output on a real HF model.

    Uses ``sshleifer/tiny-gpt2`` (the same model the existing HF loss test
    uses) so we don't pay the cost of downloading a real LM. This is the
    integration test the unit tests can't replicate: it exercises the shift,
    mask, and re-embed paths together with a real (B, T, V) shape.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

    from hessian_eigenthings.loss_fns.huggingface import hf_lm_forward, hf_lm_loss_of_output
    from hessian_eigenthings.operators import GGNOperator

    name = "sshleifer/tiny-gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    enc = tok(["hello world", "foo bar baz"], padding=True, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    batch = dict(enc)

    op_eager = GGNOperator(
        model=model,
        dataloader=[batch],
        forward_fn=hf_lm_forward(),
        loss_of_output_fn=hf_lm_loss_of_output(fused="eager"),
    )
    op_fused = GGNOperator(
        model=model,
        dataloader=[batch],
        forward_fn=hf_lm_forward(),
        loss_of_output_fn=hf_lm_loss_of_output(fused="auto"),
    )

    g = torch.Generator().manual_seed(0)
    v = torch.randn(op_eager.size, generator=g)
    out_eager = op_eager.matvec(v)
    out_fused = op_fused.matvec(v)

    assert out_eager.shape == out_fused.shape == (op_eager.size,)
    assert torch.all(torch.isfinite(out_eager))
    assert torch.all(torch.isfinite(out_fused))
    # Use a slightly looser tolerance than the per-kernel tests: the FD JVP
    # in `GGNOperator` introduces O(fd_eps^2) truncation and roundoff, and we
    # are comparing two passes through the operator (not one).
    _check_close(out_fused, out_eager, rtol=1e-5, atol=1e-4)

# hessian-eigenthings v1.1 roadmap

Status: design doc, written 2026-05-10 after extensive use of v1.0 in the
[llm-hessian-spectra](https://github.com/noahgolmant/llm-hessian-spectra)
Muon-vs-natural-gradient methodology test on Pythia-160M.

The v1.0 release covers top-k Hessian / GGN spectra, trace, and spectral
density — the "PyHessian replacement at modern scale" story. v1.1 should
position hessian-eigenthings as the canonical tool for **comparing curvature
operators and optimizer preconditioners**, not just analyzing one operator
in isolation.

## Motivation: what v1.0 didn't support

The Muon-vs-NGD experiment (see [llm-hessian-spectra/.../muon_vs_ngd_pythia160m.md](https://github.com/noahgolmant/llm-hessian-spectra/blob/master/experiments/proof_of_concept/muon_vs_ngd_pythia160m.md))
needed three things v1.0 doesn't provide:

1. **Empirical/MC Fisher operators**. Both are PSD by construction, both are
   the operators K-FAC/SOAP/Sophia/Muon all implicitly approximate. v1.0 has
   `HessianOperator`, `GGNOperator`, `EmpiricalFisherOperator` (per-sample
   outer products) — but the latter does a different computation than what
   we needed (full-batch matvec via the operator definition, not stored
   per-sample gradient collection).

2. **A way to express optimizer preconditioners as operators**. Muon's
   `(GGᵀ)⁻¹/² ⊗ I_in`, Shampoo's `L ⊗ R`, K-FAC's `A ⊗ G` — all of these
   are linear operators on parameter space, and the analysis question is
   "do they approximate Fisher / GGN / Hessian?" v1.0 has no canonical way
   to wrap an optimizer's preconditioner as a CurvatureOperator for
   subspace-level comparison.

3. **Operator-comparison utilities**. Cosine similarity of action on a
   specific vector (`u_a ⋅ u_b`), principal angles between top-k eigenspaces,
   operator-norm of difference — all generic, all reusable.

Additionally, `GGNOperator._matvec_one_batch` OOMs on A100-80GB at LM-scale
vocab (50,304+) due to a `torch.func.jvp + functional_call + autograd.grad
create_graph=True` interaction. This is a real bug, not just a missing
feature.

## Proposed additions, ranked by leverage

### 1. `FisherOperator` family with stored per-sample gradients

**New module**: `hessian_eigenthings/operators/fisher_lowrank.py`

```python
class StoredGradientFisherOperator(CurvatureOperator):
    """Fisher-style PSD operator F̂ = Gᵀ G / N, where G is the (N, |W|)
    matrix of stored per-sample gradients of a parameter (or subset)
    over a batch of N samples.

    Rank ≤ N, so:
      matvec      O(N|W|) via Gᵀ (G v) / N
      solve_damped via Sherman-Morrison-Woodbury (one N×N solve)
      top_eigvals via singular_values(G)^2 / N (no Lanczos)
    """

    def __init__(self, G: torch.Tensor, size_hint: int, device, dtype):
        ...

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self._G.t() @ (self._G @ v) / self._N

    def solve_damped(self, v: torch.Tensor, damping: float) -> torch.Tensor:
        """(F̂ + damping·I)⁻¹ v via SMW. Cost: one (N×N) linear solve."""
        ...

    def top_eigvals(self, k: int) -> torch.Tensor:
        sv = torch.linalg.svdvals(self._G)
        return (sv * sv)[:k] / self._N

    def to_dense_within_rowspan(self) -> torch.Tensor:
        """Returns G itself, useful for downstream Krylov/SVD analyses."""
        return self._G


class EmpiricalFisherOperator(StoredGradientFisherOperator):
    """Builds G from per-sample gradients using *true* labels.
    Biased estimator of true Fisher; equals true Fisher when model fits data.
    What K-FAC, SOAP, Sophia actually use in practice."""

    @classmethod
    def from_model(cls, model, dataloader, loss_fn, *, param_filter, n_samples, device): ...


class MCFisherOperator(StoredGradientFisherOperator):
    """Builds G from per-sample gradients with labels sampled from model softmax.
    Unbiased Monte Carlo estimator of true Fisher = GGN for cross-entropy +
    softmax losses."""

    @classmethod
    def from_model(cls, model, dataloader, *, param_filter, n_samples, device,
                   sample_per_token=True): ...
```

**Why it's important**: these are the *practically deployed* curvature
operators across the post-2023 optimizer literature. Sophia (diagonal
Hutchinson F̂), K-FAC (block-Kronecker F̂), Muon (row-only Kronecker
F̂-ish), Shampoo/SOAP (factored F̂-ish) — they all approximate some flavor
of Fisher. The library should ship the operator they approximate as a
first-class citizen so users can do `cos(approx, exact)` measurements.

**Code I already wrote** in [llm-hessian-spectra/scripts/muon_ngd_methodology.py](https://github.com/noahgolmant/llm-hessian-spectra/blob/master/scripts/muon_ngd_methodology.py)
is ~80% of the library form. Mostly need to: (a) generalize from
GPTNeoXForCausalLM to arbitrary HF/torch modules, (b) handle multi-param
filter cleanly (sub-tensor of a full gradient), (c) add docstrings + tests
+ cross-validation against true Fisher on small models.

### 2. `OptimizerPreconditionerOperator` family

**New module**: `hessian_eigenthings/operators/preconditioners.py`

```python
class MuonPreconditionerOperator(CurvatureOperator):
    """Muon's per-matrix preconditioner, lifted to flattened-W space as
    (g_W g_Wᵀ)⁻¹/² ⊗ I_in, where g_W is the batch-mean gradient.

    Provided so users can call `compare_subspaces(MuonOp, FisherOp, k)` and
    measure how well Muon's structured approximation tracks Fisher's
    eigenstructure.
    """

    def __init__(self, batch_grad: torch.Tensor, *, ns_iters: int = 5):
        ...

    def matvec(self, v): ...  # applies Muon's preconditioner


class ShampooPreconditionerOperator(CurvatureOperator):
    """Per-layer L ⊗ R preconditioner from Shampoo. L, R built from gradient
    second-moment statistics."""
    ...


class KFACPreconditionerOperator(CurvatureOperator):
    """K-FAC's per-layer A ⊗ G structured approximation, A from layer-input
    activations and G from output-pre-activation gradients (captured via
    forward/backward hooks)."""
    ...
```

**Why**: this is the cleanest way to make hessian-eigenthings the "central
hub for second-order analysis." Right now the library has operators for the
*targets* of approximation; v1.1 should also have operators for the
*approximations themselves*, so the comparison is symmetric.

### 3. Operator-comparison utilities

**New module**: `hessian_eigenthings/compare.py`

```python
def cosine_on_vector(op_a, op_b, v, *, damping_a=0.0, damping_b=0.0):
    """cos((op_a + da*I)⁻¹ v, (op_b + db*I)⁻¹ v) — the per-vector test."""

def principal_angles(op_a, op_b, *, k=10, max_iter=30):
    """Top-k eigenspace principal angles. Uses Lanczos on each to get top-k
    eigvecs, then SVD on V_aᵀ V_b → singular values are cos(angles)."""

def operator_relative_error(op_a, op_b, *, n_probes=10, seed=0):
    """E_v[||(op_a - op_b) v|| / ||op_a v||] via Hutchinson over random v.
    Useful for global "how different are these operators" without committing
    to a specific gradient direction."""

def project_to_rowspan(v, op_with_rowspan):
    """For low-rank Fisher-like operators that expose to_dense_within_rowspan(),
    project v onto the row span. Useful for the row-span-projected NGD
    comparison (Hypothesis E in the Muon-vs-NGD writeup)."""
```

**Why**: every "is X a good approximation to Y" question reduces to one of
these. Right now users have to write the comparison inline. With these in
the library, "Muon vs Fisher" becomes a 3-line script.

### 4. Fix `GGNOperator` OOM at LM-scale vocab

`hessian_eigenthings/operators/ggn.py:97-125` does:
```
jvp_result = torch.func.jvp(model_call, (param_dict,), (v_dict,))
...
grad_loss = torch.autograd.grad(loss, output_leaf, create_graph=True)[0]
h_loss_jvp = torch.autograd.grad(grad_loss, output_leaf, grad_outputs=jvp_out)[0]
...
_, vjp_fn = torch.func.vjp(model_call, param_dict)
```

This OOMs on A100-80GB for Pythia-160M (vocab=50,304) at any batch ≥ 1.
Smokes showed peak ~78 GB allocated for what should be a ~15 GB workload.
The `torch.func.jvp + functional_call` path with `requires_grad=True` on
all params seems to instantiate a tangent infrastructure proportional to
total params, not just to params we're filtering to. Even setting
`requires_grad=False` on non-target params didn't bound it.

**Likely fix**: replace the `autograd.grad(..., create_graph=True)` step
with the analytical cross-entropy Hessian (for LM-style losses):

```
H_psi @ Jv = p * Jv - p * (p · Jv)        where p = softmax(logits)
```

This kills the second-derivative graph entirely and reduces matvec memory
to one forward graph + the analytical computation. **Code skeleton already
in `llm-hessian-spectra/scripts/muon_ngd_methodology.py:InlineGGNOperator`**
(replaced by the Fisher path in our specific experiment because
`torch.func.jvp` itself OOMed even without the create_graph step — that
may have been a different issue with eager attention, since `sdpa` doesn't
have forward AD).

Alternative path: replace `torch.func.jvp` with finite-difference JVP:
```
Jv ≈ (model(W + εv) - model(W - εv)) / (2ε)        # 2 forwards, no graph
```
4 model passes per matvec instead of 2, but provably memory-bounded and
torch.func-free. Probably the right default for `method="finite_difference"`
on GGNOperator.

Either fix is ~50-80 LOC + tests. Should be paired with a CI test that
runs a Pythia-160M GGN matvec to prevent regression.

**Status (2026-05-11)**: implemented on `feature/ggn-oom-fix` branch, tests CPU-green, awaiting GPU validation.

### 5. Per-matrix-parameter ParamFilter convenience

Common pattern: "treat one weight matrix as the parameter of interest, all
others as fixed." Currently requires `match_names("model.layers.6.mlp.dense_h_to_4h.weight")`
which is verbose. Add:

```python
from hessian_eigenthings.param_utils import single_param

filt = single_param("layers.6.mlp.dense_h_to_4h.weight")
filt = single_param("layers.6.*.weight")  # all matrix params in layer 6
```

Also useful: `match_module_type(nn.Linear)` for "all linear layer weights."

### 6. CurvatureOperator base: optional `solve_damped`

The default Fisher / structured-preconditioner operators have **closed-form
damped solves** (SMW for low-rank Fisher, eigendecomposition for K-FAC's
A⊗G). The base class should have an optional `solve_damped(v, damping)`
method that defaults to CG-via-matvec but lets operators override with
faster paths.

```python
class CurvatureOperator(ABC):
    @abstractmethod
    def matvec(self, v): ...

    def solve_damped(self, v: torch.Tensor, damping: float, *, tol=1e-3,
                     max_iter=50) -> torch.Tensor:
        """(op + damping·I)⁻¹ v. Default: CG. Override for closed-form."""
        return _cg_solve(lambda u: self.matvec(u) + damping * u, v, tol, max_iter)
```

### 7. Polar decomposition utility

For Hypothesis B in the Muon-vs-NGD analysis (and elsewhere), exact polar
decomposition `g_W = U Σ Vᵀ → U Vᵀ` is the *target* that Newton-Schulz
approximates. Useful as a CurvatureOperator-style baseline:

```python
def polar_decomposition(g: torch.Tensor) -> torch.Tensor:
    """Exact polar decomposition U V^T from SVD. The NS-iteration limit."""
    u, _, vh = torch.linalg.svd(g, full_matrices=False)
    return u @ vh
```

Used in `MuonPreconditionerOperator` to allow `ns_iters=∞` mode (i.e., the
exact polar) for separating NS-truncation artifacts from operator-structure
mismatch.

## Sequence

If we ship items 1-3 + the GGN fix (item 4), hessian-eigenthings becomes
the canonical tool for the Muon/Shampoo/K-FAC/SOAP comparison work. That's
the smallest publishable v1.1 surface.

1. **First PR (1 week)**: ports `EmpiricalFisherOperator` /
   `MCFisherOperator` / `StoredGradientFisherOperator` from
   llm-hessian-spectra + SMW solve + top-eigvals via SVD. Adds 6 tests:
   - SMW solve matches dense linalg.solve on small model
   - SVD-via-G top eigvals match Lanczos top eigvals
   - MC Fisher cross-validates to true Fisher on a toy logistic regression
     with known analytical Fisher
   - Empirical vs MC convergence behavior on a converged model (should
     coincide) vs non-converged (should differ)
   - Memory bounded at LM scale (regression test)
   - Closed-form damped solve overrides CG correctly via base class
2. **Second PR (1 week)**: `OptimizerPreconditionerOperator` family
   (Muon, Shampoo, K-FAC). Each is a CurvatureOperator subclass whose
   `matvec` applies the preconditioner. Adds cross-library tests against
   pytorch-muon / distributed-shampoo reference implementations.
3. **Third PR (3-5 days)**: `compare.py` utilities. Adds tests showing the
   pairwise comparison metrics (cosine, principal angles, operator
   relative error) on known-equal operators (cos=1, angle=0) and known-
   orthogonal operators (cos=0).
4. **Fourth PR (1 week)**: GGN OOM fix via analytical CE Hessian + finite-
   difference JVP. Includes a CI test that runs `GGN_op.matvec(v)` on
   Pythia-160M at full vocab to prevent regression.

Total: ~3-4 weeks engineering across 4 reviewed PRs. Ships as v1.1.

## What this enables

After v1.1, the Muon-vs-NGD experiment becomes:

```python
from hessian_eigenthings import (
    EmpiricalFisherOperator, MCFisherOperator,
    MuonPreconditionerOperator,
    cosine_on_vector, project_to_rowspan,
)
from hessian_eigenthings.param_utils import single_param

# Setup (replaces 50+ lines of inline collection):
emp = EmpiricalFisherOperator.from_model(
    model, dataloader, hf_lm_loss(),
    param_filter=single_param("layers.6.mlp.dense_h_to_4h.weight"),
    n_samples=64,
)
mc = MCFisherOperator.from_model(model, dataloader,
    param_filter=single_param("layers.6.mlp.dense_h_to_4h.weight"),
    n_samples=64)
muon = MuonPreconditionerOperator(emp.batch_mean_gradient(), ns_iters=5)

# The methodology question, one line each:
cos_grad_emp = cosine_on_vector(muon, emp, emp.batch_mean_gradient(), damping_b=0.01)
cos_grad_mc  = cosine_on_vector(muon, mc,  mc.batch_mean_gradient(),  damping_b=0.01)
angles_emp   = principal_angles(muon, emp, k=10)
```

That's the "library as proper tool" target.

## What this doesn't include

For honesty:
- FSDP-compatible Fisher operators. Per the v1.0 plan, FSDP is a v1.1
  goal *for HessianOperator*, but extending to per-sample-gradient
  collection under FSDP is non-trivial (requires `no_sync` + per-rank
  gradient reductions to recover per-sample). Defer to v1.2 unless
  there's specific demand.
- Block-Kronecker variants of K-FAC (Martens-Grosse, EKFAC, Eschenhagen).
  Item 2 ships *vanilla* K-FAC; the variants are research-paper-specific
  and out of scope for v1.1.
- True Fisher via natural gradient solve in deep curvature (i.e., Krylov
  methods on the full GGN/Fisher when n_W >> N). Already supported via
  v1.0's Lanczos; we just need to thread it through the new operator
  family.

## Reference: where the work came from

This roadmap is informed by:
- `llm-hessian-spectra/scripts/muon_ngd_methodology.py` — the
  `FisherOperator` + `collect_per_sample_grads` implementation that
  should be ported upstream.
- `llm-hessian-spectra/experiments/proof_of_concept/muon_vs_ngd_pythia160m.md`
  — concrete numerical evidence that the analysis path is useful (240-cell
  trajectory with clean findings).
- `llm-hessian-spectra/scripts/analyze_muon_ngd.py` — the plotting code
  could become a `hessian-eigenthings` example notebook.

# Fused CE Hessian-vector product

When you analyze a HuggingFace causal LM with `GGNOperator` and
`hf_lm_loss_of_output()`, the library uses a *fused* kernel for the
core cross-entropy Hessian-vector product (CE HVP) instead of relying
on autograd to differentiate the loss twice. This page explains why,
what the kernel does, and what trade-offs each backend has.

## Why fuse the CE HVP

For a softmax + mean-reduced cross-entropy loss, the loss-Hessian
applied to a tangent `u` has a closed form. Per non-ignored position,
with `p = softmax(logits)` and `n` non-ignored positions:

$$H_{\text{loss}} \, u \;=\; \frac{p \odot u \;-\; p \,(p \cdot u)}{n}$$

This is the only place in a `GGNOperator` matvec where the
`(batch × seq × vocab)` tensor appears. With `V = 50,304` and
`(B, T) = (64, 256)`, the unfused version materializes several
`(N, V) = (16384, 50304)` float32 tensors — about 3.3 GB *per copy*,
and a double-backward over softmax allocates 5–6 of them. That's the
~20 GB peak you'd see from naïve autograd, and it dominates the wall
time of a Lanczos iteration on real LLMs.

The fused kernel turns those `O(N · V)` intermediates into a single
output buffer plus an online-softmax reduction, eliminating the
softmax-graph rebuild step entirely.

## Backends

`hf_lm_loss_of_output(fused=...)` selects which backend computes the
fused CE HVP:

| backend       | how                                                              | when to pick                                                |
| ---           | ---                                                              | ---                                                         |
| `"auto"` *(default)* | Triton if the inputs are on CUDA and `triton` imports; else `torch.compile`. | The right answer almost always. Auto-falls-back on CPU.    |
| `"triton"`    | CUDA Triton kernel using an online-softmax reduction. | When you specifically want the Triton path and inputs are on CUDA. |
| `"compile"`   | `torch.compile`-fused; Inductor folds softmax + elementwise + reduction into a small number of kernels and eliminates the `(N, V)` intermediates. Works on CPU, CUDA, and MPS. | When Triton isn't available, or for non-CUDA hardware.     |
| `"eager"`     | Plain PyTorch reference; full `(N, V)` materialization.            | For debugging — or for tiny vocab where the fused path costs more than it saves. |

The closed-form HVP itself (independent of backend) lets `GGNOperator`
skip the autograd double-backward through the loss entirely — it walks
the model graph once for `J @ u` (a JVP), then applies this closed-form
`H_loss @ J u`, then walks the graph backward once for `J^T (H_loss J u)`
(a VJP). One forward, one backward, no second-order graph through
softmax.

## Measured speedups

Numbers from an A100-40GB at `B=64, T=256, V=50304, fp32`:

| backend       | wall time | peak memory | speedup | memory reduction |
| ---           | ---       | ---         | ---     | ---              |
| eager         | 48.6 ms   | 19.78 GB    | 1.00×   | 1.00×            |
| compile       | 18.0 ms   | 9.89 GB     | 2.70×   | 2.00×            |
| triton        | 14.5 ms   | 9.89 GB     | **3.35×** | **2.00×**       |

Memory reduction is the same for `compile` and `triton` because both
eliminate the `(N, V)` intermediates; Triton wins on wall time because
it can stream the softmax in one pass and avoid even the temporary
activation storage that Inductor still produces.

The benefit scales with vocab size: for shapes where the unfused eager
path OOMs (e.g. `V = 100,000`, common for Llama-3 / Qwen / Mistral),
the fused backends remain the only viable option.

## Numerical correctness

The test suite asserts cross-backend agreement to within a vocab-dependent
tolerance band:

- **fp32**: max abs deviation between `eager`, `compile`, and `triton`
  is below `1e-5` across vocab sizes 7, 100, 50,257, and 130,000.
- **bf16** / **fp16**: max abs deviation is below the dtype's own
  rounding noise — i.e. the fused kernels are no less accurate than
  the eager reference at the same dtype.
- **Analytical ground truth**: the closed-form HVP itself is checked
  against the full Hessian (built via `torch.autograd.functional.hessian`)
  on a small model. All three backends match the analytical Hessian to
  within `1e-5`.

These checks run on every commit in CI, and on an A100 for changes
touching the fused CE path.

## When the fused path is *not* what you want

- **Tiny vocabularies (`V ≲ 1000`)**: the fused-kernel launch and
  reduction overhead can outweigh its memory savings. Pass
  `fused="eager"` for these — though you usually don't need
  `GGNOperator` for a tiny-vocab model in the first place.
- **Debugging a numerical anomaly**: force `fused="eager"` to remove
  Triton/compile from the picture and confirm the bug isn't kernel-side.
- **Verifying a custom `loss_of_output_fn`**: when you write your own
  HVP closure for a non-CE loss, compare against the eager reference
  on a tiny example before switching to fused for performance.

## Reference

The kernel and closed-form HVP live in
[`hessian_eigenthings/loss_fns/_fused_ce_hvp.py`](https://github.com/noahgolmant/pytorch-hessian-eigenthings/blob/master/hessian_eigenthings/loss_fns/_fused_ce_hvp.py)
and the user-facing `loss_of_output_fn` factory in
[`hessian_eigenthings/loss_fns/huggingface.py`](https://github.com/noahgolmant/pytorch-hessian-eigenthings/blob/master/hessian_eigenthings/loss_fns/huggingface.py).

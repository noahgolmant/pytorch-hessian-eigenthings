# pytorch-hessian-eigenthings

The `hessian-eigenthings` module provides an efficient (and scalable!) way to compute the eigendecomposition of the Hessian, along with other curvature matrices like the Generalized Gauss-Newton and empirical Fisher, for an arbitrary PyTorch model. It uses PyTorch's Hessian-vector product and your choice of (a) the Lanczos method or (b) stochastic power iteration with deflation to compute the top eigenvalues and eigenvectors. There's also Hutch++ for trace estimation and Stochastic Lanczos Quadrature for the spectral density.

> **v1.0.0a1**: alpha release. The original `0.x` API has been removed; pin `hessian-eigenthings==0.0.2` if you depend on it.

## Why use this?

The eigenvalues and eigenvectors of the Hessian have been implicated in many generalization properties of neural networks. People hypothesize that "flat minima" with lower eigenvalues generalize better, that the Hessians of large models are very low-rank, that certain optimizers lead to flatter or sharper minima, and so on. But computing and storing the full Hessian requires memory that is quadratic in the number of parameters, which is infeasible for anything but toy models.

Iterative methods like Lanczos and power iteration can find the eigendecomposition of arbitrary linear operators given just a matrix-vector multiplication function. The Hessian-vector product (HVP) is exactly that, and it can be computed with linear memory by taking the derivative of the inner product between the gradient and the vector $v$. This library combines the HVP with iterative algorithms to compute the eigendecomposition without the quadratic memory bottleneck.

You can use this for HVP computation, the more general iterative algorithms on any linear operator you provide, or the conjunction of the two for Hessian spectrum analysis on real models, including HuggingFace and TransformerLens transformers.

## Installation

```bash
pip install hessian-eigenthings
```

If you want the HuggingFace or TransformerLens helpers:

```bash
pip install "hessian-eigenthings[transformers,transformer-lens]"
```

## Usage

The pattern is: build a `CurvatureOperator` from your model, run any algorithm against it. Most of the time you want the Hessian:

```python
import torch
from torch import nn

from hessian_eigenthings.algorithms import lanczos, trace, spectral_density
from hessian_eigenthings.loss_fns import supervised_loss
from hessian_eigenthings.operators import HessianOperator

model = nn.Sequential(nn.Linear(20, 32), nn.Tanh(), nn.Linear(32, 1)).to(torch.float64)
x, y = torch.randn(128, 20, dtype=torch.float64), torch.randn(128, 1, dtype=torch.float64)
data = [(x[i:i+32], y[i:i+32]) for i in range(0, 128, 32)]

H = HessianOperator(model, data, supervised_loss(nn.functional.mse_loss))

eig = lanczos(H, k=5, seed=0)
print("top eigenvalues:", eig.eigenvalues)

t = trace(H, num_matvecs=99, seed=0)        # Hutch++ by default
print(f"trace ≈ {t.estimate:.3f} ± {t.stderr:.3f}")

density = spectral_density(H, num_runs=8, lanczos_steps=40, seed=0)
# density.density is a probability density over density.grid; integrates to ~1
```

If you'd rather have the GGN (PSD by construction, often what people mean by "the Hessian" on classification losses), use `GGNOperator`. For per-sample-gradient outer products, `EmpiricalFisherOperator`. They all share the same interface, so the algorithms above work on any of them.

There's also a finite-difference HVP path (`HessianOperator(method="finite_difference")`) for cases where double-backward is impractical. It's useful with FSDP and similar setups where the autograd graph gets detached during gradient communication.

You can restrict any operator to a subset of parameters with `param_filter`, e.g. `param_filter=match_names("blocks.*.attn.*")` to compute the Hessian of just the attention layers. Useful for per-block sharpness studies.

## Examples

The [`examples/`](examples/) directory has three runnable scripts:

- `supervised_mlp.py`: top-k eigenvalues, Hutch++ trace, and SLQ density on a small synthetic-data MLP. No downloads, runs in seconds.
- `huggingface_tiny_gpt2.py`: full and attention-only Hessian of a HuggingFace causal LM (`sshleifer/tiny-gpt2`).
- `transformer_lens_attention_only.py`: per-block (attention vs MLP) Hessian on a TransformerLens model.

## Documentation

Full docs at <https://noahgolmant.github.io/pytorch-hessian-eigenthings>. There are concept pages explaining the math behind each algorithm, how-to recipes for common workflows, and an auto-generated API reference. The [GGN vs Fisher vs Hessian](https://noahgolmant.github.io/pytorch-hessian-eigenthings/concepts/ggn-vs-fisher-vs-hessian/) page is worth reading before deciding which operator to instantiate. They're easy to conflate.

## Working on the library

This project uses [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/noahgolmant/pytorch-hessian-eigenthings
cd pytorch-hessian-eigenthings
uv sync --group dev --group docs --extra transformers --extra transformer-lens --extra curvlinops
uv run pytest
uv run mkdocs serve
```

## Citing this work

If you find this repo useful and would like to cite it in a publication (as [others](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=18039594054930134223) have done, thank you!), here is a BibTeX entry:

```bibtex
@misc{hessian-eigenthings,
    author       = {Noah Golmant and Zhewei Yao and Amir Gholami and Michael Mahoney and Joseph Gonzalez},
    title        = {pytorch-hessian-eigenthings: efficient PyTorch Hessian eigendecomposition},
    month        = oct,
    year         = 2018,
    version      = {1.0},
    url          = {https://github.com/noahgolmant/pytorch-hessian-eigenthings}
}
```

## Acknowledgements

The original 2018 implementation was written in collaboration with Zhewei Yao, Amir Gholami, Michael Mahoney, and Joseph Gonzalez at UC Berkeley's [RISELab](https://rise.cs.berkeley.edu).

The deflated power iteration routine is based on code in the [HessianFlow](https://github.com/amirgholami/HessianFlow) repository, described in: Z. Yao, A. Gholami, Q. Lei, K. Keutzer, M. Mahoney. *"Hessian-based Analysis of Large Batch Training and Robustness to Adversaries"*, NeurIPS 2018 ([arXiv:1802.08241](https://arxiv.org/abs/1802.08241)).

The accelerated stochastic power iteration is based on: C. De Sa, B. He, I. Mitliagkas, C. Ré, P. Xu. *"Accelerated Stochastic Power Iteration"*, PMLR 2017 ([arXiv:1707.02670](https://arxiv.org/abs/1707.02670)).

The v1 refresh borrows ideas from [PyHessian](https://github.com/amirgholami/PyHessian), [curvlinops](https://github.com/f-dangel/curvlinops), and [HessFormer](https://github.com/PureStrength-AI/HessFormer).

## License

MIT.

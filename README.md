# pytorch-hessian-eigenthings

[![PyPI](https://img.shields.io/pypi/v/hessian-eigenthings.svg)](https://pypi.org/project/hessian-eigenthings/)
[![Documentation](https://img.shields.io/badge/docs-noahgolmant.github.io-blue)](https://noahgolmant.github.io/pytorch-hessian-eigenthings/)
[![CI](https://github.com/noahgolmant/pytorch-hessian-eigenthings/actions/workflows/ci.yml/badge.svg)](https://github.com/noahgolmant/pytorch-hessian-eigenthings/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

The `hessian-eigenthings` module provides an efficient (and scalable!) way to compute the eigendecomposition of the Hessian, plus other curvature matrices like the Generalized Gauss-Newton and empirical Fisher, for an arbitrary PyTorch model. You get top eigenvalues and eigenvectors via Lanczos or stochastic power iteration, trace estimates via Hutch++, and the spectral density via Stochastic Lanczos Quadrature.

> **v1.0.0a1**: alpha release. The `0.x` API has been removed; pin `hessian-eigenthings==0.0.2` if you depend on it.

## Why use this?

The eigenvalues and eigenvectors of the Hessian have been implicated in many generalization properties of neural networks. People hypothesize that "flat minima" generalize better, that Hessians of large models are very low-rank, that certain optimizers lead to flatter minima, and so on. But the full Hessian costs memory quadratic in the number of parameters, infeasible for anything but toy models.

Iterative methods like Lanczos and power iteration only need a matrix-vector product. The Hessian-vector product (HVP) is exactly that, and it costs linear memory. This library combines the HVP with iterative algorithms to compute the eigendecomposition without the quadratic memory bottleneck, and works on real models including HuggingFace and TransformerLens transformers.

## Installation

```bash
pip install hessian-eigenthings
# or with HuggingFace / TransformerLens helpers:
pip install "hessian-eigenthings[transformers,transformer-lens]"
```

## Usage

Build a `CurvatureOperator` from your model, run any algorithm against it.

```python
import torch
from torch import nn

from hessian_eigenthings import (
    HessianOperator, lanczos, trace, spectral_density, supervised_loss,
)

model = nn.Sequential(nn.Linear(20, 32), nn.Tanh(), nn.Linear(32, 1)).to(torch.float64)
x, y = torch.randn(128, 20, dtype=torch.float64), torch.randn(128, 1, dtype=torch.float64)
data = [(x[i:i+32], y[i:i+32]) for i in range(0, 128, 32)]

H = HessianOperator(model, data, supervised_loss(nn.functional.mse_loss))

eig = lanczos(H, k=5, seed=0)             # top-5 eigenvalues + eigenvectors
t = trace(H, num_matvecs=99, seed=0)      # Hutch++ trace estimate
density = spectral_density(H, num_runs=8, lanczos_steps=40, seed=0)
```

If you'd rather use the GGN (PSD by construction, often what's meant by "the Hessian" on classification losses), swap in `GGNOperator`. For per-sample-gradient outer products, `EmpiricalFisherOperator`. They share the same interface so all the algorithms above work on any of them.

There's a finite-difference HVP path (`HessianOperator(method="finite_difference")`) for when double-backward is impractical, useful with FSDP and similar setups. You can restrict to a parameter subset with `param_filter=match_names("blocks.*.attn.*")` for per-block analysis.

For LM-scale work (large vocabulary), `hf_lm_loss_of_output()` auto-selects a fused CE Hessian-vector kernel: Triton on CUDA (~3.4× speedup, 2× peak-memory reduction over eager), else `torch.compile` (~2.6× speedup, 2× peak-memory reduction). Pass `fused="eager"` to force the unfused reference for debugging.

See [`examples/`](examples/) for runnable scripts on a small MLP, HuggingFace tiny-GPT2, and a TransformerLens model. Full docs at <https://noahgolmant.github.io/pytorch-hessian-eigenthings>.

## Working on the library

Uses [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/noahgolmant/pytorch-hessian-eigenthings
cd pytorch-hessian-eigenthings
uv sync --group dev --group docs --extra transformers --extra transformer-lens --extra curvlinops
uv run pytest
uv run mkdocs serve
```

## Citing this work

If you find this repo useful and would like to cite it (as [others](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=18039594054930134223) have done, thank you!):

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

The original 2018 implementation was written with Zhewei Yao, Amir Gholami, Michael Mahoney, and Joseph Gonzalez at UC Berkeley's [RISELab](https://rise.cs.berkeley.edu).

The deflated power iteration is based on code from [HessianFlow](https://github.com/amirgholami/HessianFlow) (Z. Yao, A. Gholami, Q. Lei, K. Keutzer, M. Mahoney. *"Hessian-based Analysis of Large Batch Training and Robustness to Adversaries"*, NeurIPS 2018, [arXiv:1802.08241](https://arxiv.org/abs/1802.08241)). Accelerated stochastic power iteration is from C. De Sa et al., *"Accelerated Stochastic Power Iteration"*, PMLR 2017 ([arXiv:1707.02670](https://arxiv.org/abs/1707.02670)). The v1 refresh borrows ideas from [PyHessian](https://github.com/amirgholami/PyHessian), [curvlinops](https://github.com/f-dangel/curvlinops), and [HessFormer](https://github.com/PureStrength-AI/HessFormer).

## License

MIT.

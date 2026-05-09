# pytorch-hessian-eigenthings

The `hessian-eigenthings` module provides an efficient (and scalable!) way to compute the eigendecomposition of the Hessian for an arbitrary PyTorch model. It uses PyTorch's Hessian-vector product and your choice of the Lanczos method or stochastic power iteration with deflation to compute the top eigenvalues and eigenvectors of the Hessian.

> **v1 alpha**: under active development. The previous `0.x` API has been removed; pin `hessian-eigenthings==0.0.2` if you depend on it.

## Documentation

Full documentation lives at <https://noahgolmant.github.io/pytorch-hessian-eigenthings>.

## Installation

```bash
pip install hessian-eigenthings
```

Optional extras:

```bash
pip install "hessian-eigenthings[transformers,transformer-lens]"
```

## Working on the library

This project uses [`uv`](https://docs.astral.sh/uv/) for environment management.

```bash
git clone https://github.com/noahgolmant/pytorch-hessian-eigenthings
cd pytorch-hessian-eigenthings
uv sync --group dev --group docs
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

The v1 refresh builds on ideas from [PyHessian](https://github.com/amirgholami/PyHessian), [curvlinops](https://github.com/f-dangel/curvlinops), and [HessFormer](https://github.com/PureStrength-AI/HessFormer).

## License

MIT.

# pytorch-hessian-eigenthings

The `hessian-eigenthings` module provides an efficient way to compute the eigendecomposition of the Hessian for an arbitrary PyTorch model. It uses PyTorch's Hessian-vector product and stochastic power iteration with deflation to compute the top eigenvalues and eigenvectors of the Hessian.

## Installation

For now, you have to install from this repo. It's a tiny thing so why put it on pypi.

`pip install --upgrade git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master#egg=hessian-eigenthings`

## Usage

The main function you're probably interested in is `compute_hessian_eigenthings`.
Sample usage is like so:

```
import torch
from hessian_eigenthings import compute_hessian_eigenthings

model = ResNet18()
dataloader = ...
loss = torch.nn.functional.cross_entropy

num_eigenthings = 20  # compute top 20 eigenvalues/eigenvectors

eigenvals, eigenvecs = compute_hessian_eigenthings(model, dataloader,
                                                   loss, num_eigenthings)
```

This also includes a more general power iteration with deflation implementation in `power_iter.py`.

## Acknowledgements

This code was written in collaboration with Zhewei Yao, Amir Gholami, and Michael Mahoney in UC Berkeley's [RISELab](https://rise.cs.berkeley.edu).

The deflated power iteration routine is based on code in the [HessianFlow](https://github.com/amirgholami/HessianFlow) repository recently described in the following paper: Z. Yao, A. Gholami, Q. Lei, K. Keutzer, M. Mahoney. "Hessian-based Analysis of Large Batch Training and Robustness to Adversaries", *NIPS'18* ([arXiv:1802.08241](https://arxiv.org/abs/1802.08241))

Stochastic power iteration with acceleration is based on the following paper: C. De Sa, B. He, I. Mitliagkas, C. Ré, P. Xu. "Accelerated Stochastic Power Iteration", *PMLR-21* ([arXiv:1707.02670](https://arxiv.org/abs/1707.02670))



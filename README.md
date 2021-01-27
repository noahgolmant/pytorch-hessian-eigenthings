# pytorch-hessian-eigenthings

The `hessian-eigenthings` module provides an efficient (and scalable!) way to compute the eigendecomposition of the Hessian for an arbitrary PyTorch model. It uses PyTorch's Hessian-vector product and your choice of (a) the Lanczos method or (b) stochastic power iteration with deflation in order to compute the top eigenvalues and eigenvectors of the Hessian.

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

This also includes a more general power iteration with deflation implementation in `power_iter.py`. `lanczos.py` calls a [`scipy` hook](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.eigsh.html) to a battle-tested ARPACK implementation.

## Example file

The example file in `example/main.py` utilizes [`skeletor`](https://github.com/noahgolmant/skeletor) version `0.1.4` for experiment orchestration, which can be installed via `pip install skeletor-ml`, but the rest of this library does not depend on it. You can execute the example via a command like `python example/main.py  --mode=power_iter <experimentname>`, where `<experimentname>` is a useful name like `resnet18_cifar10`. But it may just be easier to use a simpler codebase to instantiate PyTorch models and dataloaders (such as [`pytorch-cifar`](https://github.com/kuangliu/pytorch-cifar)).

## Citing this work
If you find this repo useful and would like to cite it in a publication (as [others](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=18039594054930134223) have done, thank you!), here is a BibTeX entry:

    @misc{hessian-eigenthings,
        author       = {Noah Golmant, Zhewei Yao, Amir Gholami, Michael Mahoney, Joseph Gonzalez},
        title        = {pytorch-hessian-eigenthings: efficient PyTorch Hessian eigendecomposition},
        month        = oct,
        year         = 2018,
        version      = {1.0},
        url          = {https://github.com/noahgolmant/pytorch-hessian-eigenthings}
        }


## Acknowledgements

This code was written in collaboration with Zhewei Yao, Amir Gholami, Michael Mahoney, and Joseph Gonzalez in UC Berkeley's [RISELab](https://rise.cs.berkeley.edu).

The deflated power iteration routine is based on code in the [HessianFlow](https://github.com/amirgholami/HessianFlow) repository recently described in the following paper: Z. Yao, A. Gholami, Q. Lei, K. Keutzer, M. Mahoney. "Hessian-based Analysis of Large Batch Training and Robustness to Adversaries", *NIPS'18* ([arXiv:1802.08241](https://arxiv.org/abs/1802.08241))

Stochastic power iteration with acceleration is based on the following paper: C. De Sa, B. He, I. Mitliagkas, C. Ré, P. Xu. "Accelerated Stochastic Power Iteration", *PMLR-21* ([arXiv:1707.02670](https://arxiv.org/abs/1707.02670))

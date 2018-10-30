# pytorch-hessian-eigenthings

The `hessian-eigenthings` module provides an efficient way to compute the eigendecomposition of the Hessian for an arbitrary PyTorch model. It uses PyTorch's finite difference Hessian-vector product and subsampled power iteration with deflation to compute the top eigenvalues and eigenvectors of the Hessian.

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

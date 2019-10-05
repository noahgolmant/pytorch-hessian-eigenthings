import numpy as np
import torch
from hessian_eigenthings import compute_hessian_eigenthings
from utils import plot_eigenval_estimates, plot_eigenvec_errors

from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
from variance_tests import get_full_hessian

import scipy


def test_principal_eigenvec(model, criterion, x, y, ntrials):
    loss = criterion(model(x), y)
    loss_grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    print("computing real hessian")
    real_hessian = get_full_hessian(loss_grad, model)
    #
    real_hessian += 1e-4 * np.eye(len(real_hessian))

    samples = [(x_i, y_i) for x_i, y_i in zip(x, y)]
    # full dataset
    dataloader = DataLoader(samples, batch_size=len(x))

    print("computing numpy principal eigenvec of hessian")
    num_params = len(real_hessian)
    real_eigenvals, real_eigenvecs = scipy.linalg.eigh(
        real_hessian, eigvals=(num_params - 1, num_params - 1)
    )
    real_eigenvec, real_eigenval = real_eigenvecs[0], real_eigenvals[0]

    eigenvals = []
    eigenvecs = []

    nparams = len(real_hessian)

    # for _ in range(ntrials):
    est_eigenvals, est_eigenvecs = compute_hessian_eigenthings(
        model,
        dataloader,
        criterion,
        num_eigenthings=1,
        power_iter_steps=10,
        power_iter_err_threshold=1e-5,
        momentum=0,
        use_gpu=False,
    )
    est_eigenval, est_eigenvec = est_eigenvecs[0], est_eigenvals[0]

    # compute cosine similarity
    print(real_eigenvec, est_eigenvec)

    dotted = np.dot(real_eigenvec, est_eigenvec)
    if dotted == 0.0:
        score = 1.0  # both in nullspace... nice...
    else:
        norm = scipy.linalg.norm(real_eigenvec) * scipy.linalg.norm(est_eigenvec)
        score = abs(dotted / norm)
    print(score)


if __name__ == "__main__":
    indim = 10
    outdim = 10
    nsamples = 10
    ntrials = 1
    bs = 10
    hidden = 1000

    model = nn.Sequential(
        nn.Linear(indim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, outdim),
        nn.ReLU(inplace=True),
    )
    criterion = torch.nn.MSELoss()

    x = torch.rand((nsamples, indim))
    y = torch.rand((nsamples, outdim))

    test_principal_eigenvec(model, criterion, x, y, ntrials)

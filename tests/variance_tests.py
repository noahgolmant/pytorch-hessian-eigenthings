"""
This test looks at the variance of eigenvalue/eigenvector estimates
    (1) Full dataset should have deterministic results
    (2) Compute variance of repeated trials and the effect of averaging, error
        relative to full dataset
    (3) Compute variance of full power iteration on a fixed mini-batch (vs.
        varying the mini-batch at each step) compared to full dataset
"""

import numpy as np
import torch
from hessian_eigenthings import compute_hessian_eigenthings
from utils import plot_eigenval_estimates, plot_eigenvec_errors

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def get_full_hessian(loss_grad, model):
    # from https://discuss.pytorch.org/t/compute-the-hessian-matrix-of-a-network/15270/3
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):
        grad2rd = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian.cpu().data.numpy()


def test_full_hessian(model, criterion, x, y, ntrials=10):
    loss = criterion(model(x), y)
    loss_grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    real_hessian = get_full_hessian(loss_grad, model)

    samples = [(x_i, y_i) for x_i, y_i in zip(x, y)]
    # full dataset
    dataloader = DataLoader(samples, batch_size=len(x))

    eigenvals = []
    eigenvecs = []

    nparams = len(real_hessian)

    for _ in range(ntrials):
        est_eigenvals, est_eigenvecs = compute_hessian_eigenthings(
            model,
            dataloader,
            criterion,
            num_eigenthings=nparams,
            power_iter_steps=10,
            power_iter_err_threshold=1e-5,
            momentum=0,
            use_gpu=False)
        est_eigenvals = np.array(est_eigenvals)
        est_eigenvecs = np.array([t.numpy() for t in est_eigenvecs])

        est_inds = np.argsort(est_eigenvals)
        est_eigenvals = np.array(est_eigenvals)[est_inds][::-1]
        est_eigenvecs = np.array(est_eigenvecs)[est_inds][::-1]

        eigenvals.append(est_eigenvals)
        eigenvecs.append(est_eigenvecs)

    eigenvals = np.array(eigenvals)
    eigenvecs = np.array(eigenvecs)

    real_eigenvals, real_eigenvecs = np.linalg.eig(real_hessian)
    real_inds = np.argsort(real_eigenvals)
    real_eigenvals = np.array(real_eigenvals)[real_inds][::-1]
    real_eigenvecs = np.array(real_eigenvecs)[real_inds][::-1]

    # Plot eigenvalue error
    plt.suptitle('Hessian eigendecomposition errors: %d trials' % ntrials)
    plt.subplot(1, 2, 1)
    plt.title('Eigenvalues')
    plt.plot(list(range(nparams)), real_eigenvals, label='True Eigenvals')
    plot_eigenval_estimates(eigenvals, label='Estimates')
    plt.legend()
    # Plot eigenvector L2 norm error
    plt.subplot(1, 2, 2)
    plt.title('Eigenvector cosine simliarity')
    plot_eigenvec_errors(real_eigenvecs, eigenvecs, label='Estimates')
    plt.legend()
    plt.savefig('full.png')
    plt.clf()
    return real_hessian


def test_stochastic_hessian(model, criterion, real_hessian, x, y, bs=10, ntrials=10):
    samples = [(x_i, y_i) for x_i, y_i in zip(x, y)]
    # full dataset
    dataloader = DataLoader(samples, batch_size=bs)

    eigenvals = []
    eigenvecs = []

    nparams = len(real_hessian)

    for _ in range(ntrials):
        est_eigenvals, est_eigenvecs = compute_hessian_eigenthings(
            model,
            dataloader,
            criterion,
            num_eigenthings=nparams,
            power_iter_steps=10,
            power_iter_err_threshold=1e-5,
            momentum=0,
            use_gpu=False)
        est_eigenvals = np.array(est_eigenvals)
        est_eigenvecs = np.array([t.numpy() for t in est_eigenvecs])

        est_inds = np.argsort(est_eigenvals)
        est_eigenvals = np.array(est_eigenvals)[est_inds][::-1]
        est_eigenvecs = np.array(est_eigenvecs)[est_inds][::-1]

        eigenvals.append(est_eigenvals)
        eigenvecs.append(est_eigenvecs)

    eigenvals = np.array(eigenvals)
    eigenvecs = np.array(eigenvecs)

    real_eigenvals, real_eigenvecs = np.linalg.eig(real_hessian)
    real_inds = np.argsort(real_eigenvals)
    real_eigenvals = np.array(real_eigenvals)[real_inds][::-1]
    real_eigenvecs = np.array(real_eigenvecs)[real_inds][::-1]

    # Plot eigenvalue error
    plt.suptitle('Stochastic Hessian eigendecomposition errors: %d trials' % ntrials)
    plt.subplot(1, 2, 1)
    plt.title('Eigenvalues')
    plt.plot(list(range(nparams)), real_eigenvals, label='True Eigenvals')
    plot_eigenval_estimates(eigenvals, label='Estimates')
    plt.legend()
    # Plot eigenvector L2 norm error
    plt.subplot(1, 2, 2)
    plt.title('Eigenvector cosine simliarity')
    plot_eigenvec_errors(real_eigenvecs, eigenvecs, label='Estimates')
    plt.legend()
    plt.savefig('stochastic.png')
    plt.clf()


def test_fixed_mini(model, criterion, real_hessian, x, y, bs=10, ntrials=10):
    x = x[:bs]
    y = y[:bs]


    samples = [(x_i, y_i) for x_i, y_i in zip(x, y)]
    # full dataset
    dataloader = DataLoader(samples, batch_size=len(x))

    eigenvals = []
    eigenvecs = []

    nparams = len(real_hessian)

    for _ in range(ntrials):
        est_eigenvals, est_eigenvecs = compute_hessian_eigenthings(
            model,
            dataloader,
            criterion,
            num_eigenthings=nparams,
            mode='lanczos',
            power_iter_steps=10,
            power_iter_err_threshold=1e-5,
            momentum=0,
            use_gpu=False)
        est_eigenvals = np.array(est_eigenvals)
        est_eigenvecs = np.array([t.numpy() for t in est_eigenvecs])

        est_inds = np.argsort(est_eigenvals)
        est_eigenvals = np.array(est_eigenvals)[est_inds][::-1]
        est_eigenvecs = np.array(est_eigenvecs)[est_inds][::-1]

        eigenvals.append(est_eigenvals)
        eigenvecs.append(est_eigenvecs)

    eigenvals = np.array(eigenvals)
    eigenvecs = np.array(eigenvecs)

    real_eigenvals, real_eigenvecs = np.linalg.eig(real_hessian)
    real_inds = np.argsort(real_eigenvals)
    real_eigenvals = np.array(real_eigenvals)[real_inds][::-1]
    real_eigenvecs = np.array(real_eigenvecs)[real_inds][::-1]

    # Plot eigenvalue error
    plt.suptitle('Fixed mini-batch Hessian eigendecomposition errors: %d trials' % ntrials)
    plt.subplot(1, 2, 1)
    plt.title('Eigenvalues')
    plt.plot(list(range(nparams)), real_eigenvals, label='True Eigenvals')
    plot_eigenval_estimates(eigenvals, label='Estimates')
    plt.legend()
    # Plot eigenvector L2 norm error
    plt.subplot(1, 2, 2)
    plt.title('Eigenvector cosine simliarity')
    plot_eigenvec_errors(real_eigenvecs, eigenvecs, label='Estimates')
    plt.legend()
    plt.savefig('fixed.png')


if __name__ == '__main__':
    indim = 100
    outdim = 1
    nsamples = 100
    ntrials = 10
    bs = 10

    model = torch.nn.Linear(indim, outdim)
    criterion = torch.nn.MSELoss()

    x = torch.rand((nsamples, indim))
    y = torch.rand((nsamples, outdim))

    hessian = test_full_hessian(model, criterion, x, y, ntrials=ntrials)
    test_stochastic_hessian(model, criterion, hessian, x, y, bs=bs, ntrials=ntrials)
    # test_fixed_mini(model, criterion, hessian, x, y, bs=bs, ntrials=ntrials)

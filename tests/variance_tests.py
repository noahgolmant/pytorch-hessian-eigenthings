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

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def compute_eigenvec_cos_similarity(actual, estimated):
    scores = []
    for estimate in estimated:
        score = np.abs(np.dot(actual, estimate))
        scores.append(score)
    return scores


def plot_eigenval_estimates(estimates, label):
    """
    estimates = 2D array (num_trials x num_eigenvalues)

    x-axis = eigenvalue index
    y-axis = eigenvalue estimate
    """
    if len(estimates.shape) == 1:
        var = np.zeros_like(estimates)
    else:
        var = np.var(estimates, axis=0)
    y = np.mean(estimates, axis=0)
    x = list(range(len(y)))
    plt.errorbar(x, y, np.sqrt(var), marker='^', label=label)


def plot_eigenvec_errors(true, estimates, label):
    """
    plots error for all eigenvector estimates in L2 norm
    estimates = (num_trials x num_eigenvalues x num_params)
    true = (num_eigenvalues x num_params)
    """
    diffs = []
    num_eigenvals = true.shape[0]
    for i in range(num_eigenvals):
        cur_estimates = estimates[:, i, :]
        cur_eigenvec = true[i]
        diff = compute_eigenvec_cos_similarity(cur_eigenvec, cur_estimates)
        diffs.append(diff)
    diffs = np.array(diffs).T
    var = np.var(diffs, axis=0)
    y = np.mean(diffs, axis=0)
    x = list(range(len(y)))

    plt.errorbar(x, y, np.sqrt(var), marker='^', label=label)


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
            power_iter_steps=100,
            power_iter_err_threshold=1e-5,
            momentum=0,
            use_gpu=False)
        est_eigenvals = np.array(est_eigenvals)
        est_eigenvecs = np.array([t.numpy() for t in est_eigenvecs])

        eigenvals.append(est_eigenvals)
        eigenvecs.append(est_eigenvecs)

    eigenvals = np.array(eigenvals)
    eigenvecs = np.array(eigenvecs)

    real_eigenvals, real_eigenvecs = np.linalg.eig(real_hessian)

    # Plot eigenvalue error
    plt.subplot(1, 2, 1)
    plt.title('Eigenvalue errors')
    plt.ylim(-10, 10)
    plt.plot(list(range(nparams)), real_eigenvals, label='True Eigenvals')
    plot_eigenval_estimates(eigenvals, label='%d trials' % ntrials)
    plt.legend()
    # Plot eigenvector L2 norm error
    plt.subplot(1, 2, 2)
    plt.title('Eigenvector cos simliarity')
    plot_eigenvec_errors(real_eigenvecs, eigenvecs, label='%d trials' % ntrials)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    indim = 10
    outdim = 1
    nsamples = 100
    ntrials = 50

    model = torch.nn.Linear(indim, outdim)
    criterion = torch.nn.MSELoss()

    x = torch.rand((nsamples, indim))
    y = torch.rand((nsamples, outdim))

    test_full_hessian(model, criterion, x, y, ntrials=ntrials)

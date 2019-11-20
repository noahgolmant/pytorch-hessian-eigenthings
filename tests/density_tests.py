"""
This file tests the accuracy of the power iteration methods by comparing
against np.linalg.eig results for various random matrix configurations
"""

import argparse
import numpy as np
import torch
from hessian_eigenthings.power_iter import LambdaOperator, deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from hessian_eigenthings.spectral_density import spectral_density
import matplotlib.pyplot as plt
from utils import plot_eigenval_estimates, plot_eigenvec_errors

from skeletor.utils import seed_all

parser = argparse.ArgumentParser(description="power iteration tester")

parser.add_argument(
    "--matrix_dim", type=int, default=100, help="number of rows/columns in matrix"
)
parser.add_argument(
    "--num_eigenthings", type=int, default=10, help="number of eigenvalues to compute"
)
parser.add_argument(
    "--power_iter_steps",
    default=20,
    type=int,
    help="number of steps of power iteration",
)
parser.add_argument(
    "--momentum",
    default=0,
    type=float,
    help="acceleration term for stochastic power iter",
)
parser.add_argument(
    "--num_trials", default=30, type=int, help="number of matrices per test"
)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--mode", default="power_iter", choices=["power_iter", "lanczos"])
args = parser.parse_args()

seed_all(args.seed)


def test_matrix(mat, ntrials, mode):
    """
    Tests the accuracy of deflated power iteration on the given matrix.
    It computes the average percent eigenval error and eigenvec simliartiy err
    """
    tensor = torch.from_numpy(mat).float()
    op = LambdaOperator(lambda x: torch.matmul(tensor, x), tensor.size()[:1])
    real_eigenvals, true_eigenvecs = np.linalg.eig(mat)
    real_eigenvecs = [true_eigenvecs[:, i] for i in range(len(real_eigenvals))]

    max_support = 1.0
    density, support = spectral_density(
        real_eigenvals,
        real_eigenvecs,
        min_eigenvalue_support=0,
        max_eigenvalue_support=max_support,
        num_support_points=1e5,
        sigma_squared=1e-5,
    )
    fix, ax = plt.subplots(figsize=(15, 7))

    # density /= sum(density)

    ax.plot(support, density, color="r", label="density")
    ax.hist(real_eigenvals, bins=100, range=(0, max_support), normed=True)

    # ax.plot(support, 10 * real_eigenvals, color="b", label="real")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("unnormalized density")
    ax.set_title("wishart matrix eigenvalue density estimation")
    plt.legend()

    plt.show()

    # eigenvals = []
    # eigenvecs = []
    # for _ in range(ntrials):
    # if mode == 'lanczos':
    # method = lanczos
    # else:
    # method = deflated_power_iteration
    # est_eigenvals, est_eigenvecs = method(
    # op,
    # num_eigenthings=args.num_eigenthings,
    # use_gpu=False
    # )
    # est_inds = np.argsort(est_eigenvals)
    # est_eigenvals = np.array(est_eigenvals)[est_inds][::-1]
    # est_eigenvecs = np.array(est_eigenvecs)[est_inds][::-1]

    # eigenvals.append(est_eigenvals)
    # eigenvecs.append(est_eigenvecs)

    # eigenvals = np.array(eigenvals)
    # eigenvecs = np.array(eigenvecs)

    # # truncate estimates
    # real_inds = np.argsort(real_eigenvals)
    # real_eigenvals = np.array(real_eigenvals)[real_inds][-args.num_eigenthings:][::-1]
    # real_eigenvecs = np.array(real_eigenvecs)[real_inds][-args.num_eigenthings:][::-1]

    # # Plot eigenvalue error
    # plt.suptitle('Random Matrix Eigendecomposition Errors: %d trials' % ntrials)
    # plt.subplot(1, 2, 1)
    # plt.title('Eigenvalues')
    # plt.plot(list(range(len(real_eigenvals))), real_eigenvals, label='True Eigenvals')
    # plot_eigenval_estimates(eigenvals, label='Estimates')
    # plt.legend()
    # # Plot eigenvector L2 norm error
    # plt.subplot(1, 2, 2)
    # plt.title('Eigenvector cosine simliarity')
    # plot_eigenvec_errors(real_eigenvecs, eigenvecs, label='Estimates')
    # plt.legend()
    # plt.show()


def generate_wishart(n, offset=0.0):
    """
    Generates a wishart PSD matrix with n rows/cols.
    Adds offset * I for conditioning testing.
    """
    matrix = np.random.random(size=(n, n)).astype(float)
    matrix = matrix.transpose().dot(matrix)
    matrix = matrix + offset * np.eye(n)
    return (1.0 / n) * matrix


def test_wishart():
    m = generate_wishart(args.matrix_dim)
    test_matrix(m, args.num_trials, mode=args.mode)


if __name__ == "__main__":
    test_wishart()

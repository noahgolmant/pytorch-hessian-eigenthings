"""
This file tests the accuracy of the power iteration methods by comparing
against np.linalg.eig results for various random matrix configurations
"""

import argparse
import numpy as np
import torch
from hessian_eigenthings.power_iter import LambdaOperator, deflated_power_iteration

parser = argparse.ArgumentParser(description='power iteration tester')

parser.add_argument('--matrix_dim', type=int, default=100,
                    help='number of rows/columns in matrix')
parser.add_argument('--num_eigenthings', type=int, default=100,
                    help='number of eigenvalues to compute')
parser.add_argument('--power_iter_steps', default=20, type=int,
                    help='number of steps of power iteration')
parser.add_argument('--momentum', default=0, type=float,
                    help='acceleration term for stochastic power iter')
parser.add_argument('--num_trials', default=10, type=int,
                    help='number of matrices per test')
args = parser.parse_args()


def compute_eigenval_err(true, estimated):
    """ Compute mean percent difference between eigenvalue estimates """
    percent_differences = []
    for actual, estimate in zip(true, estimated):
        err = np.abs(actual - estimate) / actual
        percent_differences.append(err)
    return np.mean(percent_differences)


def compute_eigenvec_err(true, estimated):
    """ Percent difference in similarity """
    errs = []
    for actual, estimate in zip(true, estimated):
        score = np.abs(np.dot(actual, estimate))
        err = np.abs(np.dot(actual, actual) - score) / np.dot(actual, actual)
        errs.append(err)
    return np.mean(errs)


def test_matrix(mat):
    """
    Tests the accuracy of deflated power iteration on the given matrix.
    It computes the average percent eigenval error and eigenvec simliartiy err
    """
    tensor = torch.from_numpy(mat).float()
    op = LambdaOperator(lambda x: torch.matmul(tensor, x), tensor.size()[:1])
    true_eigenvals, true_eigenvecs = np.linalg.eig(mat)
    true_eigenvecs = [true_eigenvecs[:, i] for i in range(len(true_eigenvals))]

    estimated_eigenvals, estimated_eigenvecs = deflated_power_iteration(
        op,
        num_eigenthings=args.num_eigenthings,
        power_iter_steps=args.power_iter_steps,
        momentum=args.momentum,
        use_gpu=False
    )
    estimated_eigenvecs = list(map(lambda t: t.numpy(), estimated_eigenvecs))

    # truncate estimates
    true_inds = np.argsort(true_eigenvals)
    true_eigenvals = np.array(true_eigenvals)[true_inds][-args.num_eigenthings:]
    true_eigenvecs = np.array(true_eigenvecs)[true_inds][-args.num_eigenthings:]

    est_inds = np.argsort(estimated_eigenvals)
    estimated_eigenvals = np.array(estimated_eigenvals)[est_inds]
    estimated_eigenvecs = np.array(estimated_eigenvecs)[est_inds]

    eigenval_err = compute_eigenval_err(true_eigenvals,
                                        estimated_eigenvals)
    eigenvec_err = compute_eigenvec_err(true_eigenvecs,
                                        estimated_eigenvecs)
    return eigenval_err, eigenvec_err


def generate_wishart(n, offset=0.0):
    """
    Generates a wishart PSD matrix with n rows/cols.
    Adds offset * I for conditioning testing.
    """
    matrix = np.random.random(size=(n, n)).astype(float)
    matrix = matrix.transpose().dot(matrix)
    matrix = matrix + offset * np.eye(n)
    return matrix


def test_wishart():
    for _ in range(args.num_trials):
        m = generate_wishart(args.matrix_dim)
        print(test_matrix(m))


if __name__ == '__main__':
    test_wishart()

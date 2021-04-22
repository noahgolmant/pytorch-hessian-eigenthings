import argparse
import numpy as np
import torch
from hessian_eigenthings.operator import LambdaOperator
from hessian_eigenthings.stochastic_lanczos import stochastic_lanczos, \
    eigv_to_density


parser = argparse.ArgumentParser(description='spectral density tester')
parser.add_argument('--matrix_dim', type=int, default=100,
                    help='number of rows/columns in matrix')
parser.add_argument('--num_eigenthings', type=int, default=10,
                    help='number of eigenvalues to compute')
parser.add_argument('--seed', default=1, type=int)
args = parser.parse_args()


def test(ntrials=1):
    """
    Test to check the statistical distance between the true spectral density
    and spectral density computed from lanczos method.
    """

    n = 100
    for _ in range(ntrials):

        # generate a random matrix
        matrix = np.random.random(size=(n, n)).astype(float)
        matrix = matrix.T @ matrix / 2
        _, eig_vecs = np.linalg.eig(matrix)

        # we generate random eigen values and x by 100 to get a wide range
        matrix = eig_vecs.T @ np.diag(np.random.randn(n)*100) @ eig_vecs
        tensor = torch.from_numpy(matrix).type(torch.float64)

        op = LambdaOperator(lambda x: torch.matmul(tensor, x), tensor.size()[:1])
        density, grids = stochastic_lanczos(op, 100, 99, 1, get_density=True)

        real_eigenvals, _ = torch.symeig(tensor, eigenvectors=False)
        real_eigenvals = real_eigenvals.reshape(1, -1)
        true_density, _ = eigv_to_density(real_eigenvals, grids=grids)

        # statistical distance as per
        # https://github.com/google/spectral-density/blob/master/jax/lanczos_test.py
        assert(np.mean(np.abs(np.array(density)-np.array(true_density))) < 5e-2)


if __name__ == '__main__':
    test(ntrials=10)


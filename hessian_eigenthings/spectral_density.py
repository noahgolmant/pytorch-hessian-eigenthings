"""
Implements lanczos gaussian quadrature step for spectrum density estimation.

Uses the scipy fortran lanczos hook for tridiagonalization because I don't
trust myself to make a numerically stable / scalable lanczos routine

Density estimation is based on the pytorch implementation by Levi Viana:
https://github.com/LeviViana/torchessian

With the original paper by Ghorbani et al.

"An Investigation into Neural Net Optimization via Hessian Eigenvalue Density": https://arxiv.org/abs/1901.10159
"""
import math
import numpy as np

from hessian_eigenthings.lanczos import lanczos


def _gaussian_density(x, mean, sigma_squared):
    sig = math.sqrt(sigma_squared)
    return np.exp(-(x - mean) ** 2 / (2 * sig)) / (sig * math.sqrt(2 * math.pi))


def spectral_density(
    eigenvals,
    eigenvecs,
    min_eigenvalue_support=-1e5,
    max_eigenvalue_support=1e5,
    num_support_points=1e6,
    sigma_squared=1e-5,  # TODO: what is this from the paper?
):
    """
    computes the spectral density of a linear operator using gaussian quadrature.
    see hessian_eigenthings.lanczos for eigenvalue,eigenvector computation.


    min_eigenvalue_support: float
        min eigenvalue for which we can estimate a non-zero density
    max_eigenvalue_support: float
        max eigenvalue for which we can estimate a non-zero density
    num_support_points: float
        number of interpolation points for quadrature coefficients.

    returns:
    ----------------
    density: np.ndarray
        array of length `num_eigenthings` which contains density estimates for
    support: np.ndarray
    """
    num_support_points = int(num_support_points)  # cast in case
    # TODO figure out what's going on
    num_support_points = len(eigenvecs)
    if isinstance(eigenvecs, list):
        eigenvecs = np.array(eigenvecs)
    support = np.linspace(
        min_eigenvalue_support, max_eigenvalue_support, num_support_points
    )
    coeffs = list(eigenvecs[i, 0] for i in range(num_support_points))
    num_eigenthings = len(eigenvals)
    assert len(eigenvecs) == num_eigenthings, "did I do something stupid here"
    density = np.zeros_like(support)
    for i in range(num_eigenthings):
        density += _gaussian_density(support, eigenvals[i], sigma_squared) * coeffs[i]
    return np.array(density), support

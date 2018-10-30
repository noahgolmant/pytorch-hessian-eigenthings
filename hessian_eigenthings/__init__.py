""" Top-level module for hessian eigenvec computation """
from . import power_iter
from .hvp_operator import HVPOperator, compute_hessian_eigenthings

__all__ = ['power_iter', 'HVPOperator', 'compute_hessian_eigenthings']

name = 'hessian_eigenthings'

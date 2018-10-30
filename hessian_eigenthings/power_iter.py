"""
This module contains functions to perform power iteration with deflation
to compute the top eigenvalues and eigenvectors of a linear operator
"""
import numpy as np
import torch


class Operator:
    """
    maps x -> Lx for a linear operator L
    """

    def __init__(self, size):
        self.size = size

    def apply(self, vec):
        """
        Function mapping vec -> L vec where L is a linear operator
        """
        raise NotImplementedError


class LambdaOperator(Operator):
    """
    Linear operator based on a provided lambda function
    """
    def __init__(self, apply_fn, size):
        super(LambdaOperator, self).__init__(size)
        self.apply_fn = apply_fn

    def apply(self, x):
        return self.apply_fn(x)


def deflated_power_iteration(operator,
                             num_eigenthings=10,
                             power_iter_steps=20,
                             power_iter_err_threshold=1e-4,
                             use_gpu=True):
    """
    Compute top k eigenvalues by repeatedly subtracting out dyads
    operator: linear operator that gives us access to matrix vector product
    num_eigenvals number of eigenvalues to compute
    power_iter_steps: number of steps per run of power iteration
    power_iter_err_threshold: early stopping threshold for power iteration
    returns: np.ndarray of top eigenvalues, np.ndarray of top eigenvectors
    """
    eigenvals = []
    eigenvecs = []
    current_op = operator

    def _deflate(x, val, vec):
        return val * vec.dot(x) * vec

    for _ in range(num_eigenthings):
        eigenval, eigenvec = power_iteration(current_op, power_iter_steps,
                                             power_iter_err_threshold,
                                             use_gpu=use_gpu)

        def _new_op_fn(x, op=current_op, val=eigenval, vec=eigenvec):
            return op.apply(x) - _deflate(x, val, vec)
        current_op = LambdaOperator(_new_op_fn, operator.size)
        eigenvals.append(eigenval.item())
        eigenvecs.append(eigenvec.cpu())

    return eigenvals, eigenvecs


def power_iteration(operator, steps=20, error_threshold=1e-4, use_gpu=True):
    """
    Compute dominant eigenvalue/eigenvector of a matrix
    operator: linear Operator giving us matrix-vector product access
    steps: number of update steps to take
    returns: (principal eigenvalue, principal eigenvector) pair
    """
    vector_size = operator.size  # input dimension of operator
    vec = torch.rand(vector_size)
    if use_gpu:
        vec = vec.cuda()

    prev_lambda = 0.
    for _ in range(steps):
        new_vec = operator.apply(vec)
        lambda_estimate = vec.dot(new_vec)
        diff = lambda_estimate - prev_lambda
        vec = new_vec.detach() / torch.norm(new_vec)
        error = np.abs(diff / lambda_estimate)
        if error < error_threshold:
            return lambda_estimate, vec
        prev_lambda = lambda_estimate

    return lambda_estimate, vec

"""
This module contains functions to perform power iteration with deflation
to compute the top eigenvalues and eigenvectors of a linear operator
"""
import numpy as np
import torch

from hessian_eigenthings.operator import LambdaOperator
import hessian_eigenthings.utils as utils




def deflated_power_iteration(
    operator,
    num_eigenthings=10,
    power_iter_steps=20,
    power_iter_err_threshold=1e-4,
    momentum=0.0,
    use_gpu=True,
    fp16=False,
    to_numpy=True,
):
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
    prev_vec = None

    def _deflate(x, val, vec):
        return val * vec.dot(x) * vec

    utils.log("beginning deflated power iteration")
    for i in range(num_eigenthings):
        utils.log("computing eigenvalue/vector %d of %d" % (i + 1, num_eigenthings))
        eigenval, eigenvec = power_iteration(
            current_op,
            power_iter_steps,
            power_iter_err_threshold,
            momentum=momentum,
            use_gpu=use_gpu,
            fp16=fp16,
            init_vec=prev_vec,
        )
        utils.log("eigenvalue %d: %.4f" % (i + 1, eigenval))

        def _new_op_fn(x, op=current_op, val=eigenval, vec=eigenvec):
            return utils.maybe_fp16(op.apply(x), fp16) - _deflate(x, val, vec)


        current_op = LambdaOperator(_new_op_fn, operator.size)
        prev_vec = eigenvec
        eigenvals.append(eigenval)
        eigenvec = eigenvec.cpu()
        if to_numpy:
            eigenvecs.append(eigenvec.numpy())
        else:
            eigenvecs.append(eigenvec)

    eigenvals = np.array(eigenvals)
    eigenvecs = np.array(eigenvecs)

    # sort them in descending order
    sorted_inds = np.argsort(eigenvals)
    eigenvals = eigenvals[sorted_inds][::-1]
    eigenvecs = eigenvecs[sorted_inds][::-1]
    return eigenvals, eigenvecs


def power_iteration(
    operator, steps=20, error_threshold=1e-4, momentum=0.0, use_gpu=True, fp16=False, init_vec=None
):
    """
    Compute dominant eigenvalue/eigenvector of a matrix
    operator: linear Operator giving us matrix-vector product access
    steps: number of update steps to take
    returns: (principal eigenvalue, principal eigenvector) pair
    """
    vector_size = operator.size  # input dimension of operator
    if init_vec is None:
        vec = torch.rand(vector_size)
    else:
        vec = init_vec

    vec = utils.maybe_fp16(vec, fp16)

    if use_gpu:
        vec = vec.cuda()

    prev_lambda = 0.0
    prev_vec = utils.maybe_fp16(torch.randn_like(vec), fp16)
    for i in range(steps):
        prev_vec = vec / (torch.norm(vec) + 1e-6)
        new_vec = utils.maybe_fp16(operator.apply(vec), fp16) - momentum * prev_vec
        # need to handle case where we end up in the nullspace of the operator.
        # in this case, we are done.
        if torch.sum(new_vec).item() == 0.0:
            return 0.0, new_vec
        lambda_estimate = vec.dot(new_vec).item()
        diff = lambda_estimate - prev_lambda
        vec = new_vec.detach() / torch.norm(new_vec)
        if lambda_estimate == 0.0:  # for low-rank
            error = 1.0
        else:
            error = np.abs(diff / lambda_estimate)
        utils.progress_bar(i, steps, "power iter error: %.4f" % error)
        if error < error_threshold:
            return lambda_estimate, vec
        prev_lambda = lambda_estimate

    return lambda_estimate, vec

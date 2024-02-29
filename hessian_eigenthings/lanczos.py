""" Use scipy/ARPACK implicitly restarted lanczos to find top k eigenthings """
from typing import Tuple, Union

import numpy as np
import torch
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from warnings import warn

import hessian_eigenthings.utils as utils

from hessian_eigenthings.operator import Operator


def lanczos(
    operator: Operator,
    num_eigenthings: int = 10,
    which: str = "LM",
    max_steps: int = 20,
    tol: float = 1e-6,
    num_lanczos_vectors: Union[int, None] = None,
    init_vec: Union[np.ndarray, None] = None,
    device: utils.Device = "cpu",
    fp16: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use the scipy.sparse.linalg.eigsh hook to the ARPACK lanczos algorithm
    to find the top k eigenvalues/eigenvectors.

    Please see scipy documentation for details on specific parameters
    such as 'which'.

    Parameters
    -------------
    operator: operator.Operator
        linear operator to solve.
    num_eigenthings : int
        number of eigenvalue/eigenvector pairs to compute
    which : str ['LM', SM', 'LA', SA']
        L,S = largest, smallest. M, A = in magnitude, algebriac
        SM = smallest in magnitude. LA = largest algebraic.
    max_steps : int
        maximum number of arnoldi updates
    tol : float
        relative accuracy of eigenvalues / stopping criterion
    num_lanczos_vectors : int
        number of lanczos vectors to compute. if None, > 2*num_eigenthings
        for stability.
    init_vec: [torch.Tensor, torch.cuda.Tensor]
        if None, use random tensor. this is the init vec for arnoldi updates.
    use_gpu: bool
        if true, use cuda tensors.
    fp16: bool
        if true, keep operator input/output in fp16 instead of fp32.

    Returns
    ----------------
    eigenvalues : np.ndarray
        array containing `num_eigenthings` eigenvalues of the operator
    eigenvectors : np.ndarray
        array containing `num_eigenthings` eigenvectors of the operator
    """
    if isinstance(operator.size, int):
        size = operator.size
    else:
        size = operator.size[0]
    shape = (size, size)

    if num_lanczos_vectors is None:
        num_lanczos_vectors = min(2 * num_eigenthings, size - 1)
    if num_lanczos_vectors < 2 * num_eigenthings:
        warn(
            "[lanczos] number of lanczos vectors should usually be > 2*num_eigenthings"
        )

    def _scipy_apply(x):
        x = torch.from_numpy(x)
        x = utils.maybe_fp16(x, fp16)
        if device == "cuda":
            x = x.cuda()
        if device == "mps":
            x = x.to("mps")
        out = operator.apply(x)
        out = utils.maybe_fp16(out, fp16)
        out = out.cpu().numpy()
        return out

    scipy_op = ScipyLinearOperator(shape, _scipy_apply)
    if init_vec is None:
        init_vec = np.random.rand(size)

    eigenvals, eigenvecs = linalg.eigsh(
        A=scipy_op,
        k=num_eigenthings,
        which=which,
        maxiter=max_steps,
        tol=int(tol),
        ncv=num_lanczos_vectors,
        return_eigenvectors=True,
    )
    return eigenvals, eigenvecs.T

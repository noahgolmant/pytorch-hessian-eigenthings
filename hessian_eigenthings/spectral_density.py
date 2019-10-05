import numpy as np
import torch


def _lanczos_step(vec, size, current_draw):

    pass


def lanczos(
    operator,
    max_steps=20,
    tol=1e-6,
    num_lanczos_vectors=None,
    init_vec=None,
    use_gpu=False,
):
    """
    Use the scipy.sparse.linalg.eigsh hook to the ARPACK lanczos algorithm
    to find the top k eigenvalues/eigenvectors.

    Parameters
    -------------
    operator: power_iter.Operator
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
    init_vec: [torch.Tensor, torch.cuda.Tensor]
        if None, use random tensor. this is the init vec for arnoldi updates.
    use_gpu: bool
        if true, use cuda tensors.

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

    if num_lanczos_vectors is None:
        num_lanczos_vectors = min(2 * num_eigenthings, size - 1)
    if num_lanczos_vectors < 2 * num_eigenthings:
        warn(
            "[lanczos] number of lanczos vectors should usually be > 2*num_eigenthings"
        )

    return eigenvals

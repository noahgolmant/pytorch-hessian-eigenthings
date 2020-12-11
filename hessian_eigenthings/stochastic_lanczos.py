import torch
import hessian_eigenthings.utils as utils
import time
import math


def stochastic_lanczos(operator, dim, order, draws, get_density=False, use_cuda=False):
    r'''
        Base function to manage stochastic lanczos experiment over multiple
        draws.

        Parameters
        -------------
            operator    : linear Operator giving us matrix-vector product access
            dim         : dimensionality of Hessian
            order       : An integer corresponding to the number of Lanczos
                          steps to take.
            draws       : Total number of draws
            use_gpu     : use_gpu
        Returns
        ----------------
            density: Array of size [grid_len]. Spectral density averaged over
                     the draws
            grids: Array of size [grid_len].
    '''

    tri = torch.zeros((draws, order, order))
    for draw_idx in range(draws):
        vecs, tridiag = lanczos_one_draw(operator, dim, order, use_cuda)
        tri[draw_idx, :, :] = tridiag

    if get_density:
        return tridiag_to_density(tri)
    else:
        return tridiag_to_eigv(tri)


def lanczos_one_draw(operator, dim, order, use_gpu=False):
    r'''
        Lanczos iteration following the wikipedia article here
            https://en.wikipedia.org/wiki/Lanczos_algorithm
        Parameters
        -------------
            operator    : linear Operator giving us matrix-vector product access
            dim         : dimensionality of Hessian
            order       : An integer corresponding to the number of Lanczos
                          steps to take.
            use_gpu     : use_gpu
        Returns
        ----------------
            eigven values
            weights
    '''
    float_dtype = torch.float64

    # Initializing empty arrays for storing
    tridiag = torch.zeros((order, order), dtype=float_dtype)
    vecs = torch.zeros((dim, order), dtype=float_dtype)

    # intialize a random unit norm vector
    init_vec = torch.zeros((dim), dtype=float_dtype).uniform_(-1, 1)
    init_vec /= torch.norm(init_vec)
    vecs[:, 0] = init_vec

    # placeholders for data
    beta = 0.0
    v_old = torch.zeros((dim), dtype=float_dtype)

    for k in range(order):
        t = time.time()

        v = vecs[:, k]
        if use_gpu:
            v = v.type(torch.float32).cuda()
        time_mvp = time.time()
        w = operator.apply(v)
        if use_gpu:
            v = v.cpu().type(float_dtype)
            w = w.cpu().type(float_dtype)
        time_mvp = time.time() - time_mvp

        w -= beta * v_old
        alpha = torch.dot(w, v)
        tridiag[k, k] = alpha
        w -= alpha*v

        # Reorthogonalization
        for j in range(k):
            tau = vecs[:, j]
            coeff = torch.dot(w, tau)
            w -= coeff * tau

        beta = torch.norm(w)

        if beta < 1e-6:
            raise ZeroDivisionError
            quit()

        if k + 1 < order:
            tridiag[k, k+1] = beta
            tridiag[k+1, k] = beta
            vecs[:, k+1] = w / beta

        v_old = v

        if k % 10 == 0:
            info = f"Iteration {k} / {order} done in {time.time()-t:.2f}s (HVP: {time_mvp:.2f}s)"
            utils.log(info)

    return vecs, tridiag


def tridiag_to_density(tridiag_list, sigma_sq=1e-5, grid_len=10000):
    r'''
    This function estimates the smoothed density from the output of lanczos.
    It is a direct implementation from [1].

    [1] https://github.com/google/spectral-density/blob/master/jax/density.py

    Parameters
    -------------
    tridiag_list: Tridiagonal matrices computed from running num_draws
                  independent runs of lanczos.
    sigma_sq    : Controls the smoothing of the density.
    grid_len    : Controls the granularity of the density.
    Returns
    ----------------
    density     : The smoothed density estimate averaged over all draws.
    grids       : The grid on which the density is estimates
    '''
    eig_vals, all_weights = tridiag_to_eigv(tridiag_list)
    density, grids = eigv_to_density(eig_vals, all_weights,
                                     grid_len=grid_len,
                                     sigma_sq=sigma_sq)
    return density, grids


def tridiag_to_eigv(tridiag_list):
    """
    Computes eigen values of the list of tridiagonal matrices
    Parameters
    -------------
    tridiag_list: Tridiagonal matrices computed from running num_draws
                  independent runs of lanczos.
    Returns
    ----------------
    eig_vals    : Eigen values
    all_weights : Weights
    """
    # Calculating the node / weights from Jacobi matrices.
    num_draws = len(tridiag_list)
    num_lanczos = tridiag_list[0].shape[0]
    eig_vals = torch.zeros((num_draws, num_lanczos))
    all_weights = torch.zeros((num_draws, num_lanczos))

    eig_vals, evecs = torch.symeig(tridiag_list, eigenvectors=True)
    for i in range(num_draws):
        all_weights[i, :] = evecs[i, 0, :] ** 2

    return eig_vals, all_weights


def eigv_to_density(eig_vals, all_weights=None, grids=None,
                    grid_len=10000, sigma_sq=None):
    r'''Helper function to compute spectral density from a set of eigen values.
    It is a direct implementation from [1].

    [1] https://github.com/google/spectral-density/blob/master/jax/density.py

    Parameters
    -------------
    eig_vals    : Eigenvalues returned from function tridiag_to_eigv.
    all_weights : Weights returned from function tridiag_to_eigv.
    grids       : Grid over which to evaluate the density.
                  If None, an appropriate value is inferred.
    grid_len    : Int to specify grid spacing.
    sigma_sq    : Sigma squared of the gaussian kernel places at each eigen-value.
                  If None, an appropriate value is inferred.
    Returns:
    density: Array of shape [grid_len], the estimated density, averaged over
      all draws.
    grids: Array of shape [grid_len]. The values the density is estimated on.
    '''

    if all_weights is None:
        all_weights = torch.ones(eig_vals.shape) * 1.0 / float(eig_vals.shape[1])
    num_draws = eig_vals.shape[0]

    if torch.isnan(eig_vals).any():
        raise ValueError('tridiag has nan values.')

    lambda_max = torch.max(eig_vals) + 1e-2
    lambda_min = torch.min(eig_vals) - 1e-2

    if grids is None:
        assert grid_len is not None, 'grid_len is required if grids is None.'
        grids = torch.linspace(lambda_min, lambda_max, grid_len)

    grid_len = grids.shape[0]
    if sigma_sq is None:
        sigma = 10 ** -5 * max(1, (lambda_max - lambda_min))
    else:
        sigma = sigma_sq * max(1, (lambda_max - lambda_min))

    density_each_draw = torch.zeros((num_draws, grid_len))
    for i in range(num_draws):
        for j in range(grid_len):
            x = grids[j]
            vals = _kernel(eig_vals[i, :], x, sigma)
            density_each_draw[i, j] = torch.sum(vals * all_weights[i, :])

    if torch.isnan(density_each_draw).any():
        raise ValueError('density has nan values.')

    density = torch.mean(density_each_draw, axis=0)
    norm_fact = torch.sum(density) * (grids[1] - grids[0])
    density = density / norm_fact
    return density, grids


def _kernel(x, x0, variance):
    """
    Gaussian kernel for computing the spectral density as per
    gaussian quadratures.
    """
    coeff = 1.0 / torch.sqrt(2 * math.pi * variance)
    val = -(x0 - x) ** 2
    val = val / (2.0 * variance)
    val = torch.exp(val)
    point_estimate = coeff * val
    return point_estimate



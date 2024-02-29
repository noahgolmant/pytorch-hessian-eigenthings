import numpy as np
import matplotlib.pyplot as plt


def compute_eigenvec_cos_similarity(actual, estimated):
    scores = []
    for estimate in estimated:
        score = np.abs(np.dot(actual, estimate))
        scores.append(score)
    return scores


def plot_eigenval_estimates(estimates, label):
    """
    estimates = 2D array (num_trials x num_eigenvalues)

    x-axis = eigenvalue index
    y-axis = eigenvalue estimate
    """
    if len(estimates.shape) == 1:
        var = np.zeros_like(estimates)
    else:
        var = np.var(estimates, axis=0)
    y = np.mean(estimates, axis=0)
    x = list(range(len(y)))
    error = np.sqrt(var)
    plt.plot(x, y, label=label)
    plt.fill_between(x, y - error, y + error, alpha=0.2)


def plot_eigenvec_errors(true, estimates, label):
    """
    plots error for all eigenvector estimates in L2 norm
    estimates = (num_trials x num_eigenvalues x num_params)
    true = (num_eigenvalues x num_params)
    """
    diffs = []
    num_eigenvals = true.shape[0]
    for i in range(num_eigenvals):
        cur_estimates = estimates[:, i, :]
        cur_eigenvec = true[i]
        diff = compute_eigenvec_cos_similarity(cur_eigenvec, cur_estimates)
        diffs.append(diff)
    diffs = np.array(diffs).T
    var = np.var(diffs, axis=0)
    y = np.mean(diffs, axis=0)
    x = list(range(len(y)))

    error = np.sqrt(var)
    plt.plot(x, y, label=label)
    plt.fill_between(x, y - error, y + error, alpha=0.2)

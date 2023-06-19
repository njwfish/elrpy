import jax
from jax import numpy as np
from jax.scipy.stats import multivariate_normal


def lyapunov_binary_loss(p, Y, N, weights=None):
    """Lyapunov Central Limit loss for binary outcomes.
    
    Args:
        p (np.ndarray): Array of probabilities of shape (n, k).
        Y (int): Number of successes.
        N (int): Size of group.
    
    Returns:
        float: a normal approximation to the combinatorial log-likelihood.
    """
    phi2 = np.sum(p * (1 - p), axis=0)
    mu = np.sum(p, axis=0)
    logp = 1/2 * np.log(phi2) - (1 / phi2) * (Y - mu)**2
    if weights is not None:
        logp = weights * logp
    return -logp


def lyapunov_categorical_loss(probs, Y, N, eps=1e-2, weights=None):
    """Lyapunov Central Limit loss for categorical outcomes.

    Args:
        probs (np.ndarray): Array of probabilities of shape (n, k).
        Y (np.ndarray): Array of outcomes of shape (n, k).
        N (int): Size of group.
        eps (float): Regularization parameter to ensure positive definiteness of covariance matrix.
        weights (np.ndarray): Array of weights of shape (n,).

    Returns:
        float: a normal approximation to the combinatorial log-likelihood.
    """
    mu = np.sum(probs, axis=0)
    demeaned_p = probs - np.mean(probs, axis=0)
    Sigma = np.sum(np.einsum('...c,...d->...cd', demeaned_p, demeaned_p), axis=0) + eps * np.eye(Y.shape[0])
    logp = multivariate_normal.logpdf(mu, mean=Y, cov=Sigma)
    if weights is not None:
        logp = weights * logp
    return -logp



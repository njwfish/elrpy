import jax
from jax import numpy as np
from jax.scipy.stats import multivariate_normal


def lyapunov_binary_loss(p, Y, N):
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
    loss = -np.mean(logp)
    return loss

def lyapunov_categorical_loss(probs, Y, N):
    """Lyapunov Central Limit loss for categorical outcomes.
    
    Args:
        probs (np.ndarray): Array of probabilities of shape (n, [r,] k, p).
        Y (np.ndarray): Number of outcomes of each type ([r,] p).
        N (int): Size of group.
        
    Returns:
        float: a normal approximation to the combinatorial log-likelihood.
    """
    Sigma = np.sum(np.einsum('...c,...d->...cd', probs, probs), axis=0)
    mu = np.sum(probs, axis=0)
    logp = multivariate_normal.logpdf(mu, mean=Y, cov=Sigma)
    loss = -np.mean(logp)
    return loss
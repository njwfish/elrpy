import jax
from jax import numpy as np
from jax.nn import sigmoid

from jax.experimental import sparse
def normal_sim_binary(rng, n, d, k, beta=None, gamma=None, intercept=True):
    """Simulate data from a normal mixture model with binary outcomes.

    Args:
        rng (jax.random.PRNGKey): random number generator
        n (int): number of observations
        d (int): number of features
        k (int): number of groups
        beta (np.ndarray): regression coefficients
        gamma (np.ndarray): group coefficients
    
    Returns:
        np.ndarray: data matrix
        np.ndarray: outcome vector
        dict: group indices
        dict: group data matrices
        dict: group outcome vectors
    """
    rng, next_rng = jax.random.split(rng)
    if intercept:
        X = jax.random.normal(next_rng, shape=(n, d - 1))
        X = np.hstack([np.ones((n, 1)), X])
    else:
        X = jax.random.normal(next_rng, shape=(n, d))

    if beta is None:
        rng, next_rng = jax.random.split(rng)
        beta = jax.random.normal(next_rng, shape=(d ,1))

    if gamma is None:
        rng, next_rng = jax.random.split(rng)
        gamma = jax.random.normal(next_rng, shape=(d, k))

    rng, next_rng = jax.random.split(rng)
    g = jax.random.categorical(rng, X @ gamma)
    G = sparse.BCOO((np.ones(g.shape[0]), np.vstack([np.arange(g.shape[0]), g]).T), shape=(g.shape[0], k)).T

    p = sigmoid(X @ beta)
    rng, next_rng = jax.random.split(rng)
    Y = jax.random.bernoulli(next_rng, p)

    return beta, gamma, X, Y, G @ Y, G


def normal_sim_categorical(rng, n, d, k, p, eps=1e-6, beta=None, gamma=None):
    """Simulate data from a normal mixture model with categorical outcomes.
    
    Args:
        rng (jax.random.PRNGKey): random number generator
        n (int): number of observations
        d (int): number of features
        k (int): number of groups
        p (int): number of outcomes
        eps (float): small constant to avoid numerical issues
        beta (np.ndarray): regression coefficients
        gamma (np.ndarray): group coefficients

    Returns:
        np.ndarray: data matrix
        np.ndarray: outcome vector
        dict: group indices
        dict: group data matrices
        dict: group outcome vectors
    """
    rng, next_rng = jax.random.split(rng)
    X = jax.random.normal(next_rng, shape=(n, d - 1))
    X = np.hstack([np.ones((n, 1)), X])

    if beta is None:
        rng, next_rng = jax.random.split(rng)
        beta = jax.random.normal(next_rng, shape=(d, p - 1))
        beta = np.concatenate([beta, np.zeros((d, 1))], axis=-1)

    if gamma is None:
        rng, next_rng = jax.random.split(rng)
        gamma = jax.random.normal(next_rng, shape=(d, k))

    rng, next_rng = jax.random.split(rng)
    g = jax.random.categorical(rng, X @ gamma)
    G = jax.nn.one_hot(g, k)
    G = sparse.BCOO.fromdense(G).T

    rng, next_rng = jax.random.split(rng)
    Y = jax.random.categorical(rng, np.tensordot(X, beta, axes=1))
    Y = jax.nn.one_hot(Y, p)
    return beta, gamma, X, Y, G @ Y, G
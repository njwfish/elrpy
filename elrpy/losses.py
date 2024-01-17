import jax
from jax import numpy as np

def get_wrapped_loss(loss_fn, model_fn, axis=None):
    """Wrap loss function to take model parameters and group data.

    Args:
        loss_fn: loss function.
        model_fn: model function.
        num_groups: number of groups.

    Returns:
        function: wrapped loss function.
    """
    def wrapped_loss(model_params, X, Y, G, weights=None):
        """Wrapped loss function.

        Args:
            model_params: model parameters.
            group_X: group covariates.
            group_Y: group outcomes.
            group_N: group number of observations.
            group_weight: group weights.

        Returns:
            float: loss.
        """
        return np.mean(loss_fn(
            model_fn(model_params, X), Y, G, weights=weights
        ), axis=axis)
    
    return wrapped_loss

def lyapunov_binary_loss(p, Y, G, weights=None, eps=1e-6):
    """Lyapunov Central Limit loss for binary outcomes.
    
    Args:
        p (np.ndarray): Array of probabilities of shape (n, k).
        Y (int): Number of successes.
        N (int): Size of group.
    
    Returns:
        float: a normal approximation to the combinatorial log-likelihood.
    """
    mu = G @ p
    phi2 = np.maximum(G @ (p * (1 - p)), eps)
    logp = -np.log(phi2) - (1 / phi2) * (Y - mu)**2
    if weights is not None:
        logp = weights * logp
    return -1/2 * logp

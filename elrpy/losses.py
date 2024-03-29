import jax
from jax import numpy as np

def get_wrapped_loss(loss_fn, model_fn, num_groups):
    """Wrap loss function to take model parameters and group data.

    Args:
        loss_fn: loss function.
        model_fn: model function.
        num_groups: number of groups.

    Returns:
        function: wrapped loss function.
    """
    def wrapped_loss(model_params, group_X, group_Y, group_N, group_weight=None):
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
        return loss_fn(
            model_fn(model_params, group_X), group_Y, group_N, weights=group_weight
        ) / num_groups
    
    return wrapped_loss


def lyapunov_binary_loss(p, Y, N, weights=None, eps=1e-6):
    """Lyapunov Central Limit loss for binary outcomes.
    
    Args:
        p (np.ndarray): Array of probabilities of shape (n, k).
        Y (int): Number of successes.
        N (int): Size of group.
    
    Returns:
        float: a normal approximation to the combinatorial log-likelihood.
    """
    phi2 = np.maximum(np.sum(p * (1 - p), axis=0), eps)
    mu = np.sum(p, axis=0)
    logp = -1/2 * np.log(phi2) - (1 / phi2) * (Y - mu)**2
    if weights is not None:
        logp = weights * logp
    return -logp

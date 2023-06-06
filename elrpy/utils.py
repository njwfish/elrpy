import jax
import jax.numpy as np

def get_dims(group_data):
    """Returns the number of groups, the number of covariates, and the number of outcomes.
    
    Args:
        group_data (tuple): tuple of group covariates, group outcomes, and group sizes
        
    Returns:
        tuple: number of groups, number of covariates, and number of outcomes
    """
    group_Xs, group_Ys, group_Ns = group_data
    num_groups = len(group_Ns)
    dim = next(iter(group_Xs.values())).shape[1]
    num_outcomes = next(iter(group_Ys.values())).shape[0]
    return num_groups, dim, num_outcomes

def get_bootstrap_weights(rng, group_data, num_boots, estimate_on_all=False):
    """Returns bootstrap weights for each group.
    
    Args:
        rng (jax.random.PRNGKey): random number generator
        group_data (tuple): tuple of group covariates, group outcomes, and group sizes
        num_boots (int): number of bootstrap samples
        estimate_on_all (bool): whether to include replicate estimated on all data, e.g. all ones
        
    Returns:
        dict: dictionary of bootstrap weights for each group
    """
    num_boots -= estimate_on_all
    num_groups = len(group_data[0])
    group_idx = jax.random.choice(rng, num_groups, shape=(num_groups, num_boots))
    weights = np.zeros((num_groups, num_boots))
    weights = jax.vmap(lambda w, idx:w.at[idx].add(1), in_axes=(1, 1))(weights, group_idx).T
    if estimate_on_all:
        weights = np.hstack([np.ones((num_groups, 1)), weights])
    group_weights = {g: w[:, None] for g, w in zip(group_data[0], weights)}
    return group_weights
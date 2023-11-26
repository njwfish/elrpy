import jax
import jax.numpy as np

def reduce_sum(tree):
    """Reduce sum of tree.
    
    Args:
        tree: tree to sum.
        
    Returns:
        float: sum of tree.
    """
    return jax.tree_util.tree_reduce(
        jax.jit(lambda x, y: x + y if isinstance(x, np.ndarray) else [xi + yi for (xi, yi) in zip(x, y)]), 
        tree, is_leaf=lambda x: not isinstance(x, dict)
    )


def get_mapped_fn(fn):
    """Get mapped function.
    
    Args:
        fn: function to map.
            
    Returns:
        function: mapped function."""
    def mapped_fn(model_params, args):
        """Map function over tree, taking model_params as first argument and summing results.
        
        Args:
            model_params: model parameters.
            args: arguments to pass to fn.
            
        Returns:
            float: sum of mapped function.
        """
        group_values = jax.tree_util.tree_map(
            lambda *args: fn(model_params, *args), *args
        )
        return reduce_sum(group_values)
    return mapped_fn


def get_mean_fn(fn):
    def mean_fn(*args, **kwargs):
        return np.mean(fn(*args, **kwargs))
    return mean_fn

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

def get_bootstrap_weights(rng, G, num_boots, estimate_on_all=False, sample_by_group_size=True):
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
    num_groups = G.shape[0]
    group_sizes = np.squeeze(G @ np.ones((G.shape[1], 1)))
    if sample_by_group_size:
        group_idx = jax.random.choice(rng, num_groups, shape=(num_groups, num_boots), p=group_sizes / np.sum(group_sizes))
    else:
        group_idx = jax.random.choice(rng, num_groups, shape=(num_groups, num_boots))
    weights = np.zeros((num_groups, num_boots))
    weights = jax.vmap(lambda w, idx: w.at[idx].add(1), in_axes=(1, 1))(weights, group_idx).T
    if estimate_on_all:
        weights = np.hstack([np.ones((num_groups, 1)), weights])
    return weights.T
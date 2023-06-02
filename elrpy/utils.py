import jax
import jax.numpy as np

def get_dims(group_data):
    group_Xs, group_Ys, group_Ns = group_data
    k = len(group_Ns)
    d = next(iter(group_Xs.values())).shape[1]
    p = next(iter(group_Ys.values())).shape[0]
    return k, d, p

def get_bootstrap_weights(rng, group_data, num_boots, estimate_on_all=True):
    num_groups = len(group_data[0])
    group_idx = jax.random.choice(rng, num_groups, shape=(num_groups, num_boots))
    weights = np.zeros((num_groups, num_boots))
    weights = jax.vmap(lambda w, idx:w.at[idx].add(1), in_axes=(1, 1))(weights, group_idx).T
    if estimate_on_all:
        weights = np.hstack([np.ones((num_groups, 1)), weights])
    group_weights = {g: w for g, w in zip(group_data[0], weights)}
    return group_weights
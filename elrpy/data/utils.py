import jax
import numpy as np

from typing import Iterable, Tuple, Optional


def group_data_to_dicts(
        groups: Iterable, Y: Iterable, N: Iterable
) -> Tuple[dict, dict]: 
    """Convert group data to dictionaries. All inputs must be the same length.
    
    Args:
        groups: Group labels.
        Y: Outcomes.
        N: Number of observations per group.
        
    Returns:
        tuple: dict, group_frame_Ns
    """
    group_frame_Ys = dict(zip(groups, Y))
    group_frame_Ns = dict(zip(groups, N))
    return group_frame_Ys, group_frame_Ns

def indidual_data_to_dicts(groups: np.ndarray, X: np.ndarray) -> Tuple[dict, dict]:
    group_indices = {
        group: np.where(group == groups)[0] for group in np.unique(groups)
    }
    individual_frame_Xs = jax.tree_util.tree_map(lambda idx: X[idx], group_indices)
    individual_frame_Ns = jax.tree_util.tree_map(lambda idx: idx.shape[0], group_indices)
    return individual_frame_Xs, individual_frame_Ns

def intersect_frames(
        individual_frame_Xs: dict, individual_frame_Ns: dict,
        group_frame_Ys: dict, group_frame_Ns: dict, 
        verbose: Optional[int] = 1
) -> Tuple[dict, dict, dict, dict]:
    """Remove groups that are not in both individual and group frame.
    
    Args:
        individual_frame_Xs: Individual frame covariates.
        individual_frame_Ns: Individual frame number of observations.
        group_frame_Ys: Group frame outcomes.
        group_frame_Ns: Group frame number of observations.
        verbose: Verbosity level (0: silent, 1: number of groups removed, 2: names of groups removed).
        
    Returns:
        tuple: individual_frame_Xs, individual_frame_Ns, group_frame_Ys, group_frame_Ns
    """
    individual_frame_groups = set(individual_frame_Xs.keys())
    group_frame_groups = set(group_frame_Ys.keys())
    # remove groups that are not in both individual and group frame
    n_dropped_individual_frame = 0
    not_in_group_frame = individual_frame_groups.difference(group_frame_groups)
    for g in not_in_group_frame:
        if verbose == 2:
            print(f"Dropping group {g} because it is not in the group frame")
        n_dropped_individual_frame += np.max(individual_frame_Ns[g])
        individual_frame_Xs.pop(g)
        individual_frame_Ns.pop(g)
    
    # remove groups that are not in both individual and group frame
    n_dropped_group_frame = 0
    not_in_individual_frame = group_frame_groups.difference(individual_frame_groups)
    for g in not_in_individual_frame:
        if verbose == 2:
            print(f"Dropping group {g} because it is not in the individual frame")
        n_dropped_group_frame += np.max(group_frame_Ns[g])
        group_frame_Ys.pop(g)
        group_frame_Ns.pop(g)
    
    if verbose > 0:
        print(
            f"Dropped {len(not_in_group_frame)} groups with "
            f"{n_dropped_individual_frame} individuals from individual frame."
        )
        print(
            f"Dropped {len(not_in_individual_frame)} groups with "
            f"{n_dropped_group_frame} individual outcomes from group frame"
        )
    
    return individual_frame_Xs, individual_frame_Ns, group_frame_Ys, group_frame_Ns

def get_individual_frame_Ys(
        individual_frame_Ns: dict, group_frame_Ys: dict, group_frame_Ns: dict
) -> dict:
    """Normalize group frame outcomes to individual frame.
    
    Args:
        individual_frame_Ns: Individual frame number of observations.
        group_frame_Ys: Group frame outcomes.
        group_frame_Ns: Group frame number of observations.
        
    Returns:
        dict: Individual frame outcomes."""
    individual_frame_Ys = jax.tree_util.tree_map(
        lambda N_i, Y_g, N_g: Y_g * N_i / N_g,
        individual_frame_Ns, group_frame_Ys, group_frame_Ns
    )
    return individual_frame_Ys
import os
from elrpy.data.pd_utils import load_from_csv, transform_from_df
from elrpy.data.np_utils import save_transform_data, save_group_data, load_group_data

from typing import Optional, Iterable, Tuple 

def load(
    covars_path: str, 
    results_path: str,
    covars_group_col: Optional[str]="group_id", 
    results_group_col: Optional[str]="group_id",
    Y_cols: Optional[str]=None, 
    N_cols: Optional[str]=None,
    save_npz: Optional[bool]=True,
    force_from_csv: Optional[bool]=False,
    verbose: Optional[bool]=1
) -> Tuple[dict, dict, dict]:
    """Load covariates and results from csvs and split into groups, covars and results.

    Args:
        covars_path: Path to covariates csv.
        results_path: Path to results csv.
        covars_group_col: Name of column in covars csv that contains group labels.
        results_group_col: Name of column in results csv that contains group labels.
        Y_cols: Names of columns in results that contain outcomes.
        N_cols: Names of columns in results that contain number of observations.
        save_npz: Whether to save npz file of group data.
        force_from_csv: Whether to force loading from csvs.
        verbose: Verbosity level (0: silent, 1: standard prints, 2: debug prints).

    Returns:
        tuple: individual_frame_Xs, individual_frame_Ns, individual_frame_Ys, group_frame_Ns
    """
    
    data_dir = os.path.dirname(results_path)
    covars_name = os.path.basename(covars_path).split(".")[0]
    results_name = os.path.basename(results_path).split(".")[0]
    npz_path = os.path.join(data_dir, f"{results_name}.{covars_name}.npz")
    if os.path.exists(npz_path) and not force_from_csv:
        if verbose > 0:
            print(f"Loading data from {npz_path}.\n"
                  "\tSet force_from_csv=True to reload the data using updated files or group_cols/Y_cols/N_cols.")
        return load_group_data(npz_path)
    
    if verbose > 0:
        print("Loading data from csvs")

    groups, covars, results, Y_cols, N_cols = load_from_csv(
        covars_path, results_path, 
        covars_group_col=covars_group_col, results_group_col=results_group_col,
        Y_cols=Y_cols, N_cols=N_cols
    )

    if verbose > 0:
        print("Transforming data from csvs.")
    group_Xs, group_Ys, group_Ns, columns, scale = transform_from_df(
        groups, covars, results, Y_cols=Y_cols, N_cols=N_cols, verbose=verbose
    )

    if save_npz:
        if verbose > 0:
            print("Saving data to npz.")
        save_transform_data(os.path.join(os.path.dirname(covars_path), f"transforms.npz"), columns, scale)
        save_group_data(npz_path, group_Xs, group_Ys, group_Ns)
    
    return group_Xs, group_Ys, group_Ns


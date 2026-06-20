import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from collections import defaultdict

from typing import Optional, Callable, Iterable, Tuple

from elrpy.data.np_utils import *
from elrpy.data.utils import *

def drop_non_numeric_cols(df: pd.DataFrame):
    """Drop non-numeric columns from dataframe.
    
    Args:
        df: Dataframe to drop columns from.

    Returns:
        pd.DataFrame: Dataframe with non-numeric columns dropped.
    """
    return df.select_dtypes(include=["number", "bool"])

def fill_na(df: pd.DataFrame):
    """Fill missing values with 0.
    
    Args:
        df: Dataframe to fill missing values in.
        
    Returns:
        pd.DataFrame: Dataframe with missing values filled with 0.
    """
    return df.fillna(0)

def drop_constant_cols(df: pd.DataFrame):
    """Drop constant columns from dataframe.
    
    Args:
        df: Dataframe to drop columns from.
    
    Returns:
        pd.DataFrame: Dataframe with constant columns dropped.
    """
    return df.loc[:, (df != df.iloc[0]).any()]


def transform_covars(
        covars: pd.DataFrame, 
        pd_transforms: Optional[Iterable[Callable]]=None
) -> np.ndarray:
    """Transform covariates into a numpy array.

    Default pandas transformations:
        drop_non_numeric_cols: Drop non-numeric columns.
        fill_na: Fill missing values with 0.
        drop_constant_cols: Drop constant columns.

    Args:
        covars (pd.DataFrame): Covariates.
        pd_transforms: List of functions to apply to covars dataframe.
        np_transforms: List of functions to apply to covars numpy array.

    Returns:
        np.ndarray: Covariates as a numpy array.
    """
    if pd_transforms is None:
        pd_transforms = [
            drop_non_numeric_cols, fill_na, drop_constant_cols
        ]
    
    for transform in pd_transforms:
        covars = transform(covars)
    covars = covars[[col for col in covars.columns if 'geo' not in col and col[:5] != 'meta_']]

    cols = covars.columns
    dd = defaultdict(list)
    for c in cols:
        if "_var_" in c:
            dd[c.split("_var_")[0]].append(c)

    X = covars.values
    X, scale = standardize(X)
    X = add_intercept(X)

    skip = [v[-2] for _, v in dd.items()]
    cc = np.array([1] + [c not in skip for i, c in enumerate(cols)]).astype(bool)
    # skip_corr = np.where(np.triu(np.abs(np.corrcoef(X.T[cc])), k=1) >= (1 - 1e-5))[0]
    # cc[cc] = np.array([True] + [(i not in skip_corr) for i, _ in enumerate(cols[cc[1:]])])
    X = X[:, cc]
    return X, covars.columns[cc[1:]], scale


def  load_from_csv(
        covars_path: str, 
        results_path: str,
        covars_group_col: Optional[str]="group_id", 
        results_group_col: Optional[str]="group_id",
        Y_cols: Optional[Iterable[str]]=["y"],
        N_cols: Optional[Iterable[str]]=["n"]
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Load covariates and results from csvs and split into groups, covars and results.
    Args:
        covars_path: Path to covariates csv.
        results_path: Path to results csv.
        covars_group_col: Name of column in covars csv that contains group labels.
        results_group_col: Name of column in results csv that contains group labels.
    
    Returns:
        tuple: groups, covars, results
    """
    groups_and_covars = pd.read_csv(covars_path)
    groups = groups_and_covars[covars_group_col]
    covars = groups_and_covars.drop(columns=[covars_group_col])
    results = pd.read_csv(results_path).set_index(results_group_col)

    return groups, covars, results, Y_cols, N_cols


def  load_from_long_csv(
        covars_path: str, 
        results_path: str,
        covars_group_col: Optional[str]="group_id", 
        results_group_col: Optional[str]="group_id",
        pivot_col: Optional[str]="outcome",
        Y_col: Optional[str]="y",
        N_col: Optional[str]="n"
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Load covariates and results from csvs and split into groups, covars and results.
    Args:
        covars_path: Path to covariates csv.
        results_path: Path to results csv.
        covars_group_col: Name of column in covars csv that contains group labels.
        results_group_col: Name of column in results csv that contains group labels.
        pivot_col: Name of column in results csv that contains outcome names.
        Y_col: Name of column in results csv that contains outcome values.
        N_col: Name of column in results csv that contains number of observations for the outcome.
    
    Returns:
        tuple: groups, covars, results
    """
    groups, covars, results, _, _ = load_from_csv(
        covars_path, results_path, 
        covars_group_col=covars_group_col, results_group_col=results_group_col
    )
    results = results.pivot(columns=pivot_col)
    
    Y_cols = [col for col in results.columns if Y_col in col and N_col not in col]
    N_cols = [col for col in results.columns if N_col in col]
    return groups, covars, results, Y_cols, N_cols


def transform_from_df(
        groups: pd.Series, covars: pd.DataFrame, results: pd.DataFrame,
        Y_cols: Optional[Iterable[str]]=None, 
        N_cols: Optional[Iterable[str]]=None,
        verbose: Optional[int]=1
) -> Tuple[dict, dict, dict, dict]:
    """Process csvs into dictionaries of covariates and outcomes.
    
    Args:
        groups: group labels corresponding to rows in covars.
        covars: covariates dataframe.
        results: results dataframe with index of group labels, outcomes and number of observations.
        Y_cols: Names of columns in results that contain outcomes.
        N_cols: Names of columns in results that contain number of observations.
        verbose: Verbosity level (0: silent, 1: standard prints, 2: debug prints).
        
    Returns:
        tuple: individual_frame_Xs, individual_frame_Ns, individual_frame_Ys, group_frame_Ns
    """
    if Y_cols is None:
        Y_cols = ["y"]
    if N_cols is None:
        N_cols = ["n"]

    # transform covars
    if verbose > 0:
        print("Transforming covars")
    X, columns, scale = transform_covars(covars)

    # transform to dicts
    if verbose > 0:
        print("Transforming group frame data to dicts")
    group_frame_Ys, group_frame_Ns = group_data_to_dicts(
        results.index, 
        jnp.array(results[Y_cols].values), 
        jnp.array(results[N_cols].values), 
    )
    if verbose > 0:
        print("Transforming individual frame data to dicts")
    individual_frame_Xs, individual_frame_Ns = indidual_data_to_dicts(groups, X)

    # intersect frames, dropping groups that are not in both
    if verbose > 0:
        print("Intersecting frames")
    individual_frame_Xs, individual_frame_Ns, group_frame_Ys, group_frame_Ns = intersect_frames(
        individual_frame_Xs, individual_frame_Ns, group_frame_Ys, group_frame_Ns, verbose=verbose
    )

    # get individual frame Ys by scaling group frame Ys by N_i/N_g
    individual_frame_Ys = get_individual_frame_Ys(
        individual_frame_Ns, group_frame_Ys, group_frame_Ns
    )

    return individual_frame_Xs, individual_frame_Ys, individual_frame_Ns, columns, scale
    

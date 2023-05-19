import numpy as np

def standardize(X, axis=0):
    """Standardize data along axis.

    Args:
        X: Data to standardize.
        axis: Axis along which to standardize.

    Returns:
        tuple: Standardized data and mean along axis.
    """
    mean = np.mean(X, axis=axis)
    std = np.std(X, axis=axis)
    return (X - mean) / std

def save_group_data(path, group_Xs, group_Ys, group_Ns):
    """Save group data to npz file.
    
    Args:
        path: Path to save npz file.
        group_Xs: Group covariates.
        group_Ys: Group outcomes.
        group_Ns: Group number of observations.
    """
    np.savez_compressed(
        path, group_Xs=group_Xs, group_Ys=group_Ys, group_Ns=group_Ns
    )

def load_group_data(path):
    """Load group data from npz file.
    
    Args:
        path: Path to save npz file.

    Returns:
        tuple: group_Xs, group_Ys, group_Ns
    """
    data = np.load(path, allow_pickle=True)
    return data['group_Xs'][()], data['group_Ys'][()], data['group_Ns'][()]

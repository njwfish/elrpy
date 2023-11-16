import numpy as np

def standardize(X, axis=0, eps=1e-8):
    """Standardize data along axis.

    Args:
        X: Data to standardize.
        axis: Axis along which to standardize.

    Returns:
        tuple: Standardized data and mean along axis.
    """
    # max_mask = np.max(np.abs(X), axis=axis) > (1.0 + eps)
    # mean = np.mean(X, axis=axis)
    # std = np.std(X, axis=axis)
    # print(np.max(std))
    # print(np.min(np.abs( (max_mask * std + (1 - max_mask)))))
    # return (X - mean * max_mask) / (max_mask * std + (1 - max_mask)), (mean, std, max_mask)
    max_val = np.max(np.abs(X), axis=axis)
    max_mask = max_val > (1.0 + eps)
    return X / (max_val * max_mask + (1 - max_mask)) , max_val

def add_intercept(X):
    """Add intercept to data.

    Args:
        X: Data to add intercept to.

    Returns:
        tuple: Data with intercept added and intercept.
    """
    intercept = np.ones((X.shape[0], 1))
    return np.hstack((intercept, X))

def save_transform_data(path, columns, scale):
    """Save transform data to npz file.
    
    Args:
        path: Path to save npz file.
        columns: Column names.
        scale: Scale of covariates.
    """
    np.savez(
        path, columns=columns, scale=scale
    )

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

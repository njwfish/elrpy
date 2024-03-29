import jax
import jax.numpy as np
from jax.nn import sigmoid, softmax
from jax.scipy.stats.norm import cdf as probit

def binary_model(model_params, X):
    """Returns the sigmoid of the linear model.

    Args:
        model_params (np.ndarray): model parameters
        X (np.ndarray): data matrix

    Returns:
        np.ndarray: sigmoid of the linear model
    """
    return sigmoid(np.tensordot(X, model_params, axes=1))

def probit_model(model_params, X):
    """Returns the sigmoid of the linear model.

    Args:
        model_params (np.ndarray): model parameters
        X (np.ndarray): data matrix

    Returns:
        np.ndarray: sigmoid of the linear model
    """
    return probit(np.tensordot(X, model_params, axes=1))

def init_binary(group_data):
    """Initializes the model parameters for the binary model.

    Args:
        d (int): number of features
        num_outcomes (int): number of outcomes
        reps (int): number of repetitions
        rng (jax.random.PRNGKey): random seed 

    Returns:
        tuple: model and model parameters
    """
    d = next(iter(group_data[0].values())).shape[1]
    return binary_model, np.zeros((1, d))

def categorical_model(model_params, X):
    """Returns the softmax of the logits of the linear model.
    Note that the last column of the logits is always zero to ensure identifiability.
    
    Args:
        model_params (np.ndarray): model parameters
        X (np.ndarray): data matrix
        
    Returns:
        np.ndarray: softmax of the logits of the linear model
    """
    model_params = model_params.reshape((X.shape[1], -1))
    
    logits = np.tensordot(X, model_params, axes=1)
    logits = np.concatenate([logits, np.zeros((*logits.shape[:-1], 1))], axis=-1)
    return softmax(logits)[..., :-1]

def init_categorical(d, p):
    """Initializes the model parameters for the categorical model.

    Args:
        d (int): number of features
        num_outcomes (int): number of outcomes
        p (int): number of logits
        reps (int): number of repetitions
        rng (jax.random.PRNGKey): random seed
    
    Returns:
        tuple: model and model parameters
    """
    return categorical_model, np.zeros((d * (p - 1),))
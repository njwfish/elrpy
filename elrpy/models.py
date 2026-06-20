import jax
import jax.numpy as np
from jax.nn import sigmoid, softmax

def binary_model(model_params, X):
    """Returns the sigmoid of the linear model.

    Args:
        model_params (np.ndarray): model parameters
        X (np.ndarray): data matrix

    Returns:
        np.ndarray: sigmoid of the linear model
    """
    p = sigmoid(X @ model_params)
    if len(p.shape) == 1:
        p = p[:, None]
    return p

def init_binary(X):
    """Initializes the model parameters for the binary model.

    Args:
        d (int): number of features
        num_outcomes (int): number of outcomes
        reps (int): number of repetitions
        rng (jax.random.PRNGKey): random seed 

    Returns:
        tuple: model and model parameters
    """
    d = X.shape[1]
    return binary_model, np.zeros((d,))

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
    return softmax(logits)

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
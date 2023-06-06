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
    return sigmoid(np.tensordot(X, model_params, axes=1))

def init_binary(d, num_outcomes=1, reps=None, rng=None):
    """Initializes the model parameters for the binary model.

    Args:
        d (int): number of features
        num_outcomes (int): number of outcomes
        reps (int): number of repetitions
        rng (jax.random.PRNGKey): random seed 

    Returns:
        tuple: model and model parameters
    """
    if reps is None:
        beta = jax.random.normal(rng, shape=(d, num_outcomes)) if rng is not None else np.zeros((d, num_outcomes))
    else:
        beta = jax.random.normal(rng, shape=(d, reps, num_outcomes)) if rng is not None else np.zeros((d, reps, num_outcomes))
    return binary_model, beta

def categorical_model(model_params, X):
    """Returns the softmax of the logits of the linear model.
    Note that the last column of the logits is always zero to ensure identifiability.
    
    Args:
        model_params (np.ndarray): model parameters
        X (np.ndarray): data matrix
        
    Returns:
        np.ndarray: softmax of the logits of the linear model
    """
    logits = np.tensordot(X, model_params, axes=1)
    logits = np.concatenate([logits, np.zeros((*logits.shape[:-1], 1))], axis=-1)
    return softmax(logits)

def init_categorical(d, num_outcomes, p, reps=None, rng=None):
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
    p = p - 1
    if reps is None:
        beta = jax.random.normal(rng, shape=(d, num_outcomes, p)) if rng is not None else np.zeros((d, num_outcomes, p))
    else:
        beta = jax.random.normal(rng, shape=(d, reps, num_outcomes, p)) if rng is not None else np.zeros((d, reps, num_outcomes, p))
    return categorical_model, beta
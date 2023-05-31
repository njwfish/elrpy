import jax
import jax.numpy as np
from jax.nn import sigmoid, softmax

def binary_model(model_params, X):
    return sigmoid(np.tensordot(X, model_params, axes=1))

def init_binary(d, num_outcomes=1, reps=None, rng=None):
    if reps is None:
        beta = jax.random.normal(rng, shape=(d, num_outcomes)) if rng is not None else np.zeros((d, num_outcomes))
    else:
        beta = jax.random.normal(rng, shape=(d, reps, num_outcomes)) if rng is not None else np.zeros((d, reps, num_outcomes))
    return binary_model, beta

def categorical_model(model_params, X, reps=1):
    logits = np.tensordot(X, model_params, axes=1)
    return softmax(np.concatenate([logits, np.zeros((logits.shape[0], reps, 1))], axis=-1))

def init_categorical(d, num_outcomes, reps=None, rng=None):
    if reps is None:
        beta = jax.random.normal(rng, shape=(d, num_outcomes)) if rng is not None else np.zeros((d, num_outcomes))
    else:
        beta = jax.random.normal(rng, shape=(d, reps, num_outcomes)) if rng is not None else np.zeros((d, reps, num_outcomes))
    return categorical_model, beta
import jax
import jax.numpy as jnp
from jax.nn import sigmoid

def logistic(model_params, X):
    return sigmoid(X @ model_params)

def init_logistic(d, p=1, reps=None, rng=None):
    if reps is None:
        beta = jax.random.normal(rng, shape=(d, p)) if rng is not None else jnp.zeros((d, p))
    else:
        beta = jax.random.normal(rng, shape=(d, p, reps)) if rng is not None else jnp.zeros((d, p, reps))
    return logistic, beta
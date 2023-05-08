import jax
from jax.nn import sigmoid

def logistic(model_params, X):
    return sigmoid(X @ model_params["beta"])

def init_logistic(rng, d):
    beta = jax.random.normal(rng, shape=(d ,1))
    return logistic, {"beta": beta}
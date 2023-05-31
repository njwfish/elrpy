
from losses import disagg_ce_loss, ecoreg_loss, lyapunov_binary_loss
from optim import fit

from jax import numpy as np

from dgp import normal_sim
from models import binary_model, init_binary

import jax

rng = jax.random.PRNGKey(1)

n, d, k = 100_000, 50, 300

true_params, X, Y, group_data = normal_sim(rng, n, d, k)

losses = [lyapunov_binary_loss]

for loss in losses:
    model_fn, model_params = init_binary(rng, d)
    print("test")
    model_params, grad_norm = fit(
        loss, binary_model, model_params, group_data,
        verbose=2, lr=1e-2, print_every=100
    )
    print(
        np.sqrt(np.sum((true_params - model_params)**2)) / d, 
        np.sqrt(np.sum((model_fn(true_params, X) - model_fn(model_params, X))**2)) / n,
        grad_norm
    )

from losses import disagg_ce_loss, ecoreg_loss, lyapunov_loss
from optim import fit

from jax import numpy as np

from dgp import normal_sim
from models import logistic, init_logistic

import jax

rng = jax.random.PRNGKey(0)

n, d, k = 100_000, 10, 300

beta, X, Y, group_data = normal_sim(rng, n, d, k)

losses = [disagg_ce_loss, lyapunov_loss]

for loss in losses:
    model_fn, model_params = init_logistic(rng, d)
    model_params, grad_norm = fit(
        loss, logistic, model_params, group_data,
        verbose=True, lr=1e-2
    )
    print(np.sum((beta - model_params["beta"])**2), grad_norm)
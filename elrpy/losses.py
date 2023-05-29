import jax
from jax import numpy as np
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


def disagg_ce_loss(p, Y, N):
    p_disagg = Y / N * np.ones_like(p)
    loss = p_disagg * np.log(p) + (1 - p_disagg) * np.log(1-p)
    loss = np.mean(loss)
    return loss


def ecoreg_loss(p, Y, N, eps=1e-6):
    pavg = np.mean(p)
    pavg = (1 - eps) * pavg + eps / 2
    dist = tfd.Binomial(total_count=N, probs=pavg)
    loss = dist.log_prob(Y)
    return loss


def lyapunov_loss(p, Y, N, eps=1e-6):
    # p = (1 - eps) * p + eps / 2
    phi2 = np.sum(p * (1 - p), axis=0)
    mu = np.sum(p, axis=0)
    loss = -1/2 * np.log(phi2) + (1 / phi2) * (Y - mu)**2
    loss = np.mean(loss)
    return loss
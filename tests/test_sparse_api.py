import jax.numpy as jnp

from elrpy.losses import get_wrapped_loss, lyapunov_binary_loss
from elrpy.models import categorical_model, init_categorical
from elrpy.optim import fit


def test_categorical_sparse_loss_shapes():
    X = jnp.array(
        [
            [1.0, -0.5],
            [1.0, 0.2],
            [1.0, 0.8],
            [1.0, -1.0],
        ]
    )
    G = jnp.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    Y = jnp.array(
        [
            [1.2, 0.5, 0.3],
            [0.4, 1.3, 0.3],
        ]
    )

    model_fn, params = init_categorical(X.shape[1], Y.shape[1])
    probs = model_fn(params, X)
    group_loss = lyapunov_binary_loss(probs, Y, G)
    wrapped_loss = get_wrapped_loss(lyapunov_binary_loss, categorical_model)

    assert probs.shape == (X.shape[0], Y.shape[1])
    assert group_loss.shape == Y.shape
    assert jnp.isfinite(wrapped_loss(params, X, Y, G))


def test_sparse_fit_gd_smoke():
    X = jnp.array(
        [
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.5],
            [1.0, 1.0],
        ]
    )
    G = jnp.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    Y = jnp.array(
        [
            [1.0, 0.7, 0.3],
            [0.4, 1.2, 0.4],
        ]
    )

    model_fn, params = init_categorical(X.shape[1], Y.shape[1])
    params, grad_norm, history = fit(
        lyapunov_binary_loss,
        model_fn,
        params,
        (X, Y, G),
        dir_type="gd",
        lr=1e-2,
        maxit=3,
        keep_history=True,
    )

    assert params.shape == (X.shape[1] * (Y.shape[1] - 1),)
    assert jnp.isfinite(grad_norm)
    assert len(history[1]) == 3


def test_scipy_trust_ncg_smoke():
    from elrpy.scipy_optim import fit_scipy

    X = jnp.array(
        [
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.5],
            [1.0, 1.0],
        ]
    )
    G = jnp.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    Y = jnp.array(
        [
            [1.0, 0.7, 0.3],
            [0.4, 1.2, 0.4],
        ]
    )

    model_fn, params = init_categorical(X.shape[1], Y.shape[1])
    result = fit_scipy(
        lyapunov_binary_loss,
        model_fn,
        params,
        (X, Y, G),
        method="trust-ncg",
        hessian="dense",
        dtype="float32",
        gtol=1e-3,
        maxiter=5,
    )

    assert result.params.shape == (X.shape[1] * (Y.shape[1] - 1),)
    assert jnp.isfinite(result.loss)
    assert result.nfev > 0


def test_scipy_hvp_smoke():
    from elrpy.scipy_optim import fit_scipy

    X = jnp.array(
        [
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.5],
            [1.0, 1.0],
        ]
    )
    G = jnp.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    Y = jnp.array(
        [
            [1.0, 0.7, 0.3],
            [0.4, 1.2, 0.4],
        ]
    )

    model_fn, params = init_categorical(X.shape[1], Y.shape[1])
    result = fit_scipy(
        lyapunov_binary_loss,
        model_fn,
        params,
        (X, Y, G),
        method="trust-ncg",
        hessian="hvp",
        dtype="float32",
        gtol=1e-3,
        maxiter=2,
    )

    assert result.params.shape == (X.shape[1] * (Y.shape[1] - 1),)
    assert jnp.isfinite(result.loss)
    assert result.nhev is not None


def test_logit_shifts_match_group_targets():
    from elrpy.shifting import fit_logit_shifts, shifted_group_means

    base_probs = jnp.array(
        [
            [0.7, 0.2, 0.1],
            [0.4, 0.4, 0.2],
            [0.2, 0.5, 0.3],
            [0.2, 0.2, 0.6],
        ]
    )
    G = jnp.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    target = jnp.array(
        [
            [0.55, 0.30, 0.15],
            [0.20, 0.35, 0.45],
        ]
    )

    result = fit_logit_shifts(base_probs, G, target, tol=1e-6, maxiter=20)
    means = shifted_group_means(base_probs, G, result.shifts)

    assert result.converged
    assert result.shifts.shape == (2, 2)
    assert jnp.max(jnp.abs(jnp.asarray(means) - target)) < 1e-5


def test_logit_shift_logit_init_can_converge_without_newton_steps():
    from elrpy.shifting import fit_logit_shifts

    base_probs = jnp.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )
    G = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    target = jnp.array(
        [
            [0.40, 0.40, 0.20],
            [0.10, 0.30, 0.60],
        ]
    )

    initialized = fit_logit_shifts(base_probs, G, target, tol=1e-6, maxiter=0)
    unpolished_zero = fit_logit_shifts(
        base_probs,
        G,
        target,
        tol=1e-6,
        maxiter=0,
        init="zero",
    )
    polished_zero = fit_logit_shifts(
        base_probs,
        G,
        target,
        tol=1e-6,
        maxiter=20,
        init="zero",
    )

    assert initialized.converged
    assert initialized.n_iter == 0
    assert not unpolished_zero.converged
    assert polished_zero.converged

"""Precinct-level logit shifting for categorical EI predictions."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as onp
from jax.experimental import sparse as jsparse
from scipy import sparse as ssparse
from scipy.optimize import least_squares
from scipy.special import logsumexp

ShiftInit = Literal["logit", "zero"]


@dataclass(frozen=True)
class ShiftResult:
    """Result from fitting precinct logit shifts."""

    shifts: onp.ndarray
    loss: float
    grad_norm: float
    n_iter: int
    converged: bool
    seconds: float
    n_refined: int = 0


def _jax_dtype(dtype) -> jnp.dtype:
    np_dtype = onp.dtype(dtype)
    if np_dtype == onp.dtype(onp.float64) and not bool(jax.config.jax_enable_x64):
        raise RuntimeError(
            "Float64 logit shifting requires `jax_enable_x64=True`. "
            "Set `jax.config.update('jax_enable_x64', True)` before calling "
            "the polished shift solver."
        )
    return jnp.dtype(np_dtype.name)


def _as_bcoo(matrix, dtype):
    if isinstance(matrix, jsparse.BCOO):
        return jsparse.BCOO(
            (jnp.asarray(matrix.data, dtype=dtype), matrix.indices),
            shape=matrix.shape,
        )
    if ssparse.issparse(matrix):
        return jsparse.BCOO.from_scipy_sparse(matrix.astype(onp.dtype(dtype).name))
    return jnp.asarray(matrix, dtype=dtype)


def _normalize_probs(probs, dtype, eps: float):
    probs = jnp.asarray(probs, dtype=dtype)
    probs = jnp.clip(probs, eps, 1.0)
    return probs / jnp.sum(probs, axis=1, keepdims=True)


def _normalize_targets(target_means, dtype, eps: float):
    target = jnp.asarray(target_means, dtype=dtype)
    target = jnp.clip(target, eps, 1.0 - eps)
    return target / jnp.sum(target, axis=1, keepdims=True)


def _reference_logits(means, eps: float):
    means = jnp.clip(means, eps, 1.0)
    return jnp.log(means[:, :-1]) - jnp.log(means[:, [-1]])


def _group_indices(matrix, n_groups: int) -> list[onp.ndarray] | None:
    if ssparse.issparse(matrix):
        csr = matrix.tocsr()
        return [
            csr.indices[csr.indptr[group] : csr.indptr[group + 1]]
            for group in range(n_groups)
        ]
    if isinstance(matrix, jsparse.BCOO):
        return None
    dense = onp.asarray(matrix)
    return [onp.flatnonzero(dense[group]) for group in range(n_groups)]


def _tilted_mean(log_probs: onp.ndarray, shifts: onp.ndarray) -> onp.ndarray:
    scores = log_probs + onp.concatenate([shifts, onp.array([0.0])])
    scores = scores - logsumexp(scores, axis=1, keepdims=True)
    return onp.exp(scores).mean(axis=0)


def _refine_problem_groups(
    probs: onp.ndarray,
    group_indices: list[onp.ndarray] | None,
    target: onp.ndarray,
    shifts: onp.ndarray,
    bad_groups: onp.ndarray,
    *,
    eps: float,
    tol: float,
) -> tuple[onp.ndarray, int]:
    if group_indices is None or bad_groups.size == 0:
        return shifts, 0

    refined = shifts.copy()
    n_refined = 0
    for group in bad_groups:
        voters = group_indices[int(group)]
        if voters.size == 0:
            continue
        log_probs = onp.log(onp.clip(probs[voters], eps, 1.0))
        target_group = target[int(group)]
        current_error = onp.max(
            onp.abs(_tilted_mean(log_probs, refined[int(group)]) - target_group)
        )

        def residual(candidate):
            return _tilted_mean(log_probs, candidate)[:-1] - target_group[:-1]

        result = least_squares(
            residual,
            refined[int(group)],
            xtol=min(tol, 1e-12),
            ftol=min(tol, 1e-12),
            gtol=min(tol, 1e-12),
            max_nfev=2000,
        )
        candidate_error = onp.max(
            onp.abs(_tilted_mean(log_probs, result.x) - target_group)
        )
        if candidate_error < current_error:
            refined[int(group)] = result.x
            n_refined += 1
    return refined, n_refined


def fit_logit_shifts(
    base_probs,
    G,
    target_means,
    *,
    dtype="float32",
    tol: float = 1e-5,
    maxiter: int = 100,
    eps: float = 1e-8,
    ridge: float = 1e-6,
    max_step: float = 5.0,
    init: ShiftInit = "logit",
    initial_shifts=None,
) -> ShiftResult:
    """Fit group-level additive logit shifts.

    The shift for each precinct solves

    `mean_i softmax(log(base_prob_i) + shift_g) = target_mean_g`.

    The final category is the reference, so each precinct has `K - 1` free shift
    parameters. The default initializer matches target and current group logits,
    then a grouped Newton method polishes the precinct equations.
    """

    start = perf_counter()
    jax_dtype = _jax_dtype(dtype)
    probs = _normalize_probs(base_probs, jax_dtype, eps)
    log_probs = jnp.log(probs)
    target = _normalize_targets(target_means, jax_dtype, eps)
    n_groups, n_categories = target.shape
    n_free = n_categories - 1
    group_indices = _group_indices(G, n_groups)
    G = _as_bcoo(G, jax_dtype)
    counts = jnp.squeeze(G @ jnp.ones((probs.shape[0], 1), dtype=jax_dtype), axis=1)
    counts = jnp.maximum(counts, 1.0)
    target_counts = counts[:, None] * target

    def append_reference(shifts):
        reference = jnp.zeros((shifts.shape[0], 1), dtype=shifts.dtype)
        return jnp.concatenate([shifts, reference], axis=1)

    def shifted_probs(shifts):
        scores = log_probs + G.T @ append_reference(shifts)
        scores = scores - jnp.max(scores, axis=1, keepdims=True)
        exp_scores = jnp.exp(scores)
        return exp_scores / jnp.sum(exp_scores, axis=1, keepdims=True)

    def diagnostics(shifts):
        shifted = shifted_probs(shifts)
        totals = G @ shifted
        means = totals / counts[:, None]
        mean_error = jnp.max(jnp.abs(means - target))
        grad = totals[:, :n_free] - target_counts[:, :n_free]
        scaled_grad_norm = jnp.max(jnp.linalg.norm(grad / counts[:, None], axis=1))
        return shifted, totals, mean_error, scaled_grad_norm

    def build_hessian(shifted):
        rows = []
        for a in range(n_free):
            cols = []
            for b in range(n_free):
                if a == b:
                    values = shifted[:, a] * (1.0 - shifted[:, a])
                else:
                    values = -shifted[:, a] * shifted[:, b]
                cols.append(G @ values)
            rows.append(jnp.stack(cols, axis=1))
        return jnp.stack(rows, axis=1)

    @jax.jit
    def newton_step(shifts):
        shifted, totals, mean_error, scaled_grad_norm = diagnostics(shifts)
        grad = totals[:, :n_free] - target_counts[:, :n_free]
        hessian = build_hessian(shifted)
        eye = jnp.eye(n_free, dtype=jax_dtype)[None, :, :]
        hessian = hessian + ridge * counts[:, None, None] * eye
        update = jnp.linalg.solve(hessian, grad[:, :, None])[:, :, 0]
        update_norm = jnp.linalg.norm(update, axis=1, keepdims=True)
        update = update * jnp.minimum(1.0, max_step / jnp.maximum(update_norm, eps))
        return shifts - update, mean_error, scaled_grad_norm

    @jax.jit
    def final_diagnostics(shifts):
        _, _, mean_error, scaled_grad_norm = diagnostics(shifts)
        return mean_error, scaled_grad_norm

    if initial_shifts is not None:
        shifts = jnp.asarray(initial_shifts, dtype=jax_dtype)
        if shifts.shape != (n_groups, n_free):
            raise ValueError(
                "initial_shifts must have shape "
                f"{(n_groups, n_free)}, got {shifts.shape}"
            )
    elif init == "zero":
        shifts = jnp.zeros((n_groups, n_free), dtype=jax_dtype)
    elif init == "logit":
        base_means = (G @ probs) / counts[:, None]
        shifts = _reference_logits(target, eps) - _reference_logits(base_means, eps)
    else:
        raise ValueError("init must be 'logit' or 'zero'")

    loss = onp.inf
    grad_norm = onp.inf
    converged = False
    n_iter = 0
    for n_iter in range(maxiter + 1):
        loss_value, grad_value = final_diagnostics(shifts)
        jax.block_until_ready(loss_value)
        loss = float(jax.device_get(loss_value))
        grad_norm = float(jax.device_get(grad_value))
        if loss <= tol:
            converged = True
            break
        if n_iter == maxiter:
            break
        shifts, _, _ = newton_step(shifts)
        jax.block_until_ready(shifts)

    n_refined = 0
    if not converged and maxiter > 0:
        shifted, totals, _, _ = diagnostics(shifts)
        means = totals / counts[:, None]
        errors = jnp.max(jnp.abs(means - target), axis=1)
        bad_groups = onp.flatnonzero(onp.asarray(jax.device_get(errors)) > tol)
        refined_shifts, n_refined = _refine_problem_groups(
            onp.asarray(jax.device_get(probs), dtype=onp.float64),
            group_indices,
            onp.asarray(jax.device_get(target), dtype=onp.float64),
            onp.asarray(jax.device_get(shifts), dtype=onp.float64),
            bad_groups,
            eps=eps,
            tol=tol,
        )
        if n_refined:
            shifts = jnp.asarray(refined_shifts, dtype=jax_dtype)
            loss_value, grad_value = final_diagnostics(shifts)
            jax.block_until_ready(loss_value)
            loss = float(jax.device_get(loss_value))
            grad_norm = float(jax.device_get(grad_value))
            converged = loss <= tol

    return ShiftResult(
        shifts=onp.asarray(jax.device_get(shifts), dtype=onp.float64),
        loss=loss,
        grad_norm=grad_norm,
        n_iter=n_iter,
        converged=converged,
        seconds=perf_counter() - start,
        n_refined=n_refined,
    )


def apply_logit_shifts(base_probs, G, shifts, *, dtype="float32", eps: float = 1e-8):
    """Apply fitted precinct shifts and return shifted probabilities."""

    jax_dtype = _jax_dtype(dtype)
    probs = _normalize_probs(base_probs, jax_dtype, eps)
    G = _as_bcoo(G, jax_dtype)
    shifts = jnp.asarray(shifts, dtype=jax_dtype)
    reference = jnp.zeros((shifts.shape[0], 1), dtype=jax_dtype)
    full_shifts = jnp.concatenate([shifts, reference], axis=1)
    scores = jnp.log(probs) + G.T @ full_shifts
    scores = scores - jnp.max(scores, axis=1, keepdims=True)
    exp_scores = jnp.exp(scores)
    shifted = exp_scores / jnp.sum(exp_scores, axis=1, keepdims=True)
    return onp.asarray(jax.device_get(shifted))


def shifted_group_means(base_probs, G, shifts, *, dtype="float32"):
    """Return shifted precinct means without materializing custom group ids."""

    shifted = apply_logit_shifts(base_probs, G, shifts, dtype=dtype)
    jax_dtype = _jax_dtype(dtype)
    if isinstance(G, jsparse.BCOO):
        totals = G @ jnp.asarray(shifted, dtype=jax_dtype)
        counts = G @ jnp.ones((shifted.shape[0], 1), dtype=jax_dtype)
        means = totals / counts
        return onp.asarray(jax.device_get(means))
    if ssparse.issparse(G):
        counts = onp.asarray(G.sum(axis=1)).ravel()
        return onp.asarray(G @ shifted) / counts[:, None]
    G_dense = onp.asarray(G)
    counts = G_dense.sum(axis=1)
    return (G_dense @ shifted) / counts[:, None]

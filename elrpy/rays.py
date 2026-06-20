"""Ray checks for ELR objectives."""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.experimental import sparse as jsparse


def make_ray_directions(
    n_params: int,
    *,
    n_random: int = 0,
    seed: int = 0,
    include_antipodes: bool = True,
    include_coordinates: bool = False,
    gradient: Sequence[float] | None = None,
) -> np.ndarray:
    """Return unit directions for ray checks."""

    if n_params <= 0:
        raise ValueError("n_params must be positive")
    if n_random < 0:
        raise ValueError("n_random must be nonnegative")

    pieces = []
    if n_random:
        rng = np.random.default_rng(seed)
        random = rng.normal(size=(n_random, n_params))
        norms = np.linalg.norm(random, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("random generator produced a zero direction")
        random = random / norms
        pieces.append(random)
        if include_antipodes:
            pieces.append(-random)

    if include_coordinates:
        coordinates = np.eye(n_params)
        pieces.append(coordinates)
        if include_antipodes:
            pieces.append(-coordinates)

    if gradient is not None:
        gradient = np.asarray(gradient, dtype=np.float64)
        if gradient.shape != (n_params,):
            raise ValueError(f"gradient must have shape {(n_params,)}")
        norm = np.linalg.norm(gradient)
        if norm > 0:
            direction = gradient / norm
            pieces.append(np.vstack([-direction, direction]))

    if not pieces:
        raise ValueError("requested direction set is empty")
    return np.vstack(pieces).astype(np.float64, copy=False)


def _jax_dtype(dtype) -> jnp.dtype:
    np_dtype = np.dtype(dtype)
    if np_dtype == np.dtype(np.float64) and not bool(jax.config.jax_enable_x64):
        raise RuntimeError("Float64 ray checks require jax_enable_x64=True")
    return jnp.dtype(np_dtype.name)


def _as_bcoo(matrix, dtype):
    if isinstance(matrix, jsparse.BCOO):
        return jsparse.BCOO(
            (jnp.asarray(matrix.data, dtype=dtype), matrix.indices),
            shape=matrix.shape,
        )
    if sp.issparse(matrix):
        return jsparse.BCOO.from_scipy_sparse(matrix.astype(np.dtype(dtype).name))
    return jnp.asarray(matrix, dtype=dtype)


def evaluate_categorical_loss_rays(
    X,
    Y,
    G,
    center_params,
    directions,
    radii,
    *,
    n_outcomes: int,
    dtype: str = "float32",
    batch_size: int = 8,
    eps: float = 1e-6,
) -> np.ndarray:
    """Evaluate the categorical normal loss on `center_params + r * direction`."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    radii = np.asarray(radii, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if radii.ndim != 1 or radii.size == 0:
        raise ValueError("radii must be a nonempty one-dimensional array")
    if directions.ndim != 2 or directions.shape[0] == 0:
        raise ValueError("directions must be a nonempty two-dimensional array")
    if n_outcomes < 2:
        raise ValueError("n_outcomes must be at least two")

    jax_dtype = _jax_dtype(dtype)
    X = jnp.asarray(X, dtype=jax_dtype)
    Y = jnp.asarray(Y, dtype=jax_dtype)
    G = _as_bcoo(G, jax_dtype)
    center_params = jnp.asarray(center_params, dtype=jax_dtype)
    directions = jnp.asarray(directions, dtype=jax_dtype)

    n_features = X.shape[1]
    n_free = n_outcomes - 1
    n_params = int(n_features * n_free)
    if center_params.shape != (n_params,):
        raise ValueError(f"center_params must have shape {(n_params,)}")
    if directions.shape[1] != n_params:
        raise ValueError(f"directions must have {n_params} columns")

    base_logits = X @ center_params.reshape((n_features, n_free))
    jax.block_until_ready(base_logits)

    @jax.jit
    def losses_for_radii(direction, radius_batch):
        direction_logits = X @ direction.reshape((n_features, n_free))

        def loss_at_radius(radius):
            free_logits = base_logits + radius * direction_logits
            reference = jnp.zeros((free_logits.shape[0], 1), dtype=jax_dtype)
            logits = jnp.concatenate([free_logits, reference], axis=1)
            probs = jax.nn.softmax(logits, axis=1)
            means = G @ probs
            variances = jnp.maximum(G @ (probs * (1.0 - probs)), eps)
            return jnp.mean(
                0.5 * (jnp.log(variances) + ((Y - means) ** 2) / variances)
            )

        return jax.vmap(loss_at_radius)(radius_batch)

    radius_batch = np.empty(batch_size, dtype=np.dtype(dtype))
    rows = []
    for direction in directions:
        pieces = []
        for start in range(0, radii.size, batch_size):
            stop = min(start + batch_size, radii.size)
            count = stop - start
            radius_batch[:count] = radii[start:stop]
            if count < batch_size:
                radius_batch[count:] = radii[stop - 1]
            values = losses_for_radii(direction, jnp.asarray(radius_batch))
            jax.block_until_ready(values)
            pieces.append(np.asarray(jax.device_get(values[:count]), dtype=np.float64))
        rows.append(np.concatenate(pieces))
    return np.vstack(rows)


def summarize_ray_losses(
    losses: np.ndarray,
    radii: Sequence[float],
    *,
    directions: np.ndarray | None = None,
    gradient: Sequence[float] | None = None,
    tolerance: float = 1e-8,
    derivative_tolerance: float | None = None,
    top_k: int = 5,
) -> dict[str, object]:
    """Summarize whether sampled rays contradict local optimality."""

    losses = np.asarray(losses, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if losses.ndim != 2:
        raise ValueError("losses must be two-dimensional")
    if radii.ndim != 1 or radii.size != losses.shape[1]:
        raise ValueError("radii must match the loss columns")
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    if derivative_tolerance is None:
        derivative_tolerance = tolerance

    diffs = np.diff(losses, axis=1)
    min_index = np.argmin(losses, axis=1)
    drop = losses[np.arange(losses.shape[0]), min_index] - losses[:, 0]
    step_drop = np.min(diffs, axis=1) if diffs.size else np.zeros(losses.shape[0])

    non_monotone = np.any(diffs < -tolerance, axis=1)
    lower = drop < -tolerance

    def top(indices, key):
        return [
            {
                "direction_index": int(index),
                "drop_from_center": float(drop[index]),
                "min_loss_radius": float(radii[min_index[index]]),
                "worst_step_delta": float(step_drop[index]),
            }
            for index in sorted(indices, key=key)[:top_k]
        ]

    summary: dict[str, object] = {
        "n_directions": int(losses.shape[0]),
        "n_radii": int(losses.shape[1]),
        "radius_max": float(radii.max()),
        "tolerance": float(tolerance),
        "lower_than_center_directions": int(lower.sum()),
        "non_monotone_directions": int(non_monotone.sum()),
        "monotone_directions": int((~non_monotone).sum()),
        "worst_drop_from_center": float(drop.min()),
        "worst_step_delta": float(step_drop.min()),
        "top_lower_than_center": top(np.where(lower)[0], lambda index: drop[index]),
        "top_non_monotone": top(
            np.where(non_monotone)[0],
            lambda index: (drop[index], step_drop[index]),
        ),
    }

    if gradient is not None:
        if directions is None:
            raise ValueError("directions are required to summarize derivatives")
        directions = np.asarray(directions, dtype=np.float64)
        gradient = np.asarray(gradient, dtype=np.float64)
        if directions.shape[0] != losses.shape[0]:
            raise ValueError("directions and losses must have the same rows")
        if gradient.shape != (directions.shape[1],):
            raise ValueError("gradient shape must match the directions")
        derivatives = directions @ gradient
        descending = derivatives < -derivative_tolerance
        summary.update(
            {
                "derivative_tolerance": float(derivative_tolerance),
                "negative_initial_derivative_directions": int(descending.sum()),
                "worst_initial_derivative": float(derivatives.min()),
                "top_negative_initial_derivative": [
                    {
                        "direction_index": int(index),
                        "initial_derivative": float(derivatives[index]),
                    }
                    for index in sorted(
                        np.where(descending)[0],
                        key=lambda index: derivatives[index],
                    )[:top_k]
                ],
            }
        )
    return summary

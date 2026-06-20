"""Delta-method and Wald inference utilities for ELR estimates."""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from statistics import NormalDist

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.stats import multivariate_normal


def _binary_probs(probs, outcome: int | None) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        return probs
    if probs.ndim != 2:
        raise ValueError("shifted_probs must be one- or two-dimensional")
    if outcome is None:
        if probs.shape[1] != 2:
            raise ValueError(
                "pass outcome when shifted_probs has more than two columns"
            )
        outcome = 0
    if not 0 <= outcome < probs.shape[1]:
        raise ValueError("outcome is out of bounds for shifted_probs")
    return probs[:, outcome]


def _group_ids(groups, n_voters: int) -> np.ndarray:
    if sp.issparse(groups):
        groups = groups.tocsr()
        if groups.shape[1] != n_voters:
            raise ValueError("group matrix must have one column per voter")
        counts = np.diff(groups.tocsc().indptr)
        if np.any(counts != 1):
            raise ValueError("each voter must belong to exactly one group")
        coo = groups.tocoo()
        return np.asarray(coo.row[np.argsort(coo.col)])

    groups = np.asarray(groups)
    if groups.ndim == 1:
        if groups.shape != (n_voters,):
            raise ValueError("group_ids must have one entry per row of X")
        return groups
    if groups.ndim == 2:
        if groups.shape[1] != n_voters:
            raise ValueError("group matrix must have one column per voter")
        membership = groups != 0
        if np.any(membership.sum(axis=0) != 1):
            raise ValueError("each voter must belong to exactly one group")
        return np.argmax(membership, axis=0)
    raise ValueError("groups must be group ids or a group-by-voter matrix")


def _group_matrix(groups, n_voters: int) -> sp.csr_matrix:
    if sp.issparse(groups):
        groups = groups.tocsr()
        if groups.shape[1] != n_voters:
            raise ValueError("group matrix must have one column per voter")
        return groups.astype(np.float64)

    group_ids = np.asarray(groups)
    if group_ids.ndim != 1 or group_ids.shape != (n_voters,):
        raise ValueError("group_ids must have one entry per voter")
    if np.any(group_ids < 0):
        raise ValueError("group ids must be nonnegative")
    n_groups = int(group_ids.max()) + 1 if group_ids.size else 0
    return sp.csr_matrix(
        (np.ones(n_voters, dtype=np.float64), (group_ids, np.arange(n_voters))),
        shape=(n_groups, n_voters),
    )


def _cov_matrix(value, n_features: int, name: str) -> np.ndarray:
    if value is None:
        return np.zeros((n_features, n_features), dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (n_features, n_features):
        raise ValueError(f"{name} must have shape {(n_features, n_features)}")
    return value


def _cov_vector(value, n_features: int, name: str) -> np.ndarray:
    if value is None:
        return np.zeros(n_features, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (n_features,):
        raise ValueError(f"{name} must have shape {(n_features,)}")
    return value


def invert_information(hessian, *, rcond: float = 1e-10) -> np.ndarray:
    """Return a symmetric generalized inverse for a positive information matrix."""

    hessian = np.asarray(hessian, dtype=np.float64)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be a square matrix")
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    largest = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    keep = eigenvalues > largest * rcond
    if not np.any(keep):
        return np.zeros_like(hessian)
    inverse = (eigenvectors[:, keep] / eigenvalues[keep]) @ eigenvectors[:, keep].T
    return 0.5 * (inverse + inverse.T)


def load_covariance_and_bread(path, *, covariance_name: str = "covariance") -> tuple[np.ndarray, np.ndarray]:
    """Load a saved beta covariance and Hessian bread from an NPZ artifact."""

    with np.load(path, allow_pickle=False) as npz:
        if covariance_name not in npz.files:
            raise ValueError(f"{path} does not contain {covariance_name!r}")
        if "hessian" not in npz.files:
            raise ValueError(f"{path} does not contain 'hessian'")
        covariance = np.asarray(npz[covariance_name], dtype=np.float64)
        hessian = np.asarray(npz["hessian"], dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, invert_information(hessian)


def _gradient_covariance_pair(gradients, covariance) -> tuple[np.ndarray, np.ndarray]:
    gradients = np.asarray(gradients, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if gradients.ndim != 2:
        raise ValueError("gradients must be two-dimensional")
    if covariance.shape != (gradients.shape[1], gradients.shape[1]):
        raise ValueError(
            "covariance shape does not match gradients: "
            f"{covariance.shape} vs {gradients.shape}"
        )
    return gradients, covariance


def _gradient_covariance_blocks(
    gradients,
    covariance,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if isinstance(gradients, (list, tuple)) or isinstance(covariance, (list, tuple)):
        if not isinstance(gradients, (list, tuple)) or not isinstance(
            covariance,
            (list, tuple),
        ):
            raise ValueError("gradients and covariance must both be block sequences")
        if len(gradients) != len(covariance):
            raise ValueError(
                "gradient and covariance block sequences must have the same length"
            )
        if not gradients:
            raise ValueError("at least one covariance block is required")
        return [
            _gradient_covariance_pair(block_gradients, block_covariance)
            for block_gradients, block_covariance in zip(gradients, covariance)
        ]
    return [_gradient_covariance_pair(gradients, covariance)]


def wald_interval(
    estimate: float,
    se: float,
    *,
    level: float = 0.95,
) -> tuple[float, float]:
    """Return an untruncated two-sided Wald interval."""

    if not 0.0 < level < 1.0:
        raise ValueError("level must be between zero and one")
    if se < 0:
        raise ValueError("se must be nonnegative")
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    return float(estimate - z * se), float(estimate + z * se)


def delta_covariance(gradients, covariance) -> np.ndarray:
    """Return the delta-method covariance of a target vector.

    Pass a single gradient matrix and covariance matrix, or matching lists of
    gradient and covariance blocks for independent parameter blocks.
    """

    pieces = [
        block_gradients @ block_covariance @ block_gradients.T
        for block_gradients, block_covariance in _gradient_covariance_blocks(
            gradients,
            covariance,
        )
    ]
    shape = pieces[0].shape
    if any(piece.shape != shape for piece in pieces):
        raise ValueError("all target covariance pieces must have the same shape")
    out = np.sum(pieces, axis=0)
    return 0.5 * (out + out.T)


def delta_variance(gradients, covariance) -> np.ndarray:
    """Return the diagonal of the delta-method covariance."""

    pieces = [
        np.einsum(
            "ip,pq,iq->i",
            block_gradients,
            block_covariance,
            block_gradients,
            optimize=True,
        )
        for block_gradients, block_covariance in _gradient_covariance_blocks(
            gradients,
            covariance,
        )
    ]
    return np.sum(pieces, axis=0)


def _covariance_to_correlation(
    covariance,
    standard_errors=None,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix.

    Zero-variance coordinates are left with zero off-diagonal correlations and
    a zero diagonal entry so simultaneous procedures can drop them cleanly.
    """

    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if standard_errors is None:
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    standard_errors = np.asarray(standard_errors, dtype=np.float64)
    if standard_errors.shape != (covariance.shape[0],):
        raise ValueError("standard_errors must match the covariance dimension")

    correlation = np.zeros_like(covariance, dtype=np.float64)
    active = standard_errors > eps
    if np.any(active):
        denominator = np.outer(standard_errors[active], standard_errors[active])
        correlation[np.ix_(active, active)] = (
            covariance[np.ix_(active, active)] / denominator
        )
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, np.where(active, 1.0, 0.0))
    return correlation


def _max_t_critical_value(
    correlation,
    *,
    standard_errors=None,
    level: float = 0.95,
    n_draws: int = 200_000,
    seed: int | None = None,
    batch_size: int = 50_000,
    eps: float = 1e-12,
) -> float:
    """Return the Gaussian max-|t| critical value for simultaneous Wald bands."""

    if not 0.0 < level < 1.0:
        raise ValueError("level must be between zero and one")
    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    correlation = np.asarray(correlation, dtype=np.float64)
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation must be square")
    if standard_errors is None:
        active = np.ones(correlation.shape[0], dtype=bool)
    else:
        standard_errors = np.asarray(standard_errors, dtype=np.float64)
        if standard_errors.shape != (correlation.shape[0],):
            raise ValueError("standard_errors must match the correlation dimension")
        active = standard_errors > eps
    if not np.any(active):
        return 0.0

    active_correlation = correlation[np.ix_(active, active)]
    active_correlation = 0.5 * (active_correlation + active_correlation.T)
    eigenvalues, eigenvectors = np.linalg.eigh(active_correlation)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    transform = eigenvectors * np.sqrt(eigenvalues)
    rng = np.random.default_rng(seed)
    max_values = np.empty(n_draws, dtype=np.float64)
    written = 0
    while written < n_draws:
        size = min(batch_size, n_draws - written)
        draws = rng.standard_normal((size, active.sum())) @ transform.T
        max_values[written : written + size] = np.max(np.abs(draws), axis=1)
        written += size
    return float(np.quantile(max_values, level))


def simultaneous_wald_intervals(
    estimates,
    covariance,
    *,
    level: float = 0.95,
    n_draws: int = 200_000,
    seed: int | None = None,
    lower_bound=None,
    upper_bound=None,
    eps: float = 1e-12,
) -> dict[str, np.ndarray | float]:
    """Return simultaneous Gaussian max-t Wald intervals for a target vector."""

    estimates = np.asarray(estimates, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if estimates.ndim != 1:
        raise ValueError("estimates must be one-dimensional")
    if covariance.shape != (estimates.size, estimates.size):
        raise ValueError("covariance shape must match estimates")

    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    correlation = _covariance_to_correlation(
        covariance,
        standard_errors,
        eps=eps,
    )
    critical_value = _max_t_critical_value(
        correlation,
        standard_errors=standard_errors,
        level=level,
        n_draws=n_draws,
        seed=seed,
        eps=eps,
    )
    lower = estimates - critical_value * standard_errors
    upper = estimates + critical_value * standard_errors
    if lower_bound is not None:
        lower = np.maximum(lower, np.asarray(lower_bound, dtype=np.float64))
    if upper_bound is not None:
        upper = np.minimum(upper, np.asarray(upper_bound, dtype=np.float64))
    return {
        "lower": lower,
        "upper": upper,
        "se": standard_errors,
        "correlation": correlation,
        "critical_value": critical_value,
    }


def _grouped_multinomial_covariances(
    probs: np.ndarray,
    groups: sp.csr_matrix,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return grouped multinomial covariance sums.

    For voter-level probabilities `p_i`, let
    `C_i = diag(p_i) - p_i p_i'`. The three returned arrays are
    `sum C_i`, `sum w_i C_i`, and `sum w_i^2 C_i` within each group.
    """

    n_groups = groups.shape[0]
    n_categories = probs.shape[1]
    precinct = np.empty((n_groups, n_categories, n_categories), dtype=np.float64)
    subgroup = np.empty_like(precinct)
    subgroup_noise = np.empty_like(precinct)
    for a in range(n_categories):
        for b in range(n_categories):
            if a == b:
                voter_cov = probs[:, a] * (1.0 - probs[:, a])
            else:
                voter_cov = -probs[:, a] * probs[:, b]
            precinct[:, a, b] = np.asarray(groups @ voter_cov).ravel()
            subgroup[:, a, b] = np.asarray(groups @ (weights * voter_cov)).ravel()
            subgroup_noise[:, a, b] = np.asarray(
                groups @ (weights * weights * voter_cov)
            ).ravel()
    return precinct, subgroup, subgroup_noise


def _grouped_softmax_beta_jacobians(
    probs: np.ndarray,
    groups: sp.csr_matrix,
    weights: np.ndarray,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return derivatives of precinct and subgroup totals with respect to beta."""

    n_groups = groups.shape[0]
    n_features = X.shape[1]
    n_categories = probs.shape[1]
    n_free = n_categories - 1
    n_params = n_features * n_free
    precinct = np.empty((n_groups, n_free, n_params), dtype=np.float64)
    subgroup = np.empty((n_groups, n_categories, n_params), dtype=np.float64)
    for free_category in range(n_free):
        for feature in range(n_features):
            param = feature * n_free + free_category
            x_col = X[:, feature]
            for outcome in range(n_categories):
                jacobian = probs[:, outcome] * (
                    (1.0 if outcome == free_category else 0.0)
                    - probs[:, free_category]
                )
                if outcome < n_free:
                    precinct[:, outcome, param] = np.asarray(
                        groups @ (x_col * jacobian)
                    ).ravel()
                subgroup[:, outcome, param] = np.asarray(
                    groups @ (weights * x_col * jacobian)
                ).ravel()
    return precinct, subgroup


def _grouped_softmax_score_jacobians(
    probs: np.ndarray,
    groups: sp.csr_matrix,
    X: np.ndarray,
    *,
    variance_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return grouped Normal-score moment derivatives for categorical fits."""

    groups = groups.tocsr()
    n_groups = groups.shape[0]
    n_features = X.shape[1]
    n_categories = probs.shape[1]
    n_free = n_categories - 1
    n_params = n_features * n_free
    means = np.asarray(groups @ probs)
    raw_variances = np.asarray(groups @ (probs * (1.0 - probs)))
    variances = np.maximum(raw_variances, variance_floor)
    mean_jacobian = np.empty((n_groups, n_categories, n_params), dtype=np.float64)
    variance_jacobian = np.empty_like(mean_jacobian)

    for free_category in range(n_free):
        params = slice(free_category, n_params, n_free)
        for outcome in range(n_categories):
            derivative = probs[:, outcome] * (
                (1.0 if outcome == free_category else 0.0)
                - probs[:, free_category]
            )
            grouped_mean = np.asarray(groups @ (X * derivative[:, None]))
            grouped_variance = np.asarray(
                groups @ (X * (derivative * (1.0 - 2.0 * probs[:, outcome]))[:, None])
            )
            inactive = raw_variances[:, outcome] <= variance_floor
            if np.any(inactive):
                grouped_variance[inactive] = 0.0
            mean_jacobian[:, outcome, params] = grouped_mean
            variance_jacobian[:, outcome, params] = grouped_variance

    return means, variances, mean_jacobian, variance_jacobian


def _grouped_count_third_moments(
    probs: np.ndarray,
    groups: sp.csr_matrix,
    n_free: int,
) -> np.ndarray:
    """Return E[(Y_k-EY_k)^2 (Y_l-EY_l)] for grouped categorical counts."""

    n_groups = groups.shape[0]
    n_categories = probs.shape[1]
    moments = np.empty((n_groups, n_categories, n_free), dtype=np.float64)
    for outcome in range(n_categories):
        p_outcome = probs[:, outcome]
        for free_category in range(n_free):
            p_free = probs[:, free_category]
            if outcome == free_category:
                voter_moment = p_outcome * (1.0 - p_outcome) * (
                    1.0 - 2.0 * p_outcome
                )
            else:
                voter_moment = p_outcome * p_free * (2.0 * p_outcome - 1.0)
            moments[:, outcome, free_category] = np.asarray(
                groups @ voter_moment
            ).ravel()
    return moments


def _normal_score_count_covariance(
    count_probs: np.ndarray,
    score_probs: np.ndarray,
    count_covariance: np.ndarray,
    groups: sp.csr_matrix,
    X: np.ndarray,
    *,
    variance_floor: float = 1e-6,
) -> np.ndarray:
    """Return Cov(count noise, grouped Normal score) for each precinct."""

    if score_probs.shape != count_probs.shape:
        raise ValueError("score_probs must have the same shape as shifted_probs")
    if np.any((score_probs < 0.0) | (score_probs > 1.0)):
        raise ValueError("score_probs must be probabilities")
    row_sums = score_probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("rows of score_probs must have positive mass")
    score_probs = score_probs / row_sums

    n_categories = count_probs.shape[1]
    n_free = n_categories - 1
    count_means = np.asarray(groups @ count_probs)
    score_means, score_variances, mean_jacobian, variance_jacobian = (
        _grouped_softmax_score_jacobians(
            score_probs,
            groups,
            X,
            variance_floor=variance_floor,
        )
    )
    third_moments = _grouped_count_third_moments(count_probs, groups, n_free)
    score_count = np.zeros(
        (groups.shape[0], n_free, X.shape[1] * n_free),
        dtype=np.float64,
    )
    mean_offset = count_means - score_means

    for outcome in range(n_categories):
        cov_with_outcome = count_covariance[:, outcome, :n_free]
        variance = score_variances[:, outcome]
        mean_scale = -cov_with_outcome / variance[:, None]
        mixed_third = (
            third_moments[:, outcome, :]
            + 2.0 * mean_offset[:, outcome, None] * cov_with_outcome
        )
        variance_scale = -0.5 * mixed_third / (variance[:, None] ** 2)
        score_count += np.einsum(
            "gl,gp->glp",
            mean_scale,
            mean_jacobian[:, outcome, :],
            optimize=True,
        )
        score_count += np.einsum(
            "gl,gp->glp",
            variance_scale,
            variance_jacobian[:, outcome, :],
            optimize=True,
        )

    return score_count / n_categories


def _grouped_softmax_mean_score_jacobians(
    probs: np.ndarray,
    groups: sp.csr_matrix,
    X: np.ndarray,
    *,
    variance_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grouped mean-score Jacobians and variances for vote-linear scores."""

    groups = groups.tocsr()
    n_groups = groups.shape[0]
    n_features = X.shape[1]
    n_categories = probs.shape[1]
    n_free = n_categories - 1
    n_params = n_features * n_free
    raw_variances = np.asarray(groups @ (probs * (1.0 - probs)))
    variances = np.maximum(raw_variances, variance_floor)
    mean_jacobian = np.empty((n_groups, n_categories, n_params), dtype=np.float64)

    for free_category in range(n_free):
        params = slice(free_category, n_params, n_free)
        for outcome in range(n_categories):
            derivative = probs[:, outcome] * (
                (1.0 if outcome == free_category else 0.0)
                - probs[:, free_category]
            )
            mean_jacobian[:, outcome, params] = np.asarray(
                groups @ (X * derivative[:, None])
            )

    return variances, mean_jacobian


def categorical_score_linearization(
    score_probs,
    groups,
    X,
    *,
    variance_floor: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Precompute grouped score pieces used by exact compatible covariance.

    The returned object is intentionally a plain dictionary so scripts can cache
    it alongside a loaded fit and pass it back to
    ``shifted_categorical_endpoint_linear_coefficients``.  It represents the
    vote-linear part of the aggregate Normal score.
    """

    probs = np.asarray(score_probs, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("score_probs must be two-dimensional")
    if X.shape[0] != probs.shape[0]:
        raise ValueError("X must have one row per voter")
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("rows of score_probs must have positive mass")
    probs = probs / row_sums
    G = _group_matrix(groups, probs.shape[0])
    variances, mean_jacobian = _grouped_softmax_mean_score_jacobians(
        probs,
        G,
        X,
        variance_floor=variance_floor,
    )
    return {
        "group_ids": _group_ids(G, probs.shape[0]),
        "variances": variances,
        "mean_jacobian": mean_jacobian,
        "n_groups": np.asarray(G.shape[0], dtype=np.int64),
        "n_categories": np.asarray(probs.shape[1], dtype=np.int64),
    }


def _endpoint_shift_jacobians(
    probs: np.ndarray,
    groups: sp.csr_matrix,
    weights: np.ndarray,
    outcomes: np.ndarray,
) -> np.ndarray:
    """Return endpoint derivatives with respect to free logit shifts."""

    n_targets = weights.shape[1]
    n_groups = groups.shape[0]
    n_categories = probs.shape[1]
    n_free = n_categories - 1
    out = np.zeros((n_targets, n_groups, n_free), dtype=np.float64)
    for target, outcome in enumerate(outcomes):
        if not 0 <= outcome < n_categories:
            continue
        target_weight = weights[:, target]
        for free_category in range(n_free):
            derivative = probs[:, outcome] * (
                (1.0 if outcome == free_category else 0.0)
                - probs[:, free_category]
            )
            out[target, :, free_category] = np.asarray(
                groups @ (target_weight * derivative)
            ).ravel()
    return out


def shifted_categorical_endpoint_covariance(
    shifted_probs,
    groups,
    endpoint_weights,
    endpoint_outcomes,
    *,
    X,
    beta_cov,
    beta_bread,
    score_probs,
    beta_gradients=None,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Return covariance pieces for linear logit-shifted endpoint functionals.

    Each target is

    ``sum_i endpoint_weights[i, m] * shifted_probs[i, endpoint_outcomes[m]]``.

    The returned covariance uses the calibrated-count derivative, the supplied
    beta sandwich covariance, and the same-target slope-count reuse covariance
    implied by ``beta_bread`` and the aggregate Normal score probabilities.
    """

    probs = np.asarray(shifted_probs, dtype=np.float64)
    weights = np.asarray(endpoint_weights, dtype=np.float64)
    outcomes = np.asarray(endpoint_outcomes, dtype=np.int64)
    X = np.asarray(X, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("shifted_probs must be two-dimensional")
    if weights.ndim == 1:
        weights = weights[:, None]
    if weights.ndim != 2 or weights.shape[0] != probs.shape[0]:
        raise ValueError("endpoint_weights must be voter-by-target")
    if outcomes.shape != (weights.shape[1],):
        raise ValueError("endpoint_outcomes must have one entry per target")
    if X.shape[0] != probs.shape[0]:
        raise ValueError("X must have one row per voter")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("shifted_probs must be probabilities")
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= eps):
        raise ValueError("rows of shifted_probs must have positive mass")
    probs = probs / row_sums

    G = _group_matrix(groups, probs.shape[0])
    n_groups = G.shape[0]
    n_categories = probs.shape[1]
    n_free = n_categories - 1
    n_features = X.shape[1]
    n_params = n_features * n_free
    beta_cov = _cov_matrix(beta_cov, n_params, "beta_cov")
    beta_bread = _cov_matrix(beta_bread, n_params, "beta_bread")

    precinct_covariance, _, _ = _grouped_multinomial_covariances(
        probs,
        G,
        np.ones(probs.shape[0], dtype=np.float64),
    )
    shift_hessian = precinct_covariance[:, :n_free, :n_free]
    precision = np.linalg.pinv(shift_hessian, rcond=eps, hermitian=True)
    shift_jacobian = _endpoint_shift_jacobians(probs, G, weights, outcomes)
    count_jacobian = np.einsum(
        "mga,gab->mgb",
        shift_jacobian,
        precision,
        optimize=True,
    )

    if beta_gradients is None:
        precinct_beta, endpoint_beta = _grouped_softmax_beta_jacobians_batch(
            probs,
            G,
            weights,
            X,
        )
        gradients = np.zeros((weights.shape[1], n_params), dtype=np.float64)
        for target, outcome in enumerate(outcomes):
            if not 0 <= outcome < n_categories:
                continue
            direct = endpoint_beta[target, :, outcome, :].sum(axis=0)
            shift_part = np.einsum(
                "ga,gab,gbp->p",
                shift_jacobian[target],
                precision,
                precinct_beta,
                optimize=True,
            )
            gradients[target] = direct - shift_part
    else:
        gradients = np.asarray(beta_gradients, dtype=np.float64)
        if gradients.shape != (weights.shape[1], n_params):
            raise ValueError(
                "beta_gradients must have shape "
                f"{(weights.shape[1], n_params)}, got {gradients.shape}"
            )

    count_covariance = np.einsum(
        "mga,gab,ngb->mn",
        count_jacobian,
        shift_hessian,
        count_jacobian,
        optimize=True,
    )
    slope_covariance = gradients @ beta_cov @ gradients.T
    score_count_covariance = _normal_score_count_covariance(
        probs,
        np.asarray(score_probs, dtype=np.float64),
        precinct_covariance,
        G,
        X,
    )
    count_beta_covariance = -(
        score_count_covariance @ beta_bread.T
    ) / n_groups
    count_beta_cross = np.einsum(
        "mga,gap,np->mn",
        count_jacobian,
        count_beta_covariance,
        gradients,
        optimize=True,
    )
    reuse_covariance = count_beta_cross + count_beta_cross.T
    total = count_covariance + slope_covariance + reuse_covariance
    return {
        "total": 0.5 * (total + total.T),
        "count": 0.5 * (count_covariance + count_covariance.T),
        "slope": 0.5 * (slope_covariance + slope_covariance.T),
        "reuse": 0.5 * (reuse_covariance + reuse_covariance.T),
        "gradients": gradients,
        "shift_jacobian": shift_jacobian,
        "count_jacobian": count_jacobian,
    }


def shifted_categorical_endpoint_linear_coefficients(
    shifted_probs,
    groups,
    endpoint_weights,
    endpoint_outcomes,
    *,
    X,
    beta_bread,
    score_probs,
    beta_gradients=None,
    score_linearization=None,
    endpoint_derivatives=None,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Return vote-linear endpoint coefficients for compatible covariance.

    The coefficient array has shape ``(n_voters, n_targets, n_categories)``.
    For a realized category ``Y_i`` it gives the first-order coefficient whose
    centered sum is the endpoint linearization.  The count part is the exact
    calibrated-count derivative; the slope part uses the vote-linear aggregate
    Normal score passed through the fitted Hessian bread.
    """

    probs = np.asarray(shifted_probs, dtype=np.float64)
    weights = np.asarray(endpoint_weights, dtype=np.float64)
    outcomes = np.asarray(endpoint_outcomes, dtype=np.int64)
    X = np.asarray(X, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("shifted_probs must be two-dimensional")
    if weights.ndim == 1:
        weights = weights[:, None]
    if weights.ndim != 2 or weights.shape[0] != probs.shape[0]:
        raise ValueError("endpoint_weights must be voter-by-target")
    if outcomes.shape != (weights.shape[1],):
        raise ValueError("endpoint_outcomes must have one entry per target")
    if X.shape[0] != probs.shape[0]:
        raise ValueError("X must have one row per voter")
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= eps):
        raise ValueError("rows of shifted_probs must have positive mass")
    probs = probs / row_sums

    G = _group_matrix(groups, probs.shape[0])
    group_ids = _group_ids(G, probs.shape[0])
    n_groups = G.shape[0]
    n_categories = probs.shape[1]
    n_free = n_categories - 1
    n_features = X.shape[1]
    n_params = n_features * n_free
    beta_bread = _cov_matrix(beta_bread, n_params, "beta_bread")

    if endpoint_derivatives is None:
        covariance = shifted_categorical_endpoint_covariance(
            probs,
            G,
            weights,
            outcomes,
            X=X,
            beta_cov=np.zeros((n_params, n_params), dtype=np.float64),
            beta_bread=beta_bread,
            score_probs=score_probs,
            beta_gradients=beta_gradients,
            eps=eps,
        )
        gradients = covariance["gradients"]
        count_jacobian = covariance["count_jacobian"]
    else:
        gradients = np.asarray(endpoint_derivatives["gradients"], dtype=np.float64)
        count_jacobian = np.asarray(
            endpoint_derivatives["count_jacobian"],
            dtype=np.float64,
        )
        if gradients.shape != (weights.shape[1], n_params):
            raise ValueError("endpoint_derivatives gradients have incompatible shape")
        if count_jacobian.shape != (weights.shape[1], n_groups, n_free):
            raise ValueError("endpoint_derivatives count_jacobian has incompatible shape")

    coefficients = np.zeros(
        (probs.shape[0], weights.shape[1], n_categories),
        dtype=np.float64,
    )
    for free_category in range(n_free):
        coefficients[:, :, free_category] += count_jacobian[
            :,
            group_ids,
            free_category,
        ].T

    if score_linearization is None:
        score_linearization = categorical_score_linearization(
            score_probs,
            G,
            X,
        )
    score_group_ids = np.asarray(score_linearization["group_ids"], dtype=np.int64)
    if score_group_ids.shape != (probs.shape[0],) or not np.array_equal(
        score_group_ids,
        group_ids,
    ):
        raise ValueError("score_linearization does not match groups")
    variances = np.asarray(score_linearization["variances"], dtype=np.float64)
    mean_jacobian = np.asarray(
        score_linearization["mean_jacobian"],
        dtype=np.float64,
    )
    if variances.shape != (n_groups, n_categories):
        raise ValueError("score_linearization variances have incompatible shape")
    if mean_jacobian.shape != (n_groups, n_categories, n_params):
        raise ValueError("score_linearization mean_jacobian has incompatible shape")

    directions = beta_bread.T @ gradients.T
    for target in range(weights.shape[1]):
        grouped_direction = np.einsum(
            "gkp,p->gk",
            mean_jacobian,
            directions[:, target],
            optimize=True,
        )
        slope_by_group_category = (
            grouped_direction / variances / (n_groups * n_categories)
        )
        coefficients[:, target, :] += slope_by_group_category[group_ids]

    return {
        "coefficients": coefficients,
        "count_jacobian": count_jacobian,
        "gradients": gradients,
    }


def shifted_categorical_endpoint_pair(
    shifted_probs,
    groups,
    lower_weights,
    upper_weights,
    outcome: int,
    *,
    X,
    beta_cov,
    beta_bread,
    score_probs,
    lower_beta_gradient=None,
    upper_beta_gradient=None,
    score_linearization=None,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Return canonical logit-shifted sandwich pieces for a bound endpoint pair.

    The lower and upper endpoints are two linear functionals of the same fitted
    categorical margin and the same outcome.  This helper keeps the endpoint
    covariance and compatible-coupling linear coefficients on one path, which is
    the object needed before applying exact two-margin Stoye inference.
    """

    weights = np.column_stack(
        [
            np.asarray(lower_weights, dtype=np.float64),
            np.asarray(upper_weights, dtype=np.float64),
        ]
    )
    outcomes = np.asarray([outcome, outcome], dtype=np.int64)
    beta_gradients = None
    if lower_beta_gradient is not None or upper_beta_gradient is not None:
        if lower_beta_gradient is None or upper_beta_gradient is None:
            raise ValueError(
                "lower_beta_gradient and upper_beta_gradient must be supplied together"
            )
        beta_gradients = np.vstack(
            [
                np.asarray(lower_beta_gradient, dtype=np.float64),
                np.asarray(upper_beta_gradient, dtype=np.float64),
            ]
        )

    endpoint = shifted_categorical_endpoint_covariance(
        shifted_probs,
        groups,
        weights,
        outcomes,
        X=X,
        beta_cov=beta_cov,
        beta_bread=beta_bread,
        score_probs=score_probs,
        beta_gradients=beta_gradients,
        eps=eps,
    )
    linearization = shifted_categorical_endpoint_linear_coefficients(
        shifted_probs,
        groups,
        weights,
        outcomes,
        X=X,
        beta_bread=beta_bread,
        score_probs=score_probs,
        beta_gradients=beta_gradients,
        score_linearization=score_linearization,
        endpoint_derivatives=endpoint,
        eps=eps,
    )
    coefficients = linearization["coefficients"]
    return {
        "covariance": endpoint["total"],
        "count_covariance": endpoint["count"],
        "slope_covariance": endpoint["slope"],
        "reuse_covariance": endpoint["reuse"],
        "lower_coefficients": coefficients[:, 0, :],
        "upper_coefficients": coefficients[:, 1, :],
        "count_jacobian": endpoint["count_jacobian"],
        "gradients": endpoint["gradients"],
    }


def _bvn_cdf(x: float, y: float, rho: float) -> float:
    if np.isneginf(x) or np.isneginf(y):
        return 0.0
    if np.isposinf(x):
        return float(NormalDist().cdf(y))
    if np.isposinf(y):
        return float(NormalDist().cdf(x))
    rho = float(np.clip(rho, -0.999, 0.999))
    return float(
        multivariate_normal.cdf(
            [x, y],
            mean=[0.0, 0.0],
            cov=[[1.0, rho], [rho, 1.0]],
        )
    )


def _stoye_rectangle_probability(a: float, b_lower: float, rho: float) -> float:
    return max(0.0, min(1.0, NormalDist().cdf(a) - _bvn_cdf(a, b_lower, rho)))


def stoye_radii(
    width: float,
    lower_se: float,
    upper_se: float,
    *,
    level: float = 0.95,
    rho_grid=None,
) -> tuple[float, float, float]:
    """Return Stoye radii for one scalar interval-identified target.

    The calibration minimizes the sum of lower/upper radii subject to Stoye's
    two Gaussian boundary-coverage inequalities. ``rho_grid`` represents the
    endpoint covariance set through standardized correlations; passing the full
    ``[-1,1]`` grid is conservative when only endpoint variances are used.
    """

    if not 0.0 < level < 1.0:
        raise ValueError("level must be between zero and one")
    if rho_grid is None:
        rho_grid = np.linspace(-0.999, 0.999, 81)
    rho_grid = np.asarray(rho_grid, dtype=np.float64)
    width = max(float(width), 0.0)
    lower_se = max(float(lower_se), 0.0)
    upper_se = max(float(upper_se), 0.0)
    if lower_se == 0.0 and upper_se == 0.0:
        return 0.0, 0.0, 1.0
    z0 = NormalDist().inv_cdf(0.5 + level / 2.0)
    one_sided = NormalDist().inv_cdf(level)
    scale = max(lower_se, upper_se, 1e-12)
    upper_bound = width + 8.0 * scale

    def probabilities(radii: np.ndarray) -> tuple[float, float]:
        r_lower, r_upper = np.maximum(radii, 0.0)
        a1 = np.inf if lower_se == 0.0 else r_lower / lower_se
        a2 = np.inf if lower_se == 0.0 else (width + r_lower) / lower_se
        b1 = -np.inf if upper_se == 0.0 else -(width + r_upper) / upper_se
        b2 = -np.inf if upper_se == 0.0 else -r_upper / upper_se
        p1 = min(_stoye_rectangle_probability(a1, b1, rho) for rho in rho_grid)
        p2 = min(_stoye_rectangle_probability(a2, b2, rho) for rho in rho_grid)
        return p1, p2

    constraints = [
        {"type": "ineq", "fun": lambda x, index=index: probabilities(x)[index] - level}
        for index in range(2)
    ]
    starts = [
        np.array([one_sided * lower_se, one_sided * upper_se]),
        np.array([z0 * lower_se, z0 * upper_se]),
        np.array([0.0, z0 * upper_se]),
        np.array([z0 * lower_se, 0.0]),
    ]
    best = None
    for start in starts:
        result = minimize(
            lambda x: float(np.maximum(x, 0.0).sum()),
            np.clip(start, 0.0, upper_bound),
            method="SLSQP",
            bounds=[(0.0, upper_bound), (0.0, upper_bound)],
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 200},
        )
        candidate = result.x if result.success else np.array([z0 * lower_se, z0 * upper_se])
        p1, p2 = probabilities(candidate)
        if p1 + 1e-7 < level or p2 + 1e-7 < level:
            candidate = np.array([z0 * lower_se, z0 * upper_se])
            p1, p2 = probabilities(candidate)
        score = float(np.maximum(candidate, 0.0).sum())
        if best is None or score < best[0]:
            best = (score, candidate, min(p1, p2))
    _, radii, min_probability = best
    return float(radii[0]), float(radii[1]), float(min_probability)


def stoye_interval(
    lower,
    upper,
    lower_se,
    upper_se,
    *,
    level: float = 0.95,
    lower_bound=None,
    upper_bound=None,
    rho_grid=None,
) -> dict[str, np.ndarray]:
    """Return Stoye partial-ID intervals for vectors of lower/upper endpoints."""

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    lower_se = np.asarray(lower_se, dtype=np.float64)
    upper_se = np.asarray(upper_se, dtype=np.float64)
    if lower.shape != upper.shape or lower.shape != lower_se.shape or lower.shape != upper_se.shape:
        raise ValueError("lower, upper, lower_se, and upper_se must have the same shape")
    widths = np.maximum(upper - lower, 0.0)
    radii = np.asarray(
        [
            stoye_radii(width, lo, hi, level=level, rho_grid=rho_grid)
            for width, lo, hi in zip(widths, lower_se, upper_se)
        ],
        dtype=np.float64,
    )
    ci_lower = lower - radii[:, 0]
    ci_upper = upper + radii[:, 1]
    if lower_bound is not None:
        ci_lower = np.maximum(ci_lower, np.asarray(lower_bound, dtype=np.float64))
    if upper_bound is not None:
        ci_upper = np.minimum(ci_upper, np.asarray(upper_bound, dtype=np.float64))
    return {
        "lower": ci_lower,
        "upper": ci_upper,
        "width": ci_upper - ci_lower,
        "lower_radius": radii[:, 0],
        "upper_radius": radii[:, 1],
        "min_gaussian_probability": radii[:, 2],
    }


def worst_case_sum_standard_error(component_variances, *, axis=0) -> np.ndarray:
    """Return the largest standard error for a sum with fixed component variances.

    This is the Cauchy-Schwarz upper envelope over all cross-component
    covariance matrices compatible with the supplied marginal variances:
    ``sup sd(sum_k X_k) = sum_k sd(X_k)``.  Use it when separately fitted
    margins enter the same partially identified endpoint and their joint
    covariance is not identified by the marginal aggregate fits.
    """

    variances = np.asarray(component_variances, dtype=np.float64)
    if np.any(~np.isfinite(variances)):
        raise ValueError("component_variances must be finite")
    return np.sum(np.sqrt(np.maximum(variances, 0.0)), axis=axis)


def _normalize_probability_rows(values: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= eps):
        raise ValueError("probability rows must have positive mass")
    return values / row_sums


def compatible_scalar_product_bounds(
    first_probs,
    second_probs,
    first_values,
    second_values,
    *,
    batch_size: int = 250_000,
) -> tuple[float, float]:
    """Sharp bounds for ``sum_i E[a_i(Y_i)b_i(Z_i)]`` over compatible couplings.

    This is the exact rearrangement bound for scalar functions of two marginal
    categorical outcomes, evaluated voter by voter and summed.
    """

    first_probs = _normalize_probability_rows(np.asarray(first_probs, dtype=np.float64))
    second_probs = _normalize_probability_rows(np.asarray(second_probs, dtype=np.float64))
    first_values = np.asarray(first_values, dtype=np.float64)
    second_values = np.asarray(second_values, dtype=np.float64)
    if first_values.shape != first_probs.shape:
        raise ValueError("first_values must match first_probs")
    if second_values.shape != second_probs.shape:
        raise ValueError("second_values must match second_probs")
    if first_probs.shape[0] != second_probs.shape[0]:
        raise ValueError("probability matrices must have the same row count")

    lower = 0.0
    upper = 0.0
    for start in range(0, first_probs.shape[0], batch_size):
        stop = min(start + batch_size, first_probs.shape[0])
        p1 = first_probs[start:stop]
        p2 = second_probs[start:stop]
        v1 = first_values[start:stop]
        v2 = second_values[start:stop]

        order1 = np.argsort(v1, axis=1)
        p1s = np.take_along_axis(p1, order1, axis=1)
        v1s = np.take_along_axis(v1, order1, axis=1)
        c1_hi = np.cumsum(p1s, axis=1)
        c1_lo = c1_hi - p1s

        order2_hi = np.argsort(v2, axis=1)
        p2h = np.take_along_axis(p2, order2_hi, axis=1)
        v2h = np.take_along_axis(v2, order2_hi, axis=1)
        c2h_hi = np.cumsum(p2h, axis=1)
        c2h_lo = c2h_hi - p2h

        order2_lo = np.argsort(-v2, axis=1)
        p2l = np.take_along_axis(p2, order2_lo, axis=1)
        v2l = np.take_along_axis(v2, order2_lo, axis=1)
        c2l_hi = np.cumsum(p2l, axis=1)
        c2l_lo = c2l_hi - p2l

        upper_block = np.zeros(stop - start, dtype=np.float64)
        lower_block = np.zeros_like(upper_block)
        for i in range(v1s.shape[1]):
            for j in range(v2h.shape[1]):
                overlap = np.maximum(
                    0.0,
                    np.minimum(c1_hi[:, i], c2h_hi[:, j])
                    - np.maximum(c1_lo[:, i], c2h_lo[:, j]),
                )
                upper_block += overlap * v1s[:, i] * v2h[:, j]
                overlap = np.maximum(
                    0.0,
                    np.minimum(c1_hi[:, i], c2l_hi[:, j])
                    - np.maximum(c1_lo[:, i], c2l_lo[:, j]),
                )
                lower_block += overlap * v1s[:, i] * v2l[:, j]
        lower += float(lower_block.sum())
        upper += float(upper_block.sum())

    return lower, upper


@lru_cache(maxsize=None)
def _transport_bases(n_left: int, n_right: int):
    rows = n_left + n_right
    n_edges = n_left * n_right
    basis_size = rows - 1
    all_edges = [(i, j) for i in range(n_left) for j in range(n_right)]
    bases = []
    for edge_indices in combinations(range(n_edges), basis_size):
        matrix = np.zeros((rows, basis_size), dtype=np.float64)
        for column, edge_index in enumerate(edge_indices):
            left, right = all_edges[edge_index]
            matrix[left, column] = 1.0
            matrix[n_left + right, column] = 1.0
        reduced = matrix[:-1]
        if np.linalg.matrix_rank(reduced) != basis_size:
            continue
        bases.append(
            (
                np.asarray(edge_indices, dtype=np.int64),
                np.linalg.inv(reduced),
            )
        )
    if not bases:
        raise RuntimeError("no transportation bases were generated")
    return tuple(bases)


def compatible_transport_cost_bounds(
    first_probs,
    second_probs,
    costs,
    *,
    batch_size: int = 100_000,
    tolerance: float = 1e-9,
) -> tuple[float, float]:
    """Sharp bounds for summed two-margin transport costs.

    ``costs`` has shape ``(n_voters, K, L)``.  The implementation enumerates
    transportation-polytope bases, which is exact and practical for the
    three-outcome margins used by the production ticket/Sankey fits.
    """

    first_probs = _normalize_probability_rows(np.asarray(first_probs, dtype=np.float64))
    second_probs = _normalize_probability_rows(np.asarray(second_probs, dtype=np.float64))
    costs = np.asarray(costs, dtype=np.float64)
    if costs.shape != (first_probs.shape[0], first_probs.shape[1], second_probs.shape[1]):
        raise ValueError("costs must have shape (n, first_categories, second_categories)")
    if first_probs.shape[0] != second_probs.shape[0]:
        raise ValueError("probability matrices must have the same row count")
    if first_probs.shape[1] > 3 or second_probs.shape[1] > 3:
        raise ValueError("exact basis enumeration is currently limited to 3x3 margins")

    bases = _transport_bases(first_probs.shape[1], second_probs.shape[1])
    lower = 0.0
    upper = 0.0
    for start in range(0, first_probs.shape[0], batch_size):
        stop = min(start + batch_size, first_probs.shape[0])
        b = np.concatenate([first_probs[start:stop], second_probs[start:stop]], axis=1)
        b = b[:, :-1]
        cost_flat = costs[start:stop].reshape(stop - start, -1)
        best_lower = np.full(stop - start, np.inf, dtype=np.float64)
        best_upper = np.full(stop - start, -np.inf, dtype=np.float64)
        for edge_indices, inverse in bases:
            flows = b @ inverse.T
            feasible = np.all(flows >= -tolerance, axis=1)
            if not np.any(feasible):
                continue
            values = np.einsum(
                "bi,bi->b",
                np.maximum(flows[feasible], 0.0),
                cost_flat[np.ix_(feasible, edge_indices)],
                optimize=True,
            )
            best_lower[feasible] = np.minimum(best_lower[feasible], values)
            best_upper[feasible] = np.maximum(best_upper[feasible], values)
        if np.any(~np.isfinite(best_lower)) or np.any(~np.isfinite(best_upper)):
            raise RuntimeError("transport basis enumeration found infeasible rows")
        lower += float(best_lower.sum())
        upper += float(best_upper.sum())
    return lower, upper


def _cross_covariance_bounds(
    first_probs: np.ndarray,
    second_probs: np.ndarray,
    first_values: np.ndarray,
    second_values: np.ndarray,
    *,
    batch_size: int,
) -> tuple[float, float]:
    product_lower, product_upper = compatible_scalar_product_bounds(
        first_probs,
        second_probs,
        first_values,
        second_values,
        batch_size=batch_size,
    )
    first_mean = np.sum(first_probs * first_values, axis=1)
    second_mean = np.sum(second_probs * second_values, axis=1)
    mean_product = float(np.sum(first_mean * second_mean))
    return product_lower - mean_product, product_upper - mean_product


def _cross_covariance_sum_bounds(
    first_probs: np.ndarray,
    second_probs: np.ndarray,
    term_pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    batch_size: int,
) -> tuple[float, float]:
    costs = np.zeros(
        (first_probs.shape[0], first_probs.shape[1], second_probs.shape[1]),
        dtype=np.float64,
    )
    mean_product = 0.0
    for first_values, second_values in term_pairs:
        costs += first_values[:, :, None] * second_values[:, None, :]
        first_mean = np.sum(first_probs * first_values, axis=1)
        second_mean = np.sum(second_probs * second_values, axis=1)
        mean_product += float(np.sum(first_mean * second_mean))
    lower, upper = compatible_transport_cost_bounds(
        first_probs,
        second_probs,
        costs,
        batch_size=batch_size,
    )
    return lower - mean_product, upper - mean_product


def exact_two_margin_endpoint_covariance_bounds(
    first_probs,
    second_probs,
    first_lower_coefficients,
    second_lower_coefficients,
    first_upper_coefficients,
    second_upper_coefficients,
    first_endpoint_covariance,
    second_endpoint_covariance,
    *,
    batch_size: int = 100_000,
) -> dict[str, np.ndarray]:
    """Exact compatible covariance bounds for two-margin lower/upper endpoints.

    The first/second coefficient arrays have shape ``(n_voters, n_targets, K)``.
    The endpoint covariance arrays have shape ``(n_targets, 2, 2)`` and contain
    the within-margin covariance for lower/upper endpoint components.  Cross
    terms are optimized exactly over each voter's compatible joint-law polytope.
    """

    first_probs = _normalize_probability_rows(np.asarray(first_probs, dtype=np.float64))
    second_probs = _normalize_probability_rows(np.asarray(second_probs, dtype=np.float64))
    first_lower = np.asarray(first_lower_coefficients, dtype=np.float64)
    second_lower = np.asarray(second_lower_coefficients, dtype=np.float64)
    first_upper = np.asarray(first_upper_coefficients, dtype=np.float64)
    second_upper = np.asarray(second_upper_coefficients, dtype=np.float64)
    first_cov = np.asarray(first_endpoint_covariance, dtype=np.float64)
    second_cov = np.asarray(second_endpoint_covariance, dtype=np.float64)
    n_targets = first_lower.shape[1]
    expected_first = (first_probs.shape[0], n_targets, first_probs.shape[1])
    expected_second = (second_probs.shape[0], n_targets, second_probs.shape[1])
    if first_lower.shape != expected_first or first_upper.shape != expected_first:
        raise ValueError("first coefficient arrays have incompatible shape")
    if second_lower.shape != expected_second or second_upper.shape != expected_second:
        raise ValueError("second coefficient arrays have incompatible shape")
    if first_probs.shape[0] != second_probs.shape[0]:
        raise ValueError("probability matrices must be aligned to the same voters")
    if first_cov.shape != (n_targets, 2, 2) or second_cov.shape != (n_targets, 2, 2):
        raise ValueError("endpoint covariance arrays must have shape (n_targets, 2, 2)")

    lower_variance = np.empty(n_targets, dtype=np.float64)
    upper_variance = np.empty(n_targets, dtype=np.float64)
    covariance_lower = np.empty(n_targets, dtype=np.float64)
    covariance_upper = np.empty(n_targets, dtype=np.float64)
    lower_cross_bounds = np.empty((n_targets, 2), dtype=np.float64)
    upper_cross_bounds = np.empty((n_targets, 2), dtype=np.float64)
    endpoint_cross_bounds = np.empty((n_targets, 2), dtype=np.float64)

    for target in range(n_targets):
        ll_min, ll_max = _cross_covariance_bounds(
            first_probs,
            second_probs,
            first_lower[:, target, :],
            second_lower[:, target, :],
            batch_size=batch_size,
        )
        uu_min, uu_max = _cross_covariance_bounds(
            first_probs,
            second_probs,
            first_upper[:, target, :],
            second_upper[:, target, :],
            batch_size=batch_size,
        )
        lu_min, lu_max = _cross_covariance_sum_bounds(
            first_probs,
            second_probs,
            [
                (first_lower[:, target, :], second_upper[:, target, :]),
                (first_upper[:, target, :], second_lower[:, target, :]),
            ],
            batch_size=batch_size,
        )
        lower_cross_bounds[target] = [ll_min, ll_max]
        upper_cross_bounds[target] = [uu_min, uu_max]
        endpoint_cross_bounds[target] = [lu_min, lu_max]
        lower_variance[target] = first_cov[target, 0, 0] + second_cov[target, 0, 0] + 2.0 * ll_max
        upper_variance[target] = first_cov[target, 1, 1] + second_cov[target, 1, 1] + 2.0 * uu_max
        within_covariance = first_cov[target, 0, 1] + second_cov[target, 0, 1]
        covariance_lower[target] = within_covariance + lu_min
        covariance_upper[target] = within_covariance + lu_max

    lower_variance = np.maximum(lower_variance, 0.0)
    upper_variance = np.maximum(upper_variance, 0.0)
    return {
        "lower_variance": lower_variance,
        "upper_variance": upper_variance,
        "lower_se": np.sqrt(lower_variance),
        "upper_se": np.sqrt(upper_variance),
        "covariance_lower": covariance_lower,
        "covariance_upper": covariance_upper,
        "lower_cross_covariance_bounds": lower_cross_bounds,
        "upper_cross_covariance_bounds": upper_cross_bounds,
        "endpoint_cross_covariance_bounds": endpoint_cross_bounds,
    }


def exact_two_margin_endpoint_pair_covariance_bounds(
    first_probs,
    second_probs,
    first_endpoint: dict[str, np.ndarray],
    second_endpoint: dict[str, np.ndarray],
    *,
    first_rows=None,
    second_rows=None,
    batch_size: int = 100_000,
) -> dict[str, np.ndarray]:
    """Exact compatible covariance bounds for two endpoint-pair objects."""

    first_probs = np.asarray(first_probs, dtype=np.float64)
    second_probs = np.asarray(second_probs, dtype=np.float64)
    first_lower = np.asarray(first_endpoint["lower_coefficients"], dtype=np.float64)
    first_upper = np.asarray(first_endpoint["upper_coefficients"], dtype=np.float64)
    second_lower = np.asarray(second_endpoint["lower_coefficients"], dtype=np.float64)
    second_upper = np.asarray(second_endpoint["upper_coefficients"], dtype=np.float64)

    if first_rows is not None:
        first_rows = np.asarray(first_rows, dtype=np.int64)
        first_probs = first_probs[first_rows]
        first_lower = first_lower[first_rows]
        first_upper = first_upper[first_rows]
    if second_rows is not None:
        second_rows = np.asarray(second_rows, dtype=np.int64)
        second_probs = second_probs[second_rows]
        second_lower = second_lower[second_rows]
        second_upper = second_upper[second_rows]

    return exact_two_margin_endpoint_covariance_bounds(
        first_probs,
        second_probs,
        first_lower[:, None, :],
        second_lower[:, None, :],
        first_upper[:, None, :],
        second_upper[:, None, :],
        np.asarray(first_endpoint["covariance"], dtype=np.float64)[None, :, :],
        np.asarray(second_endpoint["covariance"], dtype=np.float64)[None, :, :],
        batch_size=batch_size,
    )


def stoye_interval_from_covariance_bounds(
    lower,
    upper,
    lower_variance,
    upper_variance,
    covariance_lower,
    covariance_upper,
    *,
    level: float = 0.95,
    lower_bound=None,
    upper_bound=None,
    rho_grid_size: int = 81,
) -> dict[str, np.ndarray]:
    """Return Stoye intervals using exact compatible endpoint covariance bounds."""

    lower_variance = np.asarray(lower_variance, dtype=np.float64)
    upper_variance = np.asarray(upper_variance, dtype=np.float64)
    covariance_lower = np.asarray(covariance_lower, dtype=np.float64)
    covariance_upper = np.asarray(covariance_upper, dtype=np.float64)
    lower_se = np.sqrt(np.maximum(lower_variance, 0.0))
    upper_se = np.sqrt(np.maximum(upper_variance, 0.0))
    denominator = lower_se * upper_se
    rho_lower = np.full_like(lower_se, -0.999, dtype=np.float64)
    rho_upper = np.full_like(lower_se, 0.999, dtype=np.float64)
    active = denominator > 0.0
    rho_lower[active] = np.clip(covariance_lower[active] / denominator[active], -0.999, 0.999)
    rho_upper[active] = np.clip(covariance_upper[active] / denominator[active], -0.999, 0.999)
    rho_lower = np.minimum(rho_lower, rho_upper)

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    widths = np.maximum(upper - lower, 0.0)
    radii = []
    for width, lo, hi, rlo, rhi in zip(widths, lower_se, upper_se, rho_lower, rho_upper):
        if not np.isfinite(rlo) or not np.isfinite(rhi) or rhi < rlo:
            rho_grid = None
        elif abs(rhi - rlo) < 1e-12:
            rho_grid = np.asarray([rlo], dtype=np.float64)
        else:
            rho_grid = np.linspace(rlo, rhi, rho_grid_size)
        radii.append(stoye_radii(width, lo, hi, level=level, rho_grid=rho_grid))
    radii = np.asarray(radii, dtype=np.float64)
    ci_lower = lower - radii[:, 0]
    ci_upper = upper + radii[:, 1]
    if lower_bound is not None:
        ci_lower = np.maximum(ci_lower, np.asarray(lower_bound, dtype=np.float64))
    if upper_bound is not None:
        ci_upper = np.minimum(ci_upper, np.asarray(upper_bound, dtype=np.float64))
    return {
        "lower": ci_lower,
        "upper": ci_upper,
        "width": ci_upper - ci_lower,
        "lower_radius": radii[:, 0],
        "upper_radius": radii[:, 1],
        "min_gaussian_probability": radii[:, 2],
        "lower_se": lower_se,
        "upper_se": upper_se,
        "rho_lower": rho_lower,
        "rho_upper": rho_upper,
    }


def shifted_subgroup_inference(
    X,
    group_ids,
    shifted_probs,
    subgroup_mask,
    group,
    *,
    outcome: int | None = None,
    beta_cov=None,
    beta_count_cov=None,
    beta_realized_noise_cov=None,
    level: float = 0.95,
    eps: float = 1e-12,
) -> dict[str, object]:
    """Return delta-method standard errors for one shifted binary subgroup.

    Parameters
    ----------
    X:
        Voter-by-feature design matrix.
    group_ids:
        Integer precinct/group id for each voter, or a group-by-voter
        aggregation matrix such as the `G` matrix used elsewhere in `elrpy`.
    shifted_probs:
        Fitted post-shift probabilities. Pass either a one-dimensional binary
        probability vector or a voter-by-outcome probability matrix.
    subgroup_mask:
        Boolean mask selecting the target subgroup.
    group:
        Target group id.
    outcome:
        Outcome column to use when `shifted_probs` is a matrix. For a two-column
        matrix, the default is column 0, matching the first non-reference binary
        category used by the categorical model.
    beta_cov:
        Covariance matrix of the fitted slope, on the original beta scale. If
        omitted, slope uncertainty is treated as zero.
    beta_count_cov:
        Optional covariance between the fitted slope and the target precinct
        count noise. This is zero for leave-one-precinct-out fitting or when the
        target precinct has asymptotically negligible mass.
    beta_realized_noise_cov:
        Optional covariance between the fitted slope and the realized-share
        target noise `c * epsilon - A`.
    """

    X = np.asarray(X, dtype=np.float64)
    shifted_probs = _binary_probs(shifted_probs, outcome)
    group_ids = _group_ids(group_ids, X.shape[0])
    subgroup_mask = np.asarray(subgroup_mask, dtype=bool)

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if shifted_probs.shape != (X.shape[0],):
        raise ValueError("shifted_probs must have one entry per row of X")
    if subgroup_mask.shape != (X.shape[0],):
        raise ValueError("subgroup_mask must have one entry per row of X")
    if np.any((shifted_probs <= 0.0) | (shifted_probs >= 1.0)):
        raise ValueError("shifted_probs must be strictly between zero and one")

    group_mask = group_ids == group
    target_mask = group_mask & subgroup_mask
    n_group = int(group_mask.sum())
    n_subgroup = int(target_mask.sum())
    if n_group == 0:
        raise ValueError("target group has no voters")
    if n_subgroup == 0:
        raise ValueError("target subgroup has no voters in the target group")

    dot = shifted_probs * (1.0 - shifted_probs)
    W = float(dot[group_mask].sum())
    W_subgroup = float(dot[target_mask].sum())
    if W <= eps:
        raise ValueError("target group has too little vote variance")

    G = dot[group_mask] @ X[group_mask]
    G_subgroup = dot[target_mask] @ X[target_mask]
    d_beta = (G_subgroup - (W_subgroup / W) * G) / n_subgroup
    d_count = W_subgroup / (n_subgroup * W)
    estimate = float(shifted_probs[target_mask].mean())

    n_features = X.shape[1]
    beta_cov = _cov_matrix(beta_cov, n_features, "beta_cov")
    beta_count_cov = _cov_vector(beta_count_cov, n_features, "beta_count_cov")
    beta_realized_noise_cov = _cov_vector(
        beta_realized_noise_cov,
        n_features,
        "beta_realized_noise_cov",
    )

    slope_shift_derivative = -G / W
    slope_variance = float(d_beta @ beta_cov @ d_beta)
    shift_variance = float(
        slope_shift_derivative @ beta_cov @ slope_shift_derivative
        + 1.0 / W
        + 2.0 * (slope_shift_derivative @ beta_count_cov) / W
    )
    conditional_mean_variance = float(
        slope_variance
        + d_count**2 * W
        + 2.0 * d_count * (d_beta @ beta_count_cov)
    )
    realized_noise_variance = float(
        W_subgroup / n_subgroup**2 * (1.0 - W_subgroup / W)
    )
    realized_share_variance = float(
        slope_variance
        + realized_noise_variance
        + 2.0 * (d_beta @ beta_realized_noise_cov)
    )

    shift_se = float(np.sqrt(max(shift_variance, 0.0)))
    conditional_mean_se = float(np.sqrt(max(conditional_mean_variance, 0.0)))
    realized_share_se = float(np.sqrt(max(realized_share_variance, 0.0)))

    return {
        "estimate": estimate,
        "n_group": n_group,
        "n_subgroup": n_subgroup,
        "W": W,
        "W_subgroup": W_subgroup,
        "G": G,
        "G_subgroup": G_subgroup,
        "d_beta": d_beta,
        "d_count": float(d_count),
        "shift_variance": shift_variance,
        "slope_variance": slope_variance,
        "conditional_mean_variance": conditional_mean_variance,
        "realized_noise_variance": realized_noise_variance,
        "realized_share_variance": realized_share_variance,
        "shift_se": shift_se,
        "conditional_mean_se": conditional_mean_se,
        "realized_share_se": realized_share_se,
        "conditional_mean_interval": wald_interval(
            estimate,
            conditional_mean_se,
            level=level,
        ),
        "realized_share_interval": wald_interval(
            estimate,
            realized_share_se,
            level=level,
        ),
    }


def shifted_categorical_subgroup_inference(
    shifted_probs,
    groups,
    subgroup_weights,
    *,
    X=None,
    beta_cov=None,
    beta_bread=None,
    score_probs=None,
    level: float = 0.95,
    eps: float = 1e-12,
) -> dict[str, object]:
    """Return shifted categorical subgroup conditional-mean intervals.

    The target is the calibrated conditional mean for a weighted subgroup inside
    each precinct. The observed precinct-wide totals are random and enter the
    estimator through the profile logit shift. For category probabilities `p_i`,
    the multinomial covariance contribution of voter `i` is
    `diag(p_i) - p_i p_i'`. The count component is the delta-method covariance
    of the calibrated mean with respect to the observed precinct totals.

    If `X` and `beta_cov` are supplied, the returned variance also includes the
    fitted-slope contribution after differentiating through the precinct logit
    shift while holding the observed totals fixed. If `beta_bread` and
    `score_probs` are also supplied, the returned covariance includes the
    same-precinct slope-count reuse term for the fitted aggregate Normal score.
    """

    probs = np.asarray(shifted_probs, dtype=np.float64)
    weights = np.asarray(subgroup_weights, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("shifted_probs must be a two-dimensional matrix")
    if weights.shape != (probs.shape[0],):
        raise ValueError("subgroup_weights must have one entry per voter")
    if np.any(weights < 0.0):
        raise ValueError("subgroup_weights must be nonnegative")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("shifted_probs must be probabilities")
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= eps):
        raise ValueError("rows of shifted_probs must have positive mass")
    probs = probs / row_sums

    G = _group_matrix(groups, probs.shape[0])
    n_groups, _ = G.shape
    n_categories = probs.shape[1]
    n_free = n_categories - 1
    z = NormalDist().inv_cdf(0.5 + level / 2.0)

    group_n = np.asarray(G @ np.ones(probs.shape[0], dtype=np.float64)).ravel()
    subgroup_n = np.asarray(G @ weights).ravel()
    numerator = np.asarray(G @ (weights[:, None] * probs))

    estimate = np.full((n_groups, n_categories), np.nan, dtype=np.float64)
    valid = subgroup_n > eps
    estimate[valid] = numerator[valid] / subgroup_n[valid, None]

    precinct_cov, subgroup_cov, subgroup_noise_cov = _grouped_multinomial_covariances(
        probs,
        G,
        weights,
    )

    count_covariance = np.full_like(precinct_cov, np.nan)
    slope_covariance = np.full_like(precinct_cov, np.nan)
    reuse_covariance = np.full_like(precinct_cov, np.nan)
    needs_slope = beta_cov is not None
    if (beta_bread is None) != (score_probs is None):
        raise ValueError("beta_bread and score_probs must be supplied together")
    if not needs_slope and beta_bread is not None:
        raise ValueError("beta_bread requires beta_cov")
    needs_reuse = needs_slope and beta_bread is not None
    if needs_slope:
        if X is None:
            raise ValueError("X is required when beta_cov is supplied")
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] != probs.shape[0]:
            raise ValueError("X must have one row per voter")
        n_features = X.shape[1]
        beta_cov = _cov_matrix(
            beta_cov,
            n_features * n_free,
            "beta_cov",
        )
        if needs_reuse:
            beta_bread = _cov_matrix(
                beta_bread,
                n_features * n_free,
                "beta_bread",
            )
            score_probs = np.asarray(score_probs, dtype=np.float64)
            score_count_covariance = _normal_score_count_covariance(
                probs,
                score_probs,
                precinct_cov,
                G,
                X,
            )
        precinct_beta, subgroup_beta = _grouped_softmax_beta_jacobians(
            probs,
            G,
            weights,
            X,
        )

    for group in range(n_groups):
        if not valid[group]:
            continue
        shift_hessian = precinct_cov[group, :n_free, :n_free]
        precision = np.linalg.pinv(shift_hessian, rcond=eps)
        subgroup_shift_jacobian = subgroup_cov[group, :, :n_free]
        normalized_shift_jacobian = subgroup_shift_jacobian / subgroup_n[group]
        count_cov = (
            normalized_shift_jacobian
            @ precision
            @ shift_hessian
            @ precision
            @ normalized_shift_jacobian.T
        )
        count_covariance[group] = 0.5 * (count_cov + count_cov.T)
        slope_covariance[group] = 0.0
        reuse_covariance[group] = 0.0
        if needs_slope:
            normalized_beta_jacobian = subgroup_beta[group] / subgroup_n[group]
            implicit_beta_jacobian = (
                normalized_beta_jacobian
                - normalized_shift_jacobian @ precision @ precinct_beta[group]
            )
            slope_cov = implicit_beta_jacobian @ beta_cov @ implicit_beta_jacobian.T
            slope_covariance[group] = 0.5 * (slope_cov + slope_cov.T)
            if needs_reuse:
                count_jacobian = normalized_shift_jacobian @ precision
                count_beta_cov = -(
                    score_count_covariance[group] @ beta_bread.T
                ) / n_groups
                reuse_cross = (
                    count_jacobian
                    @ count_beta_cov
                    @ implicit_beta_jacobian.T
                )
                reuse_cov = reuse_cross + reuse_cross.T
                reuse_covariance[group] = 0.5 * (reuse_cov + reuse_cov.T)

    covariance = count_covariance + slope_covariance + reuse_covariance
    variance = np.diagonal(covariance, axis1=1, axis2=2)
    variance = np.maximum(variance, 0.0)
    count_variance = np.maximum(
        np.diagonal(count_covariance, axis1=1, axis2=2),
        0.0,
    )
    slope_variance = np.maximum(
        np.diagonal(slope_covariance, axis1=1, axis2=2),
        0.0,
    )
    reuse_variance = np.diagonal(reuse_covariance, axis1=1, axis2=2)
    se = np.sqrt(variance)
    lower = estimate - z * se
    upper = estimate + z * se

    return {
        "estimate": estimate,
        "se": se,
        "count_se": np.sqrt(count_variance),
        "allocation_se": np.sqrt(count_variance),
        "slope_se": np.sqrt(slope_variance),
        "lower": lower,
        "upper": upper,
        "covariance": covariance,
        "count_covariance": count_covariance,
        "allocation_covariance": count_covariance,
        "slope_covariance": slope_covariance,
        "reuse_covariance": reuse_covariance,
        "variance": variance,
        "count_variance": count_variance,
        "allocation_variance": count_variance,
        "slope_variance": slope_variance,
        "reuse_variance": reuse_variance,
        "group_n": group_n,
        "subgroup_n": subgroup_n,
        "valid": valid,
    }


def _grouped_multinomial_covariances_batch(
    probs: np.ndarray,
    groups: sp.csr_matrix,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return grouped multinomial covariance sums for many subgroup weights."""

    groups = groups.tocsr()
    n_groups = groups.shape[0]
    n_subgroups = weights.shape[1]
    n_categories = probs.shape[1]
    precinct = np.empty((n_groups, n_categories, n_categories), dtype=np.float64)
    subgroup = np.empty(
        (n_subgroups, n_groups, n_categories, n_categories),
        dtype=np.float64,
    )
    subgroup_noise = np.empty_like(subgroup)
    weights_squared = weights * weights
    covariances = np.empty(
        (probs.shape[0], n_categories * n_categories),
        dtype=np.float64,
    )
    column = 0
    for a in range(n_categories):
        for b in range(n_categories):
            if a == b:
                covariances[:, column] = probs[:, a] * (1.0 - probs[:, a])
            else:
                covariances[:, column] = -probs[:, a] * probs[:, b]
            column += 1

    precinct[:, :, :] = np.asarray(groups @ covariances).reshape(
        n_groups,
        n_categories,
        n_categories,
    )
    for column, (a, b) in enumerate(
        (a, b) for a in range(n_categories) for b in range(n_categories)
    ):
        voter_cov = covariances[:, column]
        subgroup[:, :, a, b] = np.asarray(groups @ (weights * voter_cov[:, None])).T
        subgroup_noise[:, :, a, b] = np.asarray(
            groups @ (weights_squared * voter_cov[:, None])
        ).T
    return precinct, subgroup, subgroup_noise


def _feature_block_size(
    n_voters: int,
    n_subgroups: int,
    n_features: int,
    *,
    max_bytes: int = 4_000_000_000,
) -> int:
    """Return a feature block size for voter-by-subgroup-by-feature work arrays."""

    bytes_per_feature = max(n_voters * n_subgroups * np.dtype(np.float64).itemsize, 1)
    return max(1, min(n_features, max_bytes // bytes_per_feature))


def _grouped_softmax_beta_jacobians_batch(
    probs: np.ndarray,
    groups: sp.csr_matrix,
    weights: np.ndarray,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return beta derivatives for precinct totals and many subgroup totals."""

    groups = groups.tocsr()
    n_groups = groups.shape[0]
    n_subgroups = weights.shape[1]
    n_features = X.shape[1]
    n_categories = probs.shape[1]
    n_free = n_categories - 1
    n_params = n_features * n_free
    precinct = np.zeros((n_groups, n_free, n_params), dtype=np.float64)
    subgroup = np.empty(
        (n_subgroups, n_groups, n_categories, n_params),
        dtype=np.float64,
    )
    feature_block_size = _feature_block_size(
        probs.shape[0],
        n_subgroups,
        n_features,
    )
    for free_category in range(n_free):
        params = slice(free_category, n_params, n_free)
        for outcome in range(n_categories):
            jacobian = probs[:, outcome] * (
                (1.0 if outcome == free_category else 0.0)
                - probs[:, free_category]
            )
            x_jacobian = X * jacobian[:, None]
            grouped_x = np.asarray(groups @ x_jacobian)
            if outcome < n_free:
                precinct[:, outcome, params] = grouped_x

            for feature_start in range(0, n_features, feature_block_size):
                feature_stop = min(feature_start + feature_block_size, n_features)
                block = x_jacobian[:, feature_start:feature_stop]
                weighted_block = (
                    weights[:, :, None] * block[:, None, :]
                ).reshape(probs.shape[0], -1)
                grouped_block = np.asarray(groups @ weighted_block).reshape(
                    n_groups,
                    n_subgroups,
                    feature_stop - feature_start,
                )
                block_params = slice(
                    free_category + feature_start * n_free,
                    free_category + feature_stop * n_free,
                    n_free,
                )
                subgroup[:, :, outcome, block_params] = np.swapaxes(
                    grouped_block,
                    0,
                    1,
                )
    return precinct, subgroup


def shifted_categorical_subgroups_inference(
    shifted_probs,
    groups,
    subgroup_weights,
    *,
    X=None,
    beta_cov=None,
    beta_bread=None,
    score_probs=None,
    level: float = 0.95,
    eps: float = 1e-12,
) -> dict[str, object]:
    """Return shifted categorical subgroup conditional-mean intervals in batch.

    This is algebraically the same calculation as
    :func:`shifted_categorical_subgroup_inference`, but it batches the expensive
    grouped covariance and beta-Jacobian contractions across subgroup columns.
    """

    probs = np.asarray(shifted_probs, dtype=np.float64)
    weights = np.asarray(subgroup_weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[:, None]
    if probs.ndim != 2:
        raise ValueError("shifted_probs must be a two-dimensional matrix")
    if weights.ndim != 2 or weights.shape[0] != probs.shape[0]:
        raise ValueError("subgroup_weights must be voter-by-subgroup")
    if np.any(weights < 0.0):
        raise ValueError("subgroup_weights must be nonnegative")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("shifted_probs must be probabilities")
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= eps):
        raise ValueError("rows of shifted_probs must have positive mass")
    probs = probs / row_sums

    G = _group_matrix(groups, probs.shape[0])
    n_groups, _ = G.shape
    n_subgroups = weights.shape[1]
    n_categories = probs.shape[1]
    n_free = n_categories - 1
    z = NormalDist().inv_cdf(0.5 + level / 2.0)

    group_n = np.asarray(G @ np.ones(probs.shape[0], dtype=np.float64)).ravel()
    subgroup_n = np.asarray(G @ weights).T
    numerator = np.empty((n_subgroups, n_groups, n_categories), dtype=np.float64)
    for outcome in range(n_categories):
        numerator[:, :, outcome] = np.asarray(
            G @ (weights * probs[:, outcome, None])
        ).T

    estimate = np.full((n_subgroups, n_groups, n_categories), np.nan, dtype=np.float64)
    valid = subgroup_n > eps
    estimate[valid] = numerator[valid] / subgroup_n[valid, None]

    precinct_cov, subgroup_cov, subgroup_noise_cov = (
        _grouped_multinomial_covariances_batch(probs, G, weights)
    )

    count_covariance = np.full(
        (n_subgroups, n_groups, n_categories, n_categories),
        np.nan,
        dtype=np.float64,
    )
    slope_covariance = np.full_like(count_covariance, np.nan)
    reuse_covariance = np.full_like(count_covariance, np.nan)
    needs_slope = beta_cov is not None
    if (beta_bread is None) != (score_probs is None):
        raise ValueError("beta_bread and score_probs must be supplied together")
    if not needs_slope and beta_bread is not None:
        raise ValueError("beta_bread requires beta_cov")
    needs_reuse = needs_slope and beta_bread is not None
    if needs_slope:
        if X is None:
            raise ValueError("X is required when beta_cov is supplied")
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] != probs.shape[0]:
            raise ValueError("X must have one row per voter")
        n_features = X.shape[1]
        beta_cov = _cov_matrix(
            beta_cov,
            n_features * n_free,
            "beta_cov",
        )
        if needs_reuse:
            beta_bread = _cov_matrix(
                beta_bread,
                n_features * n_free,
                "beta_bread",
            )
            score_probs = np.asarray(score_probs, dtype=np.float64)
            score_count_covariance = _normal_score_count_covariance(
                probs,
                score_probs,
                precinct_cov,
                G,
                X,
            )
        precinct_beta, subgroup_beta = _grouped_softmax_beta_jacobians_batch(
            probs,
            G,
            weights,
            X,
        )

    for group in range(n_groups):
        valid_subgroups = np.flatnonzero(valid[:, group])
        if valid_subgroups.size == 0:
            continue
        shift_hessian = precinct_cov[group, :n_free, :n_free]
        precision = np.linalg.pinv(shift_hessian, rcond=eps)
        subgroup_shift_jacobian = subgroup_cov[
            valid_subgroups,
            group,
            :,
            :n_free,
        ]
        normalized_shift_jacobian = (
            subgroup_shift_jacobian
            / subgroup_n[valid_subgroups, group, None, None]
        )
        count_cov = np.einsum(
            "ska,ab,bc,dc,sld->skl",
            normalized_shift_jacobian,
            precision,
            shift_hessian,
            precision,
            normalized_shift_jacobian,
            optimize=True,
        )
        count_covariance[valid_subgroups, group] = 0.5 * (
            count_cov + np.swapaxes(count_cov, -1, -2)
        )
        slope_covariance[valid_subgroups, group] = 0.0
        reuse_covariance[valid_subgroups, group] = 0.0
        if needs_slope:
            normalized_beta_jacobian = (
                subgroup_beta[valid_subgroups, group]
                / subgroup_n[valid_subgroups, group, None, None]
            )
            implicit_beta_jacobian = normalized_beta_jacobian - np.einsum(
                "ska,ab,bp->skp",
                normalized_shift_jacobian,
                precision,
                precinct_beta[group],
                optimize=True,
            )
            slope_cov = np.einsum(
                "skp,pq,slq->skl",
                implicit_beta_jacobian,
                beta_cov,
                implicit_beta_jacobian,
                optimize=True,
            )
            slope_covariance[valid_subgroups, group] = 0.5 * (
                slope_cov + np.swapaxes(slope_cov, -1, -2)
            )
            if needs_reuse:
                count_jacobian = np.einsum(
                    "ska,ab->skb",
                    normalized_shift_jacobian,
                    precision,
                    optimize=True,
                )
                count_beta_cov = -(
                    score_count_covariance[group] @ beta_bread.T
                ) / n_groups
                reuse_cross = np.einsum(
                    "ska,ap,slp->skl",
                    count_jacobian,
                    count_beta_cov,
                    implicit_beta_jacobian,
                    optimize=True,
                )
                reuse_cov = reuse_cross + np.swapaxes(reuse_cross, -1, -2)
                reuse_covariance[valid_subgroups, group] = 0.5 * (
                    reuse_cov + np.swapaxes(reuse_cov, -1, -2)
                )

    covariance = count_covariance + slope_covariance + reuse_covariance
    variance = np.diagonal(covariance, axis1=2, axis2=3)
    variance = np.maximum(variance, 0.0)
    count_variance = np.maximum(
        np.diagonal(count_covariance, axis1=2, axis2=3),
        0.0,
    )
    slope_variance = np.maximum(
        np.diagonal(slope_covariance, axis1=2, axis2=3),
        0.0,
    )
    reuse_variance = np.diagonal(reuse_covariance, axis1=2, axis2=3)
    se = np.sqrt(variance)
    lower = estimate - z * se
    upper = estimate + z * se

    return {
        "estimate": estimate,
        "se": se,
        "count_se": np.sqrt(count_variance),
        "allocation_se": np.sqrt(count_variance),
        "slope_se": np.sqrt(slope_variance),
        "lower": lower,
        "upper": upper,
        "covariance": covariance,
        "count_covariance": count_covariance,
        "allocation_covariance": count_covariance,
        "slope_covariance": slope_covariance,
        "reuse_covariance": reuse_covariance,
        "variance": variance,
        "count_variance": count_variance,
        "allocation_variance": count_variance,
        "slope_variance": slope_variance,
        "reuse_variance": reuse_variance,
        "group_n": group_n,
        "subgroup_n": subgroup_n,
        "valid": valid,
    }

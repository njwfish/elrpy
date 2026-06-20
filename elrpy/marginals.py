"""Marginal probabilities, flow bounds, and uncertainty summaries.

These helpers keep the post-fit layer separate from any particular election
workflow.  Given voter-level marginal probabilities for several contests, they
compute the product coupling used by independent-flow displays and the sharp
pairwise Frechet bounds implied by those marginals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd


DEFAULT_FLOW_KEYS = ("stratum", "item_1", "item_2", "outcome_1", "outcome_2")


def softmax(scores) -> np.ndarray:
    """Return row-wise softmax probabilities."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def categorical_probabilities(X, params) -> np.ndarray:
    """Return softmax probabilities for an ELR categorical parameter vector.

    `params` may be a flattened `(n_features * (n_categories - 1),)` vector or
    a two-dimensional `(n_features, n_categories - 1)` matrix.  The final
    category is the reference category with logit zero.
    """

    X = np.asarray(X, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if params.ndim == 1:
        if params.size % X.shape[1] != 0:
            raise ValueError("flat params length must be divisible by X columns")
        params = params.reshape(X.shape[1], -1)
    if params.ndim != 2 or params.shape[0] != X.shape[1]:
        raise ValueError(
            "params must have shape (n_features, n_categories - 1), "
            f"got {params.shape}"
        )
    logits = X @ params
    logits = np.hstack([logits, np.zeros((X.shape[0], 1), dtype=np.float64)])
    return softmax(logits)


def categorical_probability_draws(X, param_draws) -> np.ndarray:
    """Return categorical probabilities for several parameter draws.

    The output has shape `(n_draws, n_units, n_categories)`.
    """

    param_draws = np.asarray(param_draws, dtype=np.float64)
    if param_draws.ndim == 1:
        param_draws = param_draws[None, :]
    if param_draws.ndim != 2:
        raise ValueError("param_draws must be one- or two-dimensional")
    return np.stack(
        [categorical_probabilities(X, draw) for draw in param_draws],
        axis=0,
    )


def shifted_probabilities(base_probs, group_ids, shifts, *, eps: float = 1e-12) -> np.ndarray:
    """Apply group-level additive logit shifts to probability rows.

    This is the NumPy counterpart to `elrpy.shifting.apply_logit_shifts`.
    `shifts` may contain either `K - 1` free shift columns or all `K` category
    shifts.  In the free-shift case, the final category is the reference.
    """

    probs = _probability_matrix(base_probs, name="base_probs", eps=eps)
    group_ids = np.asarray(group_ids)
    if group_ids.shape != (probs.shape[0],):
        raise ValueError("group_ids must have one entry per probability row")
    shifts = np.asarray(shifts, dtype=np.float64)
    if shifts.ndim != 2:
        raise ValueError("shifts must be two-dimensional")
    if shifts.shape[1] == probs.shape[1] - 1:
        shifts = np.hstack([shifts, np.zeros((shifts.shape[0], 1), dtype=np.float64)])
    elif shifts.shape[1] != probs.shape[1]:
        raise ValueError("shifts must have K or K - 1 columns")
    if np.any(group_ids < 0) or np.any(group_ids >= shifts.shape[0]):
        raise ValueError("group_ids refer to a shift row that does not exist")

    logits = np.log(np.clip(probs, eps, 1.0)) + shifts[group_ids]
    return softmax(logits)


def shifted_probability_draws(base_prob_draws, group_ids, shifts, *, eps: float = 1e-12) -> np.ndarray:
    """Apply logit shifts to a stack of probability draws.

    `base_prob_draws` must have shape `(n_draws, n_units, n_categories)`.
    `shifts` may be either a fixed shift matrix with shape `(n_groups, K - 1)`
    or `(n_groups, K)`, or one shift matrix per draw with shape
    `(n_draws, n_groups, K - 1)` or `(n_draws, n_groups, K)`.
    """

    base_prob_draws = np.asarray(base_prob_draws, dtype=np.float64)
    if base_prob_draws.ndim != 3:
        raise ValueError("base_prob_draws must have shape (n_draws, n_units, n_categories)")
    shifts = np.asarray(shifts, dtype=np.float64)
    if shifts.ndim == 2:
        shifts = np.broadcast_to(shifts, (base_prob_draws.shape[0], *shifts.shape))
    if shifts.ndim != 3 or shifts.shape[0] != base_prob_draws.shape[0]:
        raise ValueError("shifts must be fixed across draws or have one matrix per draw")
    return np.stack(
        [
            shifted_probabilities(base_probs, group_ids, draw_shifts, eps=eps)
            for base_probs, draw_shifts in zip(base_prob_draws, shifts)
        ],
        axis=0,
    )


def shifted_softmax_beta_sensitivity(
    X,
    shifted_probs,
    groups,
    *,
    rcond: float = 1e-10,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Differentiate fitted logit shifts with respect to the softmax slope.

    The shifted probabilities are interpreted as

    `softmax((x_i' beta_1 + shift_{g_i,1}, ..., x_i' beta_{K-1}
    + shift_{g_i,K-1}, 0))`.

    The group shifts are implicitly defined by holding the group-level outcome
    means fixed.  The returned `shift_beta_jacobian[g, c, p]` is therefore the
    derivative of free shift `c` in group `g` with respect to beta parameter
    `p`, where beta is ordered feature-major: `(feature, free_category)`.
    """

    X = np.asarray(X, dtype=np.float64)
    probs = _probability_matrix(shifted_probs, name="shifted_probs", eps=eps)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if X.shape[0] != probs.shape[0]:
        raise ValueError("X and shifted_probs must have the same number of rows")

    group_ids, n_groups = _unit_group_ids(groups, probs.shape[0])
    if n_groups == 0:
        raise ValueError("at least one group is required")

    n_features = X.shape[1]
    n_categories = probs.shape[1]
    n_free = n_categories - 1
    n_params = n_features * n_free
    shift_hessian = np.empty((n_groups, n_free, n_free), dtype=np.float64)
    precinct_beta_jacobian = np.empty((n_groups, n_free, n_params), dtype=np.float64)

    for outcome in range(n_free):
        for free_category in range(n_free):
            derivative = probs[:, outcome] * (
                (1.0 if outcome == free_category else 0.0) - probs[:, free_category]
            )
            shift_hessian[:, outcome, free_category] = np.bincount(
                group_ids,
                weights=derivative,
                minlength=n_groups,
            )
            for feature in range(n_features):
                param = feature * n_free + free_category
                precinct_beta_jacobian[:, outcome, param] = np.bincount(
                    group_ids,
                    weights=derivative * X[:, feature],
                    minlength=n_groups,
                )

    precision = np.linalg.pinv(shift_hessian, rcond=rcond, hermitian=True)
    shift_beta_jacobian = -np.einsum(
        "gab,gbp->gap",
        precision,
        precinct_beta_jacobian,
        optimize=True,
    )
    return {
        "group_ids": group_ids,
        "n_groups": np.asarray(n_groups),
        "shift_hessian": shift_hessian,
        "precinct_beta_jacobian": precinct_beta_jacobian,
        "shift_beta_jacobian": shift_beta_jacobian,
    }


def shifted_probability_weighted_gradients(
    X,
    shifted_probs,
    groups,
    weights,
    *,
    normalizers=None,
    sensitivity: Mapping[str, np.ndarray] | None = None,
    rcond: float = 1e-10,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return beta gradients of weighted shifted probability averages.

    For each weight column `m` and outcome `k`, this differentiates

    `sum_i weights[i, m] * shifted_probs[i, k] / normalizers[m]`

    with respect to the global softmax slope, including the implicit response
    of the fitted group-level logit shifts.  This is the first-order primitive
    behind delta-method uncertainty for shifted subgroup estimates and
    model-implied marginal flows.
    """

    X = np.asarray(X, dtype=np.float64)
    probs = _probability_matrix(shifted_probs, name="shifted_probs", eps=eps)
    weights = np.asarray(weights, dtype=np.float64)
    squeeze = False
    if weights.ndim == 1:
        weights = weights[:, None]
        squeeze = True
    if weights.ndim != 2:
        raise ValueError("weights must be one- or two-dimensional")
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if X.shape[0] != probs.shape[0] or weights.shape[0] != probs.shape[0]:
        raise ValueError("X, shifted_probs, and weights must have the same number of rows")

    if normalizers is None:
        normalizers = weights.sum(axis=0)
    normalizers = np.asarray(normalizers, dtype=np.float64)
    if normalizers.ndim == 0:
        normalizers = np.full(weights.shape[1], float(normalizers))
    if normalizers.shape != (weights.shape[1],):
        raise ValueError("normalizers must have one entry per weight column")
    if np.any(np.abs(normalizers) <= eps):
        raise ValueError("normalizers must be nonzero")

    group_ids, n_groups = _unit_group_ids(groups, probs.shape[0])
    if sensitivity is None:
        sensitivity = shifted_softmax_beta_sensitivity(
            X,
            probs,
            group_ids,
            rcond=rcond,
            eps=eps,
        )
    shift_beta_jacobian = np.asarray(sensitivity["shift_beta_jacobian"], dtype=np.float64)
    if shift_beta_jacobian.shape[0] < n_groups:
        raise ValueError("sensitivity has fewer groups than the supplied group ids")

    n_weights = weights.shape[1]
    n_features = X.shape[1]
    n_categories = probs.shape[1]
    n_free = n_categories - 1
    n_params = n_features * n_free
    gradients = np.zeros((n_weights, n_categories, n_params), dtype=np.float64)

    for outcome in range(n_categories):
        for free_category in range(n_free):
            probability_derivative = probs[:, outcome] * (
                (1.0 if outcome == free_category else 0.0) - probs[:, free_category]
            )
            weighted_derivative = weights * probability_derivative[:, None]
            direct = X.T @ weighted_derivative
            gradients[:, outcome, free_category::n_free] += (
                direct.T / normalizers[:, None]
            )

            grouped_derivative = np.empty((n_groups, n_weights), dtype=np.float64)
            for column in range(n_weights):
                grouped_derivative[:, column] = np.bincount(
                    group_ids,
                    weights=weighted_derivative[:, column],
                    minlength=n_groups,
                )
            shift_part = grouped_derivative.T @ shift_beta_jacobian[:n_groups, free_category, :]
            gradients[:, outcome, :] += shift_part / normalizers[:, None]

    if squeeze:
        return gradients[0]
    return gradients


def parameter_draws(
    params,
    covariance,
    *,
    n_draws: int,
    seed: int | None = None,
    include_center: bool = False,
    rcond: float = 0.0,
) -> np.ndarray:
    """Draw parameters from a normal approximation.

    Small negative covariance eigenvalues are clipped to zero.  If `rcond` is
    positive, eigenvalues below `rcond * max_eigenvalue` are also clipped.
    """

    params = np.asarray(params, dtype=np.float64).ravel()
    if n_draws < 0:
        raise ValueError("n_draws must be nonnegative")
    if covariance is None:
        draws = np.empty((0, params.size), dtype=np.float64)
    else:
        covariance = np.asarray(covariance, dtype=np.float64)
        if covariance.shape != (params.size, params.size):
            raise ValueError("covariance shape does not match params")
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        largest = max(float(np.max(eigenvalues)), 0.0) if eigenvalues.size else 0.0
        cutoff = largest * rcond
        eigenvalues = np.where(eigenvalues > cutoff, eigenvalues, 0.0)
        transform = eigenvectors * np.sqrt(eigenvalues)
        rng = np.random.default_rng(seed)
        draws = params + rng.standard_normal((n_draws, params.size)) @ transform.T

    if include_center:
        draws = np.vstack([params[None, :], draws])
    return draws


def pairwise_frechet_bounds(first, second, weights=None) -> dict[str, float]:
    """Return product flow and sharp pairwise Frechet bounds."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("first and second probabilities must have the same shape")
    if first.ndim != 1:
        raise ValueError("first and second must be one-dimensional")

    product = first * second
    lower = np.maximum(0.0, first + second - 1.0)
    upper = np.minimum(first, second)
    return {
        "pred": _weighted_mean(product, weights),
        "lower": _weighted_mean(lower, weights),
        "upper": _weighted_mean(upper, weights),
        "width": _weighted_mean(upper - lower, weights),
    }


def pairwise_flow_table(
    probabilities: Mapping[str, object],
    *,
    strata=None,
    pairs: Sequence[tuple[str, str]] | None = None,
    outcome_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    weights=None,
) -> pd.DataFrame:
    """Compute product flows and Frechet bounds for item pairs.

    Parameters
    ----------
    probabilities:
        Mapping from item name, such as an office or year, to an `n x K`
        probability matrix.
    strata:
        Optional stratum label for each row.  A single statewide stratum is used
        when omitted.
    pairs:
        Optional item pairs.  By default, all unordered pairs are used in the
        insertion order of `probabilities`.
    outcome_names:
        Either one sequence shared by all items or a mapping from item name to
        outcome labels.  Defaults to stringified category indices.
    weights:
        Optional nonnegative row weights.  Weighted means are reported within
        each stratum.
    """

    if not probabilities:
        raise ValueError("probabilities cannot be empty")
    matrices = {
        name: _probability_matrix(probs, name=f"probabilities[{name!r}]")
        for name, probs in probabilities.items()
    }
    n_rows = _common_n_rows(matrices)
    stratum_codes, stratum_labels = _factorize_strata(strata, n_rows)
    weights = _weights(weights, n_rows)
    names = list(matrices)
    if pairs is None:
        pairs = list(combinations(names, 2))
    outcome_labels = {
        name: _outcome_names(outcome_names, name, matrices[name].shape[1])
        for name in names
    }

    rows = []
    for stratum_code, stratum_label in enumerate(stratum_labels):
        mask = stratum_codes == stratum_code
        stratum_weights = None if weights is None else weights[mask]
        n_units = int(mask.sum())
        weight_sum = float(n_units if stratum_weights is None else stratum_weights.sum())
        for first_name, second_name in pairs:
            first_probs = matrices[first_name][mask]
            second_probs = matrices[second_name][mask]
            for first_index, first_outcome in enumerate(outcome_labels[first_name]):
                for second_index, second_outcome in enumerate(outcome_labels[second_name]):
                    stats = pairwise_frechet_bounds(
                        first_probs[:, first_index],
                        second_probs[:, second_index],
                        weights=stratum_weights,
                    )
                    rows.append(
                        {
                            "stratum": stratum_label,
                            "item_1": first_name,
                            "item_2": second_name,
                            "outcome_1": first_outcome,
                            "outcome_2": second_outcome,
                            "n_units": n_units,
                            "weight_sum": weight_sum,
                            **stats,
                        }
                    )
    return pd.DataFrame(rows)


def pairwise_flow_draw_table(
    probability_draws: Mapping[str, object],
    *,
    strata=None,
    pairs: Sequence[tuple[str, str]] | None = None,
    outcome_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    weights=None,
) -> pd.DataFrame:
    """Compute pairwise flow tables for a stack of marginal-probability draws.

    Each item in `probability_draws` may be either a plug-in `n x K` probability
    matrix or a draw stack with shape `(n_draws, n, K)`.  Plug-in matrices are
    reused for every draw.
    """

    if not probability_draws:
        raise ValueError("probability_draws cannot be empty")
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in probability_draws.items()}
    n_draws = max((array.shape[0] for array in arrays.values() if array.ndim == 3), default=1)
    tables = []
    for draw in range(n_draws):
        probabilities = {}
        for name, array in arrays.items():
            if array.ndim == 2:
                probabilities[name] = array
            elif array.ndim == 3:
                if array.shape[0] != n_draws:
                    raise ValueError("all probability draw stacks must have the same number of draws")
                probabilities[name] = array[draw]
            else:
                raise ValueError("probability draws must be two- or three-dimensional")
        table = pairwise_flow_table(
            probabilities,
            strata=strata,
            pairs=pairs,
            outcome_names=outcome_names,
            weights=weights,
        )
        table.insert(0, "draw", draw)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def add_bound_diagnostics(
    table: pd.DataFrame,
    *,
    truth_col: str = "true",
    lower_col: str = "lower",
    upper_col: str = "upper",
    width_col: str = "width",
    pred_col: str = "pred",
    truth_lower_col: str | None = None,
    truth_upper_col: str | None = None,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Add point and interval diagnostics for a flow-bound table."""

    out = table.copy()
    out["covered"] = (
        (out[truth_col] >= out[lower_col] - tolerance)
        & (out[truth_col] <= out[upper_col] + tolerance)
    )
    out["lower_violation"] = np.maximum(out[lower_col] - out[truth_col], 0.0)
    out["upper_violation"] = np.maximum(out[truth_col] - out[upper_col], 0.0)
    out["bound_violation"] = out["lower_violation"] + out["upper_violation"]
    out["truth_position"] = np.divide(
        out[truth_col] - out[lower_col],
        out[width_col],
        out=np.full(len(out), np.nan),
        where=out[width_col] > tolerance,
    )
    out["product_position"] = np.divide(
        out[pred_col] - out[lower_col],
        out[width_col],
        out=np.full(len(out), np.nan),
        where=out[width_col] > tolerance,
    )
    if truth_lower_col is not None and truth_upper_col is not None:
        out["truth_interval_overlaps_bound"] = (
            (out[truth_upper_col] >= out[lower_col] - tolerance)
            & (out[truth_lower_col] <= out[upper_col] + tolerance)
        )
        out["truth_interval_bound_violation"] = np.maximum(
            out[lower_col] - out[truth_upper_col],
            0.0,
        ) + np.maximum(
            out[truth_lower_col] - out[upper_col],
            0.0,
        )
    return out


def add_rate_uncertainty(
    table: pd.DataFrame,
    *,
    rate_col: str,
    n_col: str,
    prefix: str,
    level: float = 0.95,
    method: str = "wilson",
) -> pd.DataFrame:
    """Add finite-sample uncertainty columns for observed rates."""

    out = table.copy()
    rate = out[rate_col].to_numpy(dtype=np.float64)
    n = out[n_col].to_numpy(dtype=np.float64)
    lower, upper, se = rate_interval(rate, n, level=level, method=method)
    out[f"{prefix}_se"] = se
    out[f"{prefix}_lower"] = lower
    out[f"{prefix}_upper"] = upper
    return out


def rate_interval(
    rate,
    n,
    *,
    level: float = 0.95,
    method: str = "wilson",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return confidence intervals and standard errors for binomial rates."""

    rate = np.asarray(rate, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    if np.any(n <= 0):
        raise ValueError("n must be positive")
    if np.any((rate < 0.0) | (rate > 1.0)):
        raise ValueError("rate must be in [0, 1]")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between zero and one")

    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    se = np.sqrt(np.maximum(rate * (1.0 - rate) / n, 0.0))
    if method == "wald":
        lower = rate - z * se
        upper = rate + z * se
    elif method == "wilson":
        denominator = 1.0 + z**2 / n
        center = (rate + z**2 / (2.0 * n)) / denominator
        radius = z * np.sqrt((rate * (1.0 - rate) + z**2 / (4.0 * n)) / n) / denominator
        lower = center - radius
        upper = center + radius
    else:
        raise ValueError("method must be 'wilson' or 'wald'")
    return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0), se


def summarize_flow_draws(
    draws: pd.DataFrame,
    *,
    key_cols: Sequence[str] = DEFAULT_FLOW_KEYS,
    level: float = 0.95,
) -> pd.DataFrame:
    """Summarize flow and bound draws into confidence bands.

    The resulting `identified_lower` and `identified_upper` columns combine
    dependence uncertainty with parameter uncertainty by taking quantiles of
    the lower and upper bound draws.
    """

    if not 0.0 < level < 1.0:
        raise ValueError("level must be between zero and one")
    alpha = (1.0 - level) / 2.0
    required = set(key_cols) | {"pred", "lower", "upper", "width"}
    missing = sorted(required - set(draws.columns))
    if missing:
        raise ValueError(f"draw table is missing columns: {missing}")

    grouped = draws.groupby(list(key_cols), dropna=False)
    rows = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(key_cols, key))
        row.update(
            {
                "pred_mean": float(group["pred"].mean()),
                "pred_lower": float(group["pred"].quantile(alpha)),
                "pred_upper": float(group["pred"].quantile(1.0 - alpha)),
                "identified_lower": float(group["lower"].quantile(alpha)),
                "identified_upper": float(group["upper"].quantile(1.0 - alpha)),
                "mean_width": float(group["width"].mean()),
                "n_draws": int(len(group)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _probability_matrix(probs, *, name: str, eps: float = 1e-12) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if np.any(probs < -eps):
        raise ValueError(f"{name} contains negative probabilities")
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= eps):
        raise ValueError(f"{name} contains a row with no probability mass")
    return np.clip(probs, 0.0, None) / row_sums


def _common_n_rows(matrices: Mapping[str, np.ndarray]) -> int:
    sizes = {matrix.shape[0] for matrix in matrices.values()}
    if len(sizes) != 1:
        raise ValueError("all probability matrices must have the same number of rows")
    return sizes.pop()


def _factorize_strata(strata, n_rows: int) -> tuple[np.ndarray, list[object]]:
    if strata is None:
        return np.zeros(n_rows, dtype=np.int64), ["all"]
    strata = np.asarray(strata)
    if strata.shape != (n_rows,):
        raise ValueError("strata must have one entry per probability row")
    codes, labels = pd.factorize(strata, sort=True)
    if np.any(codes < 0):
        raise ValueError("strata cannot contain missing values")
    return codes.astype(np.int64), labels.tolist()


def _unit_group_ids(groups, n_rows: int) -> tuple[np.ndarray, int]:
    if hasattr(groups, "tocsc"):
        groups = groups.tocsc()
        if groups.shape[1] != n_rows:
            raise ValueError("group matrix must have one column per row")
        counts = np.diff(groups.indptr)
        if np.any(counts != 1):
            raise ValueError("each row must belong to exactly one group")
        group_ids = np.asarray(groups.indices, dtype=np.int64)
        return group_ids, int(groups.shape[0])

    group_ids = np.asarray(groups, dtype=np.int64)
    if group_ids.ndim != 1 or group_ids.shape != (n_rows,):
        raise ValueError("groups must be group ids or a group-by-row matrix")
    if np.any(group_ids < 0):
        raise ValueError("group ids must be nonnegative")
    n_groups = int(group_ids.max()) + 1 if group_ids.size else 0
    return group_ids, n_groups


def _weights(weights, n_rows: int) -> np.ndarray | None:
    if weights is None:
        return None
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n_rows,):
        raise ValueError("weights must have one entry per probability row")
    if np.any(weights < 0.0):
        raise ValueError("weights must be nonnegative")
    return weights


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(values.mean()) if values.size else float("nan")
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return float("nan")
    return float(np.average(values, weights=weights))


def _outcome_names(
    outcome_names: Sequence[str] | Mapping[str, Sequence[str]] | None,
    item_name: str,
    n_outcomes: int,
) -> list[str]:
    if outcome_names is None:
        return [str(index) for index in range(n_outcomes)]
    if isinstance(outcome_names, Mapping):
        names = list(outcome_names[item_name])
    else:
        names = list(outcome_names)
    if len(names) != n_outcomes:
        raise ValueError(f"outcome_names for {item_name!r} must have length {n_outcomes}")
    return names

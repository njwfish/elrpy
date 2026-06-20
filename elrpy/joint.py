"""Joint ticket regions induced by fitted marginal probabilities.

The fitted marginal probabilities do not identify a unique joint ticket
distribution. They define an implicit convex set: each voter may have any joint
law with the fitted contest-level marginals, and precinct or subgroup summaries
average those voter-level joint laws. This module computes useful projections
of that set.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def joint_outcome_grid(outcome_sizes: Sequence[int]) -> np.ndarray:
    """Return all joint cells as integer outcome indices."""

    outcome_sizes = _outcome_sizes(outcome_sizes)
    return np.asarray(list(product(*[range(size) for size in outcome_sizes])), dtype=np.int64)


def sharp_joint_region(
    *,
    probabilities,
    group_ids,
    group_labels,
    subgroup_weights,
    item_names: Sequence[str],
    outcome_names: Sequence[str],
    subgroup_names: Sequence[str],
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a validated sharp joint-region NPZ payload.

    For each unit `i`, `probabilities[i, j, :]` stores the fitted marginal
    probabilities for item/race `j`. For a precinct `g` and subgroup `s`, the
    sharp joint region is

    `{ sum_i w_{is} pi_i / sum_i w_{is} : pi_i has marginals p_i }`,

    where the sum is over units in precinct `g`. This is an exact extended
    formulation of the heterogeneous region; LP projections are solved only
    when requested.
    """

    store = {
        "probabilities": np.asarray(probabilities, dtype=np.float64),
        "group_ids": np.asarray(group_ids, dtype=np.int64),
        "group_labels": np.asarray(group_labels).astype(str),
        "subgroup_weights": np.asarray(subgroup_weights, dtype=np.float64),
        "item_names": tuple(str(name) for name in item_names),
        "outcome_names": tuple(str(name) for name in outcome_names),
        "subgroup_names": tuple(str(name) for name in subgroup_names),
        "metadata": dict(metadata or {}),
    }
    return validate_sharp_joint_region(store)


def validate_sharp_joint_region(store: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize a sharp joint-region payload."""

    probabilities = np.asarray(store["probabilities"], dtype=np.float64)
    group_ids = np.asarray(store["group_ids"], dtype=np.int64)
    group_labels = np.asarray(store["group_labels"]).astype(str)
    subgroup_weights = np.asarray(store["subgroup_weights"], dtype=np.float64)
    item_names = tuple(str(name) for name in store["item_names"])
    outcome_names = tuple(str(name) for name in store["outcome_names"])
    subgroup_names = tuple(str(name) for name in store["subgroup_names"])
    metadata = dict(store.get("metadata", {}))

    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape (n_units, n_items, n_outcomes)")
    if group_ids.shape != (probabilities.shape[0],):
        raise ValueError("group_ids must have one entry per unit")
    if np.any(group_ids < 0) or np.any(group_ids >= len(group_labels)):
        raise ValueError("group_ids must index group_labels")
    if subgroup_weights.shape != (probabilities.shape[0], len(subgroup_names)):
        raise ValueError("subgroup_weights must have shape (n_units, n_subgroups)")
    if len(item_names) != probabilities.shape[1]:
        raise ValueError("item_names must have one entry per item")
    if len(outcome_names) != probabilities.shape[2]:
        raise ValueError("outcome_names must have one entry per outcome")
    if np.any(subgroup_weights < 0.0):
        raise ValueError("subgroup weights must be nonnegative")

    return {
        "probabilities": probabilities,
        "group_ids": group_ids,
        "group_labels": group_labels,
        "subgroup_weights": subgroup_weights,
        "item_names": item_names,
        "outcome_names": outcome_names,
        "subgroup_names": subgroup_names,
        "metadata": metadata,
    }


def save_sharp_joint_region(
    path: str | Path,
    *,
    probabilities,
    group_ids,
    group_labels,
    subgroup_weights,
    item_names: Sequence[str],
    outcome_names: Sequence[str],
    subgroup_names: Sequence[str],
    metadata: Mapping[str, object] | None = None,
    dtype=np.float32,
) -> None:
    """Save a sharp joint-region extended formulation to compressed NPZ."""

    store = sharp_joint_region(
        probabilities=probabilities,
        group_ids=group_ids,
        group_labels=group_labels,
        subgroup_weights=subgroup_weights,
        item_names=item_names,
        outcome_names=outcome_names,
        subgroup_names=subgroup_names,
        metadata=metadata,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probabilities=np.asarray(store["probabilities"], dtype=dtype),
        group_ids=np.asarray(store["group_ids"], dtype=np.int32),
        group_labels=np.asarray(store["group_labels"], dtype=str),
        subgroup_weights=np.asarray(store["subgroup_weights"], dtype=dtype),
        item_names=np.asarray(store["item_names"], dtype=str),
        outcome_names=np.asarray(store["outcome_names"], dtype=str),
        subgroup_names=np.asarray(store["subgroup_names"], dtype=str),
        metadata_json=np.asarray(json.dumps(store["metadata"], sort_keys=True)),
        region_type=np.asarray("sharp_heterogeneous_extended_formulation"),
    )


def load_sharp_joint_region(path: str | Path) -> dict[str, object]:
    """Load a saved sharp joint-region extended formulation."""

    with np.load(path, allow_pickle=False) as npz:
        metadata = {}
        if "metadata_json" in npz.files:
            metadata = json.loads(str(np.asarray(npz["metadata_json"]).item()))
        return sharp_joint_region(
            probabilities=np.asarray(npz["probabilities"], dtype=np.float64),
            group_ids=np.asarray(npz["group_ids"], dtype=np.int64),
            group_labels=np.asarray(npz["group_labels"]).astype(str),
            subgroup_weights=np.asarray(npz["subgroup_weights"], dtype=np.float64),
            item_names=np.asarray(npz["item_names"]).astype(str).tolist(),
            outcome_names=np.asarray(npz["outcome_names"]).astype(str).tolist(),
            subgroup_names=np.asarray(npz["subgroup_names"]).astype(str).tolist(),
            metadata=metadata,
        )


def sharp_joint_region_arrays(
    store: Mapping[str, object],
    group: int | str,
    subgroup: int | str,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return voter-level marginals and weights for one precinct/subgroup."""

    store = validate_sharp_joint_region(store)
    group_index = _resolve_label(group, store["group_labels"], "group")
    subgroup_index = _resolve_label(subgroup, store["subgroup_names"], "subgroup")
    mask = store["group_ids"] == group_index
    if not np.any(mask):
        raise ValueError("selected group has no units")
    weights = store["subgroup_weights"][mask, subgroup_index]
    if weights.sum() <= 0.0:
        raise ValueError("selected group/subgroup has zero weight")
    marginals = [
        store["probabilities"][mask, item, :]
        for item in range(store["probabilities"].shape[1])
    ]
    return marginals, weights


def sharp_joint_region_lp_bounds(
    store: Mapping[str, object],
    group: int | str,
    subgroup: int | str,
    coefficients,
    *,
    method: str = "highs",
) -> dict[str, np.ndarray]:
    """Solve exact sharp bounds for linear summaries in one saved region."""

    marginals, weights = sharp_joint_region_arrays(store, group, subgroup)
    return joint_lp_bounds(
        marginals,
        coefficients,
        weights=weights,
        method=method,
    )


def sharp_joint_region_simplex_region(
    store: Mapping[str, object],
    group: int | str,
    subgroup: int | str,
) -> dict[str, object]:
    """Return the compact aggregate-marginal relaxation for one saved region."""

    store = validate_sharp_joint_region(store)
    marginals, weights = sharp_joint_region_arrays(store, group, subgroup)
    return joint_simplex_region(
        marginals,
        weights=weights,
        item_names=store["item_names"],
        outcome_names=store["outcome_names"],
    )


def joint_indicator(
    outcome_sizes: Sequence[int],
    conditions: Mapping[int, int] | Sequence[tuple[int, int]],
) -> np.ndarray:
    """Return a coefficient vector for a joint event.

    `conditions` maps item index to the required outcome index.  For example,
    `{0: 1, 2: 0}` is the event that item 0 has outcome 1 and item 2 has
    outcome 0, regardless of all other items.
    """

    grid = joint_outcome_grid(outcome_sizes)
    if not isinstance(conditions, Mapping):
        conditions = dict(conditions)
    mask = np.ones(len(grid), dtype=bool)
    for item, outcome in conditions.items():
        if not 0 <= item < grid.shape[1]:
            raise ValueError("condition item is out of bounds")
        if not 0 <= outcome < outcome_sizes[item]:
            raise ValueError("condition outcome is out of bounds")
        mask &= grid[:, item] == outcome
    return mask.astype(np.float64)


def joint_lp_constraint_matrix(outcome_sizes: Sequence[int]) -> np.ndarray:
    """Return the equality matrix mapping joint cells to item marginals."""

    outcome_sizes = _outcome_sizes(outcome_sizes)
    grid = joint_outcome_grid(outcome_sizes)
    rows = []
    for item, size in enumerate(outcome_sizes):
        for outcome in range(size):
            rows.append((grid[:, item] == outcome).astype(np.float64))
    return np.vstack(rows)


def joint_lp_bounds(
    marginals: Sequence[object],
    coefficients,
    *,
    weights=None,
    method: str = "highs",
    tolerance: float = 1e-9,
) -> dict[str, np.ndarray]:
    """Bound linear summaries of the exact heterogeneous joint region.

    Parameters
    ----------
    marginals:
        Sequence of `n x K_j` marginal probability matrices, one matrix per
        item/race.
    coefficients:
        A vector of length `prod_j K_j`, or a matrix whose rows are such
        vectors. Each row defines a linear summary of the full joint ticket.
    weights:
        Optional nonnegative unit weights. Bounds are reported for the weighted
        average of voter-level summaries.

    Notes
    -----
    The LP separates over voters, so this is the sharp projection of the
    heterogeneous feasible set. It is intended for a modest number of arbitrary
    summaries. Common projections such as pairwise edges and full-ticket cells
    have closed forms below and should be used at production scale.
    """

    matrices = _probability_matrices(marginals)
    n_rows = _common_n_rows(matrices)
    weights = _weights(weights, n_rows)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    squeeze = False
    if coefficients.ndim == 1:
        coefficients = coefficients[None, :]
        squeeze = True
    if coefficients.ndim != 2:
        raise ValueError("coefficients must be one- or two-dimensional")

    outcome_sizes = [matrix.shape[1] for matrix in matrices]
    n_cells = int(np.prod(outcome_sizes))
    if coefficients.shape[1] != n_cells:
        raise ValueError(f"coefficients must have {n_cells} columns")

    A_eq = joint_lp_constraint_matrix(outcome_sizes)
    bounds = [(0.0, None)] * n_cells
    lower = np.zeros(coefficients.shape[0], dtype=np.float64)
    upper = np.zeros(coefficients.shape[0], dtype=np.float64)
    product_projection = np.zeros(coefficients.shape[0], dtype=np.float64)

    for row in range(n_rows):
        if weights[row] == 0.0:
            continue
        b_eq = np.concatenate([matrix[row] for matrix in matrices])
        for summary, coefficient in enumerate(coefficients):
            minimum = linprog(
                coefficient,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method=method,
            )
            if not minimum.success:
                raise RuntimeError(f"lower-bound LP failed: {minimum.message}")
            maximum = linprog(
                -coefficient,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method=method,
            )
            if not maximum.success:
                raise RuntimeError(f"upper-bound LP failed: {maximum.message}")
            lower[summary] += weights[row] * minimum.fun
            upper[summary] += weights[row] * (-maximum.fun)

        product_projection += weights[row] * independent_joint_projection(
            [matrix[row : row + 1] for matrix in matrices],
            coefficients,
        )

    weight_sum = weights.sum()
    if weight_sum <= tolerance:
        raise ValueError("weights must have positive total mass")
    lower /= weight_sum
    upper /= weight_sum
    product_projection /= weight_sum
    out = {
        "product": product_projection,
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
    }
    if squeeze:
        return {key: value[0] for key, value in out.items()}
    return out


def aggregate_marginals(marginals: Sequence[object], *, weights=None) -> list[np.ndarray]:
    """Return weighted average marginal probabilities for each item."""

    matrices = _probability_matrices(marginals)
    n_rows = _common_n_rows(matrices)
    weights = _weights(weights, n_rows)
    weight_sum = weights.sum()
    if weight_sum <= 0.0:
        raise ValueError("weights must have positive total mass")
    return [(weights @ matrix) / weight_sum for matrix in matrices]


def joint_simplex_region(
    marginals: Sequence[object] | None = None,
    *,
    marginal_means: Sequence[object] | None = None,
    weights=None,
    item_names: Sequence[str] | None = None,
    outcome_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Return the compact aggregate-marginal joint simplex region.

    This is the H-representation

    `q >= 0,  A_eq q = b_eq`,

    where `q` is a full joint ticket distribution and `b_eq` contains the
    aggregate item marginals.  It is a compact, serializable relaxation of the
    sharper heterogeneous voter-level region used by `joint_lp_bounds`.
    """

    if marginal_means is None:
        if marginals is None:
            raise ValueError("pass marginals or marginal_means")
        marginal_means = aggregate_marginals(marginals, weights=weights)
    means = [np.asarray(mean, dtype=np.float64).ravel() for mean in marginal_means]
    if not means:
        raise ValueError("at least one marginal mean vector is required")
    for index, mean in enumerate(means):
        if mean.ndim != 1:
            raise ValueError("marginal means must be one-dimensional")
        if np.any(mean < -1e-12):
            raise ValueError("marginal means cannot be negative")
        total = mean.sum()
        if total <= 0.0:
            raise ValueError("marginal means must have positive mass")
        means[index] = np.clip(mean, 0.0, None) / total

    outcome_sizes = [len(mean) for mean in means]
    grid = joint_outcome_grid(outcome_sizes)
    A_eq = joint_lp_constraint_matrix(outcome_sizes)
    b_eq = np.concatenate(means)
    if item_names is None:
        item_names = [f"item_{index}" for index in range(len(means))]
    item_names = list(item_names)
    if len(item_names) != len(means):
        raise ValueError("item_names must have one entry per marginal")
    labels = {
        name: _outcome_names(outcome_names, name, size)
        for name, size in zip(item_names, outcome_sizes)
    }
    cell_table = pd.DataFrame(
        {
            name: [labels[name][outcome] for outcome in grid[:, item]]
            for item, name in enumerate(item_names)
        }
    )
    row_items = []
    row_outcomes = []
    for name, size in zip(item_names, outcome_sizes):
        for outcome in range(size):
            row_items.append(name)
            row_outcomes.append(labels[name][outcome])
    return {
        "outcome_sizes": outcome_sizes,
        "item_names": item_names,
        "grid": grid,
        "cell_table": cell_table,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "row_items": np.asarray(row_items, dtype=object),
        "row_outcomes": np.asarray(row_outcomes, dtype=object),
        "bounds": [(0.0, None)] * len(grid),
    }


def joint_simplex_lp_bounds(
    marginals: Sequence[object] | None,
    coefficients,
    *,
    marginal_means: Sequence[object] | None = None,
    weights=None,
    method: str = "highs",
) -> dict[str, np.ndarray]:
    """Bound summaries over the aggregate-marginal simplex region."""

    region = joint_simplex_region(
        marginals,
        marginal_means=marginal_means,
        weights=weights,
    )
    coefficients = np.asarray(coefficients, dtype=np.float64)
    squeeze = False
    if coefficients.ndim == 1:
        coefficients = coefficients[None, :]
        squeeze = True
    if coefficients.ndim != 2:
        raise ValueError("coefficients must be one- or two-dimensional")
    if coefficients.shape[1] != len(region["grid"]):
        raise ValueError("coefficient length does not match the joint grid")

    lower = np.empty(coefficients.shape[0], dtype=np.float64)
    upper = np.empty(coefficients.shape[0], dtype=np.float64)
    for index, coefficient in enumerate(coefficients):
        minimum = linprog(
            coefficient,
            A_eq=region["A_eq"],
            b_eq=region["b_eq"],
            bounds=region["bounds"],
            method=method,
        )
        if not minimum.success:
            raise RuntimeError(f"lower-bound LP failed: {minimum.message}")
        maximum = linprog(
            -coefficient,
            A_eq=region["A_eq"],
            b_eq=region["b_eq"],
            bounds=region["bounds"],
            method=method,
        )
        if not maximum.success:
            raise RuntimeError(f"upper-bound LP failed: {maximum.message}")
        lower[index] = minimum.fun
        upper[index] = -maximum.fun

    product_projection = independent_joint_projection(
        [mean[None, :] for mean in np.split(region["b_eq"], np.cumsum(region["outcome_sizes"])[:-1])],
        coefficients,
    )
    out = {
        "product": product_projection,
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
    }
    if squeeze:
        return {key: value[0] for key, value in out.items()}
    return out


def joint_simplex_coordinate_bounds(
    marginals: Sequence[object] | None = None,
    *,
    marginal_means: Sequence[object] | None = None,
    weights=None,
    item_names: Sequence[str] | None = None,
    outcome_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    method: str = "highs",
) -> pd.DataFrame:
    """Return LP bounds for every full-ticket cell in the simplex region."""

    region = joint_simplex_region(
        marginals,
        marginal_means=marginal_means,
        weights=weights,
        item_names=item_names,
        outcome_names=outcome_names,
    )
    coefficients = np.eye(len(region["grid"]), dtype=np.float64)
    bounds = joint_simplex_lp_bounds(
        None,
        coefficients,
        marginal_means=np.split(
            region["b_eq"],
            np.cumsum(region["outcome_sizes"])[:-1],
        ),
        method=method,
    )
    table = region["cell_table"].copy()
    table["product"] = bounds["product"]
    table["lower"] = bounds["lower"]
    table["upper"] = bounds["upper"]
    table["width"] = bounds["width"]
    return table


def independent_joint_projection(
    marginals: Sequence[object],
    coefficients,
    *,
    weights=None,
) -> np.ndarray:
    """Evaluate linear summaries under the independent product coupling."""

    matrices = _probability_matrices(marginals)
    n_rows = _common_n_rows(matrices)
    weights = _weights(weights, n_rows)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim == 1:
        coefficients = coefficients[None, :]
    outcome_sizes = [matrix.shape[1] for matrix in matrices]
    grid = joint_outcome_grid(outcome_sizes)
    if coefficients.shape[1] != len(grid):
        raise ValueError("coefficient length does not match the joint grid")

    values = np.zeros(coefficients.shape[0], dtype=np.float64)
    for cell_index, cell in enumerate(grid):
        cell_prob = np.ones(n_rows, dtype=np.float64)
        for item, outcome in enumerate(cell):
            cell_prob *= matrices[item][:, outcome]
        values += coefficients[:, cell_index] * np.average(cell_prob, weights=weights)
    return values


def full_joint_cell_bounds(
    marginals: Sequence[object],
    *,
    weights=None,
) -> dict[str, np.ndarray]:
    """Return sharp bounds for every full joint ticket cell.

    For a cell requiring outcomes `(k_1, ..., k_m)`, the voter-level sharp bounds
    are the multi-event Frechet bounds

    `max(0, sum_j p_j(k_j) - (m - 1))` and `min_j p_j(k_j)`.
    """

    matrices = _probability_matrices(marginals)
    n_rows = _common_n_rows(matrices)
    weights = _weights(weights, n_rows)
    outcome_sizes = [matrix.shape[1] for matrix in matrices]
    grid = joint_outcome_grid(outcome_sizes)

    product_value = np.empty(len(grid), dtype=np.float64)
    lower = np.empty(len(grid), dtype=np.float64)
    upper = np.empty(len(grid), dtype=np.float64)
    for cell_index, cell in enumerate(grid):
        selected = np.column_stack(
            [matrices[item][:, outcome] for item, outcome in enumerate(cell)]
        )
        product_value[cell_index] = np.average(np.prod(selected, axis=1), weights=weights)
        lower[cell_index] = np.average(
            np.maximum(0.0, selected.sum(axis=1) - (len(matrices) - 1)),
            weights=weights,
        )
        upper[cell_index] = np.average(selected.min(axis=1), weights=weights)

    return {
        "grid": grid,
        "product": product_value,
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
    }


def full_joint_cell_bound_table(
    probabilities: Mapping[str, object],
    *,
    strata=None,
    outcome_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    weights=None,
) -> pd.DataFrame:
    """Compute production-scale full-ticket cell bounds by stratum."""

    if not probabilities:
        raise ValueError("probabilities cannot be empty")
    matrices = {
        name: _probability_matrix(probs, name=f"probabilities[{name!r}]")
        for name, probs in probabilities.items()
    }
    n_rows = _common_n_rows(list(matrices.values()))
    weights = _weights(weights, n_rows)
    stratum_codes, stratum_labels = _factorize_strata(strata, n_rows)
    n_strata = len(stratum_labels)
    n_units = np.bincount(stratum_codes, minlength=n_strata)
    weight_sum = np.bincount(stratum_codes, weights=weights, minlength=n_strata)
    names = list(matrices)
    outcome_sizes = [matrices[name].shape[1] for name in names]
    grid = joint_outcome_grid(outcome_sizes)
    labels = {
        name: _outcome_names(outcome_names, name, matrices[name].shape[1])
        for name in names
    }

    rows = []
    for cell in grid:
        selected = np.column_stack(
            [matrices[name][:, outcome] for name, outcome in zip(names, cell)]
        )
        product_value = np.prod(selected, axis=1)
        lower = np.maximum(0.0, selected.sum(axis=1) - (len(names) - 1))
        upper = selected.min(axis=1)
        product_by_stratum = _grouped_weighted_mean(
            product_value,
            stratum_codes,
            weights,
            weight_sum,
            n_strata,
        )
        lower_by_stratum = _grouped_weighted_mean(
            lower,
            stratum_codes,
            weights,
            weight_sum,
            n_strata,
        )
        upper_by_stratum = _grouped_weighted_mean(
            upper,
            stratum_codes,
            weights,
            weight_sum,
            n_strata,
        )
        for stratum_index, stratum_label in enumerate(stratum_labels):
            row = {
                "stratum": stratum_label,
                "n_units": int(n_units[stratum_index]),
                "weight_sum": float(weight_sum[stratum_index]),
                "product": product_by_stratum[stratum_index],
                "lower": lower_by_stratum[stratum_index],
                "upper": upper_by_stratum[stratum_index],
            }
            for name, outcome in zip(names, cell):
                row[name] = labels[name][outcome]
            row["width"] = row["upper"] - row["lower"]
            rows.append(row)
    return pd.DataFrame(rows)


def pairwise_edge_bound_table(
    probabilities: Mapping[str, object],
    *,
    strata=None,
    pairs: Sequence[tuple[str, str]] | None = None,
    outcome_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    weights=None,
) -> pd.DataFrame:
    """Compute pairwise Frechet bounds with vectorized stratum aggregation."""

    if not probabilities:
        raise ValueError("probabilities cannot be empty")
    matrices = {
        name: _probability_matrix(probs, name=f"probabilities[{name!r}]")
        for name, probs in probabilities.items()
    }
    n_rows = _common_n_rows(list(matrices.values()))
    weights = _weights(weights, n_rows)
    stratum_codes, stratum_labels = _factorize_strata(strata, n_rows)
    n_strata = len(stratum_labels)
    n_units = np.bincount(stratum_codes, minlength=n_strata)
    weight_sum = np.bincount(stratum_codes, weights=weights, minlength=n_strata)
    names = list(matrices)
    if pairs is None:
        pairs = list(combinations(names, 2))
    labels = {
        name: _outcome_names(outcome_names, name, matrices[name].shape[1])
        for name in names
    }

    rows = []
    for first_name, second_name in pairs:
        first = matrices[first_name]
        second = matrices[second_name]
        for first_outcome, first_label in enumerate(labels[first_name]):
            for second_outcome, second_label in enumerate(labels[second_name]):
                first_prob = first[:, first_outcome]
                second_prob = second[:, second_outcome]
                product_value = first_prob * second_prob
                lower = np.maximum(0.0, first_prob + second_prob - 1.0)
                upper = np.minimum(first_prob, second_prob)
                product_by_stratum = _grouped_weighted_mean(
                    product_value,
                    stratum_codes,
                    weights,
                    weight_sum,
                    n_strata,
                )
                lower_by_stratum = _grouped_weighted_mean(
                    lower,
                    stratum_codes,
                    weights,
                    weight_sum,
                    n_strata,
                )
                upper_by_stratum = _grouped_weighted_mean(
                    upper,
                    stratum_codes,
                    weights,
                    weight_sum,
                    n_strata,
                )
                for stratum_index, stratum_label in enumerate(stratum_labels):
                    rows.append(
                        {
                            "stratum": stratum_label,
                            "item_1": first_name,
                            "item_2": second_name,
                            "outcome_1": first_label,
                            "outcome_2": second_label,
                            "n_units": int(n_units[stratum_index]),
                            "weight_sum": float(weight_sum[stratum_index]),
                            "pred": product_by_stratum[stratum_index],
                            "lower": lower_by_stratum[stratum_index],
                            "upper": upper_by_stratum[stratum_index],
                            "width": (
                                upper_by_stratum[stratum_index]
                                - lower_by_stratum[stratum_index]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def transition_functional_bound_table(
    first_probs,
    second_probs,
    functionals: Sequence[Mapping[str, object]],
    *,
    outcome_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    strata=None,
    weights=None,
    method: str = "highs",
) -> pd.DataFrame:
    """Bound named linear functionals of a two-margin transition table.

    Each functional is a mapping with a ``component`` name and an ``estimand``.
    Supported production estimands are ``margin_delta``,
    ``turnout_margin_delta``, ``persuasion_margin_delta``, and ``event``.
    Arbitrary two-margin cost matrices can be supplied with ``estimand='lp'``;
    those use the generic heterogeneous LP and are intended for small or
    diagnostic projections rather than electorate-scale production runs.
    """

    first = _probability_matrix(first_probs, name="first_probs")
    second = _probability_matrix(second_probs, name="second_probs")
    n_rows = _common_n_rows([first, second])
    if first.shape[1] != second.shape[1]:
        raise ValueError("first_probs and second_probs must have the same outcomes")
    if not functionals:
        raise ValueError("at least one functional is required")

    weights = _weights(weights, n_rows)
    stratum_codes, stratum_labels = _factorize_strata(strata, n_rows)
    n_strata = len(stratum_labels)
    n_units = np.bincount(stratum_codes, minlength=n_strata)
    weight_sum = np.bincount(stratum_codes, weights=weights, minlength=n_strata)
    first_labels, second_labels = _two_margin_outcome_names(
        outcome_names,
        first.shape[1],
    )

    rows = []
    turnout_cache: dict[tuple[tuple[float, ...], int], dict[str, np.ndarray]] = {}
    for functional in functionals:
        component = str(functional["component"])
        kind = str(functional.get("kind", "transition_functional"))
        margin_direction = str(functional.get("margin_direction", ""))
        estimand = str(functional.get("estimand", "event"))

        if estimand == "margin_delta":
            margin = _functional_margin_values(functional, first.shape[1])
            value = second @ margin - first @ margin
            _append_functional_rows(
                rows,
                component=component,
                kind=kind,
                margin_direction=margin_direction,
                product=value,
                lower=value,
                upper=value,
                stratum_codes=stratum_codes,
                stratum_labels=stratum_labels,
                n_units=n_units,
                weight_sum=weight_sum,
                weights=weights,
            )
        elif estimand in {"turnout_margin_delta", "persuasion_margin_delta"}:
            margin = _functional_margin_values(functional, first.shape[1])
            nonvoter_index = _functional_nonvoter_index(
                functional,
                first_labels,
                first.shape[1],
            )
            cache_key = (tuple(float(x) for x in margin), nonvoter_index)
            if cache_key not in turnout_cache:
                total = second @ margin - first @ margin
                turnout = _turnout_margin_bounds_by_row(
                    first,
                    second,
                    margin,
                    nonvoter_index=nonvoter_index,
                )
                turnout_cache[cache_key] = {"total": total, **turnout}
            cached = turnout_cache[cache_key]
            if estimand == "turnout_margin_delta":
                product = cached["product"]
                lower = cached["lower"]
                upper = cached["upper"]
            else:
                product = cached["total"] - cached["product"]
                lower = cached["total"] - cached["upper"]
                upper = cached["total"] - cached["lower"]
            _append_functional_rows(
                rows,
                component=component,
                kind=kind,
                margin_direction=margin_direction,
                product=product,
                lower=lower,
                upper=upper,
                stratum_codes=stratum_codes,
                stratum_labels=stratum_labels,
                n_units=n_units,
                weight_sum=weight_sum,
                weights=weights,
            )
        elif estimand == "event":
            first_index = _resolve_outcome_indices(
                functional["first_event"],
                first_labels,
                "first_event",
            )
            second_index = _resolve_outcome_indices(
                functional["second_event"],
                second_labels,
                "second_event",
            )
            first_mass = first[:, first_index].sum(axis=1)
            second_mass = second[:, second_index].sum(axis=1)
            _append_functional_rows(
                rows,
                component=component,
                kind=kind,
                margin_direction=margin_direction,
                product=first_mass * second_mass,
                lower=np.maximum(0.0, first_mass + second_mass - 1.0),
                upper=np.minimum(first_mass, second_mass),
                stratum_codes=stratum_codes,
                stratum_labels=stratum_labels,
                n_units=n_units,
                weight_sum=weight_sum,
                weights=weights,
            )
        elif estimand == "lp":
            coefficient = _functional_cost_matrix(functional, first.shape[1]).reshape(-1)
            for stratum_index, stratum_label in enumerate(stratum_labels):
                mask = stratum_codes == stratum_index
                bounds = joint_lp_bounds(
                    [first[mask], second[mask]],
                    coefficient,
                    weights=weights[mask],
                    method=method,
                )
                rows.append(
                    {
                        "stratum": stratum_label,
                        "component": component,
                        "kind": kind,
                        "margin_direction": margin_direction,
                        "n_units": int(n_units[stratum_index]),
                        "weight_sum": float(weight_sum[stratum_index]),
                        "product": float(bounds["product"]),
                        "lower": float(bounds["lower"]),
                        "upper": float(bounds["upper"]),
                        "width": float(bounds["upper"] - bounds["lower"]),
                    }
                )
        else:
            raise ValueError(f"unknown transition estimand: {estimand}")
    return pd.DataFrame(rows)


def turnout_persuasion_margin_bound_table(
    first_probs,
    second_probs,
    margin_values,
    *,
    nonvoter_index: int,
    strata=None,
    weights=None,
) -> pd.DataFrame:
    """Sharp turnout/persuasion margin decomposition for two status margins.

    ``margin_values`` gives the signed vote margin for each outcome.  The
    turnout component is the part of the margin change carried by transitions
    touching ``nonvoter_index``; persuasion is the remaining within-voter
    transition component.  Bounds are sharp for the heterogeneous voter-level
    two-margin region.
    """

    first = _probability_matrix(first_probs, name="first_probs")
    second = _probability_matrix(second_probs, name="second_probs")
    n_rows = _common_n_rows([first, second])
    if first.shape[1] != second.shape[1]:
        raise ValueError("first_probs and second_probs must have the same outcomes")
    margin = np.asarray(margin_values, dtype=np.float64)
    if margin.shape != (first.shape[1],):
        raise ValueError("margin_values must have one entry per outcome")
    nonvoter_index = int(nonvoter_index)
    if not 0 <= nonvoter_index < first.shape[1]:
        raise ValueError("nonvoter_index is out of bounds")
    if abs(float(margin[nonvoter_index])) > 1e-12:
        raise ValueError("the nonvoter margin value must be zero")

    weights = _weights(weights, n_rows)
    stratum_codes, stratum_labels = _factorize_strata(strata, n_rows)
    n_strata = len(stratum_labels)
    n_units = np.bincount(stratum_codes, minlength=n_strata)
    weight_sum = np.bincount(stratum_codes, weights=weights, minlength=n_strata)

    first_margin = first @ margin
    second_margin = second @ margin
    total_delta = second_margin - first_margin
    turnout = _turnout_margin_bounds_by_row(
        first,
        second,
        margin,
        nonvoter_index=nonvoter_index,
    )
    product_turnout = turnout["product"]
    lower_turnout = turnout["lower"]
    upper_turnout = turnout["upper"]

    total_delta_by_stratum = _grouped_weighted_mean(
        total_delta,
        stratum_codes,
        weights,
        weight_sum,
        n_strata,
    )
    product_turnout_by_stratum = _grouped_weighted_mean(
        product_turnout,
        stratum_codes,
        weights,
        weight_sum,
        n_strata,
    )
    lower_turnout_by_stratum = _grouped_weighted_mean(
        lower_turnout,
        stratum_codes,
        weights,
        weight_sum,
        n_strata,
    )
    upper_turnout_by_stratum = _grouped_weighted_mean(
        upper_turnout,
        stratum_codes,
        weights,
        weight_sum,
        n_strata,
    )

    rows = []
    for stratum_index, stratum_label in enumerate(stratum_labels):
        total = total_delta_by_stratum[stratum_index]
        turnout_product = product_turnout_by_stratum[stratum_index]
        turnout_lower = lower_turnout_by_stratum[stratum_index]
        turnout_upper = upper_turnout_by_stratum[stratum_index]
        rows.append(
            {
                "stratum": stratum_label,
                "n_units": int(n_units[stratum_index]),
                "weight_sum": float(weight_sum[stratum_index]),
                "total_delta_margin": total,
                "turnout_product": turnout_product,
                "turnout_lower": turnout_lower,
                "turnout_upper": turnout_upper,
                "turnout_width": turnout_upper - turnout_lower,
                "persuasion_product": total - turnout_product,
                "persuasion_lower": total - turnout_upper,
                "persuasion_upper": total - turnout_lower,
                "persuasion_width": turnout_upper - turnout_lower,
            }
        )
    return pd.DataFrame(rows)


def _turnout_margin_bounds_by_row(
    first: np.ndarray,
    second: np.ndarray,
    margin: np.ndarray,
    *,
    nonvoter_index: int,
) -> dict[str, np.ndarray]:
    """Return per-row product/lower/upper values for the turnout component."""

    keep = np.asarray(
        [index for index in range(first.shape[1]) if index != nonvoter_index],
        dtype=np.int64,
    )
    first_non = first[:, keep]
    second_non = second[:, keep]
    first_nonvoter = first[:, nonvoter_index]
    second_nonvoter = second[:, nonvoter_index]
    margin_non = margin[keep]

    x_lower = np.maximum(0.0, first_nonvoter + second_nonvoter - 1.0)
    x_upper = np.minimum(first_nonvoter, second_nonvoter)
    product = (
        first_nonvoter * (second @ margin)
        - second_nonvoter * (first @ margin)
    )

    lower = _turnout_margin_extreme(
        first_non,
        second_non,
        first_nonvoter,
        second_nonvoter,
        margin_non,
        x_lower,
        x_upper,
        maximize=False,
    )
    upper = _turnout_margin_extreme(
        first_non,
        second_non,
        first_nonvoter,
        second_nonvoter,
        margin_non,
        x_lower,
        x_upper,
        maximize=True,
    )
    return {"product": product, "lower": lower, "upper": upper}


def _turnout_margin_extreme(
    first_non: np.ndarray,
    second_non: np.ndarray,
    first_nonvoter: np.ndarray,
    second_nonvoter: np.ndarray,
    margin_non: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    *,
    maximize: bool,
) -> np.ndarray:
    """Optimize turnout margin over feasible nonvoter-nonvoter mass."""

    col_order = np.argsort(-margin_non if maximize else margin_non)
    row_values = -margin_non
    row_order = np.argsort(-row_values if maximize else row_values)
    col_thresholds = np.cumsum(second_non[:, col_order], axis=1)
    row_thresholds = np.cumsum(first_non[:, row_order], axis=1)
    candidates = [x_lower, x_upper]
    for index in range(col_thresholds.shape[1]):
        candidates.append(first_nonvoter - col_thresholds[:, index])
    for index in range(row_thresholds.shape[1]):
        candidates.append(second_nonvoter - row_thresholds[:, index])
    x = np.column_stack(candidates)
    x = np.minimum(np.maximum(x, x_lower[:, None]), x_upper[:, None])
    amount_to_columns = first_nonvoter[:, None] - x
    amount_to_rows = second_nonvoter[:, None] - x
    values = _linear_allocation_value(
        amount_to_columns,
        second_non,
        margin_non,
        maximize=maximize,
    ) + _linear_allocation_value(
        amount_to_rows,
        first_non,
        row_values,
        maximize=maximize,
    )
    return values.max(axis=1) if maximize else values.min(axis=1)


def _linear_allocation_value(
    amount: np.ndarray,
    capacities: np.ndarray,
    values: np.ndarray,
    *,
    maximize: bool,
) -> np.ndarray:
    """Greedy value for bounded linear allocation with one equality total."""

    order = np.argsort(-values if maximize else values)
    remaining = np.asarray(amount, dtype=np.float64).copy()
    out = np.zeros_like(remaining)
    for category in order:
        take = np.minimum(np.maximum(remaining, 0.0), capacities[:, category, None])
        out += take * values[category]
        remaining -= take
    return out


def _append_functional_rows(
    rows: list[dict[str, object]],
    *,
    component: str,
    kind: str,
    margin_direction: str,
    product: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    stratum_codes: np.ndarray,
    stratum_labels: Sequence[object],
    n_units: np.ndarray,
    weight_sum: np.ndarray,
    weights: np.ndarray,
) -> None:
    n_strata = len(stratum_labels)
    product_by_stratum = _grouped_weighted_mean(
        product,
        stratum_codes,
        weights,
        weight_sum,
        n_strata,
    )
    lower_by_stratum = _grouped_weighted_mean(
        lower,
        stratum_codes,
        weights,
        weight_sum,
        n_strata,
    )
    upper_by_stratum = _grouped_weighted_mean(
        upper,
        stratum_codes,
        weights,
        weight_sum,
        n_strata,
    )
    for stratum_index, stratum_label in enumerate(stratum_labels):
        product_value = product_by_stratum[stratum_index]
        lower_value = lower_by_stratum[stratum_index]
        upper_value = upper_by_stratum[stratum_index]
        rows.append(
            {
                "stratum": stratum_label,
                "component": component,
                "kind": kind,
                "margin_direction": margin_direction,
                "n_units": int(n_units[stratum_index]),
                "weight_sum": float(weight_sum[stratum_index]),
                "product": product_value,
                "lower": lower_value,
                "upper": upper_value,
                "width": upper_value - lower_value,
            }
        )


def _two_margin_outcome_names(
    outcome_names: Sequence[str] | Mapping[str, Sequence[str]] | None,
    n_outcomes: int,
) -> tuple[list[str], list[str]]:
    if isinstance(outcome_names, Mapping):
        first = _outcome_names(outcome_names, "first", n_outcomes)
        second = _outcome_names(outcome_names, "second", n_outcomes)
    else:
        first = _outcome_names(outcome_names, "first", n_outcomes)
        second = first
    return first, second


def _functional_margin_values(
    functional: Mapping[str, object],
    n_outcomes: int,
) -> np.ndarray:
    if "margin_values" not in functional:
        raise ValueError("margin functional requires margin_values")
    margin = np.asarray(functional["margin_values"], dtype=np.float64)
    if margin.shape != (n_outcomes,):
        raise ValueError("margin_values must have one entry per outcome")
    return margin


def _functional_nonvoter_index(
    functional: Mapping[str, object],
    labels: Sequence[str],
    n_outcomes: int,
) -> int:
    if "nonvoter_index" in functional:
        index = int(functional["nonvoter_index"])
    elif "nonvoter" in functional:
        index = _resolve_outcome_indices(functional["nonvoter"], labels, "nonvoter")[0]
    else:
        raise ValueError("turnout functional requires nonvoter_index or nonvoter")
    if not 0 <= index < n_outcomes:
        raise ValueError("nonvoter_index is out of bounds")
    return index


def _functional_cost_matrix(
    functional: Mapping[str, object],
    n_outcomes: int,
) -> np.ndarray:
    if "cost_matrix" not in functional:
        raise ValueError("lp functional requires cost_matrix")
    cost = np.asarray(functional["cost_matrix"], dtype=np.float64)
    if cost.shape != (n_outcomes, n_outcomes):
        raise ValueError("cost_matrix must be square with one row per outcome")
    return cost


def _resolve_outcome_indices(
    values,
    labels: Sequence[str],
    name: str,
) -> list[int]:
    if isinstance(values, (str, int, np.integer)):
        values = [values]
    index_by_label = {str(label): index for index, label in enumerate(labels)}
    indices = []
    for value in values:
        if isinstance(value, (int, np.integer)):
            index = int(value)
            if not 0 <= index < len(labels):
                raise ValueError(f"{name} outcome index is out of bounds")
        else:
            key = str(value)
            if key not in index_by_label:
                raise ValueError(f"{name} contains unknown outcome label: {key}")
            index = index_by_label[key]
        indices.append(index)
    if not indices:
        raise ValueError(f"{name} must contain at least one outcome")
    return indices


def _outcome_sizes(outcome_sizes: Sequence[int]) -> list[int]:
    sizes = [int(size) for size in outcome_sizes]
    if not sizes or any(size < 2 for size in sizes):
        raise ValueError("outcome sizes must contain at least one size >= 2")
    return sizes


def _resolve_label(value: int | str, labels, name: str) -> int:
    labels = np.asarray(labels)
    if isinstance(value, (int, np.integer)):
        index = int(value)
        if not 0 <= index < len(labels):
            raise ValueError(f"{name} index is out of bounds")
        return index
    matches = np.flatnonzero(labels.astype(str) == str(value))
    if len(matches) != 1:
        raise ValueError(f"{name} label must match exactly one row")
    return int(matches[0])


def _probability_matrices(values: Sequence[object]) -> list[np.ndarray]:
    if isinstance(values, Mapping):
        values = list(values.values())
    matrices = [_probability_matrix(value, name=f"marginals[{index}]") for index, value in enumerate(values)]
    if not matrices:
        raise ValueError("at least one marginal matrix is required")
    _common_n_rows(matrices)
    return matrices


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


def _common_n_rows(matrices: Sequence[np.ndarray]) -> int:
    sizes = {matrix.shape[0] for matrix in matrices}
    if len(sizes) != 1:
        raise ValueError("all marginal matrices must have the same number of rows")
    return sizes.pop()


def _weights(weights, n_rows: int) -> np.ndarray:
    if weights is None:
        return np.ones(n_rows, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n_rows,):
        raise ValueError("weights must have one entry per row")
    if np.any(weights < 0.0):
        raise ValueError("weights must be nonnegative")
    return weights


def _factorize_strata(strata, n_rows: int) -> tuple[np.ndarray, list[object]]:
    if strata is None:
        return np.zeros(n_rows, dtype=np.int64), ["all"]
    strata = np.asarray(strata)
    if strata.shape != (n_rows,):
        raise ValueError("strata must have one entry per row")
    codes, labels = pd.factorize(strata, sort=True)
    if np.any(codes < 0):
        raise ValueError("strata cannot contain missing values")
    return codes.astype(np.int64), labels.tolist()


def _grouped_weighted_mean(
    values: np.ndarray,
    codes: np.ndarray,
    weights: np.ndarray,
    denominator: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    numerator = np.bincount(codes, weights=weights * values, minlength=n_groups)
    return np.divide(
        numerator,
        denominator,
        out=np.full(n_groups, np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )


def _outcome_names(
    outcome_names: Sequence[str] | Mapping[str, Sequence[str]] | None,
    item_name: str,
    n_outcomes: int,
) -> list[str]:
    if outcome_names is None:
        names = [str(index) for index in range(n_outcomes)]
    elif isinstance(outcome_names, Mapping):
        names = list(outcome_names[item_name])
    else:
        names = list(outcome_names)
    if len(names) != n_outcomes:
        raise ValueError(f"outcome_names for {item_name!r} must have length {n_outcomes}")
    return names

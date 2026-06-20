import numpy as np
import pandas as pd

from elrpy.joint import (
    joint_indicator,
    joint_lp_bounds,
    joint_simplex_coordinate_bounds,
    joint_simplex_lp_bounds,
    joint_simplex_region,
    load_sharp_joint_region,
    save_sharp_joint_region,
    sharp_joint_region_lp_bounds,
)
from elrpy.marginals import (
    add_bound_diagnostics,
    add_rate_uncertainty,
    categorical_probabilities,
    categorical_probability_draws,
    pairwise_flow_draw_table,
    pairwise_flow_table,
    pairwise_frechet_bounds,
    parameter_draws,
    shifted_probability_weighted_gradients,
    shifted_probability_draws,
    shifted_probabilities,
    shifted_softmax_beta_sensitivity,
    summarize_flow_draws,
)


def test_categorical_and_shifted_probabilities():
    X = np.array([[1.0, -1.0], [1.0, 1.0]])
    params = np.array([[0.2, -0.1], [0.4, 0.3]])

    probs = categorical_probabilities(X, params)
    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    shifted = shifted_probabilities(
        probs,
        np.array([0, 1]),
        np.array([[0.1, 0.0], [-0.2, 0.3]]),
    )
    assert shifted.shape == probs.shape
    np.testing.assert_allclose(shifted.sum(axis=1), 1.0)
    assert not np.allclose(shifted, probs)

    prob_draws = categorical_probability_draws(X, np.vstack([params.ravel(), params.ravel() + 0.1]))
    assert prob_draws.shape == (2, 2, 3)
    shifted_draws = shifted_probability_draws(prob_draws, np.array([0, 1]), np.array([[0.1, 0.0], [-0.2, 0.3]]))
    assert shifted_draws.shape == prob_draws.shape
    np.testing.assert_allclose(shifted_draws.sum(axis=2), 1.0)


def test_pairwise_frechet_bounds_matches_manual_formula():
    first = np.array([0.2, 0.7, 0.9])
    second = np.array([0.5, 0.6, 0.3])

    stats = pairwise_frechet_bounds(first, second)

    np.testing.assert_allclose(stats["pred"], np.mean(first * second))
    np.testing.assert_allclose(stats["lower"], np.mean(np.maximum(0.0, first + second - 1.0)))
    np.testing.assert_allclose(stats["upper"], np.mean(np.minimum(first, second)))
    np.testing.assert_allclose(stats["width"], stats["upper"] - stats["lower"])


def test_pairwise_flow_table_strata_and_weights():
    probabilities = {
        "gov": np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]]),
        "sen": np.array([[0.7, 0.3], [0.5, 0.5], [0.2, 0.8]]),
    }
    table = pairwise_flow_table(
        probabilities,
        strata=np.array(["a", "a", "b"]),
        outcome_names=["d", "r"],
        weights=np.array([1.0, 3.0, 2.0]),
    )

    assert set(table["stratum"]) == {"a", "b"}
    assert len(table) == 8
    row = table[
        (table["stratum"] == "a")
        & (table["outcome_1"] == "d")
        & (table["outcome_2"] == "d")
    ].iloc[0]
    np.testing.assert_allclose(row["pred"], np.average([0.8 * 0.7, 0.4 * 0.5], weights=[1.0, 3.0]))
    np.testing.assert_allclose(row["lower"], np.average([0.5, 0.0], weights=[1.0, 3.0]))
    np.testing.assert_allclose(row["upper"], np.average([0.7, 0.4], weights=[1.0, 3.0]))


def test_bound_diagnostics_and_rate_uncertainty():
    table = pd.DataFrame(
        {
            "true": [0.4, 0.8],
            "lower": [0.3, 0.2],
            "upper": [0.6, 0.75],
            "width": [0.3, 0.5],
            "pred": [0.45, 0.5],
            "n_true": [100, 100],
        }
    )

    table = add_rate_uncertainty(table, rate_col="true", n_col="n_true", prefix="true")
    table = add_bound_diagnostics(
        table,
        truth_lower_col="true_lower",
        truth_upper_col="true_upper",
    )

    assert bool(table.loc[0, "covered"])
    assert not bool(table.loc[1, "covered"])
    assert table.loc[1, "bound_violation"] > 0.0
    assert table.loc[0, "true_lower"] < table.loc[0, "true"] < table.loc[0, "true_upper"]
    assert bool(table.loc[1, "truth_interval_overlaps_bound"])


def test_parameter_draws_and_flow_draw_summary():
    params = np.array([1.0, -1.0])
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])

    draws = parameter_draws(params, covariance, n_draws=5, seed=0, include_center=True)
    assert draws.shape == (6, 2)
    np.testing.assert_allclose(draws[0], params)

    flow_draws = pd.DataFrame(
        {
            "draw": [0, 1, 2],
            "stratum": ["all", "all", "all"],
            "item_1": ["a", "a", "a"],
            "item_2": ["b", "b", "b"],
            "outcome_1": ["d", "d", "d"],
            "outcome_2": ["r", "r", "r"],
            "pred": [0.2, 0.3, 0.4],
            "lower": [0.1, 0.15, 0.2],
            "upper": [0.5, 0.55, 0.6],
            "width": [0.4, 0.4, 0.4],
        }
    )
    summary = summarize_flow_draws(flow_draws)
    assert len(summary) == 1
    assert summary.loc[0, "identified_lower"] < summary.loc[0, "identified_upper"]
    assert summary.loc[0, "n_draws"] == 3


def test_pairwise_flow_draw_table_accepts_plugin_and_draw_stacks():
    office_a = np.array(
        [
            [[0.8, 0.2], [0.4, 0.6]],
            [[0.7, 0.3], [0.5, 0.5]],
        ]
    )
    office_b = np.array([[0.6, 0.4], [0.3, 0.7]])

    table = pairwise_flow_draw_table(
        {"a": office_a, "b": office_b},
        outcome_names=["d", "r"],
    )

    assert set(table["draw"]) == {0, 1}
    assert len(table) == 8
    summary = summarize_flow_draws(table)
    assert len(summary) == 4


def test_heterogeneous_joint_lp_projection_recovers_pairwise_frechet_bounds():
    first = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    second = np.array([[0.4, 0.6], [0.5, 0.5], [0.9, 0.1]])
    third = np.array([[0.2, 0.8], [0.9, 0.1], [0.4, 0.6]])
    weights = np.array([1.0, 2.0, 3.0])
    coefficient = joint_indicator([2, 2, 2], {0: 1, 1: 0})

    lp = joint_lp_bounds([first, second, third], coefficient, weights=weights)
    frechet = pairwise_frechet_bounds(first[:, 1], second[:, 0], weights=weights)

    np.testing.assert_allclose(lp["product"], frechet["pred"])
    np.testing.assert_allclose(lp["lower"], frechet["lower"])
    np.testing.assert_allclose(lp["upper"], frechet["upper"])


def test_aggregate_simplex_region_and_projection_recovers_aggregate_frechet():
    first = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    second = np.array([[0.4, 0.6], [0.5, 0.5], [0.9, 0.1]])
    third = np.array([[0.2, 0.8], [0.9, 0.1], [0.4, 0.6]])
    weights = np.array([1.0, 2.0, 3.0])
    coefficient = joint_indicator([2, 2, 2], {0: 1, 1: 0})

    region = joint_simplex_region(
        [first, second, third],
        weights=weights,
        item_names=["gov", "atg", "uss"],
        outcome_names=["d", "r"],
    )
    assert region["A_eq"].shape == (6, 8)
    assert len(region["cell_table"]) == 8

    lp = joint_simplex_lp_bounds([first, second, third], coefficient, weights=weights)
    first_mean = np.average(first[:, 1], weights=weights)
    second_mean = np.average(second[:, 0], weights=weights)

    np.testing.assert_allclose(lp["product"], first_mean * second_mean)
    np.testing.assert_allclose(lp["lower"], max(0.0, first_mean + second_mean - 1.0))
    np.testing.assert_allclose(lp["upper"], min(first_mean, second_mean))

    cells = joint_simplex_coordinate_bounds(
        [first, second, third],
        weights=weights,
        item_names=["gov", "atg", "uss"],
        outcome_names=["d", "r"],
    )
    assert len(cells) == 8
    assert {"gov", "atg", "uss", "lower", "upper"}.issubset(cells.columns)


def test_sharp_joint_region_npz_round_trips_and_solves_lp(tmp_path):
    probabilities = np.array(
        [
            [[0.8, 0.2], [0.4, 0.6]],
            [[0.3, 0.7], [0.5, 0.5]],
            [[0.6, 0.4], [0.9, 0.1]],
        ]
    )
    subgroup_weights = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, 2.0]])
    path = tmp_path / "region_store.npz"
    save_sharp_joint_region(
        path,
        probabilities=probabilities,
        group_ids=np.array([0, 0, 1]),
        group_labels=np.array(["p0", "p1"]),
        subgroup_weights=subgroup_weights,
        item_names=("gov", "sen"),
        outcome_names=("d", "r"),
        subgroup_names=("black", "white"),
        metadata={"state": "xx"},
    )
    loaded = load_sharp_joint_region(path)
    coefficient = joint_indicator([2, 2], {0: 1, 1: 0})

    bounds = sharp_joint_region_lp_bounds(loaded, "p0", "black", coefficient)
    frechet = pairwise_frechet_bounds(
        probabilities[:2, 0, 1],
        probabilities[:2, 1, 0],
        weights=subgroup_weights[:2, 0],
    )

    assert loaded["metadata"]["state"] == "xx"
    np.testing.assert_allclose(bounds["product"], frechet["pred"], atol=1e-7)
    np.testing.assert_allclose(bounds["lower"], frechet["lower"], atol=1e-7)
    np.testing.assert_allclose(bounds["upper"], frechet["upper"], atol=1e-7)


def test_shifted_probability_weighted_gradients_hold_group_means_fixed():
    X = np.array(
        [
            [1.0, -1.0],
            [1.0, 0.5],
            [1.0, -0.2],
            [1.0, 1.3],
        ]
    )
    params = np.array([[0.2, -0.1], [0.5, 0.3]])
    group_ids = np.array([0, 0, 1, 1])
    shifts = np.array([[0.4, -0.2], [-0.3, 0.1]])
    logits = X @ params + shifts[group_ids]
    probs = np.column_stack([np.exp(logits), np.ones(len(X))])
    probs = probs / probs.sum(axis=1, keepdims=True)

    sensitivity = shifted_softmax_beta_sensitivity(X, probs, group_ids)
    weights = np.column_stack([group_ids == 0, group_ids == 1]).astype(float)
    gradients = shifted_probability_weighted_gradients(
        X,
        probs,
        group_ids,
        weights,
        normalizers=weights.sum(axis=0),
        sensitivity=sensitivity,
    )

    np.testing.assert_allclose(gradients, 0.0, atol=1e-10)

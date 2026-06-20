import numpy as np
import scipy.sparse as sp

from elrpy.inference import (
    delta_covariance,
    delta_variance,
    shifted_categorical_subgroup_inference,
    shifted_subgroup_inference,
    simultaneous_wald_intervals,
    wald_interval,
)
from elrpy.synthetic import (
    draw_survey_response,
    make_binary_logit_shift_case,
    make_precinct_logit_shift_case,
    shifted_binary_probs,
    shifted_binary_probs_by_group,
    shifted_inference_coverage,
    shifted_inference_sweep,
    survey_estimator_sweep,
)


def test_wald_interval():
    lo, hi = wald_interval(2.0, 0.5)
    assert lo < 2.0 < hi
    np.testing.assert_allclose(hi - 2.0, 2.0 - lo)


def test_delta_helpers_accept_independent_parameter_blocks():
    first_gradients = np.array([[1.0, 2.0], [0.0, 1.0]])
    second_gradients = np.array([[3.0], [-1.0]])
    first_covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    second_covariance = np.array([[0.25]])

    expected = (
        first_gradients @ first_covariance @ first_gradients.T
        + second_gradients @ second_covariance @ second_gradients.T
    )
    result = delta_covariance(
        [first_gradients, second_gradients],
        [first_covariance, second_covariance],
    )

    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(
        delta_variance(
            [first_gradients, second_gradients],
            [first_covariance, second_covariance],
        ),
        np.diag(expected),
    )


def test_simultaneous_wald_intervals_drop_zero_variance_coordinates():
    estimates = np.array([1.0, 2.0, 3.0])
    covariance = np.diag([1.0, 4.0, 0.0])

    intervals = simultaneous_wald_intervals(
        estimates,
        covariance,
        n_draws=20_000,
        seed=123,
        lower_bound=np.zeros(3),
        upper_bound=np.array([10.0, 10.0, 3.0]),
    )

    assert intervals["critical_value"] > 1.9
    np.testing.assert_allclose(intervals["se"], [1.0, 2.0, 0.0])
    np.testing.assert_allclose(intervals["lower"][2], 3.0)
    np.testing.assert_allclose(intervals["upper"][2], 3.0)
    np.testing.assert_allclose(
        intervals["correlation"],
        np.diag([1.0, 1.0, 0.0]),
    )


def test_shifted_subgroup_inference_matches_manual_delta_terms():
    X = np.array(
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ]
    )
    group_ids = np.zeros(4, dtype=int)
    subgroup = np.array([False, False, True, True])
    probs = np.array([0.2, 0.4, 0.6, 0.8])
    beta_cov = np.diag([0.01, 0.04])

    stats = shifted_subgroup_inference(
        X,
        group_ids,
        probs,
        subgroup,
        0,
        beta_cov=beta_cov,
    )

    dot = probs * (1.0 - probs)
    W = dot.sum()
    W_subgroup = dot[subgroup].sum()
    G = dot @ X
    G_subgroup = dot[subgroup] @ X[subgroup]
    d_beta = (G_subgroup - (W_subgroup / W) * G) / subgroup.sum()
    d_count = W_subgroup / (subgroup.sum() * W)

    np.testing.assert_allclose(stats["estimate"], probs[subgroup].mean())
    np.testing.assert_allclose(stats["d_beta"], d_beta)
    np.testing.assert_allclose(stats["d_count"], d_count)
    np.testing.assert_allclose(
        stats["conditional_mean_se"] ** 2,
        d_beta @ beta_cov @ d_beta + d_count**2 * W,
    )
    np.testing.assert_allclose(
        stats["realized_share_se"] ** 2,
        d_beta @ beta_cov @ d_beta
        + W_subgroup / subgroup.sum() ** 2 * (1.0 - W_subgroup / W),
    )


def test_shifted_subgroup_inference_accepts_sparse_groups_and_matrix_probs():
    X = np.array(
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ]
    )
    G = sp.csr_matrix(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    subgroup = np.array([True, False, True, True])
    probs = np.array([0.2, 0.4, 0.6, 0.8])
    matrix_probs = np.column_stack([probs, 1.0 - probs])

    vector_stats = shifted_subgroup_inference(
        X,
        np.array([0, 0, 1, 1]),
        probs,
        subgroup,
        1,
    )
    matrix_stats = shifted_subgroup_inference(
        X,
        G,
        matrix_probs,
        subgroup,
        1,
    )

    np.testing.assert_allclose(matrix_stats["estimate"], vector_stats["estimate"])
    np.testing.assert_allclose(matrix_stats["d_beta"], vector_stats["d_beta"])
    np.testing.assert_allclose(
        matrix_stats["conditional_mean_se"],
        vector_stats["conditional_mean_se"],
    )


def test_shifted_categorical_subgroup_inference_matches_binary_formula():
    X = np.array(
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ]
    )
    G = sp.csr_matrix([[1.0, 1.0, 1.0, 1.0]])
    subgroup = np.array([0.0, 0.0, 1.0, 1.0])
    probs = np.array([0.2, 0.4, 0.6, 0.8])
    matrix_probs = np.column_stack([probs, 1.0 - probs])
    beta_cov = np.diag([0.01, 0.04])

    binary_stats = shifted_subgroup_inference(
        X,
        G,
        probs,
        subgroup.astype(bool),
        0,
        beta_cov=beta_cov,
    )
    categorical_stats = shifted_categorical_subgroup_inference(
        matrix_probs,
        G,
        subgroup,
        X=X,
        beta_cov=beta_cov,
    )

    np.testing.assert_allclose(
        categorical_stats["estimate"][0, 0],
        binary_stats["estimate"],
    )
    np.testing.assert_allclose(
        categorical_stats["variance"][0, 0],
        binary_stats["realized_share_variance"],
    )
    np.testing.assert_allclose(
        categorical_stats["slope_variance"][0, 0],
        binary_stats["slope_variance"],
    )
    np.testing.assert_allclose(
        categorical_stats["allocation_variance"][0, 0],
        binary_stats["realized_noise_variance"],
    )
    np.testing.assert_allclose(
        categorical_stats["covariance"][0].sum(axis=1),
        np.zeros(2),
        atol=1e-12,
    )


def test_shifted_intervals_cover_conditional_and_realized_targets():
    case = make_precinct_logit_shift_case(
        seed=123,
        n_precincts=8,
        voters_per_precinct=500,
        shock_sd=0.5,
        center_sd=0.8,
        within_sd=0.9,
    )
    coverage = shifted_inference_coverage(case, seed=124, n_rep=900)

    assert 0.93 <= coverage["conditional_coverage"] <= 0.97
    assert 0.93 <= coverage["realized_coverage"] <= 0.97
    assert 0.90 <= coverage["conditional_se_ratio"] <= 1.10
    assert 0.90 <= coverage["realized_se_ratio"] <= 1.10


def test_shifted_intervals_cover_with_independent_slope_uncertainty():
    case = make_precinct_logit_shift_case(
        seed=456,
        n_precincts=6,
        voters_per_precinct=600,
        beta=(-0.1, 0.6),
        shock_sd=0.4,
        center_sd=0.7,
        within_sd=1.0,
        subgroup_quantile=0.70,
    )
    beta_cov = np.array([[0.0005, 0.0001], [0.0001, 0.0004]])
    coverage = shifted_inference_coverage(
        case,
        seed=457,
        n_rep=900,
        beta_cov=beta_cov,
    )

    assert 0.93 <= coverage["conditional_coverage"] <= 0.97
    assert 0.93 <= coverage["realized_coverage"] <= 0.97
    assert 0.90 <= coverage["conditional_se_ratio"] <= 1.10
    assert 0.90 <= coverage["realized_se_ratio"] <= 1.10


def test_synthetic_shift_matches_requested_total():
    case = make_binary_logit_shift_case(seed=789, n_voters=1000)
    rng = np.random.default_rng(790)
    votes = rng.binomial(1, case["probs"])
    shifted = shifted_binary_probs(case["linear_predictor"], votes.sum())

    np.testing.assert_allclose(shifted.sum(), votes.sum(), atol=1e-8)


def test_synthetic_group_shifts_match_requested_totals():
    case = make_precinct_logit_shift_case(
        seed=891,
        n_precincts=5,
        voters_per_precinct=300,
    )
    rng = np.random.default_rng(892)
    votes = rng.binomial(1, case["probs"])
    totals = np.bincount(case["group_ids"], weights=votes, minlength=case["n_precincts"])
    shifted = shifted_binary_probs_by_group(
        case["linear_predictor"],
        case["group_ids"],
        totals,
    )
    shifted_totals = np.bincount(
        case["group_ids"],
        weights=shifted,
        minlength=case["n_precincts"],
    )

    np.testing.assert_allclose(shifted_totals, totals, atol=1e-8)


def test_synthetic_inference_sweep_returns_setting_and_coverage():
    results = shifted_inference_sweep(
        [
            {
                "n_precincts": 3,
                "voters_per_precinct": 250,
                "shock_sd": 0.4,
                "within_sd": 1.0,
            }
        ],
        seed=901,
        n_rep=20,
    )

    assert len(results) == 1
    assert results[0]["n_precincts"] == 3
    assert results[0]["n_intervals"] == 60
    assert 0.0 <= results[0]["conditional_coverage"] <= 1.0
    assert 0.0 <= results[0]["realized_coverage"] <= 1.0


def test_survey_response_can_depend_on_vote_but_weights_do_not():
    case = make_precinct_logit_shift_case(seed=931, n_precincts=3, voters_per_precinct=250)
    rng = np.random.default_rng(932)
    votes = rng.binomial(1, case["probs"])

    survey = draw_survey_response(
        rng,
        case,
        votes,
        response_rate=0.20,
        response_x=0.4,
        response_vote=1.0,
    )

    np.testing.assert_allclose(survey["response_probs"].mean(), 0.20, atol=1e-10)
    np.testing.assert_allclose(
        survey["analysis_response_probs"].mean(),
        0.20,
        atol=1e-10,
    )
    assert survey["response_probs"][votes == 1].mean() > survey["response_probs"][
        votes == 0
    ].mean()


def test_survey_estimator_sweep_returns_expected_estimators():
    case = make_precinct_logit_shift_case(
        seed=941,
        n_precincts=4,
        voters_per_precinct=220,
    )
    rows = survey_estimator_sweep(
        case,
        seed=942,
        n_rep=8,
        response_rates=[0.25],
        response_x=0.3,
        response_vote=0.8,
        groups=[0, 1],
    )

    assert {row["estimator"] for row in rows} == {
        "elr",
        "ipw",
        "mrp",
    }
    assert all(row["response_rate"] == 0.25 for row in rows)
    assert all(0 <= row["n_intervals"] <= 16 for row in rows)

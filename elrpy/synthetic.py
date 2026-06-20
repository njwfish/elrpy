"""Small synthetic helpers for testing ELR estimators."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def sigmoid(x):
    """Numerically stable logistic link."""

    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def group_matrix(group_ids) -> sp.csr_matrix:
    """Return a sparse group-by-voter aggregation matrix."""

    group_ids = np.asarray(group_ids)
    if group_ids.ndim != 1:
        raise ValueError("group_ids must be one-dimensional")
    if group_ids.size == 0:
        raise ValueError("group_ids cannot be empty")
    if np.any(group_ids < 0):
        raise ValueError("group_ids must be nonnegative")
    rows = group_ids.astype(np.int64, copy=False)
    cols = np.arange(group_ids.size)
    data = np.ones(group_ids.size)
    return sp.csr_matrix((data, (rows, cols)), shape=(int(rows.max()) + 1, rows.size))


def _precinct_sizes(n_precincts: int, voters_per_precinct) -> np.ndarray:
    if n_precincts <= 0:
        raise ValueError("n_precincts must be positive")
    sizes = np.asarray(voters_per_precinct, dtype=np.int64)
    if sizes.ndim == 0:
        sizes = np.full(n_precincts, int(sizes))
    if sizes.shape != (n_precincts,):
        raise ValueError(
            "voters_per_precinct must be scalar or have one entry per precinct"
        )
    if np.any(sizes <= 0):
        raise ValueError("each precinct must contain at least one voter")
    return sizes


def _within_precinct_subgroup(
    x: np.ndarray,
    group_ids: np.ndarray,
    n_precincts: int,
    subgroup_quantile: float,
) -> np.ndarray:
    if not 0.0 < subgroup_quantile < 1.0:
        raise ValueError("subgroup_quantile must be between zero and one")

    subgroup_mask = np.zeros(x.shape[0], dtype=bool)
    for group in range(n_precincts):
        idx = np.flatnonzero(group_ids == group)
        cutoff = np.quantile(x[idx], subgroup_quantile)
        selected = idx[x[idx] > cutoff]
        if selected.size == 0:
            selected = idx[[np.argmax(x[idx])]]
        subgroup_mask[selected] = True
    return subgroup_mask


def make_precinct_logit_shift_case(
    *,
    seed: int = 0,
    n_precincts: int = 20,
    voters_per_precinct=400,
    beta=(-0.2, 0.7),
    shock_sd: float = 0.5,
    center_sd: float = 1.0,
    within_sd: float = 1.0,
    subgroup_quantile: float = 0.65,
    precinct_shocks=None,
) -> dict[str, object]:
    """Build a binary precinct DGP with explicit precinct logit shocks.

    The returned probabilities are generated from
    `sigmoid(X @ beta + precinct_shock[group])`. The subgroup is chosen within
    each precinct, so every precinct contributes a nonempty target subgroup.
    """

    sizes = _precinct_sizes(n_precincts, voters_per_precinct)
    if center_sd < 0.0:
        raise ValueError("center_sd must be nonnegative")
    if within_sd <= 0.0:
        raise ValueError("within_sd must be positive")
    if shock_sd < 0.0:
        raise ValueError("shock_sd must be nonnegative")

    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(n_precincts, dtype=np.int64), sizes)
    centers = rng.normal(scale=center_sd, size=n_precincts)
    x = centers[group_ids] + rng.normal(scale=within_sd, size=group_ids.size)
    X = np.column_stack([np.ones(group_ids.size), x])
    beta = np.asarray(beta, dtype=np.float64)
    if beta.shape != (X.shape[1],):
        raise ValueError(f"beta must have shape {(X.shape[1],)}")

    if precinct_shocks is None:
        precinct_shocks = rng.normal(scale=shock_sd, size=n_precincts)
    else:
        precinct_shocks = np.asarray(precinct_shocks, dtype=np.float64)
        if precinct_shocks.shape != (n_precincts,):
            raise ValueError(f"precinct_shocks must have shape {(n_precincts,)}")

    linear_predictor = X @ beta
    probs = sigmoid(linear_predictor + precinct_shocks[group_ids])
    subgroup_mask = _within_precinct_subgroup(
        x,
        group_ids,
        n_precincts,
        subgroup_quantile,
    )
    conditional_subgroup_means = np.array(
        [
            probs[(group_ids == group) & subgroup_mask].mean()
            for group in range(n_precincts)
        ]
    )

    return {
        "X": X,
        "x": x,
        "group_ids": group_ids,
        "G": group_matrix(group_ids),
        "groups": np.arange(n_precincts, dtype=np.int64),
        "subgroup_mask": subgroup_mask,
        "group": 0,
        "n_precincts": n_precincts,
        "precinct_sizes": sizes,
        "precinct_centers": centers,
        "precinct_shocks": precinct_shocks,
        "beta": beta,
        "linear_predictor": linear_predictor,
        "base_probs": sigmoid(linear_predictor),
        "probs": probs,
        "conditional_subgroup_means": conditional_subgroup_means,
        "conditional_subgroup_mean": float(conditional_subgroup_means[0]),
    }


def make_binary_logit_shift_case(
    *,
    seed: int = 0,
    n_voters: int = 2500,
    beta=(-0.2, 0.7),
    shift: float = 0.25,
    subgroup_quantile: float = 0.65,
) -> dict[str, object]:
    """Build a one-precinct binary logit-shift case."""

    case = make_precinct_logit_shift_case(
        seed=seed,
        n_precincts=1,
        voters_per_precinct=n_voters,
        beta=beta,
        subgroup_quantile=subgroup_quantile,
        precinct_shocks=np.array([shift], dtype=np.float64),
    )
    case["shift"] = float(shift)
    return case


def draw_binary_votes(rng, probs) -> np.ndarray:
    """Draw independent Bernoulli votes from a probability vector."""

    return np.asarray(rng.binomial(1, np.asarray(probs, dtype=np.float64)))


def _solve_intercept(offset, target_mean: float) -> float:
    if not 0.0 < target_mean < 1.0:
        raise ValueError("target_mean must be between zero and one")

    offset = np.asarray(offset, dtype=np.float64)
    lo = -50.0
    hi = 50.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if sigmoid(mid + offset).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _standardize(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    scale = float(np.std(x))
    if scale == 0.0:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / scale


def draw_survey_response(
    rng,
    case: dict[str, object],
    votes,
    *,
    response_rate: float,
    response_x: float = 0.0,
    response_vote: float = 0.0,
) -> dict[str, np.ndarray]:
    """Draw respondents under covariate and outcome-dependent nonresponse.

    The realized response mechanism may depend on the vote. The returned
    `analysis_response_probs` omit this vote term, matching the usual situation
    where nonresponse weights can adjust only for observed pre-election
    covariates.
    """

    votes = np.asarray(votes, dtype=np.float64)
    x = _standardize(case["x"])
    if votes.shape != x.shape:
        raise ValueError("votes must have one entry per voter")

    actual_offset = response_x * x + response_vote * (2.0 * votes - 1.0)
    actual_intercept = _solve_intercept(actual_offset, response_rate)
    response_probs = sigmoid(actual_intercept + actual_offset)
    responded = rng.binomial(1, response_probs).astype(bool)

    analysis_offset = response_x * x
    analysis_intercept = _solve_intercept(analysis_offset, response_rate)
    analysis_response_probs = sigmoid(analysis_intercept + analysis_offset)

    return {
        "responded": responded,
        "response_probs": response_probs,
        "analysis_response_probs": analysis_response_probs,
    }


def solve_binary_logit_shift(
    linear_predictor,
    total,
    *,
    maxiter: int = 100,
    tol: float = 1e-12,
) -> float:
    """Solve the scalar logit shift matching one binary aggregate total."""

    linear_predictor = np.asarray(linear_predictor, dtype=np.float64)
    if not 0 < total < linear_predictor.size:
        raise ValueError("total must be strictly inside the precinct")

    lo = -50.0 - float(np.max(linear_predictor))
    hi = 50.0 - float(np.min(linear_predictor))
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fitted_total = float(sigmoid(linear_predictor + mid).sum())
        if fitted_total < float(total):
            lo = mid
        else:
            hi = mid
        if abs(fitted_total - float(total)) <= tol:
            break
    return float(0.5 * (lo + hi))


def shifted_binary_probs(linear_predictor, total) -> np.ndarray:
    """Return binary probabilities after matching the aggregate total."""

    alpha = solve_binary_logit_shift(linear_predictor, total)
    return sigmoid(np.asarray(linear_predictor, dtype=np.float64) + alpha)


def shifted_binary_probs_by_group(linear_predictor, group_ids, totals) -> np.ndarray:
    """Return probabilities after matching each precinct aggregate total."""

    linear_predictor = np.asarray(linear_predictor, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    totals = np.asarray(totals)
    if linear_predictor.shape != group_ids.shape:
        raise ValueError("linear_predictor and group_ids must have the same shape")
    n_groups = int(group_ids.max()) + 1
    if totals.shape != (n_groups,):
        raise ValueError(f"totals must have shape {(n_groups,)}")

    shifted = np.empty_like(linear_predictor, dtype=np.float64)
    for group in range(n_groups):
        mask = group_ids == group
        shifted[mask] = shifted_binary_probs(linear_predictor[mask], totals[group])
    return shifted


def shifted_inference_coverage(
    case: dict[str, object],
    *,
    seed: int,
    n_rep: int,
    beta_cov=None,
    groups=None,
) -> dict[str, float]:
    """Estimate Wald coverage for the shifted subgroup inference formula."""

    from elrpy.inference import shifted_subgroup_inference

    rng = np.random.default_rng(seed)
    beta_cov = None if beta_cov is None else np.asarray(beta_cov, dtype=np.float64)
    conditional_hits = 0
    realized_hits = 0
    n_intervals = 0
    skipped_reps = 0
    conditional_widths = []
    realized_widths = []
    conditional_errors = []
    realized_errors = []
    conditional_ses = []
    realized_ses = []

    X = np.asarray(case["X"], dtype=np.float64)
    beta = np.asarray(case["beta"], dtype=np.float64)
    probs = np.asarray(case["probs"], dtype=np.float64)
    subgroup_mask = np.asarray(case["subgroup_mask"], dtype=bool)
    group_ids = np.asarray(case["group_ids"])
    n_groups = int(group_ids.max()) + 1
    if groups is None:
        groups = np.asarray(case.get("groups", np.arange(n_groups)), dtype=np.int64)
    else:
        groups = np.asarray(groups, dtype=np.int64)
    targets = np.asarray(
        case.get(
            "conditional_subgroup_means",
            np.array([case["conditional_subgroup_mean"]]),
        ),
        dtype=np.float64,
    )
    if targets.shape != (n_groups,):
        raise ValueError(f"conditional subgroup means must have shape {(n_groups,)}")

    for _ in range(n_rep):
        votes = draw_binary_votes(rng, probs)
        totals = np.bincount(group_ids, weights=votes, minlength=n_groups)
        if beta_cov is None:
            linear_predictor = np.asarray(case["linear_predictor"], dtype=np.float64)
        else:
            beta_hat = rng.multivariate_normal(beta, beta_cov)
            linear_predictor = X @ beta_hat
        try:
            shifted = shifted_binary_probs_by_group(linear_predictor, group_ids, totals)
        except ValueError:
            skipped_reps += 1
            continue

        for group in groups:
            target_mask = (group_ids == group) & subgroup_mask
            stats = shifted_subgroup_inference(
                X,
                group_ids,
                shifted,
                subgroup_mask,
                group,
                beta_cov=beta_cov,
            )

            estimate = float(stats["estimate"])
            target = float(targets[group])
            lo, hi = stats["conditional_mean_interval"]
            conditional_hits += lo <= target <= hi
            conditional_widths.append(hi - lo)
            conditional_errors.append(estimate - target)
            conditional_ses.append(stats["conditional_mean_se"])

            realized_share = float(votes[target_mask].mean())
            lo, hi = stats["realized_share_interval"]
            realized_hits += lo <= realized_share <= hi
            realized_widths.append(hi - lo)
            realized_errors.append(estimate - realized_share)
            realized_ses.append(stats["realized_share_se"])
            n_intervals += 1

    if n_intervals == 0:
        raise ValueError("no valid intervals were evaluated")

    conditional_coverage = conditional_hits / n_intervals
    realized_coverage = realized_hits / n_intervals
    conditional_error_sd = float(np.std(conditional_errors, ddof=1))
    realized_error_sd = float(np.std(realized_errors, ddof=1))
    mean_conditional_se = float(np.mean(conditional_ses))
    mean_realized_se = float(np.mean(realized_ses))

    return {
        "conditional_coverage": conditional_coverage,
        "realized_coverage": realized_coverage,
        "conditional_mc_se": float(
            np.sqrt(conditional_coverage * (1.0 - conditional_coverage) / n_intervals)
        ),
        "realized_mc_se": float(
            np.sqrt(realized_coverage * (1.0 - realized_coverage) / n_intervals)
        ),
        "n_intervals": n_intervals,
        "skipped_reps": skipped_reps,
        "mean_conditional_width": float(np.mean(conditional_widths)),
        "mean_realized_width": float(np.mean(realized_widths)),
        "mean_conditional_se": mean_conditional_se,
        "mean_realized_se": mean_realized_se,
        "conditional_error_sd": conditional_error_sd,
        "realized_error_sd": realized_error_sd,
        "conditional_se_ratio": mean_conditional_se / conditional_error_sd,
        "realized_se_ratio": mean_realized_se / realized_error_sd,
    }


def shifted_inference_sweep(
    settings,
    *,
    seed: int,
    n_rep: int,
    beta_cov=None,
    groups=None,
) -> list[dict[str, object]]:
    """Run shifted inference coverage over several precinct DGP settings."""

    results = []
    for index, setting in enumerate(settings):
        setting = dict(setting)
        case = make_precinct_logit_shift_case(seed=seed + 2 * index, **setting)
        coverage = shifted_inference_coverage(
            case,
            seed=seed + 2 * index + 1,
            n_rep=n_rep,
            beta_cov=beta_cov,
            groups=groups,
        )
        results.append({**setting, **coverage})
    return results


def _safe_wald_hit(estimate: float, se: float, target: float, z: float) -> bool:
    return bool(estimate - z * se <= target <= estimate + z * se)


def _summarize_estimator(records, *, level: float) -> dict[str, float]:
    from statistics import NormalDist

    if not records:
        return {
            "coverage": np.nan,
            "bias": np.nan,
            "rmse": np.nan,
            "mean_se": np.nan,
            "error_sd": np.nan,
            "se_ratio": np.nan,
            "mean_width": np.nan,
            "n_intervals": 0,
        }

    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    estimates = np.array([record["estimate"] for record in records], dtype=np.float64)
    targets = np.array([record["target"] for record in records], dtype=np.float64)
    ses = np.array([record["se"] for record in records], dtype=np.float64)
    errors = estimates - targets
    hits = np.array(
        [
            _safe_wald_hit(estimate, se, target, z)
            for estimate, se, target in zip(estimates, ses, targets)
        ],
        dtype=bool,
    )
    error_sd = float(np.std(errors, ddof=1)) if errors.size > 1 else np.nan
    mean_se = float(np.mean(ses))
    return {
        "coverage": float(np.mean(hits)),
        "bias": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean_se": mean_se,
        "error_sd": error_sd,
        "se_ratio": mean_se / error_sd if error_sd > 0.0 else np.nan,
        "mean_width": float(2.0 * z * mean_se),
        "n_intervals": int(errors.size),
    }


def _ipw_subgroup_record(votes, responded, response_probs, target_mask, target):
    respondent_mask = responded & target_mask
    n_target = int(target_mask.sum())
    if respondent_mask.sum() == 0 or n_target == 0:
        return None

    y = np.asarray(votes, dtype=np.float64)[respondent_mask]
    response_probs = np.asarray(response_probs, dtype=np.float64)[respondent_mask]
    if np.any((response_probs <= 0.0) | (response_probs > 1.0)):
        return None

    estimate = float(np.sum(y / response_probs) / n_target)
    variance = float(
        np.sum(y**2 * (1.0 - response_probs) / response_probs**2) / n_target**2
    )
    return {
        "estimate": estimate,
        "target": float(target),
        "se": float(np.sqrt(max(variance, 0.0))),
    }


def _fit_weighted_logistic(
    X,
    y,
    weights,
    *,
    maxiter: int = 100,
    tol: float = 1e-8,
    max_step: float = 5.0,
):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if y.shape != (X.shape[0],):
        raise ValueError("y must have one entry per row of X")
    if weights.shape != y.shape:
        raise ValueError("weights must have one entry per row of X")
    if y.size == 0:
        raise ValueError("weighted logistic fit needs at least one respondent")
    if np.unique(y).size < 2:
        raise ValueError("unregularized weighted logistic fit needs both outcomes")

    beta = np.zeros(X.shape[1], dtype=np.float64)
    converged = False

    for _ in range(maxiter):
        probs = sigmoid(X @ beta)
        grad = X.T @ (weights * (probs - y))
        hessian = X.T @ ((weights * probs * (1.0 - probs))[:, None] * X)
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ grad
        if not np.all(np.isfinite(step)):
            raise ValueError("unregularized weighted logistic fit became nonfinite")
        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step:
            step *= max_step / step_norm
            step_norm = max_step
        beta -= step
        if step_norm <= tol * (1.0 + float(np.linalg.norm(beta))):
            converged = True
            break
    if not converged:
        probs = sigmoid(X @ beta)
        grad_norm = float(np.linalg.norm(X.T @ (weights * (probs - y))))
        if grad_norm > 1e-4 * max(1.0, float(np.sum(weights))):
            raise ValueError("unregularized weighted logistic fit failed to converge")

    probs = sigmoid(X @ beta)
    hessian = X.T @ ((weights * probs * (1.0 - probs))[:, None] * X)
    covariance = np.linalg.pinv(hessian)
    return beta, covariance


def _mrp_subgroup_records(
    case,
    votes,
    responded,
    weights,
    targets,
    groups,
    totals,
):
    from elrpy.inference import shifted_subgroup_inference

    X = np.asarray(case["X"], dtype=np.float64)
    group_ids = np.asarray(case["group_ids"])
    subgroup_mask = np.asarray(case["subgroup_mask"], dtype=bool)
    beta, beta_cov = _fit_weighted_logistic(
        X[responded],
        np.asarray(votes)[responded],
        np.asarray(weights)[responded],
    )
    shifted = shifted_binary_probs_by_group(X @ beta, group_ids, totals)
    records = []
    for group in groups:
        stats = shifted_subgroup_inference(
            X,
            group_ids,
            shifted,
            subgroup_mask,
            group,
            beta_cov=beta_cov,
        )
        records.append(
            {
                "estimate": float(stats["estimate"]),
                "target": float(targets[group]),
                "se": float(stats["realized_share_se"]),
            }
        )
    return records


def survey_estimator_comparison(
    case: dict[str, object],
    *,
    seed: int,
    n_rep: int,
    response_rate: float,
    response_x: float = 0.0,
    response_vote: float = 0.0,
    groups=None,
    level: float = 0.95,
) -> list[dict[str, object]]:
    """Compare shifted ELR, IPW, and shifted MRP under survey nonresponse."""

    from elrpy.inference import shifted_subgroup_inference

    rng = np.random.default_rng(seed)
    X = np.asarray(case["X"], dtype=np.float64)
    group_ids = np.asarray(case["group_ids"])
    subgroup_mask = np.asarray(case["subgroup_mask"], dtype=bool)
    probs = np.asarray(case["probs"], dtype=np.float64)
    linear_predictor = np.asarray(case["linear_predictor"], dtype=np.float64)
    n_groups = int(group_ids.max()) + 1
    if groups is None:
        groups = np.arange(n_groups, dtype=np.int64)
    else:
        groups = np.asarray(groups, dtype=np.int64)

    records = {"elr": [], "ipw": [], "mrp": []}
    total_intervals = n_rep * len(groups)

    for _ in range(n_rep):
        votes = draw_binary_votes(rng, probs)
        targets = np.array(
            [
                votes[(group_ids == group) & subgroup_mask].mean()
                for group in range(n_groups)
            ],
            dtype=np.float64,
        )
        totals = np.bincount(group_ids, weights=votes, minlength=n_groups)
        try:
            shifted = shifted_binary_probs_by_group(
                linear_predictor,
                group_ids,
                totals,
            )
            for group in groups:
                stats = shifted_subgroup_inference(
                    X,
                    group_ids,
                    shifted,
                    subgroup_mask,
                    group,
                )
                records["elr"].append(
                    {
                        "estimate": float(stats["estimate"]),
                        "target": float(targets[group]),
                        "se": float(stats["realized_share_se"]),
                    }
                )
        except ValueError:
            pass

        survey = draw_survey_response(
            rng,
            case,
            votes,
            response_rate=response_rate,
            response_x=response_x,
            response_vote=response_vote,
        )
        responded = survey["responded"]
        for group in groups:
            target_mask = (group_ids == group) & subgroup_mask
            record = _ipw_subgroup_record(
                votes,
                responded,
                survey["analysis_response_probs"],
                target_mask,
                targets[group],
            )
            if record is not None:
                records["ipw"].append(record)

        try:
            records["mrp"].extend(
                _mrp_subgroup_records(
                    case,
                    votes,
                    responded,
                    1.0 / survey["analysis_response_probs"],
                    targets,
                    groups,
                    totals,
                )
            )
        except ValueError:
            pass

    rows = []
    for estimator, estimator_records in records.items():
        summary = _summarize_estimator(estimator_records, level=level)
        rows.append(
            {
                "estimator": estimator,
                "response_rate": response_rate,
                "response_x": response_x,
                "response_vote": response_vote,
                "skipped_intervals": total_intervals - summary["n_intervals"],
                **summary,
            }
        )
    return rows


def survey_estimator_sweep(
    case: dict[str, object],
    *,
    seed: int,
    n_rep: int,
    response_rates,
    response_x: float = 0.0,
    response_vote: float = 0.0,
    groups=None,
    level: float = 0.95,
) -> list[dict[str, object]]:
    """Sweep response rates for ELR, IPW, and MRP comparisons."""

    rows = []
    for index, response_rate in enumerate(response_rates):
        rows.extend(
            survey_estimator_comparison(
                case,
                seed=seed + index,
                n_rep=n_rep,
                response_rate=response_rate,
                response_x=response_x,
                response_vote=response_vote,
                groups=groups,
                level=level,
            )
        )
    return rows

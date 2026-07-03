from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SieveRieszSettings:
    basis_size_grid: tuple[int, ...]
    lambda_grid: tuple[float, ...]
    cv_folds: int
    propensity_clip: float
    riesz_bound: float
    min_rows: int
    min_unique_scores: int


def resolve_sieve_riesz_options(options, n: int) -> SieveRieszSettings:
    """Validate public Wald correction options."""

    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("`wald_options` must be a dictionary or None.")
    allowed = {
        "basis_size_grid",
        "lambda_grid",
        "cv_folds",
        "propensity_clip",
        "riesz_bound",
        "min_rows",
        "min_unique_scores",
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"Unknown `wald_options` entries: {unknown}.")

    basis_size_grid = _clean_int_grid(options.get("basis_size_grid", (8, 16, 32, 64)), "basis_size_grid")
    lambda_grid = _clean_float_grid(options.get("lambda_grid", 10.0 ** np.linspace(-8, 1, 8)), "lambda_grid")
    cv_folds = int(options.get("cv_folds", 5))
    if cv_folds < 2:
        raise ValueError("`wald_options['cv_folds']` must be at least 2.")
    cv_folds = min(cv_folds, max(2, int(n)))

    propensity_clip = float(options.get("propensity_clip", 0.025))
    if not np.isfinite(propensity_clip) or propensity_clip <= 0.0 or propensity_clip >= 0.5:
        raise ValueError("`wald_options['propensity_clip']` must lie in (0, 0.5).")
    riesz_bound = float(options.get("riesz_bound", 1.0 / propensity_clip))
    if not np.isfinite(riesz_bound) or riesz_bound <= 0.0:
        raise ValueError("`wald_options['riesz_bound']` must be positive and finite.")

    min_rows = int(options.get("min_rows", 20))
    min_unique_scores = int(options.get("min_unique_scores", 3))
    if min_rows < 2:
        raise ValueError("`wald_options['min_rows']` must be at least 2.")
    if min_unique_scores < 1:
        raise ValueError("`wald_options['min_unique_scores']` must be at least 1.")

    return SieveRieszSettings(
        basis_size_grid=basis_size_grid,
        lambda_grid=lambda_grid,
        cv_folds=cv_folds,
        propensity_clip=propensity_clip,
        riesz_bound=riesz_bound,
        min_rows=min_rows,
        min_unique_scores=min_unique_scores,
    )


def compute_sieve_riesz_wald_correction(**kwargs):
    """Compute the binary sieve-Riesz corrected Wald interval."""

    return _compute_riesz_wald_correction(aux_method="sieve_riesz", **kwargs)


def compute_levelset_riesz_wald_correction(**kwargs):
    """Compute the binary level-set Riesz corrected Wald interval."""

    return _compute_riesz_wald_correction(aux_method="levelset_riesz", **kwargs)


def _compute_riesz_wald_correction(
    *,
    aux_method,
    a_encoded,
    y,
    weights,
    levels,
    control_index,
    calibrated_mu_mat,
    calibrated_pi_mat,
    contrast_estimate,
    simple_wald_std_error,
    conf_level,
    random_state=None,
    wald_options=None,
    conservative=False,
):
    if len(levels) != 2:
        raise ValueError("Riesz Wald correction currently requires a binary treatment.")
    if aux_method not in {"sieve_riesz", "levelset_riesz"}:
        raise ValueError(f"Unknown Riesz Wald auxiliary method: {aux_method}.")

    settings = resolve_sieve_riesz_options(wald_options, len(y))
    active_index = [idx for idx in range(len(levels)) if idx != control_index][0]
    a_active = (np.asarray(a_encoded, dtype=int) == active_index).astype(float)
    a_control = (np.asarray(a_encoded, dtype=int) == control_index).astype(float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mu_active = np.asarray(calibrated_mu_mat[:, active_index], dtype=float)
    mu_control = np.asarray(calibrated_mu_mat[:, control_index], dtype=float)
    pi_active = _clip_probability(calibrated_pi_mat[:, active_index], eps=1e-8)
    pi_control = _clip_probability(calibrated_pi_mat[:, control_index], eps=1e-8)
    alpha_active = 1.0 / pi_active
    alpha_control = 1.0 / pi_control
    seed = _normalize_seed(random_state)

    alpha_active_score = 1.0 / _clip_probability(pi_active, eps=settings.propensity_clip)
    alpha_control_score = 1.0 / _clip_probability(pi_control, eps=settings.propensity_clip)
    if aux_method == "sieve_riesz":
        h_active = _fit_sieve_riesz_arm(
            score=mu_active,
            arm_indicator=a_active,
            alpha_star=alpha_active,
            weights=weights,
            seed=seed + 110,
            settings=settings,
        )
        h_control = _fit_sieve_riesz_arm(
            score=mu_control,
            arm_indicator=a_control,
            alpha_star=alpha_control,
            weights=weights,
            seed=seed + 120,
            settings=settings,
        )
        q_active = _fit_sieve_residual_arm(
            score=alpha_active_score,
            residual=y - mu_active,
            arm_indicator=a_active,
            weights=weights,
            seed=seed + 130,
            settings=settings,
        )
        q_control = _fit_sieve_residual_arm(
            score=alpha_control_score,
            residual=y - mu_control,
            arm_indicator=a_control,
            weights=weights,
            seed=seed + 140,
            settings=settings,
        )
        basis_type = "cosine"
    else:
        h_active = _fit_levelset_riesz_arm(
            score=mu_active,
            arm_indicator=a_active,
            alpha_star=alpha_active,
            weights=weights,
        )
        h_control = _fit_levelset_riesz_arm(
            score=mu_control,
            arm_indicator=a_control,
            alpha_star=alpha_control,
            weights=weights,
        )
        q_active = _fit_levelset_residual_arm(
            score=alpha_active_score,
            residual=y - mu_active,
            arm_indicator=a_active,
            weights=weights,
        )
        q_control = _fit_levelset_residual_arm(
            score=alpha_control_score,
            residual=y - mu_control,
            arm_indicator=a_control,
            weights=weights,
        )
        basis_type = "calibrated_levelset"

    active_score = (
        mu_active
        + a_active * (alpha_active + h_active["predictions"]) * (y - mu_active)
        - (a_active * alpha_active - 1.0) * q_active["predictions"]
    )
    control_score = (
        mu_control
        + a_control * (alpha_control + h_control["predictions"]) * (y - mu_control)
        - (a_control * alpha_control - 1.0) * q_control["predictions"]
    )
    corrected_score = active_score - control_score
    corrected_score = corrected_score - _weighted_mean(corrected_score, weights) + float(contrast_estimate)
    corrected_std_error = _weighted_standard_error(corrected_score, float(contrast_estimate), weights)
    conservative_std_error = max(float(simple_wald_std_error), float(corrected_std_error))
    std_error = conservative_std_error if conservative else float(corrected_std_error)
    alpha = 1.0 - float(conf_level)
    z_value = _normal_quantile(1.0 - alpha / 2.0)

    diagnostics = {
        "wald_correction": aux_method,
        "wald_aux_method": aux_method,
        "std_error_mode": "conservative" if conservative else "corrected_if",
        "simple_wald_std_error": float(simple_wald_std_error),
        "corrected_if_std_error": float(corrected_std_error),
        "conservative_std_error": float(conservative_std_error),
        "selected_std_error": float(std_error),
        "propensity_clip": settings.propensity_clip,
        "riesz_bound": settings.riesz_bound if aux_method == "sieve_riesz" else None,
        "basis_type": basis_type,
        "h_treated": _fit_metadata(h_active),
        "h_control": _fit_metadata(h_control),
        "q_treated": _fit_metadata(q_active),
        "q_control": _fit_metadata(q_control),
    }
    return {
        "std_error": float(std_error),
        "lower": float(contrast_estimate - z_value * std_error),
        "upper": float(contrast_estimate + z_value * std_error),
        "diagnostics": diagnostics,
    }


def _clean_int_grid(values, name: str) -> tuple[int, ...]:
    grid = tuple(dict.fromkeys(int(value) for value in values if np.isfinite(value) and int(value) > 0))
    if not grid:
        raise ValueError(f"`wald_options['{name}']` must contain at least one positive integer.")
    return grid


def _clean_float_grid(values, name: str) -> tuple[float, ...]:
    grid = tuple(dict.fromkeys(float(value) for value in values if np.isfinite(value) and float(value) > 0.0))
    if not grid:
        raise ValueError(f"`wald_options['{name}']` must contain at least one positive finite value.")
    return grid


def _fit_sieve_riesz_arm(score, arm_indicator, alpha_star, weights, seed: int, settings: SieveRieszSettings):
    score = np.asarray(score, dtype=float)
    arm_indicator = np.asarray(arm_indicator, dtype=float)
    alpha_star = np.asarray(alpha_star, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(score) & np.isfinite(arm_indicator) & np.isfinite(alpha_star) & np.isfinite(weights) & (weights >= 0.0)
    train_score = score[ok]
    train_arm = (arm_indicator[ok] > 0.0).astype(float)
    train_alpha = alpha_star[ok]
    train_weights = weights[ok]
    unique_score_count = _unique_score_count(train_score)
    constant_fit = _riesz_constant(train_arm, train_alpha, train_weights, settings.riesz_bound)

    def fallback(reason: str):
        return _constant_fit(
            predictions=np.repeat(constant_fit, len(score)),
            requested_method="sieve_riesz",
            reason=reason,
            train_n=len(train_score),
            arm_n=float(np.sum(train_weights * train_arm)),
            unique_score_count=unique_score_count,
            target_sd=_weighted_sd(1.0 - train_arm * train_alpha, train_weights),
            bound_lower=-settings.riesz_bound,
            bound_upper=settings.riesz_bound,
        )

    if len(train_score) < settings.min_rows:
        return fallback("too_few_rows")
    if np.sum(train_weights * train_arm) <= 0.0:
        return fallback("too_few_positive_weight_arm_rows")
    if np.sum(train_arm) < 2.0:
        return fallback("too_few_arm_rows")
    if unique_score_count < settings.min_unique_scores:
        return fallback("few_unique_scores")

    best = _select_sieve_ridge(
        score=train_score,
        target=1.0 - train_arm * train_alpha,
        weights=train_weights,
        arm_weights=train_arm,
        loss_kind="riesz",
        seed=seed,
        settings=settings,
    )
    if best is None:
        return fallback("nonfinite_cv_risks")
    predictions = _fit_and_predict_sieve_ridge(
        train_score=train_score,
        target=1.0 - train_arm * train_alpha,
        weights=train_weights,
        eval_score=score,
        basis_size=best["basis_size"],
        lambda_value=best["lambda"],
        arm_weights=train_arm,
        loss_kind="riesz",
    )
    predictions = np.where(np.isfinite(predictions), predictions, constant_fit)
    predictions = np.clip(predictions, -settings.riesz_bound, settings.riesz_bound)
    return _sieve_fit(
        predictions=predictions,
        requested_method="sieve_riesz",
        basis_size=best["basis_size"],
        lambda_value=best["lambda"],
        cv_folds=best["cv_folds"],
        selected_risk=best["risk"],
        train_n=len(train_score),
        arm_n=float(np.sum(train_weights * train_arm)),
        unique_score_count=unique_score_count,
        target_sd=_weighted_sd(1.0 - train_arm * train_alpha, train_weights),
        bound_lower=-settings.riesz_bound,
        bound_upper=settings.riesz_bound,
    )


def _fit_sieve_residual_arm(score, residual, arm_indicator, weights, seed: int, settings: SieveRieszSettings):
    score = np.asarray(score, dtype=float)
    residual = np.asarray(residual, dtype=float)
    arm_indicator = np.asarray(arm_indicator, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = (
        np.isfinite(score)
        & np.isfinite(residual)
        & np.isfinite(arm_indicator)
        & (arm_indicator > 0.0)
        & np.isfinite(weights)
        & (weights >= 0.0)
    )
    train_score = score[ok]
    train_residual = residual[ok]
    train_weights = weights[ok]
    unique_score_count = _unique_score_count(train_score)
    residual_range = _finite_range(train_residual)
    constant_fit = float(np.clip(_weighted_mean_or_zero(train_residual, train_weights), residual_range[0], residual_range[1]))

    def fallback(reason: str):
        return _constant_fit(
            predictions=np.repeat(constant_fit, len(score)),
            requested_method="sieve_residual_ridge",
            reason=reason,
            train_n=len(train_residual),
            arm_n=float(np.sum(train_weights)),
            unique_score_count=unique_score_count,
            target_sd=_weighted_sd(train_residual, train_weights),
            bound_lower=residual_range[0],
            bound_upper=residual_range[1],
        )

    if len(train_score) < settings.min_rows:
        return fallback("too_few_rows")
    if np.sum(train_weights) <= 0.0:
        return fallback("zero_positive_weight")
    target_sd = _weighted_sd(train_residual, train_weights)
    if not np.isfinite(target_sd) or target_sd < 1e-8:
        return fallback("near_constant_target")
    if unique_score_count < settings.min_unique_scores:
        return fallback("few_unique_scores")

    best = _select_sieve_ridge(
        score=train_score,
        target=train_residual,
        weights=train_weights,
        arm_weights=None,
        loss_kind="residual",
        seed=seed,
        settings=settings,
    )
    if best is None:
        return fallback("nonfinite_cv_risks")
    predictions = _fit_and_predict_sieve_ridge(
        train_score=train_score,
        target=train_residual,
        weights=train_weights,
        eval_score=score,
        basis_size=best["basis_size"],
        lambda_value=best["lambda"],
        arm_weights=None,
        loss_kind="residual",
    )
    predictions = np.where(np.isfinite(predictions), predictions, constant_fit)
    predictions = np.clip(predictions, residual_range[0], residual_range[1])
    return _sieve_fit(
        predictions=predictions,
        requested_method="sieve_residual_ridge",
        basis_size=best["basis_size"],
        lambda_value=best["lambda"],
        cv_folds=best["cv_folds"],
        selected_risk=best["risk"],
        train_n=len(train_residual),
        arm_n=float(np.sum(train_weights)),
        unique_score_count=unique_score_count,
        target_sd=target_sd,
        bound_lower=residual_range[0],
        bound_upper=residual_range[1],
    )


def _fit_levelset_riesz_arm(score, arm_indicator, alpha_star, weights):
    score = np.asarray(score, dtype=float)
    arm_indicator = np.asarray(arm_indicator, dtype=float)
    alpha_star = np.asarray(alpha_star, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(score) & np.isfinite(arm_indicator) & np.isfinite(alpha_star) & np.isfinite(weights) & (weights >= 0.0)
    train_score = score[ok]
    train_arm = (arm_indicator[ok] > 0.0).astype(float)
    train_alpha = alpha_star[ok]
    train_weights = weights[ok]
    constant_fit = _riesz_constant_unbounded(train_arm, train_alpha, train_weights)

    if len(train_score) == 0:
        return _levelset_fit(
            predictions=np.repeat(constant_fit, len(score)),
            requested_method="levelset_riesz",
            selected_method="constant",
            fallback=True,
            fallback_reason="empty_training_set",
            train_n=0,
            arm_n=0.0,
            unique_score_count=0,
            target_sd=np.nan,
            selected_risk=np.nan,
            num_levelsets=0,
            unidentified_level_count=0,
            fallback_prediction_count=len(score),
        )

    train_keys = _levelset_keys(train_score)
    level_keys, group_index = np.unique(train_keys, return_inverse=True)
    arm_n = float(np.sum(train_weights * train_arm))
    target_sd = _weighted_sd(1.0 - train_arm * train_alpha, train_weights)
    unique_score_count = _unique_score_count(train_score)
    if not np.isfinite(arm_n) or arm_n <= 0.0:
        return _levelset_fit(
            predictions=np.repeat(constant_fit, len(score)),
            requested_method="levelset_riesz",
            selected_method="constant",
            fallback=True,
            fallback_reason="too_few_positive_weight_arm_rows",
            train_n=len(train_score),
            arm_n=arm_n,
            unique_score_count=unique_score_count,
            target_sd=target_sd,
            selected_risk=np.nan,
            num_levelsets=len(level_keys),
            unidentified_level_count=len(level_keys),
            fallback_prediction_count=len(score),
        )

    numerator = np.bincount(
        group_index,
        weights=train_weights * (1.0 - train_arm * train_alpha),
        minlength=len(level_keys),
    )
    denominator = np.bincount(
        group_index,
        weights=train_weights * train_arm,
        minlength=len(level_keys),
    )
    level_values = np.repeat(constant_fit, len(level_keys))
    identified_positions = np.flatnonzero(np.isfinite(denominator) & (denominator > 0.0))
    valid_positions = np.array([], dtype=int)
    if len(identified_positions):
        candidate_values = numerator[identified_positions] / denominator[identified_positions]
        finite_candidate = np.isfinite(candidate_values)
        valid_positions = identified_positions[finite_candidate]
        level_values[valid_positions] = candidate_values[finite_candidate]
    unidentified_level_count = int(len(level_keys) - len(valid_positions))

    predictions, fallback_prediction_count = _predict_levelset_values(score, level_keys, level_values, constant_fit)
    train_predictions = level_values[group_index]
    train_loss = train_arm * train_predictions**2 + 2.0 * train_arm * train_alpha * train_predictions - 2.0 * train_predictions
    selected_risk = float(np.sum(_normalize_weights(train_weights) * train_loss))
    return _levelset_fit(
        predictions=predictions,
        requested_method="levelset_riesz",
        selected_method="levelset_objective",
        fallback=False,
        fallback_reason=None,
        train_n=len(train_score),
        arm_n=arm_n,
        unique_score_count=unique_score_count,
        target_sd=target_sd,
        selected_risk=selected_risk,
        num_levelsets=len(level_keys),
        unidentified_level_count=unidentified_level_count,
        fallback_prediction_count=fallback_prediction_count,
    )


def _fit_levelset_residual_arm(score, residual, arm_indicator, weights):
    score = np.asarray(score, dtype=float)
    residual = np.asarray(residual, dtype=float)
    arm_indicator = np.asarray(arm_indicator, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = (
        np.isfinite(score)
        & np.isfinite(residual)
        & np.isfinite(arm_indicator)
        & (arm_indicator > 0.0)
        & np.isfinite(weights)
        & (weights >= 0.0)
    )
    train_score = score[ok]
    train_residual = residual[ok]
    train_weights = weights[ok]
    constant_fit = _weighted_mean_or_zero(train_residual, train_weights)

    if len(train_score) == 0:
        return _levelset_fit(
            predictions=np.repeat(constant_fit, len(score)),
            requested_method="levelset_residual",
            selected_method="constant",
            fallback=True,
            fallback_reason="empty_training_set",
            train_n=0,
            arm_n=0.0,
            unique_score_count=0,
            target_sd=np.nan,
            selected_risk=np.nan,
            num_levelsets=0,
            unidentified_level_count=0,
            fallback_prediction_count=len(score),
        )

    train_keys = _levelset_keys(train_score)
    level_keys, group_index = np.unique(train_keys, return_inverse=True)
    arm_n = float(np.sum(train_weights))
    target_sd = _weighted_sd(train_residual, train_weights)
    unique_score_count = _unique_score_count(train_score)
    if not np.isfinite(arm_n) or arm_n <= 0.0:
        return _levelset_fit(
            predictions=np.repeat(constant_fit, len(score)),
            requested_method="levelset_residual",
            selected_method="constant",
            fallback=True,
            fallback_reason="zero_positive_weight",
            train_n=len(train_residual),
            arm_n=arm_n,
            unique_score_count=unique_score_count,
            target_sd=target_sd,
            selected_risk=np.nan,
            num_levelsets=len(level_keys),
            unidentified_level_count=len(level_keys),
            fallback_prediction_count=len(score),
        )

    numerator = np.bincount(
        group_index,
        weights=train_weights * train_residual,
        minlength=len(level_keys),
    )
    denominator = np.bincount(group_index, weights=train_weights, minlength=len(level_keys))
    level_values = np.repeat(constant_fit, len(level_keys))
    identified_positions = np.flatnonzero(np.isfinite(denominator) & (denominator > 0.0))
    if len(identified_positions):
        candidate_values = numerator[identified_positions] / denominator[identified_positions]
        finite_candidate = np.isfinite(candidate_values)
        valid_positions = identified_positions[finite_candidate]
        level_values[valid_positions] = candidate_values[finite_candidate]

    predictions, fallback_prediction_count = _predict_levelset_values(score, level_keys, level_values, constant_fit)
    train_predictions = level_values[group_index]
    selected_risk = float(np.sum(_normalize_weights(train_weights) * (train_predictions - train_residual) ** 2))
    return _levelset_fit(
        predictions=predictions,
        requested_method="levelset_residual",
        selected_method="levelset_objective",
        fallback=False,
        fallback_reason=None,
        train_n=len(train_residual),
        arm_n=arm_n,
        unique_score_count=unique_score_count,
        target_sd=target_sd,
        selected_risk=selected_risk,
        num_levelsets=len(level_keys),
        unidentified_level_count=0,
        fallback_prediction_count=fallback_prediction_count,
    )


def _levelset_keys(score):
    return np.round(np.asarray(score, dtype=float), 12)


def _predict_levelset_values(score, level_keys, level_values, fallback_value):
    score = np.asarray(score, dtype=float)
    predictions = np.repeat(float(fallback_value), len(score))
    keys = _levelset_keys(score)
    level_keys = np.asarray(level_keys, dtype=float)
    level_values = np.asarray(level_values, dtype=float)
    assigned = np.zeros(len(score), dtype=bool)
    finite_idx = np.flatnonzero(np.isfinite(keys))

    if len(level_keys) and len(finite_idx):
        positions = np.searchsorted(level_keys, keys[finite_idx])
        in_bounds = positions < len(level_keys)
        matched = np.zeros(len(finite_idx), dtype=bool)
        matched[in_bounds] = level_keys[positions[in_bounds]] == keys[finite_idx[in_bounds]]
        if np.any(matched):
            matched_idx = finite_idx[matched]
            matched_positions = positions[matched]
            finite_values = np.isfinite(level_values[matched_positions])
            if np.any(finite_values):
                assigned_idx = matched_idx[finite_values]
                predictions[assigned_idx] = level_values[matched_positions[finite_values]]
                assigned[assigned_idx] = True

    return predictions, int(len(score) - np.count_nonzero(assigned))


def _levelset_fit(
    predictions,
    requested_method,
    selected_method,
    fallback,
    fallback_reason,
    train_n,
    arm_n,
    unique_score_count,
    target_sd,
    selected_risk,
    num_levelsets,
    unidentified_level_count,
    fallback_prediction_count,
):
    return {
        "predictions": np.asarray(predictions, dtype=float),
        "requested_method": requested_method,
        "selected_method": selected_method,
        "fallback": bool(fallback),
        "fallback_reason": fallback_reason,
        "basis_size": None,
        "lambda": None,
        "cv_folds": None,
        "selected_risk": None if not np.isfinite(selected_risk) else float(selected_risk),
        "train_n": int(train_n),
        "arm_n": float(arm_n),
        "unique_score_count": int(unique_score_count),
        "target_sd": None if not np.isfinite(target_sd) else float(target_sd),
        "bound_lower": None,
        "bound_upper": None,
        "num_levelsets": int(num_levelsets),
        "unidentified_level_count": int(unidentified_level_count),
        "fallback_prediction_count": int(fallback_prediction_count),
    }


def _select_sieve_ridge(score, target, weights, arm_weights, loss_kind: str, seed: int, settings: SieveRieszSettings):
    n = len(score)
    cv_folds = min(settings.cv_folds, n)
    strata = arm_weights if arm_weights is not None else None
    fold_ids = _deterministic_fold_ids(n=n, n_folds=cv_folds, seed=seed, strata=strata)
    best = None
    for basis_size in settings.basis_size_grid:
        try:
            basis = _cosine_basis(score, score, basis_size)["train_basis"]
        except ValueError:
            continue
        penalty = _ridge_penalty(basis.shape[1])
        for lambda_value in settings.lambda_grid:
            losses = []
            for fold in np.unique(fold_ids):
                val_idx = np.flatnonzero(fold_ids == fold)
                train_idx = np.flatnonzero(fold_ids != fold)
                if len(train_idx) == 0 or len(val_idx) == 0:
                    continue
                beta = _solve_sieve(
                    basis=basis[train_idx],
                    target=target[train_idx],
                    weights=weights[train_idx],
                    arm_weights=None if arm_weights is None else arm_weights[train_idx],
                    lambda_value=lambda_value,
                    penalty=penalty,
                    loss_kind=loss_kind,
                )
                fitted = basis[val_idx] @ beta
                fitted = np.where(np.isfinite(fitted), fitted, 0.0)
                val_weights = _normalize_weights(weights[val_idx])
                if loss_kind == "riesz":
                    arm_val = arm_weights[val_idx]
                    # target is 1 - A alpha, so this equals A h^2 + 2 A alpha h - 2 h.
                    loss = arm_val * fitted**2 + 2.0 * arm_val * (1.0 - target[val_idx]) * fitted - 2.0 * fitted
                else:
                    loss = (fitted - target[val_idx]) ** 2
                losses.append(float(np.sum(val_weights * loss)))
            risk = float(np.mean(losses)) if losses else np.nan
            if np.isfinite(risk) and (best is None or risk < best["risk"]):
                best = {"risk": risk, "basis_size": int(basis_size), "lambda": float(lambda_value), "cv_folds": int(cv_folds)}
    return best


def _fit_and_predict_sieve_ridge(train_score, target, weights, eval_score, basis_size, lambda_value, arm_weights, loss_kind):
    basis_bundle = _cosine_basis(train_score, eval_score, basis_size)
    basis = basis_bundle["train_basis"]
    penalty = _ridge_penalty(basis.shape[1])
    beta = _solve_sieve(
        basis=basis,
        target=target,
        weights=weights,
        arm_weights=arm_weights,
        lambda_value=lambda_value,
        penalty=penalty,
        loss_kind=loss_kind,
    )
    return basis_bundle["eval_basis"] @ beta


def _solve_sieve(basis, target, weights, arm_weights, lambda_value, penalty, loss_kind: str):
    weights_norm = _normalize_weights(weights)
    if loss_kind == "riesz":
        lhs_weights = weights_norm * np.asarray(arm_weights, dtype=float)
        rhs_weights = weights_norm
    else:
        lhs_weights = weights_norm
        rhs_weights = weights_norm
    lhs = basis.T @ (basis * lhs_weights[:, None]) + float(lambda_value) * penalty
    lhs = (lhs + lhs.T) / 2.0
    rhs = basis.T @ (rhs_weights * target)
    return _solve_ridge(lhs, rhs)


def _cosine_basis(train_score, eval_score, basis_size: int):
    train_score = np.asarray(train_score, dtype=float)
    eval_score = np.asarray(eval_score, dtype=float)
    finite_train = train_score[np.isfinite(train_score)]
    if len(finite_train) == 0:
        raise ValueError("Cannot build a sieve basis from nonfinite scores.")
    lower = float(np.min(finite_train))
    upper = float(np.max(finite_train))
    scale = upper - lower
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Cannot build a nonconstant sieve basis from constant scores.")
    train_scaled = np.clip((train_score - lower) / scale, 0.0, 1.0)
    eval_scaled = np.clip((eval_score - lower) / scale, 0.0, 1.0)
    frequencies = np.arange(int(basis_size), dtype=float)
    train_basis = np.cos(np.pi * train_scaled[:, None] * frequencies[None, :])
    eval_basis = np.cos(np.pi * eval_scaled[:, None] * frequencies[None, :])
    train_basis[:, 0] = 1.0
    eval_basis[:, 0] = 1.0
    train_basis[~np.isfinite(train_basis)] = 0.0
    eval_basis[~np.isfinite(eval_basis)] = 0.0
    return {"train_basis": train_basis, "eval_basis": eval_basis}


def _ridge_penalty(p: int):
    penalty = np.eye(p, dtype=float)
    if p:
        penalty[0, 0] = 0.0
    return penalty


def _solve_ridge(lhs, rhs):
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        jittered = np.array(lhs, copy=True)
        jittered[np.diag_indices_from(jittered)] += 1e-8
        try:
            beta = np.linalg.solve(jittered, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    beta = np.asarray(beta, dtype=float)
    beta[~np.isfinite(beta)] = 0.0
    return beta


def _deterministic_fold_ids(n: int, n_folds: int, seed: int, strata=None):
    fold_ids = np.empty(n, dtype=int)
    keys = np.mod((np.arange(n, dtype=float) + 1.0) * 1103515245.0 + float(seed) * 12345.0, 2147483647.0)
    if strata is None:
        strata = np.zeros(n, dtype=int)
    strata = np.asarray(strata)
    for value in np.unique(strata):
        idx = np.flatnonzero(strata == value)
        order = np.lexsort((idx, keys[idx]))
        ordered = idx[order]
        fold_ids[ordered] = np.arange(len(ordered), dtype=int) % int(n_folds)
    return fold_ids


def _constant_fit(predictions, requested_method, reason, train_n, arm_n, unique_score_count, target_sd, bound_lower, bound_upper):
    return {
        "predictions": np.asarray(predictions, dtype=float),
        "requested_method": requested_method,
        "selected_method": "constant",
        "fallback": True,
        "fallback_reason": reason,
        "basis_size": None,
        "lambda": None,
        "cv_folds": None,
        "selected_risk": None,
        "train_n": int(train_n),
        "arm_n": float(arm_n),
        "unique_score_count": int(unique_score_count),
        "target_sd": None if not np.isfinite(target_sd) else float(target_sd),
        "bound_lower": float(bound_lower),
        "bound_upper": float(bound_upper),
    }


def _sieve_fit(
    predictions,
    requested_method,
    basis_size,
    lambda_value,
    cv_folds,
    selected_risk,
    train_n,
    arm_n,
    unique_score_count,
    target_sd,
    bound_lower,
    bound_upper,
):
    return {
        "predictions": np.asarray(predictions, dtype=float),
        "requested_method": requested_method,
        "selected_method": "sieve_ridge_cv",
        "fallback": False,
        "fallback_reason": None,
        "basis_size": int(basis_size),
        "lambda": float(lambda_value),
        "cv_folds": int(cv_folds),
        "selected_risk": float(selected_risk),
        "train_n": int(train_n),
        "arm_n": float(arm_n),
        "unique_score_count": int(unique_score_count),
        "target_sd": None if not np.isfinite(target_sd) else float(target_sd),
        "bound_lower": float(bound_lower),
        "bound_upper": float(bound_upper),
    }


def _fit_metadata(fit):
    return {key: value for key, value in fit.items() if key != "predictions"}


def _riesz_constant(arm_indicator, alpha_star, weights, bound):
    denom = float(np.sum(weights * arm_indicator))
    if not np.isfinite(denom) or denom <= 0.0:
        return 0.0
    value = float(np.sum(weights * (1.0 - arm_indicator * alpha_star)) / denom)
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -bound, bound))


def _riesz_constant_unbounded(arm_indicator, alpha_star, weights):
    denom = float(np.sum(weights * arm_indicator))
    if not np.isfinite(denom) or denom <= 0.0:
        return 0.0
    value = float(np.sum(weights * (1.0 - arm_indicator * alpha_star)) / denom)
    return value if np.isfinite(value) else 0.0


def _clip_probability(probability, eps):
    probability = np.asarray(probability, dtype=float)
    return np.clip(probability, float(eps), 1.0 - float(eps))


def _normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.repeat(1.0 / len(weights), len(weights))
    return weights / total


def _weighted_mean(values, weights):
    return float(np.sum(_normalize_weights(weights) * np.asarray(values, dtype=float)))


def _weighted_mean_or_zero(values, weights):
    if len(values) == 0:
        return 0.0
    return _weighted_mean(values, weights)


def _weighted_standard_error(score, estimate, weights):
    weights_norm = _normalize_weights(weights)
    return float(np.sqrt(np.sum((weights_norm * (np.asarray(score, dtype=float) - float(estimate))) ** 2)))


def _weighted_sd(values, weights):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan
    weights_norm = _normalize_weights(weights)
    mean = float(np.sum(weights_norm * values))
    return float(np.sqrt(np.sum(weights_norm * (values - mean) ** 2)))


def _finite_range(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return (0.0, 0.0)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    return (lower, upper)


def _unique_score_count(score):
    if len(score) == 0:
        return 0
    return int(len(np.unique(np.round(score, 8))))


def _normalize_seed(seed):
    if seed is None:
        return 1
    try:
        value = int(seed)
    except (TypeError, ValueError):
        return 1
    return abs(value) % 2147483646 + 1


def _normal_quantile(probability):
    from scipy.stats import norm

    return float(norm.ppf(probability))

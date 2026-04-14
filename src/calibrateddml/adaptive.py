from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator

from ._utils import (
    align_nuisance_matrix,
    as_feature_frame,
    bootstrap_indices,
    encode_treatment,
    normalize_weights,
    resolve_fold_ids,
    resolve_weights,
    validate_supplied_probability_matrix,
    validate_binary_treatment,
)
from .calibration import fit_monotone_calibrator
from .core import ResultMixin, _fit_nuisances
from .models import fit_regressor, predict_regressor, resolve_regressor


class AdaptiveCalibratedDML(BaseEstimator, ResultMixin):
    """Experimental adaptive calibrated DML estimator for binary treatment.

    The Python adaptive interface is intentionally narrower than the standard
    estimator and currently supports binary-treatment `plugin` and
    `calibrated_rlearner` modes only.

    These adaptive estimators are super-efficient methods in the sense that
    they can attain lower realized variance and lower mean-squared error than
    standard calibrated DML at favorable distributions. The tradeoff is that
    uncertainty quantification is harder, so coverage can be less stable in
    practice. For routine analyses, prefer `CalibratedDML`.
    """

    def __init__(
        self,
        control_level=0,
        mode="plugin",
        outcome_model="lasso",
        treatment_model="lasso",
        cate_model="lasso",
        stratify=("outcome", "treatment"),
        calibration_method="isotonic",
        calibration_stratify=None,
        inference="jackknife",
        conf_level=0.95,
        bootstrap_reps=200,
        jackknife_folds=20,
        random_state=None,
        n_folds=5,
        fold_ids=None,
    ):
        self.control_level = control_level
        self.mode = mode
        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
        self.cate_model = cate_model
        self.stratify = stratify
        self.calibration_method = calibration_method
        self.calibration_stratify = calibration_stratify
        self.inference = inference
        self.conf_level = conf_level
        self.bootstrap_reps = bootstrap_reps
        self.jackknife_folds = jackknife_folds
        self.random_state = random_state
        self.n_folds = n_folds
        self.fold_ids = fold_ids

    def fit(self, X, A, y, sample_weight=None):
        X_frame = as_feature_frame(X)
        treatment = encode_treatment(A, self.control_level)
        validate_binary_treatment(treatment.levels)
        y_array = np.asarray(y, dtype=float)
        weights = resolve_weights(sample_weight, len(y_array))
        fold_ids = resolve_fold_ids(treatment.encoded, self.n_folds, self.fold_ids, self.random_state)
        mu_mat, pi_mat = _fit_nuisances(
            X=X_frame,
            treatment=treatment,
            y=y_array,
            weights=weights,
            fold_ids=fold_ids,
            outcome_model=self.outcome_model,
            treatment_model=self.treatment_model,
            stratify=tuple(self.stratify),
            random_state=self.random_state,
        )
        self.n_features_in_ = X_frame.shape[1]
        self.feature_names_in_ = np.asarray(X_frame.columns, dtype=object)
        self.fold_ids_ = fold_ids
        return self._fit_from_nuisances(X_frame, treatment, y_array, mu_mat, pi_mat, weights, nuisance_source="fitted")

    def fit_from_nuisances(self, X, A, y, mu_mat, pi_mat, sample_weight=None, treatment_levels=None):
        X_frame = as_feature_frame(X)
        treatment = encode_treatment(A, self.control_level, treatment_levels=treatment_levels)
        validate_binary_treatment(treatment.levels)
        y_array = np.asarray(y, dtype=float)
        if len(y_array) != len(treatment.encoded):
            raise ValueError("`y` must have one entry per observation.")
        weights = resolve_weights(sample_weight, len(treatment.encoded))
        mu_aligned = align_nuisance_matrix(mu_mat, treatment.levels, "mu_mat")
        pi_aligned = validate_supplied_probability_matrix(pi_mat, treatment.levels, "pi_mat")
        if mu_aligned.shape[0] != len(treatment.encoded) or pi_aligned.shape[0] != len(treatment.encoded):
            raise ValueError("`mu_mat` and `pi_mat` must have one row per observation.")
        self.fold_ids_ = None if self.fold_ids is None else np.asarray(self.fold_ids, dtype=int)
        return self._fit_from_nuisances(
            X_frame,
            treatment,
            y_array,
            mu_aligned,
            pi_aligned,
            weights,
            nuisance_source="supplied",
        )

    def _fit_from_nuisances(self, X, treatment, y, mu_mat, pi_mat, weights, nuisance_source):
        if self.mode not in {"plugin", "calibrated_rlearner"}:
            raise ValueError("`mode` must be 'plugin' or 'calibrated_rlearner'.")
        adaptive_calibration_method = _resolve_adaptive_calibration_method(self.calibration_method)
        control_index = treatment.control_index
        treat_index = 1 - control_index
        a_binary = (treatment.encoded == treat_index).astype(int)
        pi_treat = pi_mat[:, treat_index]
        m_hat = np.sum(mu_mat * pi_mat, axis=1)

        if self.mode == "plugin":
            plugin_calibration_stratify = "outcome" if self.calibration_stratify is None else self.calibration_stratify
            base_fit = _compute_plugin_adaptive_fit(
                a_encoded=treatment.encoded,
                y=y,
                mu_mat=mu_mat,
                pi_mat=pi_mat,
                weights=weights,
                levels=treatment.levels,
                control_index=control_index,
                calibration_method=adaptive_calibration_method,
                calibration_stratify=plugin_calibration_stratify,
            )
            refit = lambda index: _compute_plugin_adaptive_fit(
                a_encoded=treatment.encoded[index],
                y=y[index],
                mu_mat=mu_mat[index],
                pi_mat=pi_mat[index],
                weights=weights[index],
                levels=treatment.levels,
                control_index=control_index,
                calibration_method=adaptive_calibration_method,
                calibration_stratify=plugin_calibration_stratify,
            )
        else:
            base_fit = _compute_rlearner_adaptive_fit(
                X=X,
                a_binary=a_binary,
                y=y,
                m_hat=m_hat,
                pi_treat=pi_treat,
                weights=weights,
                levels=treatment.levels,
                control_index=control_index,
                cate_model=self.cate_model,
                calibration_method=adaptive_calibration_method,
                random_state=self.random_state,
            )
            refit = lambda index: _compute_rlearner_adaptive_fit(
                X=X.iloc[index],
                a_binary=a_binary[index],
                y=y[index],
                m_hat=m_hat[index],
                pi_treat=pi_treat[index],
                weights=weights[index],
                levels=treatment.levels,
                control_index=control_index,
                cate_model=self.cate_model,
                calibration_method=adaptive_calibration_method,
                random_state=self.random_state,
            )

        interval_tables = _compute_adaptive_interval_tables(
            base_fit=base_fit,
            refit=refit,
            a_encoded=treatment.encoded,
            levels=treatment.levels,
            control_index=control_index,
            inference=self.inference,
            conf_level=self.conf_level,
            bootstrap_reps=self.bootstrap_reps,
            jackknife_folds=self.jackknife_folds,
            fold_ids=self.fold_ids_,
            random_state=self.random_state,
        )
        self.estimates_ = interval_tables["estimates"]
        self.potential_outcomes_ = interval_tables["potential_outcomes"]
        self.contrasts_ = interval_tables["contrasts"]
        self.treatment_levels_ = treatment.levels
        self.control_level_ = treatment.control_level
        self.mu_mat_ = mu_mat
        self.pi_mat_ = pi_mat
        self.calibrated_mu_mat_ = base_fit["calibrated_mu_mat"]
        self.calibrated_pi_mat_ = pi_mat
        self.calibration_ = {
            "method": adaptive_calibration_method,
            "mode": self.mode,
            "calibration_stratify": (
                "outcome" if self.mode == "plugin" and self.calibration_stratify is None else self.calibration_stratify
            ),
        }
        self.is_experimental_ = True
        self.is_fitted_ = True
        self.nuisance_source_ = nuisance_source
        self.adaptive_mode_ = self.mode
        self.calibration_method_ = adaptive_calibration_method
        return self


def _resolve_adaptive_calibration_method(method):
    if method in (None, "auto", "isotonic"):
        return "isotonic"
    raise ValueError(
        "AdaptiveCalibratedDML uses isotonic calibration internally. "
        "Set `calibration_method='isotonic'`."
    )


def _summarize_arm_scores(arm_scores, levels, control_index, weights, calibrated_mu_mat):
    normalized_weights = normalize_weights(weights)
    arm_scores = np.asarray(arm_scores, dtype=float)
    arm_estimates = np.sum(arm_scores * normalized_weights[:, None], axis=0)
    arm_se = np.sqrt(np.sum((normalized_weights[:, None] * (arm_scores - arm_estimates[None, :])) ** 2, axis=0))
    contrast_scores = arm_scores[:, [idx for idx in range(len(levels)) if idx != control_index]] - arm_scores[:, [control_index]]
    contrast_scores = np.asarray(contrast_scores, dtype=float)
    contrast_estimates = np.sum(contrast_scores * normalized_weights[:, None], axis=0)
    contrast_se = np.sqrt(np.sum((normalized_weights[:, None] * (contrast_scores - contrast_estimates[None, :])) ** 2, axis=0))
    return {
        "calibrated_mu_mat": calibrated_mu_mat,
        "calibrated_pi_mat": np.ones_like(calibrated_mu_mat),
        "calibration_bundle": None,
        "arm_scores": [arm_scores[:, idx] for idx in range(arm_scores.shape[1])],
        "arm_estimates": arm_estimates,
        "arm_standard_error": arm_se,
        "contrast_scores": [contrast_scores[:, idx] for idx in range(contrast_scores.shape[1])],
        "contrast_estimates": contrast_estimates,
        "contrast_standard_error": contrast_se,
    }


def _summarize_custom_scores(
    arm_scores, arm_estimates, contrast_scores, contrast_estimates, calibrated_mu_mat, calibrated_pi_mat, weights
):
    normalized_weights = normalize_weights(weights)
    arm_scores = np.asarray(arm_scores, dtype=float)
    arm_estimates = np.asarray(arm_estimates, dtype=float)
    contrast_scores = np.asarray(contrast_scores, dtype=float)
    contrast_estimates = np.asarray(contrast_estimates, dtype=float)
    arm_se = np.sqrt(np.sum((normalized_weights[:, None] * (arm_scores - arm_estimates[None, :])) ** 2, axis=0))
    contrast_se = np.sqrt(
        np.sum((normalized_weights[:, None] * (contrast_scores - contrast_estimates[None, :])) ** 2, axis=0)
    )
    return {
        "calibrated_mu_mat": calibrated_mu_mat,
        "calibrated_pi_mat": calibrated_pi_mat,
        "calibration_bundle": None,
        "arm_scores": [arm_scores[:, idx] for idx in range(arm_scores.shape[1])],
        "arm_estimates": arm_estimates,
        "arm_standard_error": arm_se,
        "contrast_scores": [contrast_scores[:, idx] for idx in range(contrast_scores.shape[1])],
        "contrast_estimates": contrast_estimates,
        "contrast_standard_error": contrast_se,
    }


def _estimate_gamma_weights(tau_star, residual_treatment_sq, weights, min_count=25):
    tau_star = np.asarray(tau_star, dtype=float)
    residual_treatment_sq = np.asarray(residual_treatment_sq, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(tau_star)
    if n == 0:
        return np.array([], dtype=float)

    rounded = np.round(tau_star, decimals=12)
    _, inverse, counts = np.unique(rounded, return_inverse=True, return_counts=True)
    if np.max(counts) >= max(2, min_count // 2):
        group_index = inverse
        n_groups = int(group_index.max()) + 1
    else:
        n_groups = int(min(max(5, np.sqrt(n)), max(5, n // max(5, min_count))))
        order = np.argsort(tau_star, kind="mergesort")
        cuts = np.array_split(order, n_groups)
        group_index = np.zeros(n, dtype=int)
        for group_id, idx in enumerate(cuts):
            group_index[idx] = group_id
        n_groups = len(cuts)

    gamma = np.zeros(n, dtype=float)
    for group_id in range(n_groups):
        idx = group_index == group_id
        denom = np.sum(weights[idx] * residual_treatment_sq[idx]) / np.sum(weights[idx])
        gamma[idx] = 1.0 / max(float(denom), 1e-8)
    return gamma


def _compute_plugin_adaptive_fit(a_encoded, y, mu_mat, pi_mat, weights, levels, control_index, calibration_method, calibration_stratify):
    from .calibration import calibrate_outcome_matrix

    calibrated = calibrate_outcome_matrix(
        y=y,
        mu_mat=mu_mat,
        a_encoded=a_encoded,
        weights=weights,
        method=calibration_method,
        calibration_stratify=calibration_stratify,
    )["calibrated"]
    normalized_weights = normalize_weights(weights)
    observed_mu = calibrated[np.arange(len(y)), a_encoded]
    arm_estimates = np.sum(calibrated * normalized_weights[:, None], axis=0)
    arm_scores = np.column_stack(
        [
            calibrated[:, level_index] + ((a_encoded == level_index) * (y - observed_mu) / pi_mat[:, level_index])
            for level_index in range(len(levels))
        ]
    )
    contrast_indices = [idx for idx in range(len(levels)) if idx != control_index]
    contrast_estimates = arm_estimates[contrast_indices] - arm_estimates[control_index]
    contrast_scores = arm_scores[:, contrast_indices] - arm_scores[:, [control_index]]
    return _summarize_custom_scores(
        arm_scores=arm_scores,
        arm_estimates=arm_estimates,
        contrast_scores=contrast_scores,
        contrast_estimates=contrast_estimates,
        calibrated_mu_mat=calibrated,
        calibrated_pi_mat=pi_mat,
        weights=weights,
    )


def _rlearner_pseudo(a_binary, y, m_hat, pi_treat, weights):
    residual_treatment = a_binary - pi_treat
    keep = np.abs(residual_treatment) > 1e-8
    pseudo_outcome = np.zeros_like(y, dtype=float)
    pseudo_outcome[keep] = (y[keep] - m_hat[keep]) / residual_treatment[keep]
    pseudo_weights = weights * residual_treatment ** 2
    return pseudo_outcome, pseudo_weights, keep


def _compute_rlearner_adaptive_fit(X, a_binary, y, m_hat, pi_treat, weights, levels, control_index, cate_model, calibration_method, random_state):
    pseudo_outcome, pseudo_weights, keep = _rlearner_pseudo(a_binary, y, m_hat, pi_treat, weights)
    cate_spec = resolve_regressor(cate_model, random_state=random_state)
    cate_fit = fit_regressor(cate_spec, X.iloc[keep], pseudo_outcome[keep], sample_weight=pseudo_weights[keep], random_state=random_state)
    tau_hat = predict_regressor(cate_fit, X)
    if calibration_method == "none":
        tau_star = tau_hat
    else:
        calibrator = fit_monotone_calibrator(
            x=tau_hat[keep],
            y=pseudo_outcome[keep],
            weights=pseudo_weights[keep],
            method=calibration_method,
        )
        tau_star = calibrator.predict(tau_hat)
    mu0 = m_hat - pi_treat * tau_star
    mu1 = m_hat + (1.0 - pi_treat) * tau_star
    calibrated_mu = np.column_stack([mu0, mu1]) if control_index == 0 else np.column_stack([mu1, mu0])
    normalized_weights = normalize_weights(weights)
    residual_treatment = a_binary - pi_treat
    gamma = _estimate_gamma_weights(
        tau_star=tau_star,
        residual_treatment_sq=residual_treatment**2,
        weights=weights,
    )
    correction = gamma * residual_treatment * (y - m_hat - residual_treatment * tau_hat)
    if control_index == 0:
        arm_scores = np.column_stack([mu0 - pi_treat * correction, mu1 + (1.0 - pi_treat) * correction])
        arm_estimates = np.array([np.sum(normalized_weights * mu0), np.sum(normalized_weights * mu1)])
    else:
        arm_scores = np.column_stack([mu1 + (1.0 - pi_treat) * correction, mu0 - pi_treat * correction])
        arm_estimates = np.array([np.sum(normalized_weights * mu1), np.sum(normalized_weights * mu0)])
    contrast_scores = np.column_stack([tau_star + correction])
    contrast_estimates = np.array([np.sum(normalized_weights * tau_star)])
    return _summarize_custom_scores(
        arm_scores=arm_scores,
        arm_estimates=arm_estimates,
        contrast_scores=contrast_scores,
        contrast_estimates=contrast_estimates,
        calibrated_mu_mat=calibrated_mu,
        calibrated_pi_mat=(
            np.column_stack([1.0 - pi_treat, pi_treat])
            if control_index == 0
            else np.column_stack([pi_treat, 1.0 - pi_treat])
        ),
        weights=weights,
    )


def _compute_adaptive_interval_tables(
    base_fit,
    refit,
    a_encoded,
    levels,
    control_index,
    inference,
    conf_level,
    bootstrap_reps,
    jackknife_folds,
    fold_ids,
    random_state,
):
    alpha = 1.0 - float(conf_level)
    z_value = float(norm.ppf(1.0 - alpha / 2.0))
    contrast_levels = [level for idx, level in enumerate(levels) if idx != control_index]

    potential_outcomes = pd.DataFrame(
        {
            "estimand_type": "potential_outcome",
            "estimand": [f"E[Y({level})]" for level in levels],
            "level": levels,
            "control_level": [None] * len(levels),
            "estimate": base_fit["arm_estimates"],
            "std_error": base_fit["arm_standard_error"],
            "lower": base_fit["arm_estimates"] - z_value * base_fit["arm_standard_error"],
            "upper": base_fit["arm_estimates"] + z_value * base_fit["arm_standard_error"],
        }
    )
    contrasts = pd.DataFrame(
        {
            "estimand_type": "contrast",
            "estimand": [f"E[Y({level})] - E[Y({levels[control_index]})]" for level in contrast_levels],
            "level": contrast_levels,
            "control_level": [levels[control_index]] * len(contrast_levels),
            "estimate": base_fit["contrast_estimates"],
            "std_error": base_fit["contrast_standard_error"],
            "lower": base_fit["contrast_estimates"] - z_value * base_fit["contrast_standard_error"],
            "upper": base_fit["contrast_estimates"] + z_value * base_fit["contrast_standard_error"],
        }
    )

    if inference == "bootstrap" and int(bootstrap_reps) > 0:
        rng = np.random.default_rng(random_state)
        draws = []
        for _ in range(int(bootstrap_reps)):
            index = bootstrap_indices(len(a_encoded), fold_ids, rng)
            fit = refit(index)
            draws.append(np.concatenate([fit["arm_estimates"], fit["contrast_estimates"]]))
        draws = np.vstack(draws)
        arm_draws = draws[:, : len(levels)]
        contrast_draws = draws[:, len(levels) :]
        potential_outcomes.loc[:, "std_error"] = arm_draws.std(axis=0, ddof=1)
        potential_outcomes.loc[:, "lower"] = np.quantile(arm_draws, alpha / 2.0, axis=0)
        potential_outcomes.loc[:, "upper"] = np.quantile(arm_draws, 1.0 - alpha / 2.0, axis=0)
        if len(contrast_levels):
            contrasts.loc[:, "std_error"] = contrast_draws.std(axis=0, ddof=1)
            contrasts.loc[:, "lower"] = np.quantile(contrast_draws, alpha / 2.0, axis=0)
            contrasts.loc[:, "upper"] = np.quantile(contrast_draws, 1.0 - alpha / 2.0, axis=0)

    if inference == "jackknife":
        if fold_ids is None:
            fold_ids = resolve_fold_ids(np.asarray(a_encoded), jackknife_folds, None, random_state=1)
        groups = [np.flatnonzero(fold_ids == group) for group in np.unique(fold_ids)]
        if len(groups) < 2:
            raise ValueError("Jackknife inference requires at least two groups.")
        leave_one_out = []
        for group in groups:
            keep = np.setdiff1d(np.arange(len(a_encoded)), group)
            fit = refit(keep)
            leave_one_out.append(np.concatenate([fit["arm_estimates"], fit["contrast_estimates"]]))
        loo = np.vstack(leave_one_out)
        full = np.concatenate([base_fit["arm_estimates"], base_fit["contrast_estimates"]])
        n_groups = loo.shape[0]
        pseudo = n_groups * full[None, :] - (n_groups - 1) * loo
        se = pseudo.std(axis=0, ddof=1) / np.sqrt(n_groups)
        potential_outcomes.loc[:, "std_error"] = se[: len(levels)]
        potential_outcomes.loc[:, "lower"] = potential_outcomes["estimate"] - z_value * potential_outcomes["std_error"]
        potential_outcomes.loc[:, "upper"] = potential_outcomes["estimate"] + z_value * potential_outcomes["std_error"]
        if len(contrast_levels):
            contrasts.loc[:, "std_error"] = se[len(levels) :]
            contrasts.loc[:, "lower"] = contrasts["estimate"] - z_value * contrasts["std_error"]
            contrasts.loc[:, "upper"] = contrasts["estimate"] + z_value * contrasts["std_error"]

    estimates = pd.concat([potential_outcomes, contrasts], ignore_index=True)
    return {
        "potential_outcomes": potential_outcomes,
        "contrasts": contrasts,
        "estimates": estimates,
    }

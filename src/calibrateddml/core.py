from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ._utils import (
    TreatmentEncoding,
    align_nuisance_matrix,
    as_feature_frame,
    bootstrap_indices,
    empirical_probability_matrix,
    encode_treatment,
    normalize_calibration_stratify,
    normalize_probability_matrix,
    normalize_stratify,
    normalize_weights,
    resolve_fold_ids,
    resolve_weights,
    validate_supplied_probability_matrix,
)
from .calibration import fit_calibration_bundle
from .models import (
    fit_classifier,
    fit_regressor,
    predict_classifier_binary_positive,
    predict_classifier_proba,
    predict_regressor,
    resolve_classifier,
    resolve_regressor,
)


class ResultMixin:
    def _require_fitted(self) -> None:
        check_is_fitted(self, attributes=["estimates_"])

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "estimates_")

    def summary(self) -> pd.DataFrame:
        self._require_fitted()
        return self.estimates_.copy()

    def to_frame(self) -> pd.DataFrame:
        self._require_fitted()
        return self.estimates_.copy()

    def confint(self) -> pd.DataFrame:
        self._require_fitted()
        return self.estimates_.loc[:, ["estimand", "lower", "upper"]].copy()


class CalibratedDML(BaseEstimator, ResultMixin):
    """Calibrated DML estimator for categorical treatments.

    The primary Python interface is `calibrateddml.CalibratedDML`. It supports
    either internal cross-fitted nuisance estimation from `(X, A, y)` or direct
    estimation from supplied nuisance matrices via `fit_from_nuisances(...)`.
    """

    def __init__(
        self,
        control_level=0,
        outcome_model="lasso",
        treatment_model="lasso",
        stratify=("outcome", "treatment"),
        calibration_method="auto",
        calibration_stratify=None,
        inference="wald",
        conf_level=0.95,
        bootstrap_reps=200,
        jackknife_folds=10,
        random_state=None,
        n_folds=5,
        fold_ids=None,
    ):
        self.control_level = control_level
        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
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
        y_array = np.asarray(y, dtype=float)
        treatment = encode_treatment(A, self.control_level)
        weights = resolve_weights(sample_weight, len(y_array))
        stratify = normalize_stratify(self.stratify)
        fold_ids = resolve_fold_ids(treatment.encoded, self.n_folds, self.fold_ids, self.random_state)

        mu_mat, pi_mat = _fit_nuisances(
            X=X_frame,
            treatment=treatment,
            y=y_array,
            weights=weights,
            fold_ids=fold_ids,
            outcome_model=self.outcome_model,
            treatment_model=self.treatment_model,
            stratify=stratify,
            random_state=self.random_state,
        )
        self.n_features_in_ = X_frame.shape[1]
        self.feature_names_in_ = np.asarray(X_frame.columns, dtype=object)
        nuisance_source = "fitted"
        self.fold_ids_ = fold_ids
        return self._fit_from_aligned_nuisances(
            treatment=treatment,
            y=y_array,
            mu_mat=mu_mat,
            pi_mat=pi_mat,
            weights=weights,
            nuisance_source=nuisance_source,
        )

    def fit_from_nuisances(self, A, y, mu_mat, pi_mat, sample_weight=None, treatment_levels=None):
        treatment = encode_treatment(A, self.control_level, treatment_levels=treatment_levels)
        y_array = np.asarray(y, dtype=float)
        if len(y_array) != len(treatment.encoded):
            raise ValueError("`y` must have one entry per observation.")
        weights = resolve_weights(sample_weight, len(treatment.encoded))
        mu_aligned = align_nuisance_matrix(mu_mat, treatment.levels, "mu_mat")
        pi_aligned = validate_supplied_probability_matrix(pi_mat, treatment.levels, "pi_mat")
        if mu_aligned.shape[0] != len(treatment.encoded) or pi_aligned.shape[0] != len(treatment.encoded):
            raise ValueError("`mu_mat` and `pi_mat` must have one row per observation.")
        self.fold_ids_ = None if self.fold_ids is None else np.asarray(self.fold_ids, dtype=int)
        return self._fit_from_aligned_nuisances(
            treatment=treatment,
            y=y_array,
            mu_mat=mu_aligned,
            pi_mat=pi_aligned,
            weights=weights,
            nuisance_source="supplied",
        )

    def _fit_from_aligned_nuisances(self, treatment: TreatmentEncoding, y, mu_mat, pi_mat, weights, nuisance_source):
        _validate_common_configuration(self)
        base_fit = _compute_cdml_estimates(
            a_encoded=treatment.encoded,
            y=y,
            mu_mat=mu_mat,
            pi_mat=pi_mat,
            weights=weights,
            levels=treatment.levels,
            control_index=treatment.control_index,
            calibration_method=self.calibration_method,
            calibration_stratify=self.calibration_stratify,
        )
        interval_tables = _compute_interval_tables(
            base_fit=base_fit,
            a_encoded=treatment.encoded,
            y=y,
            mu_mat=mu_mat,
            pi_mat=pi_mat,
            weights=weights,
            levels=treatment.levels,
            control_index=treatment.control_index,
            inference=self.inference,
            conf_level=self.conf_level,
            bootstrap_reps=self.bootstrap_reps,
            jackknife_folds=self.jackknife_folds,
            calibration_method=self.calibration_method,
            calibration_stratify=self.calibration_stratify,
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
        self.calibrated_pi_mat_ = base_fit["calibrated_pi_mat"]
        self.calibration_ = {
            "method": self.calibration_method,
            "calibration_stratify": normalize_calibration_stratify(self.calibration_stratify),
            "bundle": base_fit["calibration_bundle"],
        }
        self.is_fitted_ = True
        self.nuisance_source_ = nuisance_source
        return self


def _validate_common_configuration(estimator: CalibratedDML) -> None:
    if estimator.inference not in {"wald", "bootstrap", "jackknife"}:
        raise ValueError("`inference` must be one of {'wald', 'bootstrap', 'jackknife'}.")
    if estimator.calibration_method not in {"auto", "isotonic", "smooth_isotonic", "none"}:
        raise ValueError("Unsupported `calibration_method`.")
    if not (0 < float(estimator.conf_level) < 1):
        raise ValueError("`conf_level` must lie strictly between 0 and 1.")
    if int(estimator.bootstrap_reps) < 0:
        raise ValueError("`bootstrap_reps` must be non-negative.")
    if int(estimator.jackknife_folds) < 2:
        raise ValueError("`jackknife_folds` must be at least 2.")


def _fit_nuisances(
    X: pd.DataFrame,
    treatment: TreatmentEncoding,
    y: np.ndarray,
    weights: np.ndarray,
    fold_ids: np.ndarray,
    outcome_model,
    treatment_model,
    stratify,
    random_state,
):
    levels = treatment.levels
    n = len(y)
    mu_mat = np.full((n, len(levels)), np.nan, dtype=float)
    pi_mat = np.full((n, len(levels)), np.nan, dtype=float)
    outcome_spec = resolve_regressor(outcome_model, random_state=random_state)
    treatment_spec = resolve_classifier(treatment_model, random_state=random_state)

    for fold in np.unique(fold_ids):
        train = np.flatnonzero(fold_ids != fold)
        valid = np.flatnonzero(fold_ids == fold)
        mu_mat[valid] = _fit_outcome_fold(
            X_train=X.iloc[train],
            a_train=treatment.encoded[train],
            y_train=y[train],
            weights_train=weights[train],
            X_valid=X.iloc[valid],
            levels=levels,
            model_spec=outcome_spec,
            stratify=stratify,
            random_state=random_state,
        )
        pi_mat[valid] = _fit_treatment_fold(
            X_train=X.iloc[train],
            a_train=treatment.encoded[train],
            weights_train=weights[train],
            X_valid=X.iloc[valid],
            levels=levels,
            model_spec=treatment_spec,
            stratify=stratify,
            random_state=random_state,
        )

    return mu_mat, pi_mat


def _fit_outcome_fold(X_train, a_train, y_train, weights_train, X_valid, levels, model_spec, stratify, random_state):
    if "outcome" in stratify:
        predictions = np.zeros((len(X_valid), len(levels)), dtype=float)
        for level_index, level in enumerate(levels):
            arm_rows = np.flatnonzero(a_train == level_index)
            if len(arm_rows) == 0:
                raise ValueError("Every treatment level must appear in each training fold.")
            fit_obj = fit_regressor(model_spec, X_train.iloc[arm_rows], y_train[arm_rows], sample_weight=weights_train[arm_rows], random_state=random_state)
            predictions[:, level_index] = predict_regressor(fit_obj, X_valid)
        return predictions

    pooled_train = _augment_outcome_features(X_train, np.asarray(levels)[a_train], levels)
    fit_obj = fit_regressor(model_spec, pooled_train, y_train, sample_weight=weights_train, random_state=random_state)
    columns = []
    for level in levels:
        valid_features = _augment_outcome_features(X_valid, [level] * len(X_valid), levels)
        columns.append(predict_regressor(fit_obj, valid_features))
    return np.column_stack(columns)


def _fit_treatment_fold(X_train, a_train, weights_train, X_valid, levels, model_spec, stratify, random_state):
    if "treatment" not in stratify and model_spec.direct_multiclass:
        y_train = np.asarray(levels)[a_train]
        fit_obj = fit_classifier(model_spec, X_train, y_train, sample_weight=weights_train, random_state=random_state)
        return predict_classifier_proba(fit_obj, X_valid, classes=levels)

    predictions = np.zeros((len(X_valid), len(levels)), dtype=float)
    for level_index, level in enumerate(levels):
        y_binary = (a_train == level_index).astype(int)
        fit_obj = fit_classifier(model_spec, X_train, y_binary, sample_weight=weights_train, random_state=random_state)
        predictions[:, level_index] = predict_classifier_binary_positive(fit_obj, X_valid)
    fallback = empirical_probability_matrix(np.asarray(levels)[a_train], levels, len(X_valid))
    return normalize_probability_matrix(predictions, fallback=fallback)


def _augment_outcome_features(X: pd.DataFrame, treatment_values, levels):
    from ._utils import augment_features_with_treatment

    return augment_features_with_treatment(X, treatment_values, levels)


def _compute_cdml_estimates(
    a_encoded,
    y,
    mu_mat,
    pi_mat,
    weights,
    levels,
    control_index,
    calibration_method,
    calibration_stratify,
):
    bundle = fit_calibration_bundle(
        y=y,
        mu_mat=mu_mat,
        a_encoded=a_encoded,
        pi_mat=pi_mat,
        weights=weights,
        method=calibration_method,
        calibration_stratify=calibration_stratify,
    )
    calibrated_mu = bundle.calibrated_mu_mat
    calibrated_pi = bundle.calibrated_pi_mat
    observed_mu = calibrated_mu[np.arange(len(y)), a_encoded]
    normalized_weights = normalize_weights(weights)

    arm_scores = []
    arm_estimates = []
    arm_se = []
    for level_index in range(len(levels)):
        score = calibrated_mu[:, level_index] + ((a_encoded == level_index) * (y - observed_mu) / calibrated_pi[:, level_index])
        arm_scores.append(score)
        estimate = float(np.sum(normalized_weights * score))
        arm_estimates.append(estimate)
        arm_se.append(float(np.sqrt(np.sum((normalized_weights * (score - estimate)) ** 2))))

    arm_scores = [np.asarray(score, dtype=float) for score in arm_scores]
    arm_estimates = np.asarray(arm_estimates, dtype=float)
    arm_se = np.asarray(arm_se, dtype=float)

    contrast_levels = [idx for idx in range(len(levels)) if idx != control_index]
    contrast_scores = []
    contrast_estimates = []
    contrast_se = []
    for level_index in contrast_levels:
        score = arm_scores[level_index] - arm_scores[control_index]
        estimate = float(np.sum(normalized_weights * score))
        contrast_scores.append(score)
        contrast_estimates.append(estimate)
        contrast_se.append(float(np.sqrt(np.sum((normalized_weights * (score - estimate)) ** 2))))

    return {
        "calibration_bundle": bundle,
        "calibrated_mu_mat": calibrated_mu,
        "calibrated_pi_mat": calibrated_pi,
        "arm_scores": arm_scores,
        "arm_estimates": arm_estimates,
        "arm_standard_error": np.asarray(arm_se),
        "contrast_scores": contrast_scores,
        "contrast_estimates": np.asarray(contrast_estimates, dtype=float),
        "contrast_standard_error": np.asarray(contrast_se, dtype=float),
    }


def _compute_interval_tables(
    base_fit,
    a_encoded,
    y,
    mu_mat,
    pi_mat,
    weights,
    levels,
    control_index,
    inference,
    conf_level,
    bootstrap_reps,
    jackknife_folds,
    calibration_method,
    calibration_stratify,
    fold_ids,
    random_state,
):
    alpha = 1.0 - float(conf_level)
    z_value = float(norm.ppf(1.0 - alpha / 2.0))
    levels_array = np.asarray(levels)

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
    contrast_levels = [level for idx, level in enumerate(levels) if idx != control_index]
    contrast_estimates = base_fit["contrast_estimates"]
    contrast_se = base_fit["contrast_standard_error"]
    contrasts = pd.DataFrame(
        {
            "estimand_type": "contrast",
            "estimand": [f"E[Y({level})] - E[Y({levels[control_index]})]" for level in contrast_levels],
            "level": contrast_levels,
            "control_level": [levels[control_index]] * len(contrast_levels),
            "estimate": contrast_estimates,
            "std_error": contrast_se,
            "lower": contrast_estimates - z_value * contrast_se,
            "upper": contrast_estimates + z_value * contrast_se,
        }
    )

    if inference == "bootstrap" and bootstrap_reps > 0:
        draws = _bootstrap_cdml(
            a_encoded=a_encoded,
            y=y,
            mu_mat=mu_mat,
            pi_mat=pi_mat,
            weights=weights,
            levels=levels,
            control_index=control_index,
            bootstrap_reps=bootstrap_reps,
            calibration_method=calibration_method,
            calibration_stratify=calibration_stratify,
            fold_ids=fold_ids,
            random_state=random_state,
        )
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
        jackknife = _jackknife_cdml(
            base_fit=base_fit,
            a_encoded=a_encoded,
            y=y,
            mu_mat=mu_mat,
            pi_mat=pi_mat,
            weights=weights,
            levels=levels,
            control_index=control_index,
            jackknife_folds=jackknife_folds,
            calibration_method=calibration_method,
            calibration_stratify=calibration_stratify,
            fold_ids=fold_ids,
        )
        potential_outcomes.loc[:, "std_error"] = jackknife["arm_standard_error"]
        potential_outcomes.loc[:, "lower"] = potential_outcomes["estimate"] - z_value * potential_outcomes["std_error"]
        potential_outcomes.loc[:, "upper"] = potential_outcomes["estimate"] + z_value * potential_outcomes["std_error"]
        if len(contrast_levels):
            contrasts.loc[:, "std_error"] = jackknife["contrast_standard_error"]
            contrasts.loc[:, "lower"] = contrasts["estimate"] - z_value * contrasts["std_error"]
            contrasts.loc[:, "upper"] = contrasts["estimate"] + z_value * contrasts["std_error"]

    estimates = pd.concat([potential_outcomes, contrasts], ignore_index=True)
    return {
        "potential_outcomes": potential_outcomes,
        "contrasts": contrasts,
        "estimates": estimates,
    }


def _bootstrap_cdml(
    a_encoded,
    y,
    mu_mat,
    pi_mat,
    weights,
    levels,
    control_index,
    bootstrap_reps,
    calibration_method,
    calibration_stratify,
    fold_ids,
    random_state,
):
    rng = np.random.default_rng(random_state)
    draws = []
    for _ in range(int(bootstrap_reps)):
        index = bootstrap_indices(len(y), fold_ids, rng)
        fit = _compute_cdml_estimates(
            a_encoded=a_encoded[index],
            y=y[index],
            mu_mat=mu_mat[index],
            pi_mat=pi_mat[index],
            weights=weights[index],
            levels=levels,
            control_index=control_index,
            calibration_method=calibration_method,
            calibration_stratify=calibration_stratify,
        )
        draws.append(np.concatenate([fit["arm_estimates"], fit["contrast_estimates"]]))
    return np.vstack(draws)


def _jackknife_cdml(
    base_fit,
    a_encoded,
    y,
    mu_mat,
    pi_mat,
    weights,
    levels,
    control_index,
    jackknife_folds,
    calibration_method,
    calibration_stratify,
    fold_ids,
):
    if fold_ids is None:
        fold_ids = resolve_fold_ids(np.asarray(a_encoded), jackknife_folds, None, random_state=1)
    groups = [np.flatnonzero(fold_ids == group) for group in np.unique(fold_ids)]
    if len(groups) < 2:
        raise ValueError("Jackknife inference requires at least two groups.")
    leave_one_out = []
    for group in groups:
        keep = np.setdiff1d(np.arange(len(y)), group)
        fit = _compute_cdml_estimates(
            a_encoded=a_encoded[keep],
            y=y[keep],
            mu_mat=mu_mat[keep],
            pi_mat=pi_mat[keep],
            weights=weights[keep],
            levels=levels,
            control_index=control_index,
            calibration_method=calibration_method,
            calibration_stratify=calibration_stratify,
        )
        leave_one_out.append(np.concatenate([fit["arm_estimates"], fit["contrast_estimates"]]))
    loo = np.vstack(leave_one_out)
    n_groups = loo.shape[0]
    full = np.concatenate([base_fit["arm_estimates"], base_fit["contrast_estimates"]])
    pseudo = n_groups * full[None, :] - (n_groups - 1) * loo
    se = pseudo.std(axis=0, ddof=1) / np.sqrt(n_groups)
    return {
        "arm_standard_error": se[: len(levels)],
        "contrast_standard_error": se[len(levels) :],
    }


def calibrated_dml(data, outcome, treatment, covariates, control_level, sample_weight=None, **kwargs):
    frame = pd.DataFrame(data).copy()
    weights = None if sample_weight is None else frame[sample_weight] if isinstance(sample_weight, str) else sample_weight
    estimator = CalibratedDML(control_level=control_level, **kwargs)
    return estimator.fit(frame.loc[:, covariates], frame.loc[:, treatment], frame.loc[:, outcome], sample_weight=weights)


def calibrated_dml_from_nuisances(A, y, mu_mat, pi_mat, control_level, sample_weight=None, treatment_levels=None, **kwargs):
    estimator = CalibratedDML(control_level=control_level, **kwargs)
    return estimator.fit_from_nuisances(
        A=A,
        y=y,
        mu_mat=mu_mat,
        pi_mat=pi_mat,
        sample_weight=sample_weight,
        treatment_levels=treatment_levels,
    )

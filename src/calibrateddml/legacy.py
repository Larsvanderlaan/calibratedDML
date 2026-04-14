from __future__ import annotations

import warnings

import pandas as pd

from .calibration import calibrate_outcome_regression, calibrate_propensity_scores
from .core import CalibratedDML


def calibratedDML(A, Y, mu_mat, pi_mat, weights=None, control_level=0, treatment_levels=None, alpha=0.05):
    warnings.warn(
        "`calibratedDML(...)` is a legacy compatibility wrapper. "
        "Prefer `calibrateddml.CalibratedDML(...).fit_from_nuisances(...)`.",
        DeprecationWarning,
        stacklevel=2,
    )
    estimator = CalibratedDML(
        control_level=0 if control_level is None else control_level,
        inference="wald",
        conf_level=1.0 - alpha,
    ).fit_from_nuisances(
        A=A,
        y=Y,
        mu_mat=mu_mat,
        pi_mat=pi_mat,
        sample_weight=weights,
        treatment_levels=treatment_levels,
    )
    if control_level is None:
        return estimator.potential_outcomes_.loc[:, ["estimand", "estimate", "lower", "upper"]].rename(
            columns={"lower": "lower_bound", "upper": "upper_bound"}
        )
    return estimator.contrasts_.loc[:, ["estimand", "estimate", "lower", "upper"]].rename(
        columns={"lower": "lower_bound", "upper": "upper_bound"}
    )


def calibratedDML_bootstrap(A, Y, mu_mat, pi_mat, weights=None, control_level=0, treatment_levels=None, nboot=1000, alpha=0.05):
    warnings.warn(
        "`calibratedDML_bootstrap(...)` is a legacy compatibility wrapper. "
        "Prefer `CalibratedDML(..., inference='bootstrap').fit_from_nuisances(...)`.",
        DeprecationWarning,
        stacklevel=2,
    )
    estimator = CalibratedDML(
        control_level=0 if control_level is None else control_level,
        inference="bootstrap",
        conf_level=1.0 - alpha,
        bootstrap_reps=nboot,
    ).fit_from_nuisances(
        A=A,
        y=Y,
        mu_mat=mu_mat,
        pi_mat=pi_mat,
        sample_weight=weights,
        treatment_levels=treatment_levels,
    )
    if control_level is None:
        return estimator.potential_outcomes_.loc[:, ["estimand", "estimate", "lower", "upper"]].rename(
            columns={"lower": "lower_bound", "upper": "upper_bound"}
        )
    return estimator.contrasts_.loc[:, ["estimand", "estimate", "lower", "upper"]].rename(
        columns={"lower": "lower_bound", "upper": "upper_bound"}
    )


__all__ = [
    "calibratedDML",
    "calibratedDML_bootstrap",
    "calibrate_outcome_regression",
    "calibrate_propensity_scores",
]

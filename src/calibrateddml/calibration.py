from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression

from ._utils import normalize_calibration_stratify, normalize_probability_matrix, resolve_weights, weighted_mean


@dataclass
class MonotoneCalibrator:
    method: str
    model: object | None = None
    x_grid: np.ndarray | None = None
    y_grid: np.ndarray | None = None
    constant: float | None = None

    def predict(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.method == "none":
            return x
        if self.method == "constant":
            return np.repeat(self.constant, len(x))
        if self.method == "isotonic":
            return np.asarray(self.model.predict(x), dtype=float)
        if self.method == "smooth_isotonic":
            interpolator = PchipInterpolator(self.x_grid, self.y_grid, extrapolate=True)
            prediction = np.asarray(interpolator(x), dtype=float)
            return np.clip(prediction, self.y_grid.min(), self.y_grid.max())
        raise ValueError(f"Unsupported calibration method: {self.method}")


@dataclass
class CalibrationBundle:
    method: str
    calibration_stratify: str | None
    outcome_calibrators: list[MonotoneCalibrator]
    treatment_calibrators: list[MonotoneCalibrator]
    calibrated_mu_mat: np.ndarray
    calibrated_pi_mat: np.ndarray


def fit_monotone_calibrator(x, y, weights=None, method: str = "auto") -> MonotoneCalibrator:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = resolve_weights(weights, len(x))
    if method == "auto":
        method = "smooth_isotonic" if len(x) < 300 else "isotonic"
    if method == "none":
        return MonotoneCalibrator(method="none")
    if len(np.unique(x)) == 1:
        return MonotoneCalibrator(method="constant", constant=weighted_mean(y, weights))

    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(x, y, sample_weight=weights)
    if method == "isotonic":
        return MonotoneCalibrator(method="isotonic", model=iso)

    x_sorted = np.sort(np.unique(x))
    y_iso = iso.predict(x_sorted)
    if len(x_sorted) < 4:
        return MonotoneCalibrator(method="isotonic", model=iso)
    y_smooth = PchipInterpolator(x_sorted, y_iso, extrapolate=True)(x_sorted)
    y_smooth = np.maximum.accumulate(np.asarray(y_smooth, dtype=float))
    return MonotoneCalibrator(method="smooth_isotonic", x_grid=x_sorted, y_grid=y_smooth)


def isoreg_with_xgboost(x, y, weights=None, **_) -> callable:
    """Legacy compatibility alias that now uses the native isotonic engine."""
    calibrator = fit_monotone_calibrator(x=x, y=y, weights=weights, method="isotonic")
    return calibrator.predict


def calibrate_outcome_matrix(
    y,
    mu_mat,
    a_encoded,
    weights=None,
    method: str = "auto",
    calibration_stratify=None,
):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu_mat, dtype=float)
    weights = resolve_weights(weights, len(y))
    calibration_stratify = normalize_calibration_stratify(calibration_stratify)
    if method == "none":
        return {"calibrated": mu.copy(), "calibrators": [MonotoneCalibrator(method="none") for _ in range(mu.shape[1])]}
    calibrators: list[MonotoneCalibrator] = []
    calibrated = np.zeros_like(mu, dtype=float)
    for level_index in range(mu.shape[1]):
        subset = np.flatnonzero(a_encoded == level_index) if calibration_stratify == "outcome" else np.arange(len(y))
        calibrator = fit_monotone_calibrator(mu[subset, level_index], y[subset], weights=weights[subset], method=method)
        calibrators.append(calibrator)
        calibrated[:, level_index] = calibrator.predict(mu[:, level_index])
    return {"calibrated": calibrated, "calibrators": calibrators}


def calibrate_propensity_matrix(a_encoded, pi_mat, weights=None, method: str = "auto"):
    pi = normalize_probability_matrix(np.asarray(pi_mat, dtype=float))
    weights = resolve_weights(weights, pi.shape[0])
    if method == "none":
        return {"calibrated": pi.copy(), "calibrators": [MonotoneCalibrator(method="none") for _ in range(pi.shape[1])]}
    calibrators: list[MonotoneCalibrator] = []
    calibrated = np.zeros_like(pi, dtype=float)
    for level_index in range(pi.shape[1]):
        indicator = (np.asarray(a_encoded) == level_index).astype(float)
        calibrator = fit_monotone_calibrator(pi[:, level_index], indicator, weights=weights, method=method)
        calibrators.append(calibrator)
        pi_star = calibrator.predict(pi[:, level_index])
        treated = indicator == 1
        lower_bound = np.min(pi_star[treated]) if np.any(treated) else 1e-8
        if not np.isfinite(lower_bound):
            lower_bound = 1e-8
        calibrated[:, level_index] = np.clip(pi_star, lower_bound, 1.0)
    return {"calibrated": normalize_probability_matrix(calibrated), "calibrators": calibrators}


def fit_calibration_bundle(y, mu_mat, a_encoded, pi_mat, weights=None, method: str = "auto", calibration_stratify=None) -> CalibrationBundle:
    outcome = calibrate_outcome_matrix(
        y=y,
        mu_mat=mu_mat,
        a_encoded=a_encoded,
        weights=weights,
        method=method,
        calibration_stratify=calibration_stratify,
    )
    treatment = calibrate_propensity_matrix(
        a_encoded=a_encoded,
        pi_mat=pi_mat,
        weights=weights,
        method=method,
    )
    return CalibrationBundle(
        method=method,
        calibration_stratify=normalize_calibration_stratify(calibration_stratify),
        outcome_calibrators=outcome["calibrators"],
        treatment_calibrators=treatment["calibrators"],
        calibrated_mu_mat=outcome["calibrated"],
        calibrated_pi_mat=treatment["calibrated"],
    )


def calibrate_outcome_regression(y, mu_mat, a, weights=None, treatment_levels=None):
    a_array = np.asarray(a)
    if treatment_levels is None:
        treatment_levels = list(dict.fromkeys(a_array.tolist()))
    level_map = {level: index for index, level in enumerate(treatment_levels)}
    encoded = np.array([level_map[value] for value in a_array], dtype=int)
    result = calibrate_outcome_matrix(
        y=y,
        mu_mat=mu_mat,
        a_encoded=encoded,
        weights=weights,
        method="isotonic",
        calibration_stratify="outcome",
    )
    return {"mu_star": result["calibrated"]}


def calibrate_propensity_scores(a, pi_mat, weights=None, treatment_levels=None):
    a_array = np.asarray(a)
    if treatment_levels is None:
        treatment_levels = list(dict.fromkeys(a_array.tolist()))
    level_map = {level: index for index, level in enumerate(treatment_levels)}
    encoded = np.array([level_map[value] for value in a_array], dtype=int)
    result = calibrate_propensity_matrix(
        a_encoded=encoded,
        pi_mat=pi_mat,
        weights=weights,
        method="isotonic",
    )
    calibrated = result["calibrated"]
    return {"alpha_star": 1.0 / calibrated, "pi_star": calibrated}

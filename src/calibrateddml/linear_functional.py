"""Calibrated DML for linear combinations of regression evaluations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from ._utils import normalize_weights, resolve_weights
from .jackknife import vfold_jackknife


@dataclass
class StepCalibrator:
    """Right-continuous PAVA fit with constant extension beyond its knots."""

    knots: np.ndarray
    fitted: np.ndarray

    def predict(self, values):
        values = np.asarray(values, dtype=float)
        index = np.searchsorted(self.knots, values, side="right") - 1
        return self.fitted[np.clip(index, 0, len(self.knots) - 1)]

    __call__ = predict


def _pava(quadratic_mass, linear_mass):
    """Minimize sum_j a_j * u_j**2 - 2*b_j*u_j with u nondecreasing."""
    quadratic_mass = np.asarray(quadratic_mass, dtype=float)
    linear_mass = np.asarray(linear_mass, dtype=float)
    if np.any(quadratic_mass <= 0) or not np.all(np.isfinite(quadratic_mass)) or not np.all(np.isfinite(linear_mass)):
        raise ValueError("PAVA requires finite linear mass and positive finite quadratic mass.")
    blocks = []
    for index, (a, b) in enumerate(zip(quadratic_mass, linear_mass)):
        blocks.append([index, index + 1, a, b])
        while len(blocks) > 1 and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    fitted = np.empty(len(quadratic_mass), dtype=float)
    for start, end, a, b in blocks:
        fitted[start:end] = b / a
    return fitted


def _fit_outcome_pava(prediction, outcome, weights):
    positive = weights > 0
    knots, inverse = np.unique(prediction[positive], return_inverse=True)
    quadratic = np.bincount(inverse, weights=weights[positive], minlength=len(knots))
    linear = np.bincount(inverse, weights=weights[positive] * outcome[positive], minlength=len(knots))
    return StepCalibrator(knots, _pava(quadratic, linear))


def _fit_riesz_pava(prediction, evaluations, coefficients, weights):
    positive = weights > 0
    knots, inverse = np.unique(prediction[positive], return_inverse=True)
    quadratic = np.bincount(inverse, weights=weights[positive], minlength=len(knots))
    # A basis function is the indicator of one interval of the step map. Its
    # empirical functional determines the linear term of the Riesz loss.
    bins = np.searchsorted(knots, evaluations, side="right") - 1
    bins = np.clip(bins, 0, len(knots) - 1)
    linear = np.bincount(bins.ravel(), weights=(weights[:, None] * coefficients).ravel(), minlength=len(knots))
    return StepCalibrator(knots, _pava(quadratic, linear))


def _vector(value, name, n=None):
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or (n is not None and len(value) != n) or not np.all(np.isfinite(value)):
        raise ValueError(f"`{name}` must be a finite vector with one entry per observation.")
    return value


def _evaluations(value, observed, name):
    value = observed[:, None] if value is None else np.asarray(value, dtype=float)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape[0] != len(observed) or value.shape[1] < 1 or not np.all(np.isfinite(value)):
        raise ValueError(f"`{name}` must be a finite matrix with one row per observation.")
    return value


def _functional_coefficients(functional, shape):
    n, k = shape
    if callable(functional):
        def evaluate(values):
            return _vector(functional(values), "functional output", n)

        zero = np.zeros(shape)
        if not np.allclose(evaluate(zero), 0, atol=1e-12, rtol=0):
            raise ValueError("`functional` must be a row-wise linear map and send zero to zero.")
        coefficients = np.empty(shape)
        for column in range(k):
            basis = np.zeros(shape)
            basis[:, column] = 1
            coefficients[:, column] = evaluate(basis)
        # Check row-wise linearity without altering the caller's RNG state.
        probe = np.sin(np.arange(n * k).reshape(shape) + 0.3)
        for values in (probe, -0.7 * probe + np.cos(probe)):
            if not np.allclose(evaluate(values), np.sum(coefficients * values, axis=1), rtol=1e-10, atol=1e-10):
                raise ValueError("`functional` must be a row-wise linear map of the evaluation columns.")
    else:
        coefficients = np.asarray(functional, dtype=float)
        if coefficients.ndim == 0 and k == 1:
            coefficients = np.full(shape, float(coefficients))
        elif coefficients.ndim == 1 and coefficients.shape == (k,):
            coefficients = np.broadcast_to(coefficients, shape).copy()
        if coefficients.shape != shape:
            raise ValueError("`functional` coefficients must have one entry per evaluation column or shape (n, k).")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("`functional` coefficients must be finite.")
    return coefficients


@dataclass
class LinearFunctionalFit:
    estimates_: pd.DataFrame
    regression_: np.ndarray
    riesz_: np.ndarray
    calibrated_regression_: np.ndarray
    calibrated_riesz_: np.ndarray
    functional_values_: np.ndarray
    scores_: np.ndarray
    calibration_: dict
    jackknife_: dict | None
    inference: str

    def summary(self):
        return self.estimates_.copy()

    def to_frame(self):
        return self.summary()

    def confint(self):
        return self.estimates_.loc[:, ["estimand", "lower", "upper"]].copy()


def calibrated_linear_functional(
    y,
    regression,
    riesz,
    functional,
    regression_evaluations=None,
    riesz_evaluations=None,
    *,
    sample_weight=None,
    calibration_method="isotonic",
    inference="jackknife",
    jackknife_folds=10,
    bootstrap_reps=200,
    conf_level=0.95,
    random_state=None,
    name="linear_functional",
):
    """Estimate E[sum_j c_j(O) * mu(T_j(O))] from cross-fitted nuisances.

    ``regression`` and ``riesz`` give predictions at the observed inputs.
    The two evaluation matrices have shape (n, k) and give predictions at
    every input used by the functional, with the same column ordering.
    For an ATE these are predictions under control and treatment.

    ``functional`` is a row-wise linear callable mapping an (n, k) matrix to
    n contributions, a length-k coefficient vector, or an (n, k) coefficient
    matrix. For example, ``lambda values: values[:, 1] - values[:, 0]``.
    Evaluation matrices default to the observed predictions as one column.

    PAVA minimizes the regression squared-error loss and the quadratic Riesz
    loss. Signed Riesz estimates are supported. The step maps are evaluated
    at the functional inputs before forming the one-step estimate.
    Each map is nondecreasing and right-continuous, with knots at the distinct
    positive-weight observed predictions and constant extension at both tails.
    The Riesz loss is minimized over this step-function class; evaluation-only
    prediction values do not introduce additional knots.

    Inference defaults to a 10-fold jackknife with Student-t critical values.
    Initial predictions and functional coefficients stay fixed; only PAVA
    calibration and the final averages are recomputed after each deletion.
    ``calibration_method='none'`` returns the uncalibrated one-step estimator.
    """
    y = _vector(y, "y")
    n = len(y)
    if n < 2:
        raise ValueError("At least two observations are required.")
    regression = _vector(regression, "regression", n)
    riesz = _vector(riesz, "riesz", n)
    mu_eval = _evaluations(regression_evaluations, regression, "regression_evaluations")
    alpha_eval = _evaluations(riesz_evaluations, riesz, "riesz_evaluations")
    if mu_eval.shape != alpha_eval.shape:
        raise ValueError("The regression and Riesz evaluation matrices must have the same shape.")
    coefficients = _functional_coefficients(functional, mu_eval.shape)
    weights = resolve_weights(sample_weight, n)
    if calibration_method not in {"isotonic", "none"}:
        raise ValueError("`calibration_method` must be 'isotonic' or 'none'.")
    if inference not in {"jackknife", "wald", "bootstrap"}:
        raise ValueError("`inference` must be 'jackknife', 'wald', or 'bootstrap'.")
    if not 0 < conf_level < 1:
        raise ValueError("`conf_level` must lie strictly between 0 and 1.")

    def compute(index):
        w = weights[index]
        normalized = normalize_weights(w)
        if calibration_method == "isotonic":
            f = _fit_outcome_pava(regression[index], y[index], w)
            g = _fit_riesz_pava(riesz[index], alpha_eval[index], coefficients[index], w)
            mu = f(regression[index])
            alpha = g(riesz[index])
            contributions = np.sum(coefficients[index] * f(mu_eval[index]), axis=1)
        else:
            f = g = None
            mu = regression[index].copy()
            alpha = riesz[index].copy()
            contributions = np.sum(coefficients[index] * mu_eval[index], axis=1)
        scores = contributions + alpha * (y[index] - mu)
        estimate = float(np.sum(normalized * scores))
        return dict(estimate=estimate, mu=mu, alpha=alpha, contributions=contributions, scores=scores, f=f, g=g)

    full = compute(np.arange(n))
    estimate = full["estimate"]
    se = float(np.sqrt(np.sum((normalize_weights(weights) * (full["scores"] - estimate)) ** 2)))
    critical = float(norm.ppf(0.5 + conf_level / 2))
    jackknife = None
    lower, upper = estimate - critical * se, estimate + critical * se
    if inference == "jackknife":
        jackknife = vfold_jackknife([estimate], lambda index: [compute(index)["estimate"]], n, jackknife_folds, random_state)
        se = float(jackknife["standard_error"][0])
        critical = float(t.ppf(0.5 + conf_level / 2, jackknife["df"]))
        lower, upper = estimate - critical * se, estimate + critical * se
    elif inference == "bootstrap":
        if not np.isfinite(bootstrap_reps) or int(bootstrap_reps) != bootstrap_reps or bootstrap_reps < 2:
            raise ValueError("`bootstrap_reps` must be an integer of at least 2.")
        rng = np.random.default_rng(random_state)
        draws = np.array([compute(rng.integers(0, n, n))["estimate"] for _ in range(int(bootstrap_reps))])
        se = float(draws.std(ddof=1))
        lower, upper = np.quantile(draws, [(1 - conf_level) / 2, (1 + conf_level) / 2])
    estimates = pd.DataFrame([dict(estimand=name, estimate=estimate, std_error=se, lower=lower, upper=upper)])
    return LinearFunctionalFit(estimates, regression.copy(), riesz.copy(), full["mu"], full["alpha"], full["contributions"], full["scores"], {"regression": full["f"], "riesz": full["g"], "coefficients": coefficients.copy()}, jackknife, inference)

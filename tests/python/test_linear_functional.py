import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import t

from calibrateddml import calibrated_linear_functional
from calibrateddml.linear_functional import StepCalibrator, _pava


def fixture(n=120):
    rng = np.random.default_rng(23)
    x = rng.normal(size=n)
    pi = 1 / (1 + np.exp(-0.5 * x))
    a = rng.binomial(1, pi)
    mu = np.column_stack([x, x + 1])
    alpha = np.column_stack([-1 / (1 - pi), 1 / pi])
    y = x + a + rng.normal(scale=0.5, size=n)
    return y, mu[np.arange(n), a], alpha[np.arange(n), a], mu, alpha


def test_generalized_pava_matches_convex_quadratic_program():
    a = np.array([1, 2, 1, 3, 2, 1], dtype=float)
    b = np.array([-2, 4, -3, 2, 8, 0], dtype=float)
    actual = _pava(a, b)
    result = minimize(lambda u: np.sum(a * u**2 - 2 * b * u), np.zeros(len(a)),
        jac=lambda u: 2 * (a * u - b), constraints={"type": "ineq", "fun": np.diff},
        method="SLSQP", options={"ftol": 1e-12})
    assert result.success
    np.testing.assert_allclose(actual, result.x, atol=1e-6)


def test_step_extension_is_right_continuous_and_constant_at_tails():
    f = StepCalibrator(np.array([0, 1, 2]), np.array([-1, 0, 3]))
    np.testing.assert_equal(f(np.array([-2, 0, 0.9, 1, 1.9, 2, 3])), [-1, -1, -1, 0, 0, 3, 3])


def test_uncalibrated_functional_equals_aipw_and_does_not_mutate_inputs():
    y, mu, alpha, mu_eval, alpha_eval = fixture()
    before = [v.copy() for v in [mu, alpha, mu_eval, alpha_eval]]
    fit = calibrated_linear_functional(y, mu, alpha, [-1, 1], mu_eval, alpha_eval,
        calibration_method="none", inference="wald", name="ATE")
    scores = mu_eval[:, 1] - mu_eval[:, 0] + alpha * (y - mu)
    np.testing.assert_allclose(fit.summary()["estimate"], scores.mean())
    np.testing.assert_allclose(fit.scores_, scores)
    for original, value in zip(before, [mu, alpha, mu_eval, alpha_eval]):
        np.testing.assert_array_equal(original, value)


def test_pava_satisfies_both_empirical_calibration_equations_with_signed_riesz():
    y, mu, alpha, mu_eval, alpha_eval = fixture()
    weights = np.linspace(0.2, 2, len(y))
    fit = calibrated_linear_functional(y, mu, alpha, lambda p: p[:, 1] - p[:, 0],
        mu_eval, alpha_eval, sample_weight=weights, inference="wald")
    for value in np.unique(fit.calibrated_regression_):
        rows = fit.calibrated_regression_ == value
        np.testing.assert_allclose(np.sum(weights[rows] * (y[rows] - value)), 0, atol=1e-10)
    g = fit.calibration_["riesz"]
    evaluated = g(alpha_eval)
    for value in np.unique(fit.calibrated_riesz_):
        left = np.sum(weights * (fit.calibrated_riesz_ == value) * fit.calibrated_riesz_)
        right = np.sum(weights * ((evaluated[:, 1] == value).astype(float) - (evaluated[:, 0] == value).astype(float)))
        np.testing.assert_allclose(left, right, atol=1e-10)
    assert np.any(fit.calibrated_riesz_ < 0)
    assert np.any(fit.calibrated_riesz_ > 0)


def test_weighted_regression_mean_recovers_known_representer_identity():
    y, mu, _, _, _ = fixture()
    q = np.linspace(-2, 2, len(y))
    weights = np.linspace(0.2, 1.5, len(y))
    fit = calibrated_linear_functional(y, mu, q, q[:, None], sample_weight=weights, inference="wald")
    np.testing.assert_allclose(fit.calibrated_riesz_, q)
    np.testing.assert_allclose(fit.summary()["estimate"], np.average(q * y, weights=weights))


def test_ties_are_pooled_before_pava_and_zero_weight_rows_are_supported():
    y = np.array([2., 0., 8., 4., -100.])
    mu = np.array([0., 0., 1., 1., 8.])
    weights = np.array([1., 1., 1., 1., 0.])
    fit = calibrated_linear_functional(y, mu, np.ones(5), 1, sample_weight=weights, inference="wald")
    np.testing.assert_allclose(fit.calibrated_regression_[:4], [1, 1, 6, 6])
    np.testing.assert_allclose(fit.summary()["estimate"], 3.5)


def test_default_jackknife_uses_t9_and_keeps_functional_coefficients_fixed():
    y, mu, alpha, mu_eval, alpha_eval = fixture()
    calls = []

    def functional(values):
        calls.append(values.shape)
        return values[:, 1] - values[:, 0]

    fit = calibrated_linear_functional(y, mu, alpha, functional, mu_eval, alpha_eval, random_state=7)
    assert len(calls) == 5  # zero, two basis columns, and two linearity checks
    assert fit.jackknife_["df"] == 9
    tab = fit.summary().iloc[0]
    assert tab["upper"] == pytest.approx(tab["estimate"] + t.ppf(0.975, 9) * tab["std_error"])
    assert fit.confint().shape == (1, 3)


def test_bootstrap_refits_calibration_and_returns_intervals():
    y, mu, alpha, mu_eval, alpha_eval = fixture(n=60)
    fit = calibrated_linear_functional(y, mu, alpha, [-1, 1], mu_eval, alpha_eval,
        inference="bootstrap", bootstrap_reps=12, random_state=1)
    tab = fit.summary().iloc[0]
    assert np.isfinite(tab["std_error"]) and tab["std_error"] > 0
    assert tab["lower"] <= tab["upper"]


def test_rejects_nonlinear_functional_and_missing_evaluations():
    y, mu, alpha, mu_eval, alpha_eval = fixture()
    with pytest.raises(ValueError, match="row-wise linear"):
        calibrated_linear_functional(y, mu, alpha, lambda p: p[:, 0] ** 2)
    with pytest.raises(ValueError, match="same shape"):
        calibrated_linear_functional(y, mu, alpha, [-1, 1], mu_eval)
    with pytest.raises(ValueError, match="finite vector"):
        calibrated_linear_functional(y, mu[:-1], alpha, [1])

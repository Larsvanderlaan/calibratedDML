from pathlib import Path

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


def three_input_fixture():
    """A finite population shared with the R tests; no random draws."""
    path = Path(__file__).resolve().parents[1] / "fixtures" / "linear_functional_three_input.csv"
    rows = np.genfromtxt(path, delimiter=",", names=True)
    return {
        **{key: rows[key] for key in ("x", "a", "epsilon", "sample_weight", "y", "regression", "riesz")},
        "coefficients": np.column_stack([rows[f"c{j}"] for j in range(3)]),
        "regression_evaluations": np.column_stack([rows[f"mu_eval{j}"] for j in range(3)]),
        "riesz_evaluations": np.column_stack([rows[f"alpha_eval{j}"] for j in range(3)]),
    }


def fit_three_inputs(data, functional=None, **kwargs):
    return calibrated_linear_functional(
        data["y"], data["regression"], data["riesz"],
        data["coefficients"] if functional is None else functional,
        data["regression_evaluations"], data["riesz_evaluations"],
        sample_weight=data["sample_weight"], **kwargs,
    )


def interval_average_oracle(quadratic, linear):
    """The isotonic max-min formula, independent of the package's PAVA."""
    return np.array([
        max(
            min(linear[left:right + 1].sum() / quadratic[left:right + 1].sum()
                for right in range(j, len(quadratic)))
            for left in range(j + 1)
        )
        for j in range(len(quadratic))
    ])


def step_bin(knots, value):
    # Count knot comparisons directly, including constant extension at both tails.
    return max(0, sum(knot <= value for knot in knots) - 1)


def reference_step(knots, fitted, values):
    return np.array([fitted[step_bin(knots, value)] for value in values.flat]).reshape(values.shape)


def three_input_oracle(data, index=None):
    """Refit both finite quadratic problems without calling package helpers."""
    if index is None:
        index = np.arange(len(data["y"]))
    d = {key: value[index] for key, value in data.items()}
    weights = d["sample_weight"]
    models = {}
    for name in ("regression", "riesz"):
        knots = np.array(sorted(set(d[name][weights > 0])))
        quadratic = np.array([weights[d[name] == knot].sum() for knot in knots])
        if name == "regression":
            linear = np.array([(weights * d["y"])[d[name] == knot].sum() for knot in knots])
        else:
            linear = np.zeros(len(knots))
            for weight, coefficients, evaluations in zip(weights, d["coefficients"], d["riesz_evaluations"]):
                for coefficient, evaluation in zip(coefficients, evaluations):
                    linear[step_bin(knots, evaluation)] += weight * coefficient
        fitted = interval_average_oracle(quadratic, linear)
        models[name] = (knots, fitted, quadratic, linear)
    mu = reference_step(*models["regression"][:2], d["regression"])
    alpha = reference_step(*models["riesz"][:2], d["riesz"])
    mu_eval = reference_step(*models["regression"][:2], d["regression_evaluations"])
    alpha_eval = reference_step(*models["riesz"][:2], d["riesz_evaluations"])
    contributions = (d["coefficients"] * mu_eval).sum(axis=1)
    scores = contributions + alpha * (d["y"] - mu)
    return dict(estimate=np.average(scores, weights=weights), mu=mu, alpha=alpha,
                mu_eval=mu_eval, alpha_eval=alpha_eval, contributions=contributions,
                scores=scores, models=models)


@pytest.mark.parametrize("callable_functional", [False, True])
def test_three_input_signed_functional_uncalibrated_identity(callable_functional):
    d = three_input_fixture()
    before = {key: value.copy() for key, value in d.items()}
    functional = (lambda p: np.sum(d["coefficients"] * p, axis=1)) if callable_functional else None
    fit = fit_three_inputs(d, functional, calibration_method="none", inference="wald")
    # Direct algebra: the six (X,A) groups have these intercepts and slopes in epsilon.
    intercept = np.array([[3, 5, 13], [4, 6, 14]])[d["x"].astype(int), d["a"].astype(int)]
    expected_scores = intercept + (2 * d["a"] - 2) * d["epsilon"]
    np.testing.assert_allclose(fit.scores_, expected_scores, rtol=0, atol=1e-13)
    assert fit.summary().iloc[0]["estimate"] == pytest.approx(23 / 3)
    assert fit.summary().iloc[0]["std_error"] ** 2 == pytest.approx(908 / 1215)
    np.testing.assert_array_equal(fit.calibration_["coefficients"], d["coefficients"])
    for key in d:
        np.testing.assert_array_equal(d[key], before[key])


@pytest.mark.parametrize("correct_nuisance", ["regression", "riesz", "both"])
def test_three_input_population_riesz_identity_and_double_robustness(correct_nuisance):
    d = three_input_fixture()
    rows, treatments = np.arange(len(d["y"])), d["a"].astype(int)
    true_mu = d["x"][:, None] + np.array([1, 3, 5])
    true_alpha = 3 * d["coefficients"]  # P(A=a | X)=1/3 for every treatment.
    # Check the Riesz equality on a nonconstant test function at all three inputs.
    h = (d["x"][:, None] + np.array([2, -1, 3])) ** 2
    lhs = np.average(true_alpha[rows, treatments] * h[rows, treatments], weights=d["sample_weight"])
    rhs = np.average(np.sum(d["coefficients"] * h, axis=1), weights=d["sample_weight"])
    assert lhs == pytest.approx(rhs)
    assert np.average(np.sum(d["coefficients"] * true_mu, axis=1), weights=d["sample_weight"]) == pytest.approx(10)
    if correct_nuisance in ("regression", "both"):
        d["regression_evaluations"] = true_mu
        d["regression"] = true_mu[rows, treatments]
    if correct_nuisance in ("riesz", "both"):
        d["riesz_evaluations"] = true_alpha
        d["riesz"] = true_alpha[rows, treatments]
    fit = fit_three_inputs(d, calibration_method="none", inference="wald")
    assert fit.summary().iloc[0]["estimate"] == pytest.approx(10)


@pytest.mark.parametrize("off_grid", [False, True])
def test_three_input_pava_matches_oracle_and_both_calibration_equations(off_grid):
    d = three_input_fixture()
    if off_grid:
        # Exercise both tails and interior intervals, with signed row-specific masses.
        d["regression_evaluations"] += np.array([-0.25, 0.4, 4])
        d["riesz_evaluations"] += np.array([-0.25, 0.5, 4])
    fit = fit_three_inputs(d, lambda p: np.sum(d["coefficients"] * p, axis=1), inference="wald")
    oracle = three_input_oracle(d)
    np.testing.assert_allclose(fit.calibrated_regression_, oracle["mu"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(fit.calibrated_riesz_, oracle["alpha"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(fit.functional_values_, oracle["contributions"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(fit.scores_, oracle["scores"], rtol=0, atol=1e-12)
    assert fit.summary().iloc[0]["estimate"] == pytest.approx(oracle["estimate"])
    for name, (knots, expected, quadratic, linear) in oracle["models"].items():
        model = fit.calibration_[name]
        np.testing.assert_array_equal(model.knots, knots)
        np.testing.assert_allclose(model.fitted, expected, rtol=0, atol=1e-12)
        residual_prefix = np.cumsum(linear - quadratic * model.fitted)
        assert np.all(np.diff(model.fitted) >= 0)
        assert np.min(residual_prefix) >= -1e-12
        assert residual_prefix[-1] == pytest.approx(0, abs=1e-12)
        np.testing.assert_allclose(residual_prefix[:-1] * np.diff(model.fitted), 0, atol=1e-12)
    weights = d["sample_weight"]
    for value in np.unique(fit.calibrated_regression_):
        rows = fit.calibrated_regression_ == value
        assert np.sum(weights[rows] * (d["y"][rows] - value)) == pytest.approx(0, abs=1e-12)
    evaluated = fit.calibration_["riesz"](d["riesz_evaluations"])
    for value in np.unique(fit.calibrated_riesz_):
        left = np.sum(weights * fit.calibrated_riesz_ * (fit.calibrated_riesz_ == value))
        right = np.sum(weights[:, None] * d["coefficients"] * (evaluated == value))
        assert left == pytest.approx(right, abs=1e-12)
    if not off_grid:
        np.testing.assert_allclose(fit.calibration_["regression"].fitted, [1, 3, 23 / 5, 23 / 5])
        np.testing.assert_allclose(fit.calibration_["riesz"].fitted, [-4, 4, 4])
        intercept = np.array([[44, 12, 84], [74, 42, 82]])[d["x"].astype(int), d["a"].astype(int)] / 5
        np.testing.assert_allclose(fit.scores_, intercept + np.where(d["a"] == 0, -4, 4) * d["epsilon"])
        assert fit.summary().iloc[0]["estimate"] == pytest.approx(536 / 45)
        assert fit.summary().iloc[0]["std_error"] ** 2 == pytest.approx(274952 / 273375)


@pytest.mark.parametrize("folds", [10, 30])
def test_three_input_fixed_nuisance_jackknife_matches_independent_refits(folds):
    d = three_input_fixture()
    before = {key: value.copy() for key, value in d.items()}
    calls = []

    def functional(p):
        calls.append(p.shape)
        return np.sum(d["coefficients"] * p, axis=1)

    # The 10-fold case deliberately exercises the public default.
    kwargs = {} if folds == 10 else {"jackknife_folds": folds}
    fit = fit_three_inputs(d, functional, random_state=7, **kwargs)
    jackknife = fit.jackknife_
    assert calls == [(30, 3)] * 6  # Zero, three basis columns, two probes; no resample calls.
    assert jackknife["df"] == folds - 1
    np.testing.assert_array_equal(np.bincount(jackknife["fold_ids"]), np.full(folds, 30 // folds))
    draws, fixed_full_calibration = [], []
    for fold in range(folds):
        keep = np.flatnonzero(jackknife["fold_ids"] != fold)
        draws.append(three_input_oracle(d, keep)["estimate"])
        fixed_full_calibration.append(np.average(fit.scores_[keep], weights=d["sample_weight"][keep]))
    draws = np.array(draws)
    full = 536 / 45
    pseudo = folds * full - (folds - 1) * draws
    se = np.sqrt((folds - 1) / folds * np.sum((draws - draws.mean()) ** 2))
    np.testing.assert_allclose(jackknife["leaveout_estimates"][:, 0], draws, rtol=0, atol=1e-12)
    np.testing.assert_allclose(jackknife["pseudo_values"][:, 0], pseudo, rtol=0, atol=1e-12)
    np.testing.assert_allclose(jackknife["standard_error"], se, rtol=0, atol=1e-12)
    assert np.max(np.abs(draws - fixed_full_calibration)) > 0.01
    tab = fit.summary().iloc[0]
    assert tab["estimate"] == pytest.approx(full)
    assert tab["std_error"] == pytest.approx(se)
    np.testing.assert_allclose([tab["lower"], tab["upper"]], full + np.array([-1, 1]) * t.ppf(0.975, folds - 1) * se)
    if folds == 30:
        assert se == pytest.approx(1.3783388993802286, abs=1e-12)
    for key in d:
        np.testing.assert_array_equal(d[key], before[key])


def test_three_input_row_and_evaluation_column_permutations_preserve_results():
    d = three_input_fixture()
    original = fit_three_inputs(d, inference="wald")
    row_order = np.r_[np.arange(29, -1, -2), np.arange(28, -1, -2)]
    permuted = {key: value[row_order].copy() for key, value in d.items()}
    for key in ("coefficients", "regression_evaluations", "riesz_evaluations"):
        permuted[key] = permuted[key][:, [2, 0, 1]]
    fit = fit_three_inputs(permuted, lambda p: np.sum(permuted["coefficients"] * p, axis=1), inference="wald")
    np.testing.assert_allclose(fit.summary().iloc[0, 1:].to_numpy(dtype=float), original.summary().iloc[0, 1:].to_numpy(dtype=float))
    np.testing.assert_allclose(fit.scores_[np.argsort(row_order)], original.scores_, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fit.calibrated_riesz_[np.argsort(row_order)], original.calibrated_riesz_, rtol=0, atol=1e-12)


def test_three_input_rejects_functionals_that_mix_observations():
    d = three_input_fixture()
    # Linear in the whole matrix, but not a row-wise linear functional.
    with pytest.raises(ValueError, match="row-wise linear"):
        fit_three_inputs(d, lambda p: np.roll(np.sum(d["coefficients"] * p, axis=1), 1), inference="wald")

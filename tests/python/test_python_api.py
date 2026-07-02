from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError

from calibrateddml import AdaptiveCalibratedDML, CalibratedDML
from calibrateddml.calibration import fit_monotone_calibrator

NOMINAL_COVERAGE = 0.90


def make_binary_oracle_data(n: int = 320, seed: int = 1):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = -0.2 + 0.5 * x1 - 0.25 * x2
    pi1 = 1.0 / (1.0 + np.exp(-logits))
    a = rng.binomial(1, pi1)
    mu0 = 0.5 + 0.3 * x1
    tau = 0.8 + 0.2 * x2
    mu1 = mu0 + tau
    y = mu0 + a * tau + rng.normal(scale=0.4, size=n)
    mu_mat = np.column_stack([mu0, mu1])
    pi_mat = np.column_stack([1.0 - pi1, pi1])
    x = pd.DataFrame({"x1": x1, "x2": x2})
    truth = {
        "EY0": float(mu0.mean()),
        "EY1": float(mu1.mean()),
        "ATE": float(tau.mean()),
    }
    return x, a, y, mu_mat, pi_mat, truth


def make_binary_nonlinear_oracle_data(n: int = 320, seed: int = 3):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = 0.4 * np.sin(x1) + 0.35 * x2
    pi1 = 1.0 / (1.0 + np.exp(-logits))
    a = rng.binomial(1, pi1)
    mu0 = 0.2 + 0.8 * np.sin(x1) + 0.3 * x2**2
    tau = 0.7 + 0.25 * x1 - 0.15 * x2
    mu1 = mu0 + tau
    y = mu0 + a * tau + rng.normal(scale=0.4, size=n)
    mu_mat = np.column_stack([mu0, mu1])
    pi_mat = np.column_stack([1.0 - pi1, pi1])
    x = pd.DataFrame({"x1": x1, "x2": x2})
    truth = {
        "EY0": float(mu0.mean()),
        "EY1": float(mu1.mean()),
        "ATE": float(tau.mean()),
    }
    return x, a, y, mu_mat, pi_mat, truth


def make_weighted_binary_oracle_data(n: int = 320, seed: int = 11):
    x, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data(n=n, seed=seed)
    weights = np.exp(0.3 * x["x1"].to_numpy() - 0.15 * x["x2"].to_numpy())
    weights = weights / weights.mean()
    mu0 = mu_mat[:, 0]
    mu1 = mu_mat[:, 1]
    truth = {
        "EY0": float(np.average(mu0, weights=weights)),
        "EY1": float(np.average(mu1, weights=weights)),
        "ATE": float(np.average(mu1 - mu0, weights=weights)),
    }
    return x, a, y, mu_mat, pi_mat, weights, truth


def make_multiarm_oracle_data(n: int = 360, seed: int = 2):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    scores = np.column_stack(
        [
            0.2 + 0.3 * x1,
            -0.1 + 0.2 * x2,
            0.1 - 0.15 * x1 + 0.25 * x2,
        ]
    )
    scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    pi_mat = scores / scores.sum(axis=1, keepdims=True)
    a = np.array([rng.choice([0, 1, 2], p=row) for row in pi_mat], dtype=int)
    mu0 = 0.2 + 0.1 * x1
    mu1 = mu0 + 0.6
    mu2 = mu0 - 0.3 + 0.15 * x2
    y = np.choose(a, [mu0, mu1, mu2]) + rng.normal(scale=0.35, size=n)
    mu_mat = np.column_stack([mu0, mu1, mu2])
    x = pd.DataFrame({"x1": x1, "x2": x2})
    truth = {
        "EY0": float(mu0.mean()),
        "EY1": float(mu1.mean()),
        "EY2": float(mu2.mean()),
        "ATE1": float((mu1 - mu0).mean()),
        "ATE2": float((mu2 - mu0).mean()),
    }
    return x, a, y, mu_mat, pi_mat, truth


def make_weighted_multiarm_oracle_data(n: int = 360, seed: int = 13):
    x, a, y, mu_mat, pi_mat, _ = make_multiarm_oracle_data(n=n, seed=seed)
    weights = np.exp(0.25 * x["x1"].to_numpy() + 0.1 * x["x2"].to_numpy())
    weights = weights / weights.mean()
    truth = {
        "EY0": float(np.average(mu_mat[:, 0], weights=weights)),
        "EY1": float(np.average(mu_mat[:, 1], weights=weights)),
        "EY2": float(np.average(mu_mat[:, 2], weights=weights)),
        "ATE1": float(np.average(mu_mat[:, 1] - mu_mat[:, 0], weights=weights)),
        "ATE2": float(np.average(mu_mat[:, 2] - mu_mat[:, 0], weights=weights)),
    }
    return x, a, y, mu_mat, pi_mat, weights, truth


def extract_coverage_from_fit(fit, truth):
    estimate_table = fit.to_frame()
    covered = {}
    for name, truth_value in truth.items():
        if name.startswith("EY"):
            level = int(name.replace("EY", ""))
            row = estimate_table[(estimate_table["estimand_type"] == "potential_outcome") & (estimate_table["level"] == level)].iloc[0]
        else:
            level = 1 if name == "ATE" else int(name.replace("ATE", ""))
            row = estimate_table[(estimate_table["estimand_type"] == "contrast") & (estimate_table["level"] == level)].iloc[0]
        covered[name] = float(row["lower"] <= truth_value <= row["upper"])
    return covered


def mean_coverage(coverage):
    return float(np.mean(np.asarray(list(coverage.values()), dtype=float)))


def simulate_standard_coverage(data_factory, inference, n_rep, n, **kwargs):
    coverage = None
    for rep in range(n_rep):
        _, a, y, mu_mat, pi_mat, truth = data_factory(n=n, seed=1000 + rep)
        fit = CalibratedDML(
            control_level=0,
            inference=inference,
            conf_level=NOMINAL_COVERAGE,
            **kwargs,
        ).fit_from_nuisances(A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat)
        draw = extract_coverage_from_fit(fit, truth)
        if coverage is None:
            coverage = {key: 0.0 for key in draw}
        for key, value in draw.items():
            coverage[key] += value
    return {key: value / n_rep for key, value in coverage.items()}


def simulate_adaptive_coverage(data_factory, mode, n_rep, n, **kwargs):
    coverage = None
    for rep in range(n_rep):
        x, a, y, mu_mat, pi_mat, truth = data_factory(n=n, seed=2000 + rep)
        fit = AdaptiveCalibratedDML(
            control_level=0,
            mode=mode,
            conf_level=NOMINAL_COVERAGE,
            **kwargs,
        ).fit_from_nuisances(X=x, A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat)
        draw = extract_coverage_from_fit(fit, truth)
        if coverage is None:
            coverage = {key: 0.0 for key in draw}
        for key, value in draw.items():
            coverage[key] += value
    return {key: value / n_rep for key, value in coverage.items()}


def simulate_fitted_nuisance_coverage(data_factory, n_rep, n, outcome_model="linear", treatment_model="linear"):
    coverage = None
    for rep in range(n_rep):
        x, a, y, _, _, truth = data_factory(n=n, seed=3000 + rep)
        fit = CalibratedDML(
            control_level=0,
            outcome_model=outcome_model,
            treatment_model=treatment_model,
            conf_level=NOMINAL_COVERAGE,
            inference="wald",
            calibration_method="auto",
            n_folds=5,
            random_state=rep,
        ).fit(x, a, y)
        draw = extract_coverage_from_fit(fit, truth)
        if coverage is None:
            coverage = {key: 0.0 for key in draw}
        for key, value in draw.items():
            coverage[key] += value
    return {key: value / n_rep for key, value in coverage.items()}


def test_standard_fit_from_nuisances_binary_returns_expected_outputs():
    _, a, y, mu_mat, pi_mat, truth = make_binary_oracle_data()
    fit = CalibratedDML(control_level=0).fit_from_nuisances(
        A=a,
        y=y,
        mu_mat=pd.DataFrame(mu_mat, columns=[0, 1]),
        pi_mat=pd.DataFrame(pi_mat, columns=[0, 1]),
    )

    assert fit.control_level_ == 0
    assert fit.treatment_levels_ == [0, 1]
    assert list(fit.potential_outcomes_["level"]) == [0, 1]
    assert list(fit.contrasts_["level"]) == [1]
    assert np.allclose(fit.calibrated_pi_mat_.sum(axis=1), 1.0)
    assert abs(fit.potential_outcomes_.loc[0, "estimate"] - truth["EY0"]) < 0.15
    assert abs(fit.potential_outcomes_.loc[1, "estimate"] - truth["EY1"]) < 0.15
    assert abs(fit.contrasts_.loc[0, "estimate"] - truth["ATE"]) < 0.15


def test_weighted_standard_fit_from_nuisances_matches_manual_binary_formula():
    _, a, y, mu_mat, pi_mat, weights, _ = make_weighted_binary_oracle_data()
    fit = CalibratedDML(control_level=0, calibration_method="none").fit_from_nuisances(
        A=a,
        y=y,
        mu_mat=mu_mat,
        pi_mat=pi_mat,
        sample_weight=weights,
    )

    weights_norm = weights / weights.sum()
    mu_obs = mu_mat[np.arange(len(y)), a]
    score0 = mu_mat[:, 0] + ((a == 0) * (y - mu_obs) / pi_mat[:, 0])
    score1 = mu_mat[:, 1] + ((a == 1) * (y - mu_obs) / pi_mat[:, 1])

    assert np.isclose(fit.potential_outcomes_.loc[0, "estimate"], np.sum(weights_norm * score0))
    assert np.isclose(fit.potential_outcomes_.loc[1, "estimate"], np.sum(weights_norm * score1))
    assert np.isclose(fit.contrasts_.loc[0, "estimate"], np.sum(weights_norm * (score1 - score0)))


def test_weighted_standard_fit_from_nuisances_matches_weighted_multiarm_truth():
    _, a, y, mu_mat, pi_mat, weights, truth = make_weighted_multiarm_oracle_data()
    fit = CalibratedDML(control_level=0, inference="wald").fit_from_nuisances(
        A=a,
        y=y,
        mu_mat=mu_mat,
        pi_mat=pi_mat,
        sample_weight=weights,
        treatment_levels=[0, 1, 2],
    )
    estimates = fit.to_frame().set_index("estimand")["estimate"]
    assert abs(estimates["E[Y(0)]"] - truth["EY0"]) < 0.2
    assert abs(estimates["E[Y(1)]"] - truth["EY1"]) < 0.2
    assert abs(estimates["E[Y(2)]"] - truth["EY2"]) < 0.2
    assert abs(estimates["E[Y(1)] - E[Y(0)]"] - truth["ATE1"]) < 0.2
    assert abs(estimates["E[Y(2)] - E[Y(0)]"] - truth["ATE2"]) < 0.2


def test_estimator_requires_fit_before_results_access():
    fit = CalibratedDML(control_level=0)
    with pytest.raises(NotFittedError):
        fit.summary()
    with pytest.raises(NotFittedError):
        fit.confint()


def test_standard_estimator_defaults_to_jackknife_with_100_folds():
    fit = CalibratedDML(control_level=0)
    assert fit.inference == "jackknife"
    assert fit.jackknife_folds == 100


def test_estimator_round_trips_sklearn_style_params():
    fit = CalibratedDML(control_level=0, bootstrap_reps=25)
    params = fit.get_params()

    assert params["control_level"] == 0
    assert params["bootstrap_reps"] == 25

    fit.set_params(bootstrap_reps=50, calibration_method="none")
    assert fit.bootstrap_reps == 50
    assert fit.calibration_method == "none"


def test_standard_fit_from_nuisances_multiarm_reports_all_arm_means_and_contrasts():
    _, a, y, mu_mat, pi_mat, truth = make_multiarm_oracle_data()
    fit = CalibratedDML(control_level=0, inference="jackknife", jackknife_folds=5).fit_from_nuisances(
        A=a,
        y=y,
        mu_mat=mu_mat,
        pi_mat=pi_mat,
        treatment_levels=[0, 1, 2],
    )

    assert fit.treatment_levels_ == [0, 1, 2]
    assert fit.potential_outcomes_.shape[0] == 3
    assert fit.contrasts_.shape[0] == 2
    assert np.allclose(fit.calibrated_pi_mat_.sum(axis=1), 1.0)
    estimates = fit.to_frame().set_index("estimand")["estimate"]
    assert abs(estimates["E[Y(0)]"] - truth["EY0"]) < 0.2
    assert abs(estimates["E[Y(1)]"] - truth["EY1"]) < 0.2
    assert abs(estimates["E[Y(2)]"] - truth["EY2"]) < 0.2
    assert abs(estimates["E[Y(1)] - E[Y(0)]"] - truth["ATE1"]) < 0.2
    assert abs(estimates["E[Y(2)] - E[Y(0)]"] - truth["ATE2"]) < 0.2


def test_multiarm_supplied_nuisances_are_consistent_without_explicit_treatment_levels():
    _, a, y, mu_mat, pi_mat, _ = make_multiarm_oracle_data(seed=17)
    fit_implicit = CalibratedDML(control_level=0, calibration_method="none").fit_from_nuisances(
        A=a,
        y=y,
        mu_mat=mu_mat,
        pi_mat=pi_mat,
    )
    fit_explicit = CalibratedDML(control_level=0, calibration_method="none").fit_from_nuisances(
        A=a,
        y=y,
        mu_mat=mu_mat,
        pi_mat=pi_mat,
        treatment_levels=[0, 1, 2],
    )

    assert np.allclose(
        fit_implicit.to_frame()["estimate"].to_numpy(),
        fit_explicit.to_frame()["estimate"].to_numpy(),
    )


def test_standard_fit_smoke_with_internal_nuisance_fitting():
    x, a, y, _, _, _ = make_multiarm_oracle_data()
    fit = CalibratedDML(
        control_level=0,
        outcome_model="linear",
        treatment_model="linear",
        inference="wald",
        calibration_method="none",
        n_folds=3,
    ).fit(x, a, y)

    assert fit.nuisance_source_ == "fitted"
    assert fit.mu_mat_.shape == (len(y), 3)
    assert fit.pi_mat_.shape == (len(y), 3)
    assert np.allclose(fit.pi_mat_.sum(axis=1), 1.0)


def test_bootstrap_and_jackknife_run_with_fixed_nuisances():
    _, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data()
    fit_boot = CalibratedDML(
        control_level=0,
        inference="bootstrap",
        bootstrap_reps=40,
        random_state=123,
    ).fit_from_nuisances(A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat)
    fit_jack = CalibratedDML(
        control_level=0,
        inference="jackknife",
        jackknife_folds=5,
    ).fit_from_nuisances(A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat)

    assert np.isfinite(fit_boot.estimates_["lower"]).all()
    assert np.isfinite(fit_boot.estimates_["upper"]).all()
    assert np.isfinite(fit_jack.estimates_["std_error"]).all()


def test_binary_wald_uses_sieve_riesz_correction_by_default():
    _, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data(n=180, seed=21)
    options = {
        "basis_size_grid": [4, 8],
        "lambda_grid": [1e-4, 1e-2],
        "cv_folds": 3,
        "min_rows": 5,
    }

    corrected = CalibratedDML(
        control_level=0,
        inference="wald",
        calibration_method="none",
        random_state=31,
        wald_options=options,
    ).fit_from_nuisances(A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat)
    standard = CalibratedDML(
        control_level=0,
        inference="wald",
        calibration_method="none",
        wald_correction="none",
        random_state=31,
        wald_options=options,
    ).fit_from_nuisances(A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat)

    diagnostics = corrected.wald_diagnostics_
    assert diagnostics["applied"] is True
    assert diagnostics["wald_aux_method"] == "sieve_riesz"
    assert diagnostics["std_error_mode"] == "corrected_if"
    assert np.isfinite(corrected.contrasts_.loc[0, "std_error"])
    assert np.isclose(standard.contrasts_.loc[0, "std_error"], diagnostics["simple_wald_std_error"])
    assert standard.wald_diagnostics_["applied"] is False


def test_binary_wald_conservative_uses_max_of_standard_and_corrected_se():
    _, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data(n=180, seed=22)
    options = {
        "basis_size_grid": [4, 8],
        "lambda_grid": [1e-4, 1e-2],
        "cv_folds": 3,
        "min_rows": 5,
    }

    fit = CalibratedDML(
        control_level=0,
        inference="wald",
        calibration_method="none",
        random_state=32,
        wald_conservative=True,
        wald_options=options,
    ).fit_from_nuisances(A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat)

    diagnostics = fit.wald_diagnostics_
    expected = max(diagnostics["simple_wald_std_error"], diagnostics["corrected_if_std_error"])
    assert diagnostics["std_error_mode"] == "conservative"
    assert np.isclose(fit.contrasts_.loc[0, "std_error"], expected)
    assert np.isclose(diagnostics["selected_std_error"], expected)


def test_sieve_riesz_wald_supports_explicit_binary_labels():
    _, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data(n=160, seed=23)
    labels = np.where(a == 1, "treated", "control")
    mu_frame = pd.DataFrame({"treated": mu_mat[:, 1], "control": mu_mat[:, 0]})
    pi_frame = pd.DataFrame({"treated": pi_mat[:, 1], "control": pi_mat[:, 0]})

    fit = CalibratedDML(
        control_level="control",
        inference="wald",
        calibration_method="none",
        random_state=33,
        wald_options={"basis_size_grid": [4], "lambda_grid": [1e-3], "cv_folds": 3, "min_rows": 5},
    ).fit_from_nuisances(
        A=labels,
        y=y,
        mu_mat=mu_frame,
        pi_mat=pi_frame,
        treatment_levels=["treated", "control"],
    )

    assert fit.wald_diagnostics_["applied"] is True
    assert fit.contrasts_.loc[0, "level"] == "treated"
    assert fit.contrasts_.loc[0, "control_level"] == "control"
    assert np.isfinite(fit.contrasts_.loc[0, "std_error"])


def test_wald_correction_validation_and_multiarm_auto_fallback():
    _, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data(n=100, seed=24)
    with pytest.raises(ValueError, match="inference='wald'"):
        CalibratedDML(control_level=0, inference="bootstrap", wald_correction="sieve_riesz").fit_from_nuisances(
            A=a,
            y=y,
            mu_mat=mu_mat,
            pi_mat=pi_mat,
        )

    _, a3, y3, mu3, pi3, _ = make_multiarm_oracle_data(n=150, seed=25)
    auto = CalibratedDML(control_level=0, inference="wald", calibration_method="none").fit_from_nuisances(
        A=a3,
        y=y3,
        mu_mat=mu3,
        pi_mat=pi3,
    )
    assert auto.wald_diagnostics_["fallback_reason"] == "non_binary_treatment"

    with pytest.raises(ValueError, match="binary treatment"):
        CalibratedDML(
            control_level=0,
            inference="wald",
            calibration_method="none",
            wald_correction="sieve_riesz",
        ).fit_from_nuisances(A=a3, y=y3, mu_mat=mu3, pi_mat=pi3)


def test_sieve_riesz_wald_fallbacks_and_boundary_propensities_are_finite():
    _, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data(n=140, seed=26)
    fallback_fit = CalibratedDML(
        control_level=0,
        inference="wald",
        calibration_method="none",
        random_state=34,
        wald_options={
            "basis_size_grid": [4],
            "lambda_grid": [1e-3],
            "cv_folds": 2,
            "min_rows": 5,
            "min_unique_scores": 10_000,
        },
    ).fit_from_nuisances(A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat)
    assert fallback_fit.wald_diagnostics_["h_treated"]["fallback"] is True
    assert fallback_fit.wald_diagnostics_["q_control"]["fallback"] is True
    assert np.isfinite(fallback_fit.contrasts_.loc[0, "std_error"])

    boundary_pi = np.column_stack([
        np.where(a == 0, 0.998, 0.002),
        np.where(a == 1, 0.998, 0.002),
    ])
    boundary_fit = CalibratedDML(
        control_level=0,
        inference="wald",
        calibration_method="none",
        random_state=35,
        wald_options={"basis_size_grid": [4], "lambda_grid": [1e-3], "cv_folds": 2, "min_rows": 5},
    ).fit_from_nuisances(A=a, y=y, mu_mat=mu_mat, pi_mat=boundary_pi)
    assert np.isfinite(boundary_fit.contrasts_.loc[0, "std_error"])
    assert boundary_fit.wald_diagnostics_["h_treated"]["bound_upper"] == pytest.approx(40.0)


def test_sieve_riesz_wald_matches_r_reference_fixture():
    n = 80
    w = np.linspace(-1.0, 1.0, n)
    pi1 = 1.0 / (1.0 + np.exp(-(0.15 + 0.75 * w)))
    a = (((np.arange(n) * 7 + 3) % 11) < (3 + 4 * (w > 0))).astype(int)
    mu0 = 0.2 + 0.3 * w + 0.1 * w * w
    mu1 = mu0 + 0.6 + 0.2 * w
    y = mu0 + a * (mu1 - mu0) + 0.15 * np.sin(np.arange(n) * 1.7)

    fit = CalibratedDML(
        control_level=0,
        inference="wald",
        calibration_method="none",
        random_state=101,
        wald_options={"basis_size_grid": [4, 6], "lambda_grid": [1e-4, 1e-2], "cv_folds": 4, "min_rows": 5},
    ).fit_from_nuisances(
        A=a,
        y=y,
        mu_mat=np.column_stack([mu0, mu1]),
        pi_mat=np.column_stack([1.0 - pi1, pi1]),
    )

    assert fit.contrasts_.loc[0, "estimate"] == pytest.approx(0.5948840416978424)
    assert fit.contrasts_.loc[0, "std_error"] == pytest.approx(0.031816138714923135)
    assert fit.contrasts_.loc[0, "lower"] == pytest.approx(0.5325255556894626)
    assert fit.contrasts_.loc[0, "upper"] == pytest.approx(0.6572425277062222)
    assert fit.wald_diagnostics_["h_treated"]["basis_size"] == 4
    assert fit.wald_diagnostics_["q_treated"]["lambda"] == pytest.approx(0.01)


def test_calibration_auto_uses_smooth_isotonic_for_small_samples():
    x = np.linspace(0.0, 1.0, 40)
    y = x**2
    calibrator = fit_monotone_calibrator(x, y, method="auto")
    pred = calibrator.predict(x)

    assert calibrator.method == "smooth_isotonic"
    assert np.all(np.diff(pred) >= -1e-10)


def test_calibration_stratify_outcome_is_supported():
    _, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data()
    fit = CalibratedDML(
        control_level=0,
        calibration_method="isotonic",
        calibration_stratify="outcome",
    ).fit_from_nuisances(A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat)

    assert fit.calibration_["calibration_stratify"] == "outcome"
    assert np.allclose(fit.calibrated_pi_mat_.sum(axis=1), 1.0)


def test_fit_from_nuisances_validates_matrix_rows():
    _, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data()
    estimator = CalibratedDML(control_level=0)
    try:
        estimator.fit_from_nuisances(A=a, y=y, mu_mat=mu_mat[:-1], pi_mat=pi_mat)
    except ValueError as exc:
        assert "one row per observation" in str(exc)
    else:
        raise AssertionError("Expected fit_from_nuisances to reject mismatched nuisance rows.")


def test_fit_from_nuisances_rejects_invalid_control_level():
    _, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data()
    with pytest.raises(ValueError, match="control_level"):
        CalibratedDML(control_level=2).fit_from_nuisances(A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat)


def test_fit_from_nuisances_rejects_duplicate_treatment_levels():
    _, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data()
    with pytest.raises(ValueError, match="duplicates"):
        CalibratedDML(control_level=0).fit_from_nuisances(
            A=a,
            y=y,
            mu_mat=mu_mat,
            pi_mat=pi_mat,
            treatment_levels=[0, 0],
        )


def test_fit_from_nuisances_rejects_negative_propensity_entries():
    _, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data()
    broken = pi_mat.copy()
    broken[0, 0] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        CalibratedDML(control_level=0).fit_from_nuisances(A=a, y=y, mu_mat=mu_mat, pi_mat=broken)


def test_fit_rejects_bad_fold_ids():
    x, a, y, _, _, _ = make_binary_oracle_data()
    with pytest.raises(ValueError, match="at least two folds"):
        CalibratedDML(control_level=0, fold_ids=np.zeros(len(y), dtype=int)).fit(x, a, y)


def test_bootstrap_is_deterministic_with_fixed_seed():
    _, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data()
    fit1 = CalibratedDML(
        control_level=0,
        inference="bootstrap",
        bootstrap_reps=30,
        random_state=44,
    ).fit_from_nuisances(A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat)
    fit2 = CalibratedDML(
        control_level=0,
        inference="bootstrap",
        bootstrap_reps=30,
        random_state=44,
    ).fit_from_nuisances(A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat)

    pd.testing.assert_frame_equal(fit1.estimates_, fit2.estimates_)


def test_supports_custom_sklearn_estimators():
    x, a, y, _, _, _ = make_binary_oracle_data(n=180)
    fit = CalibratedDML(
        control_level=0,
        outcome_model=RandomForestRegressor(n_estimators=50, random_state=7),
        treatment_model=RandomForestClassifier(n_estimators=50, random_state=7),
        calibration_method="none",
        n_folds=3,
    ).fit(x, a, y)

    assert fit.nuisance_source_ == "fitted"
    assert fit.estimates_.shape[0] == 3


def test_adaptive_plugin_and_calibrated_rlearner_modes_run():
    x, a, y, mu_mat, pi_mat, truth = make_binary_oracle_data()

    plugin_fit = AdaptiveCalibratedDML(
        control_level=0,
        mode="plugin",
        inference="bootstrap",
        bootstrap_reps=30,
        random_state=12,
    ).fit_from_nuisances(X=x, A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat)
    rlearner_fit = AdaptiveCalibratedDML(
        control_level=0,
        mode="calibrated_rlearner",
        cate_model="linear",
        inference="jackknife",
        jackknife_folds=5,
    ).fit_from_nuisances(X=x, A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat)

    assert plugin_fit.adaptive_mode_ == "plugin"
    assert rlearner_fit.adaptive_mode_ == "calibrated_rlearner"
    assert abs(plugin_fit.contrasts_.loc[0, "estimate"] - truth["ATE"]) < 0.2
    assert abs(rlearner_fit.contrasts_.loc[0, "estimate"] - truth["ATE"]) < 0.25


def test_adaptive_constructor_no_longer_accepts_calibration_method():
    with pytest.raises(TypeError, match="calibration_method"):
        AdaptiveCalibratedDML(control_level=0, calibration_method="isotonic")


def test_adaptive_fit_from_nuisances_validates_probability_matrix():
    x, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data()
    broken = pi_mat.copy()
    broken[0, 1] = -0.05
    with pytest.raises(ValueError, match="non-negative"):
        AdaptiveCalibratedDML(control_level=0).fit_from_nuisances(
            X=x,
            A=a,
            y=y,
            mu_mat=mu_mat,
            pi_mat=broken,
        )


def test_adaptive_fit_normalizes_compact_stratify_inputs():
    x, a, y, _, _, _ = make_binary_oracle_data(n=180, seed=21)

    fit_true = AdaptiveCalibratedDML(
        control_level=0,
        stratify=True,
        outcome_model="linear",
        treatment_model="linear",
        inference="wald",
        n_folds=3,
        random_state=9,
    ).fit(x, a, y)
    fit_explicit = AdaptiveCalibratedDML(
        control_level=0,
        stratify=("outcome", "treatment"),
        outcome_model="linear",
        treatment_model="linear",
        inference="wald",
        n_folds=3,
        random_state=9,
    ).fit(x, a, y)
    fit_string = AdaptiveCalibratedDML(
        control_level=0,
        stratify="outcome",
        outcome_model="linear",
        treatment_model="linear",
        inference="wald",
        n_folds=3,
        random_state=9,
    ).fit(x, a, y)
    fit_tuple = AdaptiveCalibratedDML(
        control_level=0,
        stratify=("outcome",),
        outcome_model="linear",
        treatment_model="linear",
        inference="wald",
        n_folds=3,
        random_state=9,
    ).fit(x, a, y)

    pd.testing.assert_frame_equal(fit_true.estimates_, fit_explicit.estimates_)
    pd.testing.assert_frame_equal(fit_string.estimates_, fit_tuple.estimates_)


def test_adaptive_fit_rejects_invalid_inference_configuration():
    x, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data()

    with pytest.raises(ValueError, match="`inference`"):
        AdaptiveCalibratedDML(control_level=0, inference="bad").fit_from_nuisances(
            X=x,
            A=a,
            y=y,
            mu_mat=mu_mat,
            pi_mat=pi_mat,
        )

    with pytest.raises(ValueError, match="`conf_level`"):
        AdaptiveCalibratedDML(control_level=0, conf_level=1.0).fit_from_nuisances(
            X=x,
            A=a,
            y=y,
            mu_mat=mu_mat,
            pi_mat=pi_mat,
        )


def test_weighted_adaptive_fit_from_nuisances_runs_and_targets_weighted_ate():
    x, a, y, mu_mat, pi_mat, weights, truth = make_weighted_binary_oracle_data()
    fit = AdaptiveCalibratedDML(
        control_level=0,
        mode="calibrated_rlearner",
        inference="wald",
        cate_model="linear",
    ).fit_from_nuisances(
        X=x,
        A=a,
        y=y,
        mu_mat=mu_mat,
        pi_mat=pi_mat,
        sample_weight=weights,
    )

    assert abs(fit.contrasts_.loc[0, "estimate"] - truth["ATE"]) < 0.2


def test_estimators_reject_mismatched_observation_counts():
    x, a, y, mu_mat, pi_mat, _ = make_binary_oracle_data()

    with pytest.raises(ValueError, match="same number of observations"):
        CalibratedDML(control_level=0, n_folds=3).fit(x.iloc[:-1], a, y)

    with pytest.raises(ValueError, match="same number of observations"):
        AdaptiveCalibratedDML(control_level=0, n_folds=3).fit_from_nuisances(
            X=x.iloc[:-1],
            A=a,
            y=y,
            mu_mat=mu_mat,
            pi_mat=pi_mat,
        )


def test_fixed_fixture_matches_manual_binary_dml_formula():
    a = np.array([0, 1, 0, 1], dtype=int)
    y = np.array([1.0, 3.0, 2.0, 4.0])
    mu = np.array(
        [
            [1.1, 2.8],
            [1.3, 3.2],
            [1.9, 3.7],
            [2.1, 4.1],
        ]
    )
    pi = np.array(
        [
            [0.7, 0.3],
            [0.4, 0.6],
            [0.55, 0.45],
            [0.35, 0.65],
        ]
    )

    fit = CalibratedDML(control_level=0, calibration_method="none").fit_from_nuisances(A=a, y=y, mu_mat=mu, pi_mat=pi)

    mu_obs = mu[np.arange(len(y)), a]
    score0 = mu[:, 0] + ((a == 0) * (y - mu_obs) / pi[:, 0])
    score1 = mu[:, 1] + ((a == 1) * (y - mu_obs) / pi[:, 1])
    ate = score1.mean() - score0.mean()

    assert np.isclose(fit.potential_outcomes_.loc[0, "estimate"], score0.mean())
    assert np.isclose(fit.potential_outcomes_.loc[1, "estimate"], score1.mean())
    assert np.isclose(fit.contrasts_.loc[0, "estimate"], ate)


def test_oracle_standard_coverage_is_close_to_nominal_in_binary_settings_for_all_inference_modes():
    coverage_wald = simulate_standard_coverage(
        data_factory=make_binary_oracle_data,
        inference="wald",
        n_rep=28,
        n=600,
    )
    coverage_boot = simulate_standard_coverage(
        data_factory=make_binary_oracle_data,
        inference="bootstrap",
        n_rep=20,
        n=600,
        bootstrap_reps=50,
        random_state=7,
    )
    coverage_jack = simulate_standard_coverage(
        data_factory=make_binary_oracle_data,
        inference="jackknife",
        n_rep=20,
        n=600,
        jackknife_folds=10,
    )

    assert mean_coverage(coverage_wald) > 0.85
    assert mean_coverage(coverage_boot) > 0.68
    assert mean_coverage(coverage_jack) > 0.78


def test_oracle_standard_coverage_is_close_to_nominal_in_multiarm_settings_for_all_inference_modes():
    coverage_wald = simulate_standard_coverage(
        data_factory=make_multiarm_oracle_data,
        inference="wald",
        n_rep=24,
        n=700,
    )
    coverage_boot = simulate_standard_coverage(
        data_factory=make_multiarm_oracle_data,
        inference="bootstrap",
        n_rep=16,
        n=700,
        bootstrap_reps=40,
        random_state=9,
    )
    coverage_jack = simulate_standard_coverage(
        data_factory=make_multiarm_oracle_data,
        inference="jackknife",
        n_rep=16,
        n=700,
        jackknife_folds=10,
    )

    assert mean_coverage(coverage_wald) > 0.90
    assert mean_coverage(coverage_boot) > 0.82
    assert mean_coverage(coverage_jack) > 0.86


def test_adaptive_oracle_coverage_is_reasonable_across_clean_binary_dgps():
    coverage_plugin_linear = simulate_adaptive_coverage(
        data_factory=make_binary_oracle_data,
        mode="plugin",
        n_rep=16,
        n=600,
        inference="wald",
    )
    coverage_plugin_nonlinear = simulate_adaptive_coverage(
        data_factory=make_binary_nonlinear_oracle_data,
        mode="plugin",
        n_rep=16,
        n=600,
        inference="wald",
    )
    coverage_rlearner_linear = simulate_adaptive_coverage(
        data_factory=make_binary_oracle_data,
        mode="calibrated_rlearner",
        n_rep=16,
        n=600,
        inference="wald",
        cate_model="linear",
    )
    coverage_plugin_jackknife = simulate_adaptive_coverage(
        data_factory=make_binary_oracle_data,
        mode="plugin",
        n_rep=12,
        n=600,
        inference="jackknife",
        jackknife_folds=20,
    )
    coverage_rlearner_jackknife = simulate_adaptive_coverage(
        data_factory=make_binary_nonlinear_oracle_data,
        mode="calibrated_rlearner",
        n_rep=12,
        n=600,
        inference="jackknife",
        jackknife_folds=20,
        cate_model="linear",
    )

    assert mean_coverage(coverage_plugin_linear) > 0.90
    assert mean_coverage(coverage_plugin_nonlinear) > 0.82
    assert mean_coverage(coverage_rlearner_linear) > 0.82
    assert mean_coverage(coverage_plugin_jackknife) > 0.82
    assert mean_coverage(coverage_rlearner_jackknife) > 0.80


def test_weighted_standard_oracle_coverage_is_reasonable_in_binary_settings():
    coverage = None
    for rep in range(12):
        _, a, y, mu_mat, pi_mat, weights, truth = make_weighted_binary_oracle_data(n=600, seed=4000 + rep)
        fit = CalibratedDML(
            control_level=0,
            inference="wald",
            conf_level=NOMINAL_COVERAGE,
        ).fit_from_nuisances(A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat, sample_weight=weights)
        draw = extract_coverage_from_fit(fit, truth)
        if coverage is None:
            coverage = {key: 0.0 for key in draw}
        for key, value in draw.items():
            coverage[key] += value
    coverage = {key: value / 12 for key, value in coverage.items()}
    assert mean_coverage(coverage) > 0.80


def test_fitted_nuisance_wald_coverage_remains_reasonable_in_smaller_binary_and_multiarm_studies():
    coverage_binary = simulate_fitted_nuisance_coverage(
        data_factory=make_binary_oracle_data,
        n_rep=14,
        n=600,
        outcome_model="linear",
        treatment_model="linear",
    )
    coverage_multiarm = simulate_fitted_nuisance_coverage(
        data_factory=make_multiarm_oracle_data,
        n_rep=12,
        n=700,
        outcome_model="linear",
        treatment_model="linear",
    )

    assert mean_coverage(coverage_binary) > 0.68
    assert mean_coverage(coverage_multiarm) > 0.60

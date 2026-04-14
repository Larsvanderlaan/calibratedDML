from __future__ import annotations

import numpy as np
import pandas as pd

from calibrateddml import AdaptiveCalibratedDML, CalibratedDML

NOMINAL_COVERAGE = 0.90


def make_binary_oracle_data(n: int = 600, seed: int = 1):
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
    return pd.DataFrame({"x1": x1, "x2": x2}), a, y, np.column_stack([mu0, mu1]), np.column_stack([1.0 - pi1, pi1]), {
        "EY0": float(mu0.mean()),
        "EY1": float(mu1.mean()),
        "ATE": float(tau.mean()),
    }


def make_binary_nonlinear_oracle_data(n: int = 600, seed: int = 3):
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
    return pd.DataFrame({"x1": x1, "x2": x2}), a, y, np.column_stack([mu0, mu1]), np.column_stack([1.0 - pi1, pi1]), {
        "EY0": float(mu0.mean()),
        "EY1": float(mu1.mean()),
        "ATE": float(tau.mean()),
    }


def make_multiarm_oracle_data(n: int = 700, seed: int = 2):
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
    return pd.DataFrame({"x1": x1, "x2": x2}), a, y, np.column_stack([mu0, mu1, mu2]), pi_mat, {
        "EY0": float(mu0.mean()),
        "EY1": float(mu1.mean()),
        "EY2": float(mu2.mean()),
        "ATE1": float((mu1 - mu0).mean()),
        "ATE2": float((mu2 - mu0).mean()),
    }


def extract_coverage(fit, truth):
    table = fit.to_frame()
    covered = {}
    for key, truth_value in truth.items():
        if key.startswith("EY"):
            level = int(key.replace("EY", ""))
            row = table[(table["estimand_type"] == "potential_outcome") & (table["level"] == level)].iloc[0]
        else:
            level = 1 if key == "ATE" else int(key.replace("ATE", ""))
            row = table[(table["estimand_type"] == "contrast") & (table["level"] == level)].iloc[0]
        covered[key] = float(row["lower"] <= truth_value <= row["upper"])
    return covered


def summarize_coverages(draws):
    frame = pd.DataFrame(draws)
    summary = pd.DataFrame(
        {
            "coverage": frame.mean(axis=0),
            "nominal": NOMINAL_COVERAGE,
            "deviation": frame.mean(axis=0) - NOMINAL_COVERAGE,
        }
    )
    summary.loc["average", :] = [frame.to_numpy().mean(), NOMINAL_COVERAGE, frame.to_numpy().mean() - NOMINAL_COVERAGE]
    return summary


def run_standard(factory, inference, reps, **kwargs):
    draws = []
    for rep in range(reps):
        _, a, y, mu_mat, pi_mat, truth = factory(seed=1000 + rep)
        fit = CalibratedDML(control_level=0, inference=inference, conf_level=NOMINAL_COVERAGE, **kwargs).fit_from_nuisances(
            A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat
        )
        draws.append(extract_coverage(fit, truth))
    return summarize_coverages(draws)


def run_adaptive(factory, mode, reps, **kwargs):
    draws = []
    for rep in range(reps):
        x, a, y, mu_mat, pi_mat, truth = factory(seed=2000 + rep)
        fit = AdaptiveCalibratedDML(control_level=0, mode=mode, conf_level=NOMINAL_COVERAGE, **kwargs).fit_from_nuisances(
            X=x, A=a, y=y, mu_mat=mu_mat, pi_mat=pi_mat
        )
        draws.append(extract_coverage(fit, truth))
    return summarize_coverages(draws)


def run_fitted(factory, reps, outcome_model="linear", treatment_model="linear"):
    draws = []
    for rep in range(reps):
        x, a, y, _, _, truth = factory(seed=3000 + rep)
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
        draws.append(extract_coverage(fit, truth))
    return summarize_coverages(draws)


def main():
    studies = {
        "standard_binary_wald": run_standard(make_binary_oracle_data, "wald", reps=20),
        "standard_binary_bootstrap": run_standard(make_binary_oracle_data, "bootstrap", reps=16, bootstrap_reps=40, random_state=7),
        "standard_binary_jackknife": run_standard(make_binary_oracle_data, "jackknife", reps=16, jackknife_folds=10),
        "standard_multiarm_wald": run_standard(make_multiarm_oracle_data, "wald", reps=16),
        "standard_multiarm_bootstrap": run_standard(make_multiarm_oracle_data, "bootstrap", reps=12, bootstrap_reps=40, random_state=9),
        "standard_multiarm_jackknife": run_standard(make_multiarm_oracle_data, "jackknife", reps=12, jackknife_folds=10),
        "adaptive_plugin_linear": run_adaptive(make_binary_oracle_data, "plugin", reps=16, calibration_method="none", inference="wald"),
        "adaptive_plugin_nonlinear": run_adaptive(make_binary_nonlinear_oracle_data, "plugin", reps=16, calibration_method="none", inference="wald"),
        "adaptive_rlearner_linear": run_adaptive(make_binary_oracle_data, "calibrated_rlearner", reps=16, calibration_method="none", inference="wald", cate_model="linear"),
        "fitted_binary_wald": run_fitted(make_binary_oracle_data, reps=12, outcome_model="linear", treatment_model="linear"),
        "fitted_multiarm_wald": run_fitted(make_multiarm_oracle_data, reps=10, outcome_model="linear", treatment_model="linear"),
    }

    for name, summary in studies.items():
        print(f"\n## {name}")
        print(summary.round(3))


if __name__ == "__main__":
    main()

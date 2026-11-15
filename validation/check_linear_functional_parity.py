"""Compare the working-tree Python API with the installed R package.

First install the R package from this checkout with ``R CMD INSTALL .``.
Run ``python validation/check_linear_functional_parity.py``; use ``--rscript``
if Rscript is not on PATH. No files in the checkout are modified.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from calibrateddml import calibrated_linear_functional  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    fixture = ROOT / "tests/fixtures/linear_functional_three_input.csv"
    data = pd.read_csv(fixture)
    n = len(data)
    coefficients = data[[f"c{j}" for j in range(3)]].to_numpy()
    mu_eval = data[[f"mu_eval{j}" for j in range(3)]].to_numpy()
    alpha_eval = data[[f"alpha_eval{j}" for j in range(3)]].to_numpy()
    y, mu, alpha, weights = (
        data[column].to_numpy()
        for column in ("y", "regression", "riesz", "sample_weight")
    )

    def fit(method, index=None, inference="wald", folds=10):
        index = np.arange(n) if index is None else index
        return calibrated_linear_functional(
            y[index], mu[index], alpha[index], coefficients[index],
            mu_eval[index], alpha_eval[index], sample_weight=weights[index],
            calibration_method=method, inference=inference,
            jackknife_folds=folds, random_state=7,
        )

    expected = {}

    def add(prefix, values):
        for i, value in enumerate(np.asarray(values).ravel(), start=1):
            expected[f"{prefix}.{i}"] = float(value)

    def add_jackknife(prefix, estimate, draws):
        v = len(draws)
        pseudo = v * estimate - (v - 1) * np.asarray(draws)
        se = pseudo.std(ddof=1) / np.sqrt(v)
        add(prefix + ".leaveout", draws)
        add(prefix + ".pseudo", pseudo)
        add(prefix + ".summary", [se, v - 1, estimate - t.ppf(.975, v - 1) * se,
                                 estimate + t.ppf(.975, v - 1) * se])

    def add_native_jackknife(prefix, fitted, order=None):
        jk = fitted.jackknife_
        order = np.arange(len(jk["leaveout_estimates"])) if order is None else order
        tab = fitted.summary().iloc[0]
        add(prefix + ".leaveout", jk["leaveout_estimates"][order, 0])
        add(prefix + ".pseudo", jk["pseudo_values"][order, 0])
        add(prefix + ".summary", [jk["standard_error"][0], jk["df"],
                                 tab["lower"], tab["upper"]])
        np.testing.assert_allclose(tab["std_error"], jk["standard_error"][0])

    py_groups = None
    for method in ("none", "isotonic"):
        full = fit(method)
        add(method + ".wald", full.summary()[["estimate", "std_error", "lower", "upper"]])
        for name, values in (
            ("regression", full.calibrated_regression_),
            ("riesz", full.calibrated_riesz_),
            ("functional", full.functional_values_),
            ("scores", full.scores_),
            ("coefficients", full.calibration_["coefficients"]),
        ):
            add(method + "." + name, values)
        if method == "isotonic":
            for nuisance in ("regression", "riesz"):
                calibrator = full.calibration_[nuisance]
                add(method + "." + nuisance + ".knots", calibrator.knots)
                add(method + "." + nuisance + ".fitted", calibrator.fitted)
        ten = fit(method, inference="jackknife")
        py_groups = ten.jackknife_["fold_ids"]
        add_native_jackknife(method + ".python_groups", ten)
        loo = fit(method, inference="jackknife", folds=n)
        row_order = loo.jackknife_["fold_ids"]
        add_native_jackknife(method + ".loo", loo, row_order)

    with tempfile.TemporaryDirectory(prefix="linear-functional-parity-") as temp:
        temp = Path(temp)
        groups_path, output_path = temp / "python-groups.txt", temp / "r-results.csv"
        np.savetxt(groups_path, py_groups, fmt="%d")
        subprocess.run(
            [args.rscript, str(ROOT / "validation/linear_functional_parity.R"),
             str(fixture), str(groups_path), str(output_path)],
            check=True, cwd=ROOT,
        )
        with output_path.open(newline="") as stream:
            actual = {row["key"]: float(row["value"]) for row in csv.DictReader(stream)}

    # Compare R's native ten-fold result on its own deletion groups too.
    # Matching seeds do not mean matching random partitions across languages.
    for method in ("none", "isotonic"):
        groups = np.array([actual[f"{method}.r_groups.fold_ids.{i}"]
                           for i in range(1, n + 1)], dtype=int)
        add(method + ".r_groups.fold_ids", groups)
        draws = [fit(method, np.flatnonzero(groups != v)).summary().estimate[0]
                 for v in range(10)]
        add_jackknife(method + ".r_groups", fit(method).summary().estimate[0], draws)

    if actual.keys() != expected.keys():
        raise AssertionError(f"Mismatched result fields: {actual.keys() ^ expected.keys()}")
    for method in ("none", "isotonic"):
        keys = sorted(key for key in expected if key.startswith(method + "."))
        left = np.array([actual[key] for key in keys])
        right = np.array([expected[key] for key in keys])
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-10)
        print(f"{method}: {len(keys)} values agree; max absolute difference "
              f"{np.max(np.abs(left - right)):.3g}")
        for label in ("python_groups", "r_groups", "loo"):
            print(f"  {label}: SE = {actual[f'{method}.{label}.summary.1']:.15g}")
    print("PASS: R/Python fitted maps, row scores, Wald inference, ten-fold "
          "refits on both partitions, and leave-one-out inference agree.")


if __name__ == "__main__":
    main()

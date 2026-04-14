from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibrateddml import CalibratedDML


def make_binary_data(n: int = 180, seed: int = 101):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = 0.2 + 0.6 * x1 - 0.3 * x2
    pi1 = 1.0 / (1.0 + np.exp(-logits))
    a = rng.binomial(1, pi1)
    mu0 = 0.3 + 0.4 * x1
    tau = 0.7 + 0.1 * x2
    y = mu0 + a * tau + rng.normal(scale=0.4, size=n)
    x = pd.DataFrame({"x1": x1, "x2": x2})
    return x, a, y


def test_gam_backend_runs_when_available():
    pytest.importorskip("pygam")
    x, a, y = make_binary_data()

    fit = CalibratedDML(
        control_level=0,
        outcome_model="gam",
        treatment_model="gam",
        calibration_method="none",
        n_folds=3,
    ).fit(x, a, y)

    assert fit.estimates_.shape[0] == 3


def test_boosted_trees_backend_runs_when_available():
    pytest.importorskip("lightgbm")
    x, a, y = make_binary_data()

    fit = CalibratedDML(
        control_level=0,
        outcome_model="boosted_trees",
        treatment_model="boosted_trees",
        calibration_method="none",
        n_folds=3,
    ).fit(x, a, y)

    assert fit.estimates_.shape[0] == 3

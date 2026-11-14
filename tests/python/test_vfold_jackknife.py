import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from calibrateddml import CalibratedDML, AdaptiveCalibratedDML
from calibrateddml.jackknife import vfold_jackknife


def test_mean_pseudo_values_are_fold_means():
    y = np.arange(60, dtype=float) ** 2
    fit = vfold_jackknife([y.mean()], lambda keep: [y[keep].mean()], len(y), 6, 12)
    fold_means = np.array([y[fit["fold_ids"] == v].mean() for v in range(6)])
    np.testing.assert_allclose(fit["pseudo_values"][:, 0], fold_means)
    np.testing.assert_allclose(fit["standard_error"], fold_means.std(ddof=1) / np.sqrt(6))
    assert fit["df"] == 5


def data(n=101):
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(n, 2)))
    a = np.tile([0, 1], n // 2 + 1)[:n]
    y = x[0].to_numpy() + a + rng.normal(size=n)
    mu = np.column_stack([x[0], x[0] + 1])
    pi = np.full((n, 2), 0.5)
    return x, a, y, mu, pi


def test_jackknife_uses_t_and_ten_balanced_groups_independent_of_cross_fitting():
    x, a, y, mu, pi = data()
    fit = CalibratedDML(fold_ids=np.arange(len(a)) % 3, random_state=42).fit_from_nuisances(a, y, mu, pi)
    jk = fit.jackknife_
    counts = np.bincount(jk["fold_ids"])
    assert len(counts) == 10
    assert counts.max() - counts.min() <= 1
    assert jk["df"] == 9
    tab = fit.summary()
    critical = t.ppf(0.975, 9)
    np.testing.assert_allclose(tab["upper"], tab["estimate"] + critical * tab["std_error"])
    np.testing.assert_allclose(tab["lower"], tab["estimate"] - critical * tab["std_error"])
    np.testing.assert_allclose(jk["standard_error"], jk["pseudo_values"].std(axis=0, ddof=1) / np.sqrt(10))


def test_raw_nuisance_estimates_are_fitted_once(monkeypatch):
    import calibrateddml.core as core
    original = core._fit_nuisances
    calls = []

    def counted(*args, **kwargs):
        calls.append(len(kwargs["y"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "_fit_nuisances", counted)
    x, a, y, _, _ = data(n=120)
    fit = CalibratedDML(outcome_model="linear", treatment_model="linear", n_folds=3, random_state=7).fit(x, a, y)
    assert calls == [120]
    assert fit.jackknife_["leaveout_estimates"].shape == (10, 3)


@pytest.mark.parametrize("mode", ["plugin", "calibrated_rlearner"])
def test_adaptive_initial_learners_are_not_refitted_by_jackknife(monkeypatch, mode):
    import calibrateddml.adaptive as adaptive
    original_nuisance = adaptive._fit_nuisances
    original_cate = adaptive.fit_regressor
    counts = {"nuisance": 0, "cate": 0}

    def nuisance(*args, **kwargs):
        counts["nuisance"] += 1
        return original_nuisance(*args, **kwargs)

    def cate(*args, **kwargs):
        counts["cate"] += 1
        return original_cate(*args, **kwargs)

    monkeypatch.setattr(adaptive, "_fit_nuisances", nuisance)
    monkeypatch.setattr(adaptive, "fit_regressor", cate)
    x, a, y, _, _ = data(n=120)
    fit = AdaptiveCalibratedDML(mode=mode, outcome_model="linear", treatment_model="linear", cate_model="linear", n_folds=3, random_state=7).fit(x, a, y)
    assert counts == {"nuisance": 1, "cate": int(mode == "calibrated_rlearner")}
    assert fit.jackknife_["df"] == 9
    tab = fit.summary()
    np.testing.assert_allclose(tab["upper"], tab["estimate"] + t.ppf(0.975, 9) * tab["std_error"])


def test_invalid_number_of_groups():
    for folds in [1, 2.5, np.inf]:
        with pytest.raises(ValueError, match="jackknife_folds"):
            vfold_jackknife([1], lambda keep: [1], 20, folds)


def test_small_sample_uses_actual_group_count():
    fit = vfold_jackknife([2], lambda keep: [1], 3, 10)
    assert fit["df"] == 2
    assert len(fit["fold_ids"]) == 3

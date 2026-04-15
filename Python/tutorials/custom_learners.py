from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from calibrateddml import CalibratedDML

from tutorial_data import make_binary_data, print_table


def run_tutorial():
    x, a, y, _, _ = make_binary_data()

    fit_builtin = CalibratedDML(
        control_level=0,
        outcome_model="linear",
        treatment_model="linear",
        inference="wald",
        n_folds=3,
        random_state=4,
    ).fit(x, a, y)

    fit_custom = CalibratedDML(
        control_level=0,
        outcome_model=RandomForestRegressor(n_estimators=80, min_samples_leaf=5, random_state=5),
        treatment_model=RandomForestClassifier(n_estimators=80, min_samples_leaf=5, random_state=5),
        inference="wald",
        n_folds=3,
        random_state=5,
    ).fit(x, a, y)

    return {
        "builtin": fit_builtin.to_frame(),
        "custom": fit_custom.to_frame(),
    }


if __name__ == "__main__":
    results = run_tutorial()
    print_table("Built-in learner strings", results["builtin"], rows=3)
    print_table("Custom sklearn-compatible learners", results["custom"], rows=3)
    print("\ncustom learners tutorial completed")

from __future__ import annotations

from calibrateddml import CalibratedDML

from tutorial_data import make_binary_data, make_multiarm_data, print_table


def run_tutorial():
    x_binary, a_binary, y_binary, _, _ = make_binary_data()
    fit_wald = CalibratedDML(
        control_level=0,
        outcome_model="linear",
        treatment_model="linear",
        inference="wald",
        n_folds=3,
        random_state=1,
    ).fit(x_binary, a_binary, y_binary)

    fit_bootstrap = CalibratedDML(
        control_level=0,
        outcome_model="linear",
        treatment_model="linear",
        inference="bootstrap",
        bootstrap_reps=25,
        n_folds=3,
        random_state=2,
    ).fit(x_binary, a_binary, y_binary)

    x_multi, a_multi, y_multi, _, _ = make_multiarm_data()
    fit_multiarm = CalibratedDML(
        control_level=0,
        outcome_model="linear",
        treatment_model="linear",
        inference="jackknife",
        jackknife_folds=5,
        n_folds=3,
        random_state=3,
    ).fit(x_multi, a_multi, y_multi)

    return {
        "binary_wald": fit_wald.to_frame(),
        "binary_bootstrap": fit_bootstrap.confint(),
        "multiarm_jackknife": fit_multiarm.to_frame(),
    }


if __name__ == "__main__":
    results = run_tutorial()
    print_table("Binary raw-data fit (Wald)", results["binary_wald"], rows=3)
    print_table("Binary raw-data fit (bootstrap intervals)", results["binary_bootstrap"], rows=1)
    print_table("Multi-arm raw-data fit (jackknife)", results["multiarm_jackknife"])
    print("\nstandard workflows tutorial completed")

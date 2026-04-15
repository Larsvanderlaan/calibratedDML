from __future__ import annotations

from calibrateddml import CalibratedDML

from tutorial_data import make_binary_data, make_multiarm_data, print_table


def run_tutorial():
    _, a_binary, y_binary, mu_binary, pi_binary = make_binary_data()
    fit_binary = CalibratedDML(
        control_level=0,
        inference="bootstrap",
        bootstrap_reps=25,
        random_state=6,
    ).fit_from_nuisances(
        A=a_binary,
        y=y_binary,
        mu_mat=mu_binary,
        pi_mat=pi_binary,
        treatment_levels=[0, 1],
    )

    _, a_multi, y_multi, mu_multi, pi_multi = make_multiarm_data()
    mu_perm = mu_multi.loc[:, [2, 0, 1]]
    pi_perm = pi_multi.loc[:, [2, 0, 1]]
    fit_multiarm = CalibratedDML(
        control_level=0,
        inference="wald",
        random_state=7,
    ).fit_from_nuisances(
        A=a_multi,
        y=y_multi,
        mu_mat=mu_perm,
        pi_mat=pi_perm,
        treatment_levels=[0, 1, 2],
    )

    return {
        "binary_supplied": fit_binary.to_frame(),
        "multiarm_supplied": fit_multiarm.to_frame(),
    }


if __name__ == "__main__":
    results = run_tutorial()
    print_table("Supplied binary nuisance matrices", results["binary_supplied"], rows=3)
    print_table("Supplied multi-arm nuisance matrices", results["multiarm_supplied"])
    print("\nsupplied nuisances tutorial completed")

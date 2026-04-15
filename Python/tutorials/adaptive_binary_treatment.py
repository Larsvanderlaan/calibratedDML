from __future__ import annotations

from calibrateddml import AdaptiveCalibratedDML

from tutorial_data import make_binary_data, print_table


def run_tutorial():
    x, a, y, mu_mat, pi_mat = make_binary_data()

    fit_rlearner = AdaptiveCalibratedDML(
        control_level=0,
        mode="calibrated_rlearner",
        cate_model="linear",
        inference="wald",
        random_state=8,
    ).fit_from_nuisances(
        X=x,
        A=a,
        y=y,
        mu_mat=mu_mat,
        pi_mat=pi_mat,
    )

    fit_plugin = AdaptiveCalibratedDML(
        control_level=0,
        mode="plugin",
        inference="wald",
        random_state=9,
    ).fit_from_nuisances(
        X=x,
        A=a,
        y=y,
        mu_mat=mu_mat,
        pi_mat=pi_mat,
    )

    return {
        "calibrated_rlearner": fit_rlearner.to_frame(),
        "plugin": fit_plugin.to_frame(),
    }


if __name__ == "__main__":
    results = run_tutorial()
    print_table("Adaptive calibrated R-learner", results["calibrated_rlearner"], rows=3)
    print_table("Adaptive plug-in mode", results["plugin"], rows=3)
    print("\nadaptive tutorial completed")

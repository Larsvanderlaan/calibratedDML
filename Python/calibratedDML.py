from calibrate_nuisances import *
import pandas as pd
import numpy as np
from scipy.stats import norm


def calibratedDML(A, Y, mu_mat, pi_mat, weights=None, control_level=0, treatment_levels=None, alpha = 0.05):
    """
    Computes the ICDML (Isotonic Calibrated Debiased Machine Learning) estimator
    for causal inference, estimating the average treatment effect (ATE) as a linear functional
    of the outcome regression.

    Args:
        A (np.array): Integer array indicating treatment assignment for each individual
                      (e.g., 0 for control, 1 for treatment, etc.).
        Y (np.array): Observed outcomes.
        mu_mat (np.array): Matrix of predicted outcomes for each treatment level, where
                           each column corresponds to a treatment level.
        pi_mat (np.array): Matrix of propensity scores for each treatment level, where each
                           column corresponds to a treatment level.
        weights (np.array, optional): Sample weights. Default is None.
        control_level (int, optional): Index of the control group level in `A`. Default is 0.
         If None, then counterfactual mean outcomes are estimated instead.
        treatment_levels (list, optional): List of treatment level indices to estimate ATEs for.
                                           If None, uses all levels in `pi_mat` except `control_level`.

    Returns:
        np.array: Estimated causal effect, tau_n, using the ICDML estimator for each treatment level.
    """
    if treatment_levels is None:
        treatment_levels = [i for i in range(pi_mat.shape[1]) if i != control_level]

    levels_to_calibrate = [control_level] + treatment_levels if control_level is not None else treatment_levels
    # Calibrate outcome regression and propensity score matrices
    mu_star_mat = calibrate_outcome_regression(Y, mu_mat, A, weights=weights, treatment_levels=levels_to_calibrate)[
        'mu_star']
    pi_star_mat = calibrate_inverse_probability_weights(A, pi_mat, weights=weights, treatment_levels=levels_to_calibrate)['pi_star']

    # Reference values for the control level
    mu_reference = mu_star_mat[:, control_level] if control_level is not None else np.zeros_like(Y)
    pi_reference = pi_star_mat[:, control_level] if control_level is not None else np.ones_like(Y)

        # Calculate estimates for each treatment level
    estimates = {}
    lower_bounds = []
    upper_bounds = []
    for level in treatment_levels:
        pi_trt = pi_star_mat[:, level]
        mu_trt = mu_star_mat[:, level]

        # Calculate alpha_star and tau_star for the current treatment level
        alpha_star = (A == level) / pi_trt - (A == control_level) / pi_reference
        tau_star = np.mean(mu_trt - mu_reference + alpha_star * (Y - mu_star_mat[:, A]))
        se = np.std(mu_trt - mu_reference + alpha_star * (Y - mu_star_mat[:, A])) / np.sqrt(len(Y))
        
        # Wald-type confidence intervals
        z_alpha = norm.ppf(1 - alpha / 2)
        lower_bound = tau_star - z_alpha * se
        upper_bound = tau_star + z_alpha * se
        
        estimates[level] = tau_star
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    estimates = np.array(list(estimates.values()))
    # Set row names
    estimands = []
    for level in treatment_levels:
        if control_level is not None:
            estimands.append(f"E[Y({level})] - E[Y({control_level})]")
        else:
            estimands.append(f"E[Y({level})]")

    results = pd.DataFrame({
        'estimand': estimands,
        'estimate': estimates,
        'lower_bound': lower_bounds,
        'upper_bound': upper_bounds
    })

    return results


def calibratedDML_bootstrap(A, Y, mu_mat, pi_mat, weights=None, control_level=0, treatment_levels=None, nboot=1000, alpha=0.05):
    """
    Computes a bootstrap-assisted version of the ICDML estimator to obtain a confidence interval
    for the causal effect, leveraging calibrated debiased machine learning.

    Args:
        A (np.array): Integer array indicating treatment assignment for each individual
                      (e.g., 0 for control, 1 for treatment, etc.).
        Y (np.array): Observed outcomes.
        mu_mat (np.array): Matrix of predicted outcomes for each treatment level, where
                           each column corresponds to a treatment level.
        pi_mat (np.array): Matrix of propensity scores for each treatment level, where each
                           column corresponds to a treatment level.
        weights (np.array, optional): Sample weights. Default is None.
        control_level (int, optional): Index of the control group level in `A`. Default is 0.
        treatment_levels (list, optional): List of treatment level indices to estimate ATEs for.
                                           If None, uses all levels in `pi_mat` except `control_level`.
        nboot (int, optional): Number of bootstrap samples. Default is 1000.
        alpha (float, optional): Significance level for the confidence interval. Default is 0.05.

    Returns:
        dict: Contains the estimated causal effect and confidence interval:
              - tau_n: Estimated causal effects using ICDML for each treatment level.
              - CI: Bootstrap confidence interval for each treatment level.
    """
    # Calculate the ICDML point estimate
    results = calibratedDML(A, Y, mu_mat, pi_mat, weights=weights, control_level=control_level, treatment_levels=treatment_levels)
    estimands = results['estimand'].values
    estimates = results['estimate'].values
    # Perform bootstrap resampling
    bootstrap_estimates = []
    for _ in range(nboot):
        # Random sampling with replacement
        bootstrap_indices = np.random.choice(len(A), size=len(A), replace=True)

        # Subset data based on bootstrap indices
        A_boot = A[bootstrap_indices]
        Y_boot = Y[bootstrap_indices]
        mu_mat_boot = mu_mat[bootstrap_indices]
        pi_mat_boot = pi_mat[bootstrap_indices]
        weights_boot = weights[bootstrap_indices] if weights is not None else None

        # Compute ICDML for bootstrap sample
        estimates_boot = calibratedDML(A_boot, Y_boot, mu_mat_boot, pi_mat_boot, weights=weights_boot, control_level=control_level, treatment_levels=treatment_levels)
        bootstrap_estimates.append(estimates_boot['estimate'].values)

    # Convert bootstrap estimates to array
    bootstrap_estimates = np.array(bootstrap_estimates)

    # Compute confidence intervals for each treatment level
    bootstrap_differences = bootstrap_estimates - estimates
    lower_bounds = estimates + np.quantile(bootstrap_differences, alpha / 2, axis=0)
    upper_bounds = estimates + np.quantile(bootstrap_differences, 1 - alpha / 2, axis=0)


    # Create DataFrame with results
    results = pd.DataFrame({
        'estimand': estimands,
        'estimate': estimates,
        'lower_bound': lower_bounds,
        'upper_bound': upper_bounds
    })

    return results

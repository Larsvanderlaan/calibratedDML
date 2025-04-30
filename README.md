# Doubly robust inference for ATEs using calibration

## Overview of Calibrated Debiased Machine learning

The package provides `R` and `Python` implementations of `Calibrated Debiased Machine learning` (calibrated DML) for average treatment effects using isotonic calibration from our paper on [Automatic Doubly Robust Inference for Linear Functionals via Calibrated Debiased Machine Learning](https://arxiv.org/pdf/2411.02771v1).

The `calibratedDML` package implements tools for automatic doubly robust inference for linear functionals in causal inference problems. Many causal estimands of interest, such as average causal effects of static, dynamic, and stochastic interventions, can be expressed as linear functionals of the outcome regression function. Calibrated DML leverages modern machine learning to provide debiased estimators that are doubly robust asymptotically normality, thus providing not only doubly robust consistency but also facilitating doubly robust inference (e.g., confidence intervals and hypothesis tests). The calibrated DML estimator we implement integrates cross-fitting, isotonic calibration, and debiased machine learning estimation (AIPW). It maintains asymptotic normality when either the outcome regression or the propensity score is estimated sufficiently well, allowing the other to be estimated at arbitrarily slow rates or even inconsistently. For inference, we use a bootstrap-assisted approach to construct doubly robust confidence intervals.

Currently, the supported estimands are the Average Treatment Effect (ATE). The code can be modified to accommodate more general parameters and please reach out to me if you would like me to implement other estimands.

## Installation

To install the calibratedDML package in R, run the following command. For an example of using this package, see the `Vignette.rmd` file.

```r
devtools::install_github("Larsvanderlaan/calibratedDML")
```

To run the Python code, download the code files in the `Python` folder and refer to the `vignette.ipynb` file for usage instructions.

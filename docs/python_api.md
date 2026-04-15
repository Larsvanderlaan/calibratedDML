# Python API Reference

## Primary imports

```python
from calibrateddml import CalibratedDML, AdaptiveCalibratedDML
```

Primary import path:

- `calibrateddml`

Legacy compatibility alias:

- `calibratedDML`

For new code, prefer `calibrateddml`.

## `CalibratedDML`

Main constructor:

```python
CalibratedDML(
    control_level=0,
    outcome_model="lasso",
    treatment_model="lasso",
    stratify=("outcome", "treatment"),
    calibration_method="auto",
    calibration_stratify=None,
    inference="wald",
    conf_level=0.95,
    bootstrap_reps=200,
    jackknife_folds=10,
    random_state=None,
    n_folds=5,
    fold_ids=None,
)
```

Main methods:

- `fit(X, A, y, sample_weight=None)`
- `fit_from_nuisances(A, y, mu_mat, pi_mat, sample_weight=None, treatment_levels=None)`
- `summary()`
- `to_frame()`
- `confint()`

Main fitted attributes:

- `estimates_`
- `potential_outcomes_`
- `contrasts_`
- `treatment_levels_`
- `control_level_`
- `mu_mat_`
- `pi_mat_`
- `calibrated_mu_mat_`
- `calibrated_pi_mat_`
- `calibration_`
- `nuisance_source_`

## Built-in models

Outcome and treatment model names:

- `mean`
- `linear`
- `lasso`
- `random_forest`
- `gam`
- `boosted_trees`
- `auto`

Core installs support:

- `mean`
- `linear`
- `lasso`
- `random_forest`

Optional extras:

- `gam` requires `pygam`
- `boosted_trees` requires `lightgbm`

Advanced users may also pass sklearn-compatible:

- regressors for `outcome_model`
- probabilistic classifiers for `treatment_model`

## Calibration

Supported calibration methods:

- `"auto"`
- `"isotonic"`
- `"smooth_isotonic"`
- `"none"`

Calibration controls:

- `calibration_method`
- `calibration_stratify`

`calibration_stratify="outcome"` applies only to outcome nuisance calibration. Treatment nuisance calibration is already coordinate-wise by treatment level.

## Inference

Supported inference methods:

- `"wald"`
- `"bootstrap"`
- `"jackknife"`

For `"bootstrap"` and `"jackknife"`, nuisance matrices are held fixed and only calibration plus debiasing are refit.

## `AdaptiveCalibratedDML`

Main constructor:

```python
AdaptiveCalibratedDML(
    control_level=0,
    mode="plugin",
    outcome_model="lasso",
    treatment_model="lasso",
    cate_model="lasso",
    stratify=("outcome", "treatment"),
    calibration_stratify=None,
    inference="jackknife",
    conf_level=0.95,
    bootstrap_reps=200,
    jackknife_folds=20,
    random_state=None,
    n_folds=5,
    fold_ids=None,
)
```

Current status:

- experimental
- binary-treatment only
- modes limited to `"plugin"` and `"calibrated_rlearner"`
- isotonic calibration only
- no adaptive `calibration_method` parameter

Adaptive methods are intentionally documented separately from the standard estimator because they are super-efficient procedures. In this context, super-efficient means they can achieve lower mean-squared error and smaller realized variance than regular calibrated DML at favorable data-generating distributions. The tradeoff is that standard-error estimation is harder, so interval coverage can be less stable in practice.

Recommended usage:

- `CalibratedDML` is the recommended default for general use
- `AdaptiveCalibratedDML` is an advanced option
- `calibrated_rlearner` is especially appealing when the truth may be close to homogeneous but some heterogeneity is possible
- consult Benkeser and van der Laan, "A Super-Efficient Estimator of the Average Treatment Effect," and van der Laan, Carone, Luedtke, and van der Laan, "Adaptive debiased machine learning using data-driven model selection techniques," before relying on adaptive confidence intervals

For the standard estimator, see van der Laan, Luedtke, and Carone, "Doubly robust inference via calibration."

## Legacy wrappers

Legacy wrappers remain available for compatibility:

- `calibratedDML(...)`
- `calibratedDML_bootstrap(...)`

They are deprecated for new code. Prefer the estimator classes above.

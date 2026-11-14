# Python API Reference

## Primary imports

```python
from calibrateddml import CalibratedDML, AdaptiveCalibratedDML
```

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
    inference="jackknife",
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

- `"jackknife"`
- `"wald"`
- `"bootstrap"`

For `"bootstrap"` and `"jackknife"`, nuisance matrices are held fixed and only calibration plus debiasing are refit.

`"jackknife"` is the default, with 10 groups and Student-t critical values with 9 degrees of freedom. Jackknife and bootstrap intervals account for the calibration step and support inference when one nuisance estimator is misspecified. Use `"wald"` when both nuisance estimators are consistent, even if one converges slowly. See [confidence intervals](https://larsvanderlaan.github.io/calibratedDML/causal-setup.html#inference) for details.

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
    jackknife_folds=10,
    random_state=None,
    n_folds=5,
    fold_ids=None,
)
```

This experimental estimator supports binary treatment with `"plugin"` and `"calibrated_rlearner"` modes. Both use isotonic calibration; there is no adaptive `calibration_method` parameter.

The calibrated R-learner estimates the CATE, calibrates its predictions, and averages the calibrated effects. It can reduce variance when treatment-effect heterogeneity is well described by the learned summary. See the [adaptive guide](https://larsvanderlaan.github.io/calibratedDML/adaptive.html) for the working model and [Adaptive debiased machine learning](https://arxiv.org/abs/2307.12544) for the theory.

## General linear functionals

```python
calibrated_linear_functional(
    y, regression, riesz, functional,
    regression_evaluations=None, riesz_evaluations=None,
    sample_weight=None, calibration_method="isotonic",
    inference="jackknife", jackknife_folds=10,
    bootstrap_reps=200, conf_level=0.95, random_state=None,
    name="linear_functional",
)
```

Import `calibrated_linear_functional` from `calibrateddml`. The observed-input
regression and Riesz predictions are vectors of length n. Evaluation matrices
have n rows and one column for each input used by the functional. Use the same
column order for both nuisances and the same cross-fitting model for every
evaluation in a row.

`functional` is a row-wise linear callable, such as
`lambda values: values[:, 1] - values[:, 0]`, or a coefficient vector or
matrix. PAVA minimizes the regression squared-error loss and the quadratic
Riesz loss. The one-step estimator uses the calibrated regression at the
functional inputs and the calibrated Riesz at the observed inputs.

The returned object supports `summary()`, `to_frame()`, and `confint()` and
stores calibrated predictions and calibration maps. Its default 10-fold
jackknife holds the initial predictions fixed. See the [ATE example](https://larsvanderlaan.github.io/calibratedDML/examples.html#linear-functional).

## Legacy wrappers

Legacy wrappers remain available for compatibility:

- `calibratedDML(...)`
- `calibratedDML_bootstrap(...)`

They are deprecated for new code. Prefer the estimator classes above.

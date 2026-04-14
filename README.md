# calibratedDML

`calibratedDML` provides calibrated debiased machine learning estimators for categorical treatments, with a native Python interface for mean potential outcomes and treatment-vs-control contrasts.

The Python package currently supports:

- categorical treatments with an explicit `control_level`
- mean potential outcomes for every arm, `E[Y(a)]`
- all treatment-vs-control contrasts, `E[Y(a)] - E[Y(control)]`
- flexible nuisance handling:
  - fit nuisances internally with built-in Python models
  - pass sklearn-compatible regressors and probabilistic classifiers
  - provide cross-fitted nuisance matrices directly
- modular calibration with `isotonic`, `smooth_isotonic`, `auto`, or `none`
- `wald`, `bootstrap`, and `jackknife` inference
- an experimental binary-treatment adaptive layer with plug-in and calibrated R-learner modes

## Installation

The native Python package uses a lean core dependency set:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`

Optional extras are available for richer built-in learners:

```bash
pip install calibratedDML
pip install 'calibratedDML[gam]'
pip install 'calibratedDML[boosted]'
pip install 'calibratedDML[dev]'
```

The import path is `calibrateddml`, with a compatibility alias at `calibratedDML`.

The current release posture should be read as a strong `0.x` beta rather than a finished `1.0`.

## Python quickstart

```python
from calibrateddml import CalibratedDML

fit = CalibratedDML(
    control_level=0,
    outcome_model="lasso",
    treatment_model="lasso",
    stratify=("outcome", "treatment"),
    calibration_method="auto",
    inference="wald",
    random_state=123,
)

fit.fit(X, A, y)
fit.summary()
fit.confint()
```

Defaults are chosen for a fast, strong first pass:

- `outcome_model = "lasso"`
- `treatment_model = "lasso"`
- `stratify = ("outcome", "treatment")`
- `calibration_method = "auto"`

For multi-arm treatment, the default treatment path fits armwise one-vs-rest nuisance models and then maps predictions to valid multinomial probabilities.

Python supports the same core estimator contract:

- `CalibratedDML.fit(X, A, y, sample_weight=None)`
- `CalibratedDML.fit_from_nuisances(A, y, mu_mat, pi_mat, sample_weight=None, treatment_levels=None)`
- `summary()`, `to_frame()`, and `confint()`

The main result attributes are:

- `estimates_`
- `potential_outcomes_`
- `contrasts_`
- `treatment_levels_`
- `control_level_`
- `mu_mat_`, `pi_mat_`
- `calibrated_mu_mat_`, `calibrated_pi_mat_`

Built-in Python model names are:

- `mean`
- `linear`
- `lasso`
- `random_forest`
- `gam`
- `boosted_trees`
- `auto`

Core installs support `mean`, `linear`, `lasso`, and `random_forest`. `gam` requires `pygam`, and `boosted_trees` requires `lightgbm`.

Primary import story:

- package name: `calibratedDML`
- primary import path: `calibrateddml`
- compatibility alias: `calibratedDML`

For new code, prefer `import calibrateddml`.

## Supplying nuisance estimates directly

`mu_mat` should have one column per treatment level for `E[Y | A = a, W]`, and `pi_mat` should have one column per treatment level for `P(A = a | W)`. Columns may be named and will be aligned to the declared treatment levels. Supplied propensity matrices must be non-negative and have positive row sums.

```python
from calibrateddml import CalibratedDML

fit = CalibratedDML(
    control_level=0,
    inference="bootstrap",
    bootstrap_reps=200,
    random_state=123,
).fit_from_nuisances(
    A=A,
    y=y,
    mu_mat=mu_mat,
    pi_mat=pi_mat,
    treatment_levels=[0, 1, 2],
)
```

## Calibration controls

Calibration is modular and sits between nuisance estimation and debiasing:

- `calibration_method = "auto"` uses `smooth_isotonic` for smaller calibration samples and stepwise isotonic otherwise
- `calibration_stratify = "outcome"` calibrates outcome nuisance coordinates within the observed treatment arm
- treatment calibration is already done coordinate-wise by treatment level

## Inference

The package supports:

- `inference = "wald"`: analytic influence-function intervals
- `inference = "bootstrap"`: bootstrap intervals with nuisances held fixed and calibration plus debiasing refit
- `inference = "jackknife"`: delete-a-group jackknife intervals with nuisances held fixed and calibration plus debiasing refit

## Adaptive binary-treatment estimators

The native Python package currently exposes an experimental binary adaptive layer through `AdaptiveCalibratedDML` with:

- `mode="plugin"`
- `mode="calibrated_rlearner"`

These adaptive estimators are super-efficient methods in the sense of Benkeser and van der Laan, "A Super-Efficient Estimator of the Average Treatment Effect," and van der Laan, Carone, Luedtke, and van der Laan, "Adaptive debiased machine learning using data-driven model selection techniques." Here, "super-efficient" means the estimator can have substantially smaller variance and lower mean-squared error than regular semiparametric estimators at favorable data-generating distributions, sometimes yielding tighter point estimates than standard calibrated DML.

That extra efficiency comes with a practical tradeoff: uncertainty quantification is harder. Standard errors and confidence intervals can be less stable, and empirical coverage may be worse even when point estimates look excellent.

Practical guidance:

- `CalibratedDML` is the recommended default for general use
- `AdaptiveCalibratedDML` is an advanced option
- `calibrated_rlearner` is especially appealing when the truth may be close to homogeneous but some heterogeneity is possible
- users should consult those adaptive-method references before relying on adaptive intervals in practice

For standard calibrated DML, the main reference is van der Laan, Luedtke, and Carone, "Doubly robust inference via calibration."

Adaptive calibration now uses isotonic regression internally. `smooth_isotonic` and `none` are not supported for adaptive methods.

Example:

```python
from calibrateddml import AdaptiveCalibratedDML

fit = AdaptiveCalibratedDML(
    control_level=0,
    mode="plugin",
    calibration_method="isotonic",
    inference="jackknife",
    jackknife_folds=20,
    random_state=123,
)

fit.fit(X, A, y)
```

This layer is binary-treatment only and should be treated as experimental in Python for now.

## More docs

- [Python quickstart](docs/python_quickstart.md)
- [Python API reference](docs/python_api.md)
- [Python common workflows](docs/python_workflows.md)

## R package

The repository also contains the R package. The R interface is documented separately and is not the primary path for Python users.

## Status

The Python estimator core now targets:

- explicit multi-arm treatment handling
- modular nuisance -> calibrate -> DML flow
- production-oriented defaults
- reproducible inference behavior

The next steps after release hardening are deeper reproducibility harnessing, richer optional-backend validation, and the package website.

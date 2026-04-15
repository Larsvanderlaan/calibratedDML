# calibratedDML

`calibratedDML` implements calibrated doubly robust estimators for causal
inference with categorical treatments. The package targets:

- mean potential outcomes, `E[Y(a)]`
- treatment-versus-control contrasts, `E[Y(a)] - E[Y(control)]`
- workflows that either fit nuisance models internally or start from
  cross-fitted nuisance estimates

Python and R interfaces are both included in this repository.

## Documentation

- Website: [larsvanderlaan.github.io/calibratedDML](https://larsvanderlaan.github.io/calibratedDML/)
- Main paper: [Doubly robust inference via calibration](https://arxiv.org/abs/2411.02771)
- Companion package: [ppi_aipw](https://larsvanderlaan.github.io/ppi-aipw/index.html)

## Repository layout

The repository contains both the Python and R packages, plus tutorials, docs,
and paper-reproduction material.

- `src/calibrateddml/`: Python package source
- `src/calibratedDML.py`: Python compatibility shim for the historical import path
- `R/`, `man/`, `vignettes/`: R package source, reference docs, and R vignettes
- `Python/tutorials/`: Python tutorial notebooks and script mirrors
- `docs/`: built website pages for the project site
- `examples/`: small runnable examples
- `tests/`: Python and R package tests
- `validation/`: focused coverage and validation scripts for inference behavior
- `paper_experiment_scripts/`, `paper_experiment_results/`, `paper_data/`: paper reproduction code, outputs, and datasets

For Python package development, use `src/` as the source of truth. The
top-level `Python/` directory is for tutorial material and legacy compatibility
helpers, not the main packaged implementation.

## Installation

### Python

Install from PyPI:

```bash
pip install calibratedDML
```

Optional extras:

```bash
pip install 'calibratedDML[gam]'
pip install 'calibratedDML[boosted]'
pip install 'calibratedDML[dev]'
```

The package name on PyPI is `calibratedDML`. For imports, prefer:

```python
import calibrateddml
```

A compatibility import path at `calibratedDML` is also available.

### R

Install from GitHub:

```r
remotes::install_github("Larsvanderlaan/calibratedDML")
```

The R package can work with built-in learners and can also integrate with
`sl3` or `SuperLearner` when those packages are available.

## Python quickstart

```python
from calibrateddml import CalibratedDML

fit = CalibratedDML(
    control_level=0,
    outcome_model="lasso",
    treatment_model="lasso",
    calibration_method="auto",
    random_state=123,
)

fit.fit(X, A, y)
fit.summary()
fit.confint()
```

Main Python entry points:

- `CalibratedDML.fit(X, A, y, sample_weight=None)`
- `CalibratedDML.fit_from_nuisances(A, y, mu_mat, pi_mat, sample_weight=None, treatment_levels=None)`

Common result accessors:

- `summary()`
- `to_frame()`
- `confint()`

Built-in Python model names:

- `mean`
- `linear`
- `lasso`
- `random_forest`
- `gam`
- `boosted_trees`
- `auto`

Core installs support `mean`, `linear`, `lasso`, and `random_forest`. The
`gam` option requires `pygam`, and `boosted_trees` requires `lightgbm`.

## R quickstart

```r
library(calibratedDML)

fit <- calibrated_dml(
  data = df,
  outcome = "Y",
  treatment = "A",
  covariates = c("W1", "W2", "W3"),
  control_level = 0,
  outcome_model = "lasso",
  treatment_model = "lasso",
  calibration_method = "auto"
)

summary(fit)
confint(fit)
```

Main R entry points:

- `calibrated_dml(...)`
- `calibrated_dml_from_nuisances(...)`

The R interface supports the same standard estimator class as Python, including
multi-arm treatment, direct nuisance input, and `wald`, `bootstrap`, and
`jackknife` inference.

## Supplying nuisance estimates

Both interfaces support direct nuisance input.

- `mu_mat` should contain one column per treatment level for `E[Y | A = a, W]`
- `pi_mat` should contain one column per treatment level for `P(A = a | W)`
- nuisance estimates should usually be cross-fitted

## Calibration and inference

Calibration sits between nuisance estimation and debiasing.

Standard calibrated DML supports:

- `calibration_method = "auto"`
- `calibration_method = "isotonic"`
- `calibration_method = "smooth_isotonic"`
- `calibration_method = "none"`

Inference options:

- `inference = "jackknife"` with `jackknife_folds = 100` is the default for standard calibrated DML
- `inference = "wald"`
- `inference = "bootstrap"`

Practical guidance:

- Use the default jackknife intervals for standard calibrated DML.
- Use Wald when both nuisance estimators are consistent, even if one converges arbitrarily slowly.
- Use bootstrap when you want another valid resampling interval and can afford the extra computation.

## Adaptive binary-treatment methods

The repository also includes adaptive binary-treatment estimators through:

- Python: `AdaptiveCalibratedDML`
- R: `adaptive_calibrated_dml()`

Adaptive methods should be treated as experimental. They target the ATE
through a learned and calibrated treatment-effect summary and have a narrower,
more delicate inferential scope than standard calibrated DML.
Adaptive estimation always uses isotonic calibration internally.

Documented adaptive modes:

- `mode = "calibrated_rlearner"`
- `mode = "plugin"`

For most users, `CalibratedDML` and `calibrated_dml()` remain the default
entry points.

## Status

Current release posture:

- Python package version: `0.1.0`
- R package version: `0.1.0`
- standard calibrated DML is the primary supported workflow
- adaptive binary-treatment methods are experimental

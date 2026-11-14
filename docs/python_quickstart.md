# Python Quickstart

`calibrateddml` is the primary Python import path for the package.

## Standard estimator from raw data

```python
from calibrateddml import CalibratedDML

fit = CalibratedDML(
    control_level=0,
    outcome_model="lasso",
    treatment_model="lasso",
    stratify=("outcome", "treatment"),
    calibration_method="auto",
    random_state=123,
)

fit.fit(X, A, y)
results = fit.summary()
intervals = fit.confint()
```

The package cross-fits the outcome and treatment models, calibrates their predictions, and estimates arm means and treatment contrasts.

## Estimator from supplied nuisances

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

Supply cross-fitted nuisance predictions from models you have already trained.

`mu_mat` should contain one column per treatment level for `E[Y | A=a, W]`.
`pi_mat` should contain one column per treatment level for `P(A=a | W)`.

## Defaults

- `outcome_model="lasso"`
- `treatment_model="lasso"`
- `stratify=("outcome", "treatment")`
- `calibration_method="auto"`
- `inference="jackknife"`
- `jackknife_folds=100`

For multi-arm treatment, the default treatment path fits one-vs-rest propensity models and normalizes them to valid multinomial probabilities.

Inference guide:

- Use the default jackknife intervals for standard calibrated DML.
- Use Wald when both nuisance estimators are consistent, even if one converges slowly.
- Use `inference="bootstrap"` for bootstrap intervals.

## Adaptive estimator

```python
from calibrateddml import AdaptiveCalibratedDML

fit = AdaptiveCalibratedDML(
    control_level=0,
    mode="plugin",
    inference="jackknife",
    jackknife_folds=20,
    random_state=123,
)

fit.fit(X, A, y)
```

`AdaptiveCalibratedDML` estimates binary-treatment effects with `mode="plugin"` or `mode="calibrated_rlearner"`. Both use isotonic calibration. These methods are experimental.

The calibrated R-learner can reduce variance when treatment effects are nearly constant or well described by a learned summary. See the [adaptive guide](https://larsvanderlaan.github.io/calibratedDML/adaptive.html) for the model and the role of treatment overlap. For the standard estimator and doubly robust confidence intervals, see [Doubly robust inference via calibration](https://arxiv.org/abs/2411.02771).

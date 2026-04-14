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
    inference="wald",
    random_state=123,
)

fit.fit(X, A, y)
results = fit.summary()
intervals = fit.confint()
```

Use this path when you want the package to fit cross-fitted nuisance models for you.

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

Use this path when you already have cross-fitted nuisance predictions.

`mu_mat` should contain one column per treatment level for `E[Y | A=a, W]`.
`pi_mat` should contain one column per treatment level for `P(A=a | W)`.

## Defaults

- `outcome_model="lasso"`
- `treatment_model="lasso"`
- `stratify=("outcome", "treatment")`
- `calibration_method="auto"`
- `inference="wald"`

For multi-arm treatment, the default treatment path fits one-vs-rest propensity models and normalizes them to valid multinomial probabilities.

## Adaptive estimator

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

The adaptive Python API is binary-treatment only, experimental, limited to `mode="plugin"` and `mode="calibrated_rlearner"`, and uses isotonic calibration internally.

`CalibratedDML` is the recommended default for general use. `AdaptiveCalibratedDML` is an advanced option. In particular, `calibrated_rlearner` is especially appealing when the truth may be close to homogeneous but some heterogeneity is possible. These adaptive estimators are super-efficient methods, so they can have lower realized variance and lower MSE than standard calibrated DML in favorable settings, but inference is harder and interval coverage can be less stable. For adaptive inference details, see Benkeser and van der Laan, "A Super-Efficient Estimator of the Average Treatment Effect," and van der Laan, Carone, Luedtke, and van der Laan, "Adaptive debiased machine learning using data-driven model selection techniques." For standard calibrated DML, see van der Laan, Luedtke, and Carone, "Doubly robust inference via calibration."

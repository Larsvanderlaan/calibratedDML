# Python Common Workflows

## 1. Fit nuisances internally

Use this when you want the package to manage cross-fitting and nuisance estimation:

```python
from calibrateddml import CalibratedDML

fit = CalibratedDML(
    control_level=0,
    outcome_model="lasso",
    treatment_model="lasso",
    random_state=123,
)
fit.fit(X, A, y)
```

This is the simplest user-facing workflow.

## 2. Supply nuisance matrices directly

Use this when nuisance estimation already happens elsewhere:

```python
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

This is the preferred path for parity checks, reproducibility, and paper-style experiments.

## 3. Use sklearn-compatible custom models

You may pass sklearn-compatible estimators:

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from calibrateddml import CalibratedDML

fit = CalibratedDML(
    control_level=0,
    outcome_model=RandomForestRegressor(n_estimators=200, random_state=123),
    treatment_model=RandomForestClassifier(n_estimators=200, random_state=123),
    random_state=123,
)
fit.fit(X, A, y)
```

For treatment models, the estimator must support `predict_proba`.

## 4. Choose calibration explicitly

```python
fit = CalibratedDML(
    control_level=0,
    calibration_method="smooth_isotonic",
    calibration_stratify="outcome",
)
```

Use explicit calibration settings when you want to avoid the default `"auto"` behavior.

## 5. Use adaptive binary-treatment methods

```python
from calibrateddml import AdaptiveCalibratedDML

fit = AdaptiveCalibratedDML(
    control_level=0,
    mode="calibrated_rlearner",
    cate_model="linear",
    calibration_method="isotonic",
    inference="jackknife",
    jackknife_folds=20,
    random_state=123,
)
fit.fit(X, A, y)
```

Current adaptive Python scope:

- binary treatment only
- experimental
- `plugin` and `calibrated_rlearner` only
- isotonic calibration only

Practical guidance:

- Adaptive methods are super-efficient: they can deliver lower variance and lower mean-squared error than standard calibrated DML in favorable settings.
- That same nonregular behavior makes standard errors harder to estimate, so interval coverage can be poorer.
- `CalibratedDML` is the recommended default for general use.
- `AdaptiveCalibratedDML` is an advanced option.
- `calibrated_rlearner` is especially appealing when the truth may be close to homogeneous but some heterogeneity is possible.
- Consult Benkeser and van der Laan, "A Super-Efficient Estimator of the Average Treatment Effect," plus van der Laan, Carone, Luedtke, and van der Laan, "Adaptive debiased machine learning using data-driven model selection techniques," before relying on adaptive confidence intervals.

For the standard estimator, the main methodological reference is van der Laan, Luedtke, and Carone, "Doubly robust inference via calibration."

## 6. Read results

The main result tables are:

- `fit.summary()`
- `fit.to_frame()`
- `fit.confint()`

The full fitted object also stores nuisance and calibration artifacts for debugging and reproducibility:

- `mu_mat_`, `pi_mat_`
- `calibrated_mu_mat_`, `calibrated_pi_mat_`
- `calibration_`
- `nuisance_source_`

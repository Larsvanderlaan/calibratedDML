# Python examples

## 1. Fit nuisances internally

Fit the outcome and treatment models with cross-fitting:

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

## 2. Supply nuisance matrices directly

Supply cross-fitted predictions from your own models:

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

Use one column per treatment level in each matrix and align the columns with `treatment_levels`.

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

Set `calibration_method` to choose isotonic or smooth isotonic calibration, or `"none"` to omit calibration.

## 5. Use adaptive binary-treatment methods

```python
from calibrateddml import AdaptiveCalibratedDML

fit = AdaptiveCalibratedDML(
    control_level=0,
    mode="calibrated_rlearner",
    cate_model="linear",
    inference="jackknife",
    jackknife_folds=20,
    random_state=123,
)
fit.fit(X, A, y)
```

The experimental adaptive estimator supports binary treatment, with `plugin` and `calibrated_rlearner` modes. Both use isotonic calibration. The calibrated R-learner can reduce variance when treatment effects are nearly constant or well described by a learned summary; see the [adaptive guide](https://larsvanderlaan.github.io/calibratedDML/adaptive.html).

## 6. Read results

The main result tables are:

- `fit.summary()`
- `fit.to_frame()`
- `fit.confint()`

The fitted object also stores the original and calibrated nuisance predictions:

- `mu_mat_`, `pi_mat_`
- `calibrated_mu_mat_`, `calibrated_pi_mat_`
- `calibration_`
- `nuisance_source_`

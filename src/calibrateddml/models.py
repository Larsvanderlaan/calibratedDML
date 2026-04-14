from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from ._utils import as_feature_frame, design_matrix, empirical_probability_matrix, normalize_probability_matrix, resolve_weights


class MeanRegressor:
    def fit(self, X, y, sample_weight=None):
        weights = resolve_weights(sample_weight, len(y))
        self.value_ = float(np.average(np.asarray(y, dtype=float), weights=weights))
        return self

    def predict(self, X):
        X = as_feature_frame(X)
        return np.repeat(self.value_, len(X))


class EmpiricalClassifier:
    def fit(self, X, y, sample_weight=None):
        y = pd.Series(np.asarray(y).astype(str))
        weights = resolve_weights(sample_weight, len(y))
        self.classes_ = list(pd.unique(y))
        totals = np.array([weights[y == cls].sum() for cls in self.classes_], dtype=float)
        if totals.sum() <= 0:
            totals = np.repeat(1.0, len(self.classes_))
        self.probs_ = totals / totals.sum()
        return self

    def predict_proba(self, X):
        X = as_feature_frame(X)
        return np.tile(self.probs_[None, :], (len(X), 1))


@dataclass
class ModelSpec:
    name: str
    kind: str
    direct_multiclass: bool
    estimator_factory: callable


def _fit_model(model, X, y, sample_weight=None):
    X_dm = design_matrix(X)
    try:
        model.fit(X_dm, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X_dm, y)
    return model


def _predict_regression(model, X):
    return np.asarray(model.predict(design_matrix(X)), dtype=float)


def _predict_positive_class(model, X):
    proba = np.asarray(model.predict_proba(design_matrix(X)), dtype=float)
    if proba.ndim != 2:
        raise ValueError("Treatment models must return class probabilities.")
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, -1]


def _predict_multiclass(model, X):
    proba = np.asarray(model.predict_proba(design_matrix(X)), dtype=float)
    return normalize_probability_matrix(proba)


def _optional_pygam():
    try:
        from pygam import LinearGAM, LogisticGAM
    except ImportError as exc:
        raise ImportError("Model 'gam' requires the optional 'pygam' package.") from exc
    return LinearGAM, LogisticGAM


def _optional_lightgbm():
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:
        raise ImportError("Model 'boosted_trees' requires the optional 'lightgbm' package.") from exc
    return LGBMClassifier, LGBMRegressor


def builtin_regression_names() -> tuple[str, ...]:
    return ("mean", "linear", "lasso", "random_forest", "gam", "boosted_trees", "auto")


def builtin_classification_names() -> tuple[str, ...]:
    return ("mean", "linear", "lasso", "random_forest", "gam", "boosted_trees", "auto")


def resolve_regressor(model, random_state=None):
    if isinstance(model, str):
        return make_regression_spec(model, random_state=random_state)
    if hasattr(model, "fit") and hasattr(model, "predict"):
        return ModelSpec(
            name=getattr(model, "__class__", type(model)).__name__,
            kind="regression",
            direct_multiclass=False,
            estimator_factory=lambda: clone(model),
        )
    raise ValueError("Unsupported `outcome_model` specification.")


def resolve_classifier(model, random_state=None):
    if isinstance(model, str):
        return make_classification_spec(model, random_state=random_state)
    if hasattr(model, "fit") and hasattr(model, "predict_proba"):
        return ModelSpec(
            name=getattr(model, "__class__", type(model)).__name__,
            kind="classification",
            direct_multiclass=True,
            estimator_factory=lambda: clone(model),
        )
    raise ValueError("Unsupported `treatment_model` specification.")


def make_regression_spec(name: str, random_state=None) -> ModelSpec:
    if name not in builtin_regression_names():
        raise ValueError(f"Unsupported regression model '{name}'.")
    factories = {
        "mean": lambda: MeanRegressor(),
        "linear": lambda: LinearRegression(),
        "lasso": lambda: LassoCV(cv=5, random_state=random_state),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=random_state
        ),
        "gam": lambda: _optional_pygam()[0](),
        "boosted_trees": lambda: _optional_lightgbm()[1](
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            random_state=random_state,
            verbosity=-1,
        ),
    }
    if name == "auto":
        return ModelSpec(name=name, kind="regression", direct_multiclass=False, estimator_factory=None)
    return ModelSpec(name=name, kind="regression", direct_multiclass=False, estimator_factory=factories[name])


def make_classification_spec(name: str, random_state=None) -> ModelSpec:
    if name not in builtin_classification_names():
        raise ValueError(f"Unsupported treatment model '{name}'.")
    factories = {
        "mean": lambda: EmpiricalClassifier(),
        "linear": lambda: LogisticRegressionCV(
            cv=5,
            solver="lbfgs",
            max_iter=2000,
            l1_ratios=[0],
            random_state=random_state,
            use_legacy_attributes=False,
        ),
        "lasso": lambda: LogisticRegressionCV(
            cv=5,
            solver="saga",
            max_iter=3000,
            l1_ratios=[1],
            random_state=random_state,
            use_legacy_attributes=False,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=random_state
        ),
        "gam": lambda: _optional_pygam()[1](),
        "boosted_trees": lambda: _optional_lightgbm()[0](
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            random_state=random_state,
            verbosity=-1,
        ),
    }
    if name == "auto":
        return ModelSpec(name=name, kind="classification", direct_multiclass=False, estimator_factory=None)
    direct_multiclass = name in {"mean", "linear", "lasso", "random_forest", "boosted_trees"}
    return ModelSpec(name=name, kind="classification", direct_multiclass=direct_multiclass, estimator_factory=factories[name])


def fit_regressor(spec: ModelSpec, X, y, sample_weight=None, random_state=None):
    if spec.name == "auto":
        selected = select_regressor_name(X, y, sample_weight=sample_weight, random_state=random_state)
        return {
            "model": fit_regressor(
                make_regression_spec(selected, random_state=random_state),
                X,
                y,
                sample_weight,
                random_state,
            )["model"],
            "selected": selected,
        }
    model = spec.estimator_factory()
    y_array = np.asarray(y, dtype=float)
    if spec.name == "lasso" and len(y_array) < 5:
        model = LinearRegression() if len(y_array) < 3 else Lasso(alpha=1e-3, max_iter=5000, random_state=random_state)
    if spec.name == "gam":
        model.fit(design_matrix(X), y_array, weights=sample_weight)
    else:
        _fit_model(model, X, y_array, sample_weight=sample_weight)
    return {"model": model}


def predict_regressor(fit_obj, X):
    model = fit_obj["model"]
    if hasattr(model, "predict"):
        if model.__class__.__name__.endswith("GAM"):
            return np.asarray(model.predict(design_matrix(X)), dtype=float)
        return _predict_regression(model, X)
    raise ValueError("Regression model must implement predict.")


def fit_classifier(spec: ModelSpec, X, y, sample_weight=None, random_state=None):
    if spec.name == "auto":
        selected = select_classifier_name(X, y, sample_weight=sample_weight, random_state=random_state)
        return {
            "model": fit_classifier(
                make_classification_spec(selected, random_state=random_state),
                X,
                y,
                sample_weight,
                random_state,
            )["model"],
            "selected": selected,
        }
    y_array = np.asarray(y)
    model = spec.estimator_factory()
    unique, counts = np.unique(y_array, return_counts=True)
    min_class_count = counts.min() if len(counts) else 0
    if spec.name == "lasso" and len(unique) >= 2 and min_class_count < 5:
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=3000,
            random_state=random_state,
        )
    elif spec.name == "linear" and len(unique) >= 2 and min_class_count < 5:
        model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=random_state,
        )
    if spec.name == "gam":
        if len(np.unique(y_array)) != 2:
            raise ValueError("Built-in 'gam' treatment model supports binary fitting directly only.")
        model.fit(design_matrix(X), y_array.astype(int), weights=sample_weight)
    else:
        _fit_model(model, X, y_array, sample_weight=sample_weight)
    return {"model": model}


def predict_classifier_proba(fit_obj, X, classes: list[str] | None = None) -> np.ndarray:
    model = fit_obj["model"]
    if model.__class__.__name__.endswith("GAM"):
        proba1 = np.asarray(model.predict_proba(design_matrix(X)), dtype=float)
        proba = np.column_stack([1.0 - proba1, proba1])
    else:
        proba = np.asarray(model.predict_proba(design_matrix(X)), dtype=float)
    if classes is not None and hasattr(model, "classes_"):
        model_classes = [str(cls) for cls in model.classes_]
        aligned = np.zeros((proba.shape[0], len(classes)), dtype=float)
        for idx, level in enumerate(classes):
            if level in model_classes:
                aligned[:, idx] = proba[:, model_classes.index(level)]
        proba = aligned
    return normalize_probability_matrix(proba)


def predict_classifier_binary_positive(fit_obj, X) -> np.ndarray:
    model = fit_obj["model"]
    if model.__class__.__name__.endswith("GAM"):
        return np.asarray(model.predict_proba(design_matrix(X)), dtype=float)
    return _predict_positive_class(model, X)


def available_regression_candidates() -> list[str]:
    candidates = ["lasso", "random_forest"]
    try:
        _optional_lightgbm()
        candidates.append("boosted_trees")
    except ImportError:
        pass
    try:
        _optional_pygam()
        candidates.append("gam")
    except ImportError:
        pass
    return candidates


def available_classification_candidates() -> list[str]:
    candidates = ["lasso", "random_forest"]
    try:
        _optional_lightgbm()
        candidates.append("boosted_trees")
    except ImportError:
        pass
    try:
        _optional_pygam()
        candidates.append("gam")
    except ImportError:
        pass
    return candidates


def select_regressor_name(X, y, sample_weight=None, random_state=None) -> str:
    X_frame = as_feature_frame(X)
    y_array = np.asarray(y, dtype=float)
    weights = resolve_weights(sample_weight, len(y_array))
    if len(y_array) < 4:
        return "lasso"
    splitter = KFold(n_splits=min(3, len(y_array)), shuffle=True, random_state=random_state)
    best_name = None
    best_score = np.inf
    for candidate in available_regression_candidates():
        spec = make_regression_spec(candidate, random_state=random_state)
        scores = []
        try:
            for train, valid in splitter.split(X_frame):
                fit_obj = fit_regressor(spec, X_frame.iloc[train], y_array[train], sample_weight=weights[train], random_state=random_state)
                pred = predict_regressor(fit_obj, X_frame.iloc[valid])
                scores.append(mean_squared_error(y_array[valid], pred, sample_weight=weights[valid]))
        except Exception:
            continue
        score = float(np.mean(scores))
        if score < best_score:
            best_score = score
            best_name = candidate
    return best_name or "lasso"


def select_classifier_name(X, y, sample_weight=None, random_state=None) -> str:
    X_frame = as_feature_frame(X)
    y_series = pd.Series(np.asarray(y).astype(str))
    weights = resolve_weights(sample_weight, len(y_series))
    min_count = int(y_series.value_counts().min())
    if min_count < 2 or len(y_series) < 4:
        return "lasso"
    splitter = StratifiedKFold(n_splits=min(3, min_count), shuffle=True, random_state=random_state)
    best_name = None
    best_score = np.inf
    for candidate in available_classification_candidates():
        spec = make_classification_spec(candidate, random_state=random_state)
        scores = []
        try:
            for train, valid in splitter.split(X_frame, y_series):
                fit_obj = fit_classifier(spec, X_frame.iloc[train], y_series.iloc[train], sample_weight=weights[train], random_state=random_state)
                proba = predict_classifier_proba(fit_obj, X_frame.iloc[valid], classes=list(pd.unique(y_series)))
                scores.append(log_loss(y_series.iloc[valid], proba, sample_weight=weights[valid], labels=list(pd.unique(y_series))))
        except Exception:
            continue
        score = float(np.mean(scores))
        if score < best_score:
            best_score = score
            best_name = candidate
    return best_name or "lasso"

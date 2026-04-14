from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass
class TreatmentEncoding:
    levels: list[object]
    control_level: object
    control_index: int
    encoded: np.ndarray
    factor: np.ndarray


def as_feature_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True).copy()
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError("`X` must be a 2-dimensional array or DataFrame.")
    columns = [f"x{i}" for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


def as_1d_array(x, name: str) -> np.ndarray:
    array = np.asarray(x)
    if array.ndim != 1:
        raise ValueError(f"`{name}` must be one-dimensional.")
    return array


def resolve_weights(sample_weight, n: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != (n,):
        raise ValueError("`sample_weight` must have one entry per observation.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("`sample_weight` must be finite and non-negative.")
    if np.sum(weights) <= 0:
        raise ValueError("`sample_weight` must sum to a positive value.")
    return weights


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Weights must sum to a positive finite value.")
    return weights / total


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(normalize_weights(weights) * values))


def normalize_stratify(stratify) -> tuple[str, ...]:
    if stratify is None or stratify is False:
        return tuple()
    if stratify is True:
        return ("outcome", "treatment")
    if isinstance(stratify, str):
        values = (stratify,)
    else:
        values = tuple(str(item) for item in stratify)
    invalid = sorted(set(values) - {"outcome", "treatment"})
    if invalid:
        raise ValueError("`stratify` entries must be drawn from {'outcome', 'treatment'}.")
    return tuple(dict.fromkeys(values))


def normalize_calibration_stratify(calibration_stratify) -> str | None:
    if calibration_stratify is None or calibration_stratify is False:
        return None
    if calibration_stratify is True:
        return "outcome"
    if isinstance(calibration_stratify, str) and calibration_stratify == "outcome":
        return "outcome"
    if isinstance(calibration_stratify, Iterable):
        values = {str(item) for item in calibration_stratify}
        if values == {"outcome"}:
            return "outcome"
    raise ValueError("`calibration_stratify` must be None or 'outcome'.")


def encode_treatment(A, control_level, treatment_levels: Sequence | None = None) -> TreatmentEncoding:
    a_raw = as_1d_array(A, "A")
    a_series = pd.Series(a_raw)
    if treatment_levels is None:
        observed_levels = list(pd.unique(a_series))
        if control_level not in observed_levels:
            raise ValueError("`control_level` must appear in the observed treatment values.")
        remaining = [level for level in observed_levels if level != control_level]
        try:
            remaining = sorted(remaining)
        except TypeError:
            remaining = sorted(remaining, key=lambda value: str(value))
        levels = [control_level] + remaining
    else:
        levels = list(treatment_levels)
    if pd.Index(levels).duplicated().any():
        raise ValueError("`treatment_levels` must not contain duplicates.")
    if control_level not in levels:
        raise ValueError("`control_level` must appear in the treatment levels.")
    factor = pd.Categorical(a_series, categories=levels)
    if np.any(pd.isna(factor)):
        raise ValueError("Observed treatment values must be covered by `treatment_levels`.")
    encoded = np.asarray(factor.codes, dtype=int)
    return TreatmentEncoding(
        levels=levels,
        control_level=control_level,
        control_index=levels.index(control_level),
        encoded=encoded,
        factor=np.asarray(a_series),
    )


def align_nuisance_matrix(matrix, levels: Sequence[str], name: str) -> np.ndarray:
    if isinstance(matrix, pd.DataFrame):
        column_lookup = {str(column): column for column in matrix.columns}
        missing = [level for level in levels if str(level) not in column_lookup]
        if missing:
            raise ValueError(f"Column names of `{name}` must include all treatment levels.")
        aligned_columns = [column_lookup[str(level)] for level in levels]
        aligned = matrix.loc[:, aligned_columns].to_numpy(dtype=float)
    else:
        aligned = np.asarray(matrix, dtype=float)
        if aligned.ndim != 2:
            raise ValueError(f"`{name}` must be 2-dimensional.")
        if aligned.shape[1] != len(levels):
            raise ValueError(f"`{name}` must have one column per treatment level.")
    if aligned.shape[1] != len(levels):
        raise ValueError(f"`{name}` must have one column per treatment level.")
    if aligned.shape[0] == 0:
        raise ValueError(f"`{name}` must have at least one row.")
    if np.any(~np.isfinite(aligned)):
        raise ValueError(f"`{name}` must be finite.")
    return aligned


def validate_supplied_probability_matrix(matrix, levels: Sequence[str], name: str = "pi_mat") -> np.ndarray:
    pi = align_nuisance_matrix(matrix, levels, name)
    if np.any(pi < 0):
        raise ValueError(f"`{name}` must be non-negative.")
    row_sums = pi.sum(axis=1)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0):
        raise ValueError(f"Rows of `{name}` must sum to a positive finite value.")
    return normalize_probability_matrix(pi)


def normalize_probability_matrix(pi_mat: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    pi = np.asarray(pi_mat, dtype=float)
    if pi.ndim != 2:
        raise ValueError("Probability matrix must be 2-dimensional.")
    pi = np.maximum(pi, 1e-8)
    row_sums = pi.sum(axis=1)
    invalid = ~np.isfinite(row_sums) | (row_sums <= 0)
    if np.any(invalid):
        if fallback is None:
            raise ValueError("Probability matrix rows must have positive finite sums.")
        fallback = np.asarray(fallback, dtype=float)
        pi[invalid] = fallback[invalid]
        row_sums = pi.sum(axis=1)
    return pi / row_sums[:, None]


def empirical_probability_matrix(observed_levels: Sequence[str], levels: Sequence[str], n_rows: int) -> np.ndarray:
    observed = pd.Series(np.asarray(observed_levels))
    probs = np.array([(observed == level).mean() for level in levels], dtype=float)
    if probs.sum() <= 0:
        probs = np.repeat(1.0 / len(levels), len(levels))
    return np.tile(probs[None, :], (n_rows, 1))


def augment_features_with_treatment(X: pd.DataFrame, treatment_values: Sequence[str], levels: Sequence[str]) -> pd.DataFrame:
    augmented = X.reset_index(drop=True).copy()
    level_strings = [str(level) for level in levels]
    treatment = pd.Categorical(pd.Series(treatment_values).astype(str), categories=level_strings)
    dummies = pd.get_dummies(treatment, prefix="treatment", drop_first=False, dtype=float)
    return pd.concat([augmented, dummies.reset_index(drop=True)], axis=1)


def design_matrix(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    frame = as_feature_frame(X)
    numeric = pd.get_dummies(frame, drop_first=False, dtype=float)
    return numeric.to_numpy(dtype=float)


def resolve_fold_ids(encoded_treatment: np.ndarray, n_folds: int, fold_ids=None, random_state=None) -> np.ndarray:
    n = len(encoded_treatment)
    if fold_ids is not None:
        fold_ids = np.asarray(fold_ids, dtype=int)
        if fold_ids.shape != (n,):
            raise ValueError("`fold_ids` must have one entry per observation.")
        if np.any(fold_ids < 0):
            raise ValueError("`fold_ids` must be non-negative integers.")
        if np.unique(fold_ids).size < 2:
            raise ValueError("`fold_ids` must define at least two folds.")
        return fold_ids
    class_counts = np.bincount(encoded_treatment)
    positive_counts = class_counts[class_counts > 0]
    max_folds = int(np.min(positive_counts)) if len(positive_counts) else 0
    n_splits = min(int(n_folds), max_folds)
    if n_splits < 2:
        raise ValueError("Need at least two observations per treatment class to cross-fit nuisances.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_ids_out = np.empty(n, dtype=int)
    dummy = np.zeros(n)
    for fold, (_, valid) in enumerate(splitter.split(dummy, encoded_treatment)):
        fold_ids_out[valid] = fold
    return fold_ids_out


def bootstrap_indices(n: int, fold_ids: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    if fold_ids is None:
        return rng.choice(n, size=n, replace=True)
    resampled = []
    for group in np.unique(fold_ids):
        group_index = np.flatnonzero(fold_ids == group)
        resampled.append(rng.choice(group_index, size=len(group_index), replace=True))
    return np.concatenate(resampled)


def validate_binary_treatment(levels: Sequence[str]) -> None:
    if len(levels) != 2:
        raise ValueError("Adaptive calibrated DML currently supports binary treatment only.")

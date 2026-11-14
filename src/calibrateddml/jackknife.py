"""V-fold jackknife standard errors with Student-t confidence intervals.

Li, Ertefaie, and van der Laan (2026), Algorithm 1, arXiv:2607.22493.
The caller refits calibration and final means with initial nuisances fixed.
"""

from __future__ import annotations

import numpy as np


def vfold_jackknife(full_estimate, refit, n, folds=10, random_state=None):
    """Delete approximately equal groups and compute fold pseudo-values."""
    if not np.isfinite(folds) or int(folds) != folds or folds < 2:
        raise ValueError("`jackknife_folds` must be an integer of at least 2.")
    if n < 2:
        raise ValueError("Jackknife inference requires at least two observations.")
    v = min(int(folds), n)
    rng = np.random.default_rng(1 if random_state is None else random_state)
    groups = np.array_split(rng.permutation(n), v)
    fold_ids = np.empty(n, dtype=int)
    draws = []
    for label, group in enumerate(groups):
        fold_ids[group] = label
        keep = np.ones(n, dtype=bool)
        keep[group] = False
        draws.append(np.asarray(refit(np.flatnonzero(keep)), dtype=float))
    draws = np.vstack(draws)
    full_estimate = np.asarray(full_estimate, dtype=float)
    if draws.shape[1:] != full_estimate.shape or not np.all(np.isfinite(draws)):
        raise ValueError("Each jackknife refit must return finite estimates for every estimand.")
    pseudo = v * full_estimate[None, :] - (v - 1) * draws
    return {
        "standard_error": pseudo.std(axis=0, ddof=1) / np.sqrt(v),
        "df": v - 1,
        "fold_ids": fold_ids,
        "pseudo_values": pseudo,
        "leaveout_estimates": draws,
    }

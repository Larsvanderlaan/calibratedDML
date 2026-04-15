from __future__ import annotations

import numpy as np
import pandas as pd


def make_binary_data(n: int = 180, seed: int = 1):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = -0.15 + 0.5 * x1 - 0.25 * x2
    pi1 = 1.0 / (1.0 + np.exp(-logits))
    a = rng.binomial(1, pi1)
    mu0 = 0.4 + 0.6 * x1 - 0.2 * x2
    tau = 0.8 + 0.25 * x1
    mu1 = mu0 + tau
    y = mu0 + a * tau + rng.normal(scale=0.45, size=n)
    x = pd.DataFrame({"x1": x1, "x2": x2})
    mu_mat = pd.DataFrame(np.column_stack([mu0, mu1]), columns=[0, 1])
    pi_mat = pd.DataFrame(np.column_stack([1.0 - pi1, pi1]), columns=[0, 1])
    return x, a, y, mu_mat, pi_mat


def make_multiarm_data(n: int = 220, seed: int = 2):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    scores = np.column_stack(
        [
            0.1 + 0.2 * x1,
            -0.1 + 0.35 * x2,
            0.15 - 0.2 * x1 + 0.25 * x2,
        ]
    )
    scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    pi_mat = scores / scores.sum(axis=1, keepdims=True)
    a = np.array([rng.choice([0, 1, 2], p=row) for row in pi_mat], dtype=int)
    mu0 = 0.2 + 0.4 * x1
    mu1 = mu0 + 0.6
    mu2 = mu0 - 0.35 + 0.2 * x2
    y = np.choose(a, [mu0, mu1, mu2]) + rng.normal(scale=0.45, size=n)
    x = pd.DataFrame({"x1": x1, "x2": x2})
    mu_df = pd.DataFrame(np.column_stack([mu0, mu1, mu2]), columns=[0, 1, 2])
    pi_df = pd.DataFrame(pi_mat, columns=[0, 1, 2])
    return x, a, y, mu_df, pi_df


def print_table(title: str, frame, columns=None, rows: int | None = None) -> None:
    print(f"\n{title}")
    if columns is not None:
        frame = frame.loc[:, columns]
    if rows is not None:
        frame = frame.head(rows)
    print(frame.to_string(index=False))

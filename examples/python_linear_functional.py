import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from calibrateddml import calibrated_linear_functional

rng = np.random.default_rng(42)
n = 600
X = rng.normal(size=(n, 2))
A = rng.binomial(1, 1 / (1 + np.exp(-0.5 * X[:, 0])))
y = X[:, 0] + X[:, 1] + 2 * A + rng.normal(size=n)

mu_mat = np.empty((n, 2))
pi = np.empty(n)
folds = StratifiedKFold(5, shuffle=True, random_state=42)
for train, test in folds.split(X, A):
    outcome_model = LinearRegression().fit(np.column_stack([X[train], A[train]]), y[train])
    for treatment in (0, 1):
        inputs = np.column_stack([X[test], np.full(len(test), treatment)])
        mu_mat[test, treatment] = outcome_model.predict(inputs)
    treatment_model = LogisticRegression().fit(X[train], A[train])
    pi[test] = treatment_model.predict_proba(X[test])[:, 1]

alpha_mat = np.column_stack([-1 / (1 - pi), 1 / pi])
rows = np.arange(n)
fit = calibrated_linear_functional(
    y=y,
    regression=mu_mat[rows, A],
    riesz=alpha_mat[rows, A],
    functional=lambda values: values[:, 1] - values[:, 0],
    regression_evaluations=mu_mat,
    riesz_evaluations=alpha_mat,
    random_state=42,
    name="ATE",
)
print(fit.summary().round(3).to_string(index=False))

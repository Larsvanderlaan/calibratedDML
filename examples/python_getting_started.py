import numpy as np
from calibrateddml import CalibratedDML

rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 2))
propensity = 1 / (1 + np.exp(-0.7 * X[:, 0]))
A = rng.binomial(1, propensity)
y = 2 * A + X[:, 0] + X[:, 1] + rng.normal(size=1000)

fit = CalibratedDML(control_level=0, random_state=42).fit(X, A, y)
columns = ["estimand", "estimate", "std_error", "lower", "upper"]
print(fit.summary()[columns].round(3).to_string(index=False))

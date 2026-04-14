import numpy as np

from calibrateddml import AdaptiveCalibratedDML, CalibratedDML


def main():
    rng = np.random.default_rng(7)
    n = 400
    X = rng.normal(size=(n, 3))

    logits = np.column_stack(
        [
            np.zeros(n),
            0.6 * X[:, 0],
            -0.25 * X[:, 0] + 0.5 * X[:, 1],
        ]
    )
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    A = np.array([rng.choice(["0", "1", "2"], p=row) for row in probs])

    mu0 = 0.5 + X[:, 0] - 0.25 * X[:, 1]
    mu1 = mu0 + 1.0
    mu2 = mu0 - 0.5 + 0.3 * X[:, 2]
    mu = np.column_stack([mu0, mu1, mu2])
    Y = rng.normal(mu[np.arange(n), np.array([0, 1, 2])[np.searchsorted(["0", "1", "2"], A)]], 1.0)

    fit = CalibratedDML(control_level="0").fit_from_nuisances(A=A, y=Y, mu_mat=mu, pi_mat=probs)
    print(fit.summary())

    A_binary = (A == "1").astype(int)
    pi_binary = np.column_stack([1.0 - probs[:, 1], probs[:, 1]])
    mu_binary = np.column_stack([mu0, mu1])
    adaptive = AdaptiveCalibratedDML(control_level=0, mode="plugin").fit_from_nuisances(
        X=X,
        A=A_binary,
        y=Y,
        mu_mat=mu_binary,
        pi_mat=pi_binary,
    )
    print(adaptive.summary())


if __name__ == "__main__":
    main()

library(calibratedDML)

set.seed(42)
n <- 600
df <- data.frame(W1 = rnorm(n), W2 = rnorm(n))
df$A <- rbinom(n, 1, plogis(0.5 * df$W1))
df$Y <- df$W1 + df$W2 + 2 * df$A + rnorm(n)

mu_mat <- matrix(NA_real_, n, 2)
pi <- numeric(n)
fold_id <- sample(rep(seq_len(5), length.out = n))
for (fold in seq_len(5)) {
  train <- fold_id != fold
  test <- fold_id == fold
  outcome_model <- lm(Y ~ A + W1 + W2, data = df[train, ])
  for (treatment in 0:1) {
    inputs <- df[test, ]
    inputs$A <- treatment
    mu_mat[test, treatment + 1] <- predict(outcome_model, newdata = inputs)
  }
  treatment_model <- glm(A ~ W1 + W2, data = df[train, ], family = binomial())
  pi[test] <- predict(treatment_model, newdata = df[test, ], type = "response")
}

alpha_mat <- cbind(-1 / (1 - pi), 1 / pi)
observed <- cbind(seq_len(n), df$A + 1)
fit <- calibrated_linear_functional(
  Y = df$Y,
  regression = mu_mat[observed],
  riesz = alpha_mat[observed],
  functional = function(values) values[, 2] - values[, 1],
  regression_evaluations = mu_mat,
  riesz_evaluations = alpha_mat,
  seed = 42,
  name = "ATE"
)
print(summary(fit), row.names = FALSE, digits = 3)

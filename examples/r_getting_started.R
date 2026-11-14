library(calibratedDML)

set.seed(42)
n <- 1000
df <- data.frame(W1 = rnorm(n), W2 = rnorm(n))
df$A <- rbinom(n, 1, plogis(0.7 * df$W1))
df$Y <- 2 * df$A + df$W1 + df$W2 + rnorm(n)

fit <- calibrated_dml(
  data = df,
  outcome = "Y",
  treatment = "A",
  covariates = c("W1", "W2"),
  control_level = 0,
  seed = 42
)
columns <- c("estimand", "estimate", "std_error", "lower", "upper")
print(summary(fit)[columns], row.names = FALSE, digits = 3)

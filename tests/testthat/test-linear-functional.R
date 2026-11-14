linear_fixture <- function(n = 120) {
  set.seed(23)
  x <- rnorm(n)
  pi <- plogis(0.5 * x)
  A <- rbinom(n, 1, pi)
  mu <- cbind(x, x + 1)
  alpha <- cbind(-1 / (1 - pi), 1 / pi)
  list(Y = x + A + rnorm(n, sd = 0.5), mu = mu[cbind(seq_len(n), A + 1)],
    alpha = alpha[cbind(seq_len(n), A + 1)], mu_eval = mu, alpha_eval = alpha)
}

test_that("generalized PAVA pools quadratic masses and supports signed values", {
  expect_equal(calibratedDML:::linear_pava(c(1, 2, 1), c(3, 0, 1)), c(1, 1, 1))
  expect_equal(calibratedDML:::linear_pava(c(1, 1, 1), c(-2, 3, -1)), c(-2, 1, 1))
})

test_that("uncalibrated linear-functional estimate equals AIPW", {
  d <- linear_fixture()
  fit <- calibrated_linear_functional(d$Y, d$mu, d$alpha, c(-1, 1), d$mu_eval, d$alpha_eval,
    calibration_method = "none", inference = "wald", name = "ATE")
  scores <- d$mu_eval[, 2] - d$mu_eval[, 1] + d$alpha * (d$Y - d$mu)
  expect_equal(summary(fit)$estimate, mean(scores))
  expect_equal(fit$scores, scores)
  expect_equal(dim(confint(fit)), c(1L, 2L))
  expect_equal(as.data.frame(fit), summary(fit))
})

test_that("both calibration equations hold with signed representers", {
  d <- linear_fixture()
  weights <- seq(0.2, 2, length.out = length(d$Y))
  fit <- calibrated_linear_functional(d$Y, d$mu, d$alpha, function(p) p[, 2] - p[, 1],
    d$mu_eval, d$alpha_eval, sample_weight = weights, inference = "wald")
  for (value in unique(fit$calibrated_regression)) {
    rows <- fit$calibrated_regression == value
    expect_equal(sum(weights[rows] * (d$Y[rows] - value)), 0, tolerance = 1e-10)
  }
  evaluated <- matrix(calibratedDML:::predict_linear_pava(fit$calibration$riesz, d$alpha_eval), ncol = 2)
  for (value in unique(fit$calibrated_riesz)) {
    left <- sum(weights * (fit$calibrated_riesz == value) * fit$calibrated_riesz)
    right <- sum(weights * ((evaluated[, 2] == value) - (evaluated[, 1] == value)))
    expect_equal(left, right, tolerance = 1e-10)
  }
  expect_true(any(fit$calibrated_riesz < 0))
  expect_true(any(fit$calibrated_riesz > 0))
})

test_that("known signed representers recover weighted outcome means", {
  d <- linear_fixture()
  q <- seq(-2, 2, length.out = length(d$Y))
  weights <- seq(0.2, 1.5, length.out = length(d$Y))
  fit <- calibrated_linear_functional(d$Y, d$mu, q, matrix(q, ncol = 1),
    sample_weight = weights, inference = "wald")
  expect_equal(fit$calibrated_riesz, q)
  expect_equal(summary(fit)$estimate, weighted.mean(q * d$Y, weights))
})

test_that("PAVA pools ties and handles zero-weight observations", {
  fit <- calibrated_linear_functional(c(2, 0, 8, 4, -100), c(0, 0, 1, 1, 8), rep(1, 5), 1,
    sample_weight = c(1, 1, 1, 1, 0), inference = "wald")
  expect_equal(fit$calibrated_regression[1:4], c(1, 1, 6, 6))
  expect_equal(summary(fit)$estimate, 3.5)
})

test_that("default jackknife uses t9 and fixed functional coefficients", {
  d <- linear_fixture()
  calls <- 0L
  functional <- function(p) { calls <<- calls + 1L; p[, 2] - p[, 1] }
  fit <- calibrated_linear_functional(d$Y, d$mu, d$alpha, functional, d$mu_eval, d$alpha_eval, seed = 7)
  expect_equal(calls, 5)
  expect_equal(fit$jackknife$df, 9)
  tab <- summary(fit)
  expect_equal(tab$upper, tab$estimate + qt(0.975, 9) * tab$std_error)
})

test_that("bootstrap intervals and input validation work", {
  d <- linear_fixture(60)
  fit <- calibrated_linear_functional(d$Y, d$mu, d$alpha, c(-1, 1), d$mu_eval, d$alpha_eval,
    inference = "bootstrap", bootstrap_reps = 12, seed = 1)
  expect_true(is.finite(summary(fit)$std_error))
  expect_gt(summary(fit)$std_error, 0)
  expect_error(calibrated_linear_functional(d$Y, d$mu, d$alpha, function(p) p[, 1]^2), "row-wise linear")
  expect_error(calibrated_linear_functional(d$Y, d$mu, d$alpha, c(-1, 1), d$mu_eval), "same shape")
})

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

three_input_fixture <- function() {
  d <- read.csv(testthat::test_path("..", "fixtures", "linear_functional_three_input.csv"))
  list(
    Y = d$y, mu = d$regression, alpha = d$riesz, weights = d$sample_weight,
    coefficients = unname(as.matrix(d[paste0("c", 0:2)])),
    mu_eval = unname(as.matrix(d[paste0("mu_eval", 0:2)])),
    alpha_eval = unname(as.matrix(d[paste0("alpha_eval", 0:2)])),
    x = d$x, a = d$a, epsilon = d$epsilon
  )
}

fit_three_input <- function(d, functional = d$coefficients, ...) {
  calibrated_linear_functional(d$Y, d$mu, d$alpha, functional,
    d$mu_eval, d$alpha_eval, sample_weight = d$weights, ...)
}

# Independent isotonic oracle: the max-min interval-average formula, without
# the package's PAVA, binning, or prediction routines.
reference_linear_isotonic <- function(quadratic, linear) {
  k <- length(quadratic)
  vapply(seq_len(k), function(j) {
    max(vapply(seq_len(j), function(left) {
      min(vapply(j:k, function(right) {
        sum(linear[left:right]) / sum(quadratic[left:right])
      }, numeric(1)))
    }, numeric(1)))
  }, numeric(1))
}

reference_linear_step <- function(knots, fitted, values) {
  result <- vapply(as.numeric(values), function(value) {
    fitted[max(1L, sum(knots <= value))]
  }, numeric(1))
  if (!is.null(dim(values))) dim(result) <- dim(values)
  result
}

reference_three_input <- function(d, index = seq_along(d$Y), calibrate = TRUE) {
  Y <- d$Y[index]
  mu <- d$mu[index]
  alpha <- d$alpha[index]
  w <- d$weights[index]
  coefficients <- d$coefficients[index, , drop = FALSE]
  mu_eval <- d$mu_eval[index, , drop = FALSE]
  alpha_eval <- d$alpha_eval[index, , drop = FALSE]
  if (calibrate) {
    f_knots <- sort(unique(mu[w > 0]))
    f_quadratic <- vapply(f_knots, function(value) sum(w[mu == value]), numeric(1))
    f_linear <- vapply(f_knots, function(value) sum(w[mu == value] * Y[mu == value]), numeric(1))
    f_fitted <- reference_linear_isotonic(f_quadratic, f_linear)
    g_knots <- sort(unique(alpha[w > 0]))
    g_quadratic <- vapply(g_knots, function(value) sum(w[alpha == value]), numeric(1))
    g_linear <- numeric(length(g_knots))
    for (i in seq_along(index)) {
      for (j in seq_len(ncol(coefficients))) {
        bin <- max(1L, sum(g_knots <= alpha_eval[i, j]))
        g_linear[bin] <- g_linear[bin] + w[i] * coefficients[i, j]
      }
    }
    g_fitted <- reference_linear_isotonic(g_quadratic, g_linear)
    mu <- reference_linear_step(f_knots, f_fitted, mu)
    alpha <- reference_linear_step(g_knots, g_fitted, alpha)
    mu_eval <- reference_linear_step(f_knots, f_fitted, mu_eval)
  }
  functional_values <- rowSums(coefficients * mu_eval)
  scores <- functional_values + alpha * (Y - mu)
  list(estimate = sum(w * scores) / sum(w), scores = scores,
    mu = mu, alpha = alpha, functional_values = functional_values)
}

expect_three_input_calibration <- function(d, fit) {
  for (value in unique(fit$calibrated_regression)) {
    rows <- fit$calibrated_regression == value
    expect_equal(sum(d$weights[rows] * (d$Y[rows] - value)), 0, tolerance = 1e-10)
  }
  g <- fit$calibration$riesz
  evaluated <- reference_linear_step(g$knots, g$fitted, d$alpha_eval)
  for (value in unique(fit$calibrated_riesz)) {
    left <- sum(d$weights * (fit$calibrated_riesz == value) * fit$calibrated_riesz)
    right <- sum(d$weights * rowSums(d$coefficients * (evaluated == value)))
    expect_equal(left, right, tolerance = 1e-10)
  }
}

test_that("three signed evaluation inputs satisfy the AIPW and Riesz identities", {
  d <- three_input_fixture()
  before <- d
  raw <- fit_three_input(d, calibration_method = "none", inference = "wald")
  expected_scores <- c(3, 5, 13, 4, 6, 14)[3 * d$x + d$a + 1] + d$alpha * d$epsilon
  expect_equal(raw$scores, expected_scores, tolerance = 1e-12)
  expect_equal(summary(raw)$estimate, 23 / 3, tolerance = 1e-12)
  expect_equal(summary(raw)$std_error^2, 908 / 1215, tolerance = 1e-12)

  true_mu <- outer(d$x, c(1, 3, 5), "+")
  true_alpha <- 3 * d$coefficients
  observed <- cbind(seq_along(d$Y), d$a + 1L)
  # The finite sample enumerates the complete law with balanced noise within
  # every (X, A) cell, so either correct nuisance gives the target exactly.
  correct_mu <- d
  correct_mu$mu_eval <- true_mu
  correct_mu$mu <- true_mu[observed]
  correct_alpha <- d
  correct_alpha$alpha_eval <- true_alpha
  correct_alpha$alpha <- true_alpha[observed]
  for (correct in list(correct_mu, correct_alpha)) {
    fit <- fit_three_input(correct, calibration_method = "none", inference = "wald")
    expect_equal(summary(fit)$estimate, 10, tolerance = 1e-12)
  }
  # Verify the Riesz identity on every basis vector of the six-point (X, A)
  # domain; this establishes it for all functions on this finite domain.
  for (x in 0:1) {
    for (a in 0:2) {
      left <- sum(d$weights * true_alpha[observed] * (d$x == x & d$a == a))
      right <- sum(d$weights * d$coefficients[, a + 1L] * (d$x == x))
      expect_equal(left, right, tolerance = 1e-12)
    }
  }
  expect_identical(d, before)
})

test_that("three-input PAVA matches exact block solutions and both calibration equations", {
  d <- three_input_fixture()
  fit <- fit_three_input(d, inference = "wald")
  reference <- reference_three_input(d)
  expect_equal(fit$calibration$regression$knots, c(0, 1, 2, 3))
  expect_equal(fit$calibration$regression$fitted, c(1, 3, 23 / 5, 23 / 5))
  expect_equal(fit$calibration$riesz$knots, c(-2, 0, 2))
  expect_equal(fit$calibration$riesz$fitted, c(-4, 4, 4))
  expect_equal(fit$calibrated_regression, reference$mu, tolerance = 1e-12)
  expect_equal(fit$calibrated_riesz, reference$alpha, tolerance = 1e-12)
  expect_equal(fit$functional_values, reference$functional_values, tolerance = 1e-12)
  expected_scores <- c(44, 12, 84, 74, 42, 82)[3 * d$x + d$a + 1] / 5 +
    c(-4, 4, 4)[d$a + 1L] * d$epsilon
  expect_equal(fit$scores, expected_scores, tolerance = 1e-12)
  expect_equal(summary(fit)$estimate, 536 / 45, tolerance = 1e-12)
  expect_equal(summary(fit)$std_error^2, 274952 / 273375, tolerance = 1e-12)
  expect_three_input_calibration(d, fit)
  # Prefix residuals must be nonnegative and vanish at strict fitted jumps.
  f_prefix <- cumsum(c(5, 45, 75, 40) - c(5, 15, 15, 10) * fit$calibration$regression$fitted)
  g_prefix <- cumsum(c(-60, 75, 45) - rep(15, 3) * fit$calibration$riesz$fitted)
  expect_equal(f_prefix, c(0, 0, 6, 0), tolerance = 1e-12)
  expect_equal(g_prefix, c(0, 15, 0), tolerance = 1e-12)
})

test_that("row-specific callable coefficients and aligned permutations agree", {
  d <- three_input_fixture()
  before <- d
  calls <- 0L
  functional <- function(values) {
    calls <<- calls + 1L
    rowSums(d$coefficients * values)
  }
  fit <- fit_three_input(d, functional, inference = "wald")
  expect_equal(calls, 6L) # zero, three basis columns, two linearity checks
  expect_equal(fit$calibration$coefficients, d$coefficients)
  expect_equal(fit$scores, fit_three_input(d, inference = "wald")$scores)
  rows <- rev(seq_along(d$Y))
  columns <- c(3L, 1L, 2L)
  permuted <- lapply(d, function(value) {
    if (is.matrix(value)) value[rows, columns, drop = FALSE] else value[rows]
  })
  other <- fit_three_input(permuted, inference = "wald")
  expect_equal(other$scores, fit$scores[rows], tolerance = 1e-12)
  expect_equal(summary(other), summary(fit), tolerance = 1e-12)
  expect_identical(d, before)
})

test_that("three-input calibration handles off-knot evaluations and zero weights", {
  d <- three_input_fixture()
  d$weights[c(1, 7, 11, 19, 30)] <- 0
  d$mu_eval[2, ] <- c(-2, 0.5, 5)
  d$alpha_eval[3, ] <- c(-4, -1, 4)
  fit <- fit_three_input(d, inference = "wald")
  reference <- reference_three_input(d)
  expect_equal(fit$calibrated_regression, reference$mu, tolerance = 1e-12)
  expect_equal(fit$calibrated_riesz, reference$alpha, tolerance = 1e-12)
  expect_equal(fit$scores, reference$scores, tolerance = 1e-12)
  expect_equal(summary(fit)$estimate, reference$estimate, tolerance = 1e-12)
  expect_three_input_calibration(d, fit)
})

test_that("three-input fixed-nuisance jackknife refits match independent calculations", {
  d <- three_input_fixture()
  before <- d
  calls <- 0L
  functional <- function(values) {
    calls <<- calls + 1L
    rowSums(d$coefficients * values)
  }
  for (method in c("isotonic", "none")) {
    calls <- 0L
    fit <- fit_three_input(d, functional, calibration_method = method, seed = 7)
    expect_equal(calls, 6L)
    jk <- fit$jackknife
    expect_equal(jk$df, 9L)
    expect_equal(as.integer(table(jk$fold_id)), rep(3L, 10))
    full <- reference_three_input(d, calibrate = method == "isotonic")$estimate
    deleted <- vapply(seq_len(10), function(fold) {
      reference_three_input(d, which(jk$fold_id != fold),
        calibrate = method == "isotonic")$estimate
    }, numeric(1))
    pseudo <- 10 * full - 9 * deleted
    se <- sqrt(sum((pseudo - mean(pseudo))^2) / (10 * 9))
    expect_equal(as.numeric(jk$leaveout_estimates), deleted, tolerance = 1e-12)
    expect_equal(as.numeric(jk$pseudo_values), pseudo, tolerance = 1e-12)
    expect_equal(as.numeric(jk$standard_error), se, tolerance = 1e-12)
    tab <- summary(fit)
    expect_equal(tab$estimate, full, tolerance = 1e-12)
    expect_equal(tab$std_error, se, tolerance = 1e-12)
    expect_equal(tab$lower, full - qt(0.975, 9) * se, tolerance = 1e-12)
    expect_equal(tab$upper, full + qt(0.975, 9) * se, tolerance = 1e-12)
    if (method == "isotonic") {
      frozen <- vapply(seq_len(10), function(fold) {
        rows <- jk$fold_id != fold
        weighted.mean(fit$scores[rows], d$weights[rows])
      }, numeric(1))
      expect_gt(max(abs(deleted - frozen)), 1e-3)
    }
  }
  expect_identical(d, before)
})

test_that("three-input leave-one-out jackknife agrees with the deterministic reference", {
  d <- three_input_fixture()
  n <- length(d$Y)
  fit <- fit_three_input(d, jackknife_folds = n, seed = 7)
  deleted <- vapply(seq_len(n), function(row) {
    reference_three_input(d, setdiff(seq_len(n), row))$estimate
  }, numeric(1))
  pseudo <- n * (536 / 45) - (n - 1) * deleted
  jk <- fit$jackknife
  expect_equal(as.numeric(jk$leaveout_estimates)[jk$fold_id], deleted, tolerance = 1e-12)
  expect_equal(as.numeric(jk$pseudo_values)[jk$fold_id], pseudo, tolerance = 1e-12)
  expect_equal(summary(fit)$std_error, sd(pseudo) / sqrt(n), tolerance = 1e-12)
  expect_equal(summary(fit)$std_error, 1.3783388993802286, tolerance = 1e-12)
  expect_equal(jk$df, n - 1L)
})

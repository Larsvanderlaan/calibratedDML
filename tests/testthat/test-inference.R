test_that("wald bootstrap and jackknife all run from supplied nuisances", {
  fixture <- oracle_binary_fixture(n = 250, seed = 5)

  wald_fit <- calibrated_dml_from_nuisances(
    A = fixture$A,
    Y = fixture$Y,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    control_level = 0,
    inference = "wald"
  )
  boot_fit <- calibrated_dml_from_nuisances(
    A = fixture$A,
    Y = fixture$Y,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    control_level = 0,
    inference = "bootstrap",
    bootstrap_reps = 30,
    seed = 1
  )
  jack_fit <- calibrated_dml_from_nuisances(
    A = fixture$A,
    Y = fixture$Y,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    control_level = 0,
    inference = "jackknife",
    jackknife_folds = 8
  )

  for (fit in list(wald_fit, boot_fit, jack_fit)) {
    expect_true(all(is.finite(fit$estimates$estimate)))
    expect_true(all(is.finite(fit$estimates$lower)))
    expect_true(all(is.finite(fit$estimates$upper)))
    expect_true(all(fit$estimates$upper >= fit$estimates$lower))
  }
})

test_that("binary Wald uses sieve-Riesz correction by default", {
  fixture <- oracle_binary_fixture(n = 180, seed = 21)
  options <- list(
    basis_size_grid = c(4L, 8L),
    lambda_grid = c(1e-4, 1e-2),
    cv_folds = 3L,
    min_rows = 5L
  )

  corrected <- calibrated_dml_from_nuisances(
    A = fixture$A,
    Y = fixture$Y,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    control_level = 0,
    inference = "wald",
    calibration_method = "none",
    seed = 31L,
    wald_options = options
  )
  standard <- calibrated_dml_from_nuisances(
    A = fixture$A,
    Y = fixture$Y,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    control_level = 0,
    inference = "wald",
    calibration_method = "none",
    seed = 31L,
    wald_correction = "none",
    wald_options = options
  )

  expect_true(isTRUE(corrected$wald_diagnostics$applied))
  expect_identical(corrected$wald_diagnostics$wald_aux_method, "sieve_riesz")
  expect_identical(corrected$wald_diagnostics$std_error_mode, "corrected_if")
  expect_true(is.finite(corrected$contrasts$std_error[[1L]]))
  expect_equal(standard$contrasts$std_error[[1L]], corrected$wald_diagnostics$simple_wald_std_error)
  expect_false(isTRUE(standard$wald_diagnostics$applied))
})

test_that("conservative binary Wald uses maximum standard error", {
  fixture <- oracle_binary_fixture(n = 180, seed = 22)
  fit <- calibrated_dml_from_nuisances(
    A = fixture$A,
    Y = fixture$Y,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    control_level = 0,
    inference = "wald",
    calibration_method = "none",
    seed = 32L,
    wald_conservative = TRUE,
    wald_options = list(
      basis_size_grid = c(4L, 8L),
      lambda_grid = c(1e-4, 1e-2),
      cv_folds = 3L,
      min_rows = 5L
    )
  )

  expected <- max(
    fit$wald_diagnostics$simple_wald_std_error,
    fit$wald_diagnostics$corrected_if_std_error
  )
  expect_identical(fit$wald_diagnostics$std_error_mode, "conservative")
  expect_equal(fit$contrasts$std_error[[1L]], expected)
  expect_equal(fit$wald_diagnostics$selected_std_error, expected)
})

test_that("sieve-Riesz Wald supports explicit binary labels", {
  fixture <- oracle_binary_fixture(n = 160, seed = 23)
  labels <- ifelse(fixture$A == 1, "treated", "control")
  mu_mat <- cbind(treated = fixture$mu_mat[, "1"], control = fixture$mu_mat[, "0"])
  pi_mat <- cbind(treated = fixture$pi_mat[, "1"], control = fixture$pi_mat[, "0"])

  fit <- calibrated_dml_from_nuisances(
    A = labels,
    Y = fixture$Y,
    mu_mat = mu_mat,
    pi_mat = pi_mat,
    control_level = "control",
    treatment_levels = c("treated", "control"),
    inference = "wald",
    calibration_method = "none",
    seed = 33L,
    wald_options = list(basis_size_grid = 4L, lambda_grid = 1e-3, cv_folds = 3L, min_rows = 5L)
  )

  expect_true(isTRUE(fit$wald_diagnostics$applied))
  expect_identical(fit$contrasts$level[[1L]], "treated")
  expect_identical(fit$contrasts$control_level[[1L]], "control")
  expect_true(is.finite(fit$contrasts$std_error[[1L]]))
})

test_that("Wald correction validation and multi-arm auto fallback are explicit", {
  binary <- oracle_binary_fixture(n = 100, seed = 24)
  expect_error(
    calibrated_dml_from_nuisances(
      A = binary$A,
      Y = binary$Y,
      mu_mat = binary$mu_mat,
      pi_mat = binary$pi_mat,
      control_level = 0,
      inference = "wald",
      wald_conservative = "TRUE"
    ),
    "`wald_conservative` must be TRUE or FALSE",
    fixed = TRUE
  )

  expect_error(
    calibrated_dml_from_nuisances(
      A = binary$A,
      Y = binary$Y,
      mu_mat = binary$mu_mat,
      pi_mat = binary$pi_mat,
      control_level = 0,
      inference = "bootstrap",
      wald_correction = "sieve_riesz"
    ),
    "inference = \"wald\""
  )

  multi <- oracle_multiarm_fixture(n = 150, seed = 25)
  auto <- calibrated_dml_from_nuisances(
    A = multi$A,
    Y = multi$Y,
    mu_mat = multi$mu_mat,
    pi_mat = multi$pi_mat,
    control_level = 0,
    inference = "wald",
    calibration_method = "none"
  )
  expect_identical(auto$wald_diagnostics$fallback_reason, "non_binary_treatment")

  expect_error(
    calibrated_dml_from_nuisances(
      A = multi$A,
      Y = multi$Y,
      mu_mat = multi$mu_mat,
      pi_mat = multi$pi_mat,
      control_level = 0,
      inference = "wald",
      calibration_method = "none",
      wald_correction = "sieve_riesz"
    ),
    "binary treatment"
  )
})

test_that("sieve-Riesz Wald fallbacks and boundary propensities remain finite", {
  fixture <- oracle_binary_fixture(n = 140, seed = 26)
  fallback <- calibrated_dml_from_nuisances(
    A = fixture$A,
    Y = fixture$Y,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    control_level = 0,
    inference = "wald",
    calibration_method = "none",
    seed = 34L,
    wald_options = list(
      basis_size_grid = 4L,
      lambda_grid = 1e-3,
      cv_folds = 2L,
      min_rows = 5L,
      min_unique_scores = 10000L
    )
  )
  expect_true(isTRUE(fallback$wald_diagnostics$h_treated$fallback))
  expect_true(isTRUE(fallback$wald_diagnostics$q_control$fallback))
  expect_true(is.finite(fallback$contrasts$std_error[[1L]]))

  boundary_pi <- cbind(
    "0" = ifelse(fixture$A == 0, 0.998, 0.002),
    "1" = ifelse(fixture$A == 1, 0.998, 0.002)
  )
  boundary <- calibrated_dml_from_nuisances(
    A = fixture$A,
    Y = fixture$Y,
    mu_mat = fixture$mu_mat,
    pi_mat = boundary_pi,
    control_level = 0,
    inference = "wald",
    calibration_method = "none",
    seed = 35L,
    wald_options = list(basis_size_grid = 4L, lambda_grid = 1e-3, cv_folds = 2L, min_rows = 5L)
  )
  expect_true(is.finite(boundary$contrasts$std_error[[1L]]))
  expect_equal(boundary$wald_diagnostics$h_treated$bound_upper, 40)
})

test_that("sieve-Riesz Wald matches Python reference fixture", {
  n <- 80L
  w <- seq(-1, 1, length.out = n)
  pi1 <- stats::plogis(0.15 + 0.75 * w)
  a <- as.integer(((seq_len(n) - 1L) * 7L + 3L) %% 11L < (3L + 4L * as.integer(w > 0)))
  mu0 <- 0.2 + 0.3 * w + 0.1 * w * w
  mu1 <- mu0 + 0.6 + 0.2 * w
  y <- mu0 + a * (mu1 - mu0) + 0.15 * sin((seq_len(n) - 1L) * 1.7)

  fit <- calibrated_dml_from_nuisances(
    A = a,
    Y = y,
    mu_mat = cbind("0" = mu0, "1" = mu1),
    pi_mat = cbind("0" = 1 - pi1, "1" = pi1),
    control_level = 0,
    inference = "wald",
    calibration_method = "none",
    seed = 101L,
    wald_options = list(basis_size_grid = c(4L, 6L), lambda_grid = c(1e-4, 1e-2), cv_folds = 4L, min_rows = 5L)
  )

  expect_equal(fit$contrasts$estimate[[1L]], 0.5948840416978424, tolerance = 1e-12)
  expect_equal(fit$contrasts$std_error[[1L]], 0.031816138714923135, tolerance = 1e-12)
  expect_equal(fit$contrasts$lower[[1L]], 0.5325255556894626, tolerance = 1e-12)
  expect_equal(fit$contrasts$upper[[1L]], 0.6572425277062222, tolerance = 1e-12)
  expect_equal(fit$wald_diagnostics$h_treated$basis_size, 4L)
  expect_equal(fit$wald_diagnostics$q_treated$lambda, 0.01)
})

test_that("adaptive plugin and calibrated r-learner honor supplied nuisances", {
  fixture <- oracle_binary_fixture(n = 300, seed = 11)

  plugin_fit <- adaptive_calibrated_dml(
    data = fixture$data,
    outcome = "Y",
    treatment = "A",
    covariates = c("W1", "W2"),
    control_level = 0,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    mode = "plugin",
    inference = "jackknife",
    jackknife_folds = 6
  )

  rlearner_fit <- adaptive_calibrated_dml(
    data = fixture$data,
    outcome = "Y",
    treatment = "A",
    covariates = c("W1", "W2"),
    control_level = 0,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    mode = "calibrated_rlearner",
    inference = "wald"
  )

  expect_equal(plugin_fit$adaptive_mode, "plugin")
  expect_equal(rlearner_fit$adaptive_mode, "calibrated_rlearner")
  expect_true(all(is.finite(plugin_fit$estimates$estimate)))
  expect_true(all(is.finite(rlearner_fit$estimates$estimate)))
})

test_that("working r-learner supports lasso when glmnet is available", {
  skip_if_not_installed("glmnet")
  fixture <- oracle_binary_fixture(n = 250, seed = 9)

  fit <- adaptive_calibrated_dml(
    data = fixture$data,
    outcome = "Y",
    treatment = "A",
    covariates = c("W1", "W2"),
    control_level = 0,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    mode = "working_rlearner",
    cate_model = "lasso",
    inference = "wald"
  )

  expect_equal(fit$adaptive_mode, "working_rlearner")
  expect_true(all(is.finite(fit$estimates$estimate)))
})

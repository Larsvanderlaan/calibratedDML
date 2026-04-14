test_that("auto monotone calibration chooses smooth isotonic for smaller samples", {
  set.seed(123)
  x_small <- runif(200)
  y_small <- plogis(2 * x_small) + rnorm(200, sd = 0.05)
  cal_small <- fit_monotone_calibrator(x_small, y_small, method = "auto")
  pred_small <- predict_monotone_calibrator(cal_small, sort(x_small))

  x_large <- runif(400)
  y_large <- plogis(2 * x_large) + rnorm(400, sd = 0.05)
  cal_large <- fit_monotone_calibrator(x_large, y_large, method = "auto")
  pred_large <- predict_monotone_calibrator(cal_large, sort(x_large))

  expect_equal(cal_small$method, "smooth_isotonic")
  expect_equal(cal_large$method, "isotonic")
  expect_true(all(diff(pred_small) >= -1e-8))
  expect_true(all(diff(pred_large) >= -1e-8))
})

test_that("outcome calibration stratification changes outcome-only calibration groups", {
  a <- c(rep(0, 20), rep(1, 20))
  y <- c(rep(0, 20), rep(1, 20))
  mu_mat <- cbind("0" = rep(0.2, 40), "1" = rep(0.8, 40))

  pooled <- calibrate_outcome_matrix(
    Y = y,
    mu_mat = mu_mat,
    A_index = a + 1L,
    method = "isotonic",
    calibration_stratify = NULL
  )$calibrated
  stratified <- calibrate_outcome_matrix(
    Y = y,
    mu_mat = mu_mat,
    A_index = a + 1L,
    method = "isotonic",
    calibration_stratify = "outcome"
  )$calibrated

  expect_equal(mean(pooled[, "0"]), 0.5)
  expect_equal(mean(stratified[a == 0, "0"]), 0)
  expect_equal(mean(stratified[a == 1, "1"]), 1)
})

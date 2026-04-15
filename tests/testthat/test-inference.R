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

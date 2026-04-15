test_that("standard estimator returns tidy multi-arm results from supplied nuisances", {
  fixture <- oracle_multiarm_fixture()
  fit <- calibrated_dml_from_nuisances(
    A = fixture$A,
    Y = fixture$Y,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    control_level = 0,
    inference = "wald"
  )

  expect_s3_class(fit, "calibrated_dml_fit")
  expect_equal(nrow(fit$potential_outcomes), 3)
  expect_equal(nrow(fit$contrasts), 2)
  expect_equal(unique(fit$contrasts$control_level), "0")
  expect_true(all(c("estimand_type", "level", "control_level", "estimate", "std_error", "lower", "upper") %in% names(fit$estimates)))
  expect_equal(nrow(as.data.frame(fit)), nrow(fit$estimates))
})

test_that("standard estimator defaults to jackknife with 100 folds", {
  fixture <- oracle_binary_fixture()
  fit <- calibrated_dml_from_nuisances(
    A = fixture$A,
    Y = fixture$Y,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    control_level = 0
  )

  expect_identical(fit$inference, "jackknife")
  expect_identical(fit$jackknife_folds, 100L)
})

test_that("implicit multi-arm treatment level inference matches explicit ordering for supplied nuisances", {
  fixture <- oracle_multiarm_fixture(n = 320, seed = 17)
  permuted_A <- as.character(fixture$A)
  permuted_A[permuted_A == "1"] <- "tmp"
  permuted_A[permuted_A == "2"] <- "1"
  permuted_A[permuted_A == "tmp"] <- "2"

  fit_implicit <- calibrated_dml_from_nuisances(
    A = permuted_A,
    Y = fixture$Y,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    control_level = 0,
    calibration_method = "none"
  )
  fit_explicit <- calibrated_dml_from_nuisances(
    A = permuted_A,
    Y = fixture$Y,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    control_level = 0,
    treatment_levels = c(0, 1, 2),
    calibration_method = "none"
  )

  expect_equal(fit_implicit$treatment_levels, c("0", "1", "2"))
  expect_equal(fit_implicit$estimates$estimate, fit_explicit$estimates$estimate)
})

test_that("stratify normalization accepts compact user inputs", {
  expect_equal(calibratedDML:::normalize_stratify(NULL), character())
  expect_equal(calibratedDML:::normalize_stratify(FALSE), character())
  expect_equal(calibratedDML:::normalize_stratify(TRUE), c("outcome", "treatment"))
  expect_equal(calibratedDML:::normalize_stratify("outcome"), "outcome")
  expect_equal(sort(calibratedDML:::normalize_stratify(c("treatment", "outcome", "treatment"))), c("outcome", "treatment"))
  expect_error(calibratedDML:::normalize_stratify("bad"))
})

test_that("adaptive estimator accepts supplied oracle nuisances", {
  fixture <- oracle_binary_fixture()
  fit <- adaptive_calibrated_dml(
    data = fixture$data,
    outcome = "Y",
    treatment = "A",
    covariates = c("W1", "W2"),
    control_level = 0,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    mode = "plugin",
    inference = "wald"
  )

  expect_s3_class(fit, "calibrated_dml_fit")
  expect_equal(nrow(fit$contrasts), 1)
  expect_equal(fit$adaptive_mode, "plugin")
  expect_equal(fit$calibration_method, "isotonic")
})

test_that("weighted standard estimator matches manual weighted binary score", {
  fixture <- weighted_binary_fixture()
  fit <- calibrated_dml_from_nuisances(
    A = fixture$A,
    Y = fixture$Y,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    control_level = 0,
    calibration_method = "none",
    sample_weight = fixture$sample_weight
  )

  weights <- fixture$sample_weight / sum(fixture$sample_weight)
  a_num <- as.integer(as.character(fixture$A))
  mu_obs <- fixture$mu_mat[cbind(seq_along(fixture$Y), a_num + 1L)]
  score0 <- fixture$mu_mat[, "0"] + as.numeric(a_num == 0) * (fixture$Y - mu_obs) / fixture$pi_mat[, "0"]
  score1 <- fixture$mu_mat[, "1"] + as.numeric(a_num == 1) * (fixture$Y - mu_obs) / fixture$pi_mat[, "1"]

  expect_equal(fit$potential_outcomes$estimate[[1]], sum(weights * score0))
  expect_equal(fit$potential_outcomes$estimate[[2]], sum(weights * score1))
  expect_equal(fit$contrasts$estimate[[1]], sum(weights * (score1 - score0)))
})

test_that("weighted multi-arm estimator targets weighted oracle means and contrasts", {
  fixture <- weighted_multiarm_fixture()
  fit <- calibrated_dml_from_nuisances(
    A = fixture$A,
    Y = fixture$Y,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    control_level = 0,
    sample_weight = fixture$sample_weight,
    inference = "wald"
  )
  tab <- fit$estimates
  rownames(tab) <- ifelse(tab$estimand_type == "potential_outcome", paste0("EY", tab$level), paste0("ATE", tab$level))

  expect_lt(abs(tab["EY0", "estimate"] - fixture$ey_truth[["EY0"]]), 0.2)
  expect_lt(abs(tab["EY1", "estimate"] - fixture$ey_truth[["EY1"]]), 0.2)
  expect_lt(abs(tab["EY2", "estimate"] - fixture$ey_truth[["EY2"]]), 0.2)
  expect_lt(abs(tab["ATE1", "estimate"] - fixture$contrast_truth[["ATE1"]]), 0.2)
  expect_lt(abs(tab["ATE2", "estimate"] - fixture$contrast_truth[["ATE2"]]), 0.2)
})

test_that("weighted adaptive calibrated r-learner targets weighted oracle ate", {
  fixture <- weighted_binary_fixture()
  fit <- adaptive_calibrated_dml(
    data = fixture$data,
    outcome = "Y",
    treatment = "A",
    covariates = c("W1", "W2"),
    control_level = 0,
    mu_mat = fixture$mu_mat,
    pi_mat = fixture$pi_mat,
    sample_weight = "weight",
    mode = "calibrated_rlearner",
    inference = "wald"
  )

  expect_lt(abs(fit$contrasts$estimate[[1]] - fixture$contrast_truth[["ATE1"]]), 0.2)
})

test_that("adaptive estimator no longer exposes calibration_method", {
  fixture <- oracle_binary_fixture()
  expect_error(
    adaptive_calibrated_dml(
      data = fixture$data,
      outcome = "Y",
      treatment = "A",
      covariates = c("W1", "W2"),
      control_level = 0,
      mu_mat = fixture$mu_mat,
      pi_mat = fixture$pi_mat,
      mode = "plugin",
      calibration_method = "isotonic"
    )
  )
})

test_that("main estimator fits nuisances internally with no-dependency built-ins", {
  fixture <- oracle_multiarm_fixture(n = 240, seed = 7)
  fit <- calibrated_dml(
    data = fixture$data,
    outcome = "Y",
    treatment = "A",
    covariates = c("W1", "W2"),
    control_level = 0,
    outcome_model = "mean",
    treatment_model = "multinom",
    stratify = "outcome",
    inference = "wald"
  )

  expect_s3_class(fit, "calibrated_dml_fit")
  expect_equal(fit$nuisance_source, "builtin")
  expect_equal(nrow(fit$estimates), 5)
  expect_true(all(is.finite(fit$estimates$estimate)))
})

test_that("binary multinom treatment backend returns valid probability columns", {
  fixture <- oracle_binary_fixture(n = 220, seed = 23)
  fit <- calibrated_dml(
    data = fixture$data,
    outcome = "Y",
    treatment = "A",
    covariates = c("W1", "W2"),
    control_level = 0,
    outcome_model = "lm",
    treatment_model = "multinom",
    inference = "wald",
    n_folds = 3,
    seed = 8
  )

  expect_true(all(is.finite(fit$pi_mat)))
  expect_equal(ncol(fit$pi_mat), 2)
  expect_equal(as.numeric(rowSums(fit$pi_mat)), rep(1, nrow(fit$pi_mat)))
})

test_that("main estimator fits default lasso nuisances with balnet propensity backend", {
  skip_if_not_installed("balnet")
  skip_if_not_installed("glmnet")
  fixture <- oracle_binary_fixture(n = 240, seed = 19)
  fit <- calibrated_dml(
    data = fixture$data,
    outcome = "Y",
    treatment = "A",
    covariates = c("W1", "W2"),
    control_level = 0,
    inference = "wald",
    n_folds = 3,
    seed = 4
  )

  expect_s3_class(fit, "calibrated_dml_fit")
  expect_true(all(is.finite(fit$estimates$estimate)))
  expect_true(all(is.finite(fit$estimates$std_error)))
})

test_that("SuperLearner backend works without attaching the package", {
  skip_if_not_installed("SuperLearner")

  fixture <- oracle_binary_fixture(n = 220, seed = 29)
  sl_spec <- list(SL.library = c("SL.mean", "SL.glm"))
  fit <- calibrated_dml(
    data = fixture$data,
    outcome = "Y",
    treatment = "A",
    covariates = c("W1", "W2"),
    control_level = 0,
    outcome_model = sl_spec,
    treatment_model = sl_spec,
    inference = "wald",
    n_folds = 3,
    seed = 9
  )

  expect_true(all(is.finite(fit$estimates$estimate)))
  expect_true(all(is.finite(fit$estimates$std_error)))
})

test_that("mean pseudo-values equal fold means", {
  y <- seq_len(60)^2
  jk <- calibratedDML:::vfold_jackknife(mean(y), function(keep) mean(y[keep]), length(y), 6, 12)
  fold_means <- vapply(split(y, jk$fold_id), mean, numeric(1))
  expect_equal(as.numeric(jk$pseudo_values), unname(fold_means))
  expect_equal(as.numeric(jk$standard_error), stats::sd(fold_means) / sqrt(6))
  expect_equal(jk$df, 5)
})

test_that("jackknife uses ten balanced groups and Student-t intervals", {
  set.seed(7)
  n <- 101
  A <- rep(c(0, 1), length.out = n)
  Y <- A + rnorm(n)
  mu <- cbind(rep(0, n), rep(1, n))
  pi <- matrix(0.5, n, 2)
  fit <- calibrated_dml_from_nuisances(A, Y, mu, pi, control_level = 0,
    fold_id = rep(1:3, length.out = n), seed = 42)
  jk <- fit$jackknife
  counts <- table(jk$fold_id)
  expect_equal(length(counts), 10)
  expect_lte(max(counts) - min(counts), 1)
  expect_equal(jk$df, 9)
  tab <- summary(fit)
  expect_equal(tab$upper, tab$estimate + qt(0.975, 9) * tab$std_error)
  expect_equal(tab$lower, tab$estimate - qt(0.975, 9) * tab$std_error)
  expect_equal(unname(jk$standard_error), unname(apply(jk$pseudo_values, 2, sd) / sqrt(10)))
})

test_that("raw nuisance estimation runs once", {
  calls <- 0L
  original <- calibratedDML:::fit_cdml_nuisances
  testthat::local_mocked_bindings(fit_cdml_nuisances = function(...) {
    calls <<- calls + 1L
    original(...)
  }, .package = "calibratedDML")
  set.seed(7)
  df <- data.frame(W1 = rnorm(120), W2 = rnorm(120), A = rep(c(0, 1), 60))
  df$Y <- df$A + df$W1 + rnorm(120)
  fit <- calibrated_dml(df, "Y", "A", c("W1", "W2"), 0,
    outcome_model = "lm", treatment_model = "multinom", seed = 7)
  expect_equal(calls, 1)
  expect_equal(nrow(fit$jackknife$leaveout_estimates), 10)
})

test_that("small samples and invalid fold counts are handled", {
  jk <- calibratedDML:::vfold_jackknife(2, function(keep) 1, 3, 10)
  expect_equal(jk$df, 2)
  expect_error(calibratedDML:::vfold_jackknife(2, function(keep) 1, 20, 1), "jackknife_folds")
  expect_error(calibratedDML:::vfold_jackknife(2, function(keep) 1, 20, 2.5), "jackknife_folds")
})

test_that("adaptive jackknife fits initial learners once", {
  nuisance_calls <- 0L
  cate_calls <- 0L
  original_nuisance <- calibratedDML:::fit_cdml_nuisances
  original_cate <- calibratedDML:::fit_binary_cate
  testthat::local_mocked_bindings(
    fit_cdml_nuisances = function(...) {
      nuisance_calls <<- nuisance_calls + 1L
      original_nuisance(...)
    },
    fit_binary_cate = function(...) {
      cate_calls <<- cate_calls + 1L
      original_cate(...)
    }, .package = "calibratedDML")
  set.seed(7)
  df <- data.frame(W1 = rnorm(120), W2 = rnorm(120), A = rep(c(0, 1), 60))
  df$Y <- df$A + df$W1 + rnorm(120)
  for (mode in c("plugin", "calibrated_rlearner")) {
    nuisance_calls <- cate_calls <- 0L
    fit <- adaptive_calibrated_dml(df, "Y", "A", c("W1", "W2"), 0,
      mode = mode, outcome_model = "lm", treatment_model = "multinom",
      cate_model = "lm", seed = 7)
    expect_equal(nuisance_calls, 1)
    expect_equal(cate_calls, as.integer(mode == "calibrated_rlearner"))
    expect_equal(fit$jackknife$df, 9)
    expect_equal(fit$estimates$upper, fit$estimates$estimate + qt(0.975, 9) * fit$estimates$std_error)
  }
})

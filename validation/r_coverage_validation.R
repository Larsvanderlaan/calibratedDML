.libPaths(c("/tmp/r-lib-calibrateddml", .libPaths()))

suppressPackageStartupMessages({
  library(calibratedDML)
})

source("tests/testthat/helper-fixtures.R")

nominal_coverage <- 0.90

extract_standard_coverage <- function(fit, truth) {
  estimate_table <- fit$estimates
  result <- numeric(length(truth))
  names(result) <- names(truth)

  for (name in names(truth)) {
    if (startsWith(name, "EY")) {
      level <- sub("^EY", "", name)
      row <- estimate_table[estimate_table$estimand_type == "potential_outcome" & as.character(estimate_table$level) == level, , drop = FALSE][1, ]
    } else {
      level <- sub("^ATE", "", name)
      row <- estimate_table[estimate_table$estimand_type == "contrast" & as.character(estimate_table$level) == level, , drop = FALSE][1, ]
    }
    result[[name]] <- as.numeric(row$lower <= truth[[name]] && truth[[name]] <= row$upper)
  }

  result
}

summarize_coverages <- function(draws) {
  coverage <- colMeans(do.call(rbind, draws))
  summary <- data.frame(
    estimand = names(coverage),
    coverage = as.numeric(coverage),
    nominal = nominal_coverage,
    deviation = as.numeric(coverage) - nominal_coverage,
    row.names = NULL
  )
  rbind(summary, data.frame(
    estimand = "average",
    coverage = mean(summary$coverage),
    nominal = nominal_coverage,
    deviation = mean(summary$coverage) - nominal_coverage
  ))
}

safe_run <- function(label, expr) {
  tryCatch(
    expr,
    error = function(e) {
      data.frame(
        estimand = "error",
        coverage = NA_real_,
        nominal = nominal_coverage,
        deviation = NA_real_,
        message = paste(label, "failed:", conditionMessage(e))
      )
    }
  )
}

run_standard <- function(fixture_factory, inference, n_rep, ..., n = NULL) {
  draws <- vector("list", n_rep)
  for (rep in seq_len(n_rep)) {
    fixture <- fixture_factory(n = n, seed = 1000 + rep)
    fit <- calibrated_dml_from_nuisances(
      A = fixture$A,
      Y = fixture$Y,
      mu_mat = fixture$mu_mat,
      pi_mat = fixture$pi_mat,
      control_level = 0,
      conf_level = nominal_coverage,
      inference = inference,
      ...
    )
    draws[[rep]] <- extract_standard_coverage(fit, c(fixture$ey_truth, fixture$contrast_truth))
  }
  summarize_coverages(draws)
}

run_adaptive <- function(fixture_factory, mode, n_rep, ..., n = NULL) {
  draws <- vector("list", n_rep)
  for (rep in seq_len(n_rep)) {
    fixture <- fixture_factory(n = n, seed = 2000 + rep)
    fit <- adaptive_calibrated_dml(
      data = fixture$data,
      outcome = "Y",
      treatment = "A",
      covariates = c("W1", "W2"),
      control_level = 0,
      mu_mat = fixture$mu_mat,
      pi_mat = fixture$pi_mat,
      mode = mode,
      conf_level = nominal_coverage,
      ...
    )
    draws[[rep]] <- extract_standard_coverage(fit, c(fixture$ey_truth, fixture$contrast_truth))
  }
  summarize_coverages(draws)
}

run_fitted <- function(fixture_factory, n_rep, outcome_model = "lm", treatment_model = "lm", n = NULL) {
  draws <- vector("list", n_rep)
  for (rep in seq_len(n_rep)) {
    fixture <- fixture_factory(n = n, seed = 3000 + rep)
    fit <- calibrated_dml(
      data = fixture$data,
      outcome = "Y",
      treatment = "A",
      covariates = c("W1", "W2"),
      control_level = 0,
      outcome_model = outcome_model,
      treatment_model = treatment_model,
      conf_level = nominal_coverage,
      inference = "wald",
      calibration_method = "auto",
      n_folds = 5,
      seed = rep
    )
    draws[[rep]] <- extract_standard_coverage(fit, c(fixture$ey_truth, fixture$contrast_truth))
  }
  summarize_coverages(draws)
}

studies <- list(
  standard_binary_wald = safe_run("standard_binary_wald", run_standard(oracle_binary_fixture, "wald", n_rep = 20, n = 600)),
  standard_binary_bootstrap = safe_run("standard_binary_bootstrap", run_standard(oracle_binary_fixture, "bootstrap", n_rep = 16, n = 600, bootstrap_reps = 40, seed = 7)),
  standard_binary_jackknife = safe_run("standard_binary_jackknife", run_standard(oracle_binary_fixture, "jackknife", n_rep = 16, n = 600, jackknife_folds = 10)),
  standard_multiarm_wald = safe_run("standard_multiarm_wald", run_standard(oracle_multiarm_fixture, "wald", n_rep = 16, n = 700)),
  standard_multiarm_bootstrap = safe_run("standard_multiarm_bootstrap", run_standard(oracle_multiarm_fixture, "bootstrap", n_rep = 12, n = 700, bootstrap_reps = 40, seed = 9)),
  standard_multiarm_jackknife = safe_run("standard_multiarm_jackknife", run_standard(oracle_multiarm_fixture, "jackknife", n_rep = 12, n = 700, jackknife_folds = 10)),
  adaptive_plugin_linear = safe_run("adaptive_plugin_linear", run_adaptive(oracle_binary_fixture, "plugin", n_rep = 16, n = 600, inference = "wald")),
  adaptive_plugin_nonlinear = safe_run("adaptive_plugin_nonlinear", run_adaptive(oracle_binary_nonlinear_fixture, "plugin", n_rep = 16, n = 600, inference = "wald")),
  adaptive_rlearner_linear = safe_run("adaptive_rlearner_linear", run_adaptive(oracle_binary_fixture, "calibrated_rlearner", n_rep = 16, n = 600, inference = "wald", cate_model = "lm")),
  fitted_binary_wald = safe_run("fitted_binary_wald", run_fitted(oracle_binary_fixture, n_rep = 12, n = 600, outcome_model = "lm", treatment_model = "multinom"))
)

for (name in names(studies)) {
  cat("\n##", name, "\n")
  print(studies[[name]])
}

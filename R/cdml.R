#' Calibrated Debiased Machine Learning for Categorical Treatments
#'
#' `calibrated_dml()` estimates mean potential outcomes for every treatment arm
#' and treatment-vs-control contrasts for a categorical treatment using
#' calibrated debiased machine learning. Users can either provide cross-fitted
#' nuisance matrices directly or ask the package to fit nuisances with built-in
#' models or advanced `sl3` learners.
#'
#' @param data A `data.frame` containing the analysis variables.
#' @param outcome Name of the outcome column in `data`.
#' @param treatment Name of the categorical treatment column in `data`.
#' @param covariates Character vector naming covariate columns in `data`.
#' @param control_level The reference treatment level used for reported
#'   contrasts.
#' @param outcome_model Outcome nuisance model specification. Built-in options
#'   include `"mean"`, `"lm"`, `"lasso"`, `"gam"`, `"random_forest"`,
#'   `"boosted_trees"`, and `"auto"`, or a custom model spec list with `fit`
#'   and `predict` elements, a `SuperLearner` library, or an `sl3` learner.
#' @param treatment_model Treatment nuisance model specification. Built-in
#'   options include `"mean"`, `"lm"`, `"lasso"`, `"gam"`, `"random_forest"`,
#'   `"boosted_trees"`, `"auto"`, `"multinom"`, and `"empirical"`, or a
#'   custom model spec list with `fit` and `predict` elements, a
#'   `SuperLearner` library, or an `sl3` learner. For binary treatment, `"lm"`
#'   and `"lasso"` use balancing-weight fits via `balnet`.
#' @param mu_mat Optional matrix of cross-fitted outcome nuisance estimates with
#'   one column per treatment level.
#' @param pi_mat Optional matrix of cross-fitted treatment propensity estimates
#'   with one column per treatment level.
#' @param sample_weight Optional vector of non-negative sample weights.
#' @param treatment_levels Optional explicit ordering of treatment levels. When
#'   omitted, the observed treatment levels are used.
#' @param stratify Controls whether nuisance fitting is stratified by
#'   `"outcome"`, `"treatment"`, or both.
#' @param calibration_method One of `"auto"`, `"isotonic"`,
#'   `"smooth_isotonic"`, or `"none"`.
#' @param calibration_options Optional list of calibration tuning options.
#' @param calibration_stratify Optional calibration stratification control. Set
#'   to `"outcome"` to calibrate outcome nuisance coordinates within observed
#'   treatment arm.
#' @param n_folds Number of folds for nuisance fitting when nuisances are not
#'   supplied.
#' @param fold_id Optional fold assignment vector.
#' @param inference Inference mode. One of `"jackknife"`, `"bootstrap"`, or
#'   `"wald"`. The default is `"jackknife"` for the standard estimator.
#' @param conf_level Confidence level for reported intervals.
#' @param bootstrap_reps Number of bootstrap resamples used for interval
#'   estimation when `inference = "bootstrap"`.
#' @param jackknife_folds Number of delete-a-group folds used when
#'   `inference = "jackknife"`. The default is `100`.
#' @param wald_correction Wald standard-error correction. `"auto"` uses the
#'   corrected level-set Riesz standard error for binary-treatment Wald
#'   contrasts and the standard Wald standard error otherwise. `"none"` uses
#'   the standard Wald standard error.
#' @param wald_conservative If `TRUE`, binary corrected Wald uses the maximum
#'   of the standard Wald and corrected Wald standard errors.
#' @param alpha Legacy compatibility alias for `1 - conf_level`.
#' @param n_boot Legacy compatibility alias for `bootstrap_reps`.
#' @param seed Optional random seed used for fold creation and bootstrap
#'   resampling.
#'
#' @return An object of class `"calibrated_dml_fit"`.
#' @export
calibrated_dml <- function(
  data,
  outcome,
  treatment,
  covariates,
  control_level,
  outcome_model = "lasso",
  treatment_model = "lasso",
  mu_mat = NULL,
  pi_mat = NULL,
  sample_weight = NULL,
  treatment_levels = NULL,
  stratify = c("outcome", "treatment"),
  calibration_method = c("auto", "isotonic", "smooth_isotonic", "none"),
  calibration_options = list(),
  calibration_stratify = NULL,
  n_folds = 5,
  fold_id = NULL,
  inference = c("jackknife", "bootstrap", "wald"),
  conf_level = 0.95,
  bootstrap_reps = 200,
  jackknife_folds = 100,
  wald_correction = c("auto", "none"),
  wald_conservative = FALSE,
  alpha = NULL,
  n_boot = NULL,
  seed = NULL
) {
  inference <- match.arg(inference)
  wald_correction <- validate_wald_correction(wald_correction)
  wald_conservative <- validate_wald_conservative(wald_conservative)
  calibration_method <- match.arg(calibration_method)
  if (!is.data.frame(data)) {
    stop("`data` must be a data.frame.", call. = FALSE)
  }
  if (missing(outcome) || missing(treatment) || missing(covariates)) {
    stop("`outcome`, `treatment`, and `covariates` are required.", call. = FALSE)
  }
  assert_columns_exist(data, c(outcome, treatment, covariates))

  y <- data[[outcome]]
  a <- data[[treatment]]
  x <- as.data.frame(data[covariates], stringsAsFactors = FALSE)
  weights <- resolve_sample_weight(sample_weight, nrow(data), data)
  inference_options <- resolve_inference_options(
    conf_level = conf_level,
    alpha = alpha,
    bootstrap_reps = bootstrap_reps,
    n_boot = n_boot,
    jackknife_folds = jackknife_folds
  )
  stratify <- normalize_stratify(stratify)

  if (xor(is.null(mu_mat), is.null(pi_mat))) {
    stop("Provide either both `mu_mat` and `pi_mat` or neither of them.", call. = FALSE)
  }

  if (!is.null(mu_mat)) {
    return(calibrated_dml_from_nuisances(
      A = a,
      Y = y,
      mu_mat = mu_mat,
      pi_mat = pi_mat,
      control_level = control_level,
      sample_weight = weights,
      treatment_levels = treatment_levels,
      conf_level = inference_options$conf_level,
      inference = inference,
      bootstrap_reps = inference_options$bootstrap_reps,
      jackknife_folds = inference_options$jackknife_folds,
      wald_correction = wald_correction,
      wald_conservative = wald_conservative,
      calibration_method = calibration_method,
      calibration_options = calibration_options,
      calibration_stratify = calibration_stratify,
      fold_id = fold_id,
      seed = seed,
      nuisance_source = "supplied"
    ))
  }

  fit <- fit_cdml_nuisances(
    X = x,
    A = a,
    Y = y,
    sample_weight = weights,
    treatment_levels = treatment_levels,
    control_level = control_level,
    outcome_model = outcome_model,
    treatment_model = treatment_model,
    stratify = stratify,
    n_folds = n_folds,
    fold_id = fold_id,
    seed = seed
  )

  result <- calibrated_dml_from_nuisances(
    A = fit$A_factor,
    Y = y,
    mu_mat = fit$mu_mat,
    pi_mat = fit$pi_mat,
    control_level = control_level,
    sample_weight = weights,
    treatment_levels = fit$treatment_levels,
    conf_level = inference_options$conf_level,
    inference = inference,
    bootstrap_reps = inference_options$bootstrap_reps,
    jackknife_folds = inference_options$jackknife_folds,
    wald_correction = wald_correction,
    wald_conservative = wald_conservative,
    calibration_method = calibration_method,
    calibration_options = calibration_options,
    calibration_stratify = calibration_stratify,
    fold_id = fit$fold_id,
    seed = seed,
    nuisance_source = fit$nuisance_source
  )
  result$nuisance_fit <- fit
  result$call <- match.call()
  result
}

#' Estimate Calibrated DML from Supplied Nuisance Matrices
#'
#' @param A Treatment vector. Must be categorical/discrete.
#' @param Y Outcome vector.
#' @param mu_mat Matrix of cross-fitted outcome nuisance estimates with one
#'   column per treatment level.
#' @param pi_mat Matrix of cross-fitted propensity estimates with one column per
#'   treatment level.
#' @param control_level The reference treatment level used for reported
#'   contrasts.
#' @param sample_weight Optional non-negative sample weights.
#' @param treatment_levels Optional explicit ordering of treatment levels.
#' @param inference Inference mode. One of `"jackknife"`, `"bootstrap"`, or
#'   `"wald"`. The default is `"jackknife"` for the standard estimator.
#' @param conf_level Confidence level for reported intervals.
#' @param bootstrap_reps Number of bootstrap resamples.
#' @param jackknife_folds Number of delete-a-group folds. The default is `100`.
#' @param wald_correction Wald standard-error correction. `"auto"` uses the
#'   corrected level-set Riesz standard error for binary-treatment Wald
#'   contrasts and the standard Wald standard error otherwise. `"none"` uses
#'   the standard Wald standard error.
#' @param wald_conservative If `TRUE`, binary corrected Wald uses the maximum
#'   of the standard Wald and corrected Wald standard errors.
#' @param calibration_method One of `"auto"`, `"isotonic"`,
#'   `"smooth_isotonic"`, or `"none"`.
#' @param calibration_options Optional list of calibration tuning options.
#' @param calibration_stratify Optional calibration stratification control.
#' @param fold_id Optional fold assignment vector used for fold-wise bootstrap
#'   resampling.
#' @param seed Optional random seed used for bootstrap resampling.
#' @param alpha Legacy compatibility alias for `1 - conf_level`.
#' @param n_boot Legacy compatibility alias for `bootstrap_reps`.
#' @param nuisance_source Internal label describing how nuisance estimates were
#'   obtained.
#'
#' @return An object of class `"calibrated_dml_fit"`.
#' @export
calibrated_dml_from_nuisances <- function(
  A,
  Y,
  mu_mat,
  pi_mat,
  control_level,
  sample_weight = NULL,
  treatment_levels = NULL,
  conf_level = 0.95,
  inference = c("jackknife", "bootstrap", "wald"),
  bootstrap_reps = 200,
  jackknife_folds = 100,
  wald_correction = c("auto", "none"),
  wald_conservative = FALSE,
  calibration_method = c("auto", "isotonic", "smooth_isotonic", "none"),
  calibration_options = list(),
  calibration_stratify = NULL,
  fold_id = NULL,
  seed = NULL,
  alpha = NULL,
  n_boot = NULL,
  nuisance_source = "supplied"
) {
  inference <- match.arg(inference)
  wald_correction <- validate_wald_correction(wald_correction)
  wald_conservative <- validate_wald_conservative(wald_conservative)
  calibration_method <- match.arg(calibration_method)
  inference_options <- resolve_inference_options(
    conf_level = conf_level,
    alpha = alpha,
    bootstrap_reps = bootstrap_reps,
    n_boot = n_boot,
    jackknife_folds = jackknife_folds
  )
  standardized <- standardize_treatment(A, control_level, treatment_levels)
  weights <- resolve_sample_weight(sample_weight, length(A))
  y <- as.numeric(Y)
  if (anyNA(y)) {
    stop("`Y` cannot contain missing values.", call. = FALSE)
  }

  mu_aligned <- align_nuisance_matrix(mu_mat, standardized$levels, "mu_mat")
  pi_aligned <- align_nuisance_matrix(pi_mat, standardized$levels, "pi_mat")

  base_fit <- compute_cdml_estimates(
    A_index = standardized$index,
    Y = y,
    mu_mat = mu_aligned,
    pi_mat = pi_aligned,
    weights = weights,
    levels = standardized$levels,
    control_index = standardized$control_index,
    calibration_method = calibration_method,
    calibration_options = calibration_options,
    calibration_stratify = calibration_stratify
  )

  interval_table <- compute_cdml_intervals(
    base_fit = base_fit,
    A_index = standardized$index,
    Y = y,
    mu_mat = mu_aligned,
    pi_mat = pi_aligned,
    weights = weights,
    levels = standardized$levels,
    control_index = standardized$control_index,
    conf_level = inference_options$conf_level,
    inference = inference,
    bootstrap_reps = inference_options$bootstrap_reps,
    jackknife_folds = inference_options$jackknife_folds,
    wald_correction = wald_correction,
    wald_conservative = wald_conservative,
    calibration_method = calibration_method,
    calibration_options = calibration_options,
    calibration_stratify = calibration_stratify,
    fold_id = fold_id,
    seed = seed
  )

  result <- list(
    call = NULL,
    control_level = standardized$levels[[standardized$control_index]],
    treatment_levels = standardized$levels,
    nuisance_source = nuisance_source,
    inference = inference,
    conf_level = inference_options$conf_level,
    bootstrap_reps = inference_options$bootstrap_reps,
    jackknife_folds = inference_options$jackknife_folds,
    wald_correction = wald_correction,
    wald_conservative = isTRUE(wald_conservative),
    calibration_method = calibration_method,
    calibration_stratify = normalize_calibration_stratify(calibration_stratify),
    fold_id = fold_id,
    sample_weight = weights,
    mu_mat = mu_aligned,
    pi_mat = pi_aligned,
    calibrated_mu_mat = base_fit$calibrated_mu_mat,
    calibrated_pi_mat = base_fit$calibrated_pi_mat,
    calibration_bundle = base_fit$calibration_bundle,
    potential_outcomes = interval_table$potential_outcomes,
    contrasts = interval_table$contrasts,
    estimates = interval_table$estimates,
    wald_diagnostics = interval_table$wald_diagnostics
  )
  class(result) <- "calibrated_dml_fit"
  result
}

#' @export
print.calibrated_dml_fit <- function(x, ...) {
  cat("calibratedDML fit\n")
  cat("  treatment levels:", paste(x$treatment_levels, collapse = ", "), "\n")
  cat("  control level:", x$control_level, "\n")
  cat("  nuisance source:", x$nuisance_source, "\n")
  cat("  calibration:", x$calibration_method, "\n")
  cat("  inference:", x$inference, "\n")
  print(x$estimates, row.names = FALSE, ...)
  invisible(x)
}

#' @export
summary.calibrated_dml_fit <- function(object, ...) {
  object$estimates
}

#' @export
coef.calibrated_dml_fit <- function(object, type = c("all", "contrast", "potential_outcome"), ...) {
  type <- match.arg(type)
  table <- switch(
    type,
    all = object$estimates,
    contrast = object$contrasts,
    potential_outcome = object$potential_outcomes
  )
  stats::setNames(table$estimate, table$estimand)
}

#' @export
confint.calibrated_dml_fit <- function(object, type = c("all", "contrast", "potential_outcome"), ...) {
  type <- match.arg(type)
  table <- switch(
    type,
    all = object$estimates,
    contrast = object$contrasts,
    potential_outcome = object$potential_outcomes
  )
  out <- as.matrix(table[c("lower", "upper")])
  rownames(out) <- table$estimand
  colnames(out) <- c("2.5 %", "97.5 %")
  out
}

#' @export
predict.calibrated_dml_fit <- function(object, type = c("all", "contrast", "potential_outcome"), ...) {
  type <- match.arg(type)
  switch(
    type,
    all = object$estimates,
    contrast = object$contrasts,
    potential_outcome = object$potential_outcomes
  )
}

#' @export
as.data.frame.calibrated_dml_fit <- function(x, row.names = NULL, optional = FALSE, ...) {
  x$estimates
}

compute_cdml_estimates <- function(
  A_index,
  Y,
  mu_mat,
  pi_mat,
  weights,
  levels,
  control_index,
  calibration_method,
  calibration_options,
  calibration_stratify
) {
  calibration_bundle <- fit_calibration_bundle(
    Y = Y,
    mu_mat = mu_mat,
    A_index = A_index,
    pi_mat = pi_mat,
    weights = weights,
    calibration_method = calibration_method,
    calibration_options = calibration_options,
    calibration_stratify = calibration_stratify
  )
  calibrated_mu <- calibration_bundle$calibrated_mu_mat
  calibrated_pi <- calibration_bundle$calibrated_pi_mat

  observed_mu <- calibrated_mu[cbind(seq_along(A_index), A_index)]
  normalized_weights <- normalize_weights(weights)

  pseudo_outcomes <- vector("list", length(levels))
  arm_estimates <- numeric(length(levels))
  arm_se <- numeric(length(levels))

  for (level_index in seq_along(levels)) {
    score <- calibrated_mu[, level_index] +
      (A_index == level_index) * (Y - observed_mu) / calibrated_pi[, level_index]
    pseudo_outcomes[[level_index]] <- score
    arm_estimates[[level_index]] <- sum(normalized_weights * score)
    centered <- score - arm_estimates[[level_index]]
    arm_se[[level_index]] <- sqrt(sum((normalized_weights * centered) ^ 2))
  }

  names(arm_estimates) <- levels
  names(arm_se) <- levels

  contrast_estimates <- arm_estimates[-control_index] - arm_estimates[[control_index]]
  contrast_se <- numeric(length(contrast_estimates))
  contrast_scores <- vector("list", length(contrast_estimates))
  contrast_levels <- levels[-control_index]

  for (index in seq_along(contrast_levels)) {
    score <- pseudo_outcomes[[match(contrast_levels[[index]], levels)]] -
      pseudo_outcomes[[control_index]]
    contrast_scores[[index]] <- score
    centered <- score - contrast_estimates[[index]]
    contrast_se[[index]] <- sqrt(sum((normalized_weights * centered) ^ 2))
  }

  names(contrast_estimates) <- contrast_levels
  names(contrast_se) <- contrast_levels

  list(
    calibration_bundle = calibration_bundle,
    calibrated_mu_mat = calibrated_mu,
    calibrated_pi_mat = calibrated_pi,
    arm_estimates = arm_estimates,
    arm_standard_error = arm_se,
    arm_scores = pseudo_outcomes,
    contrast_estimates = contrast_estimates,
    contrast_standard_error = contrast_se,
    contrast_scores = contrast_scores
  )
}

compute_cdml_intervals <- function(
  base_fit,
  A_index,
  Y,
  mu_mat,
  pi_mat,
  weights,
  levels,
  control_index,
  conf_level,
  inference,
  bootstrap_reps,
  jackknife_folds,
  wald_correction,
  wald_conservative,
  calibration_method,
  calibration_options,
  calibration_stratify,
  fold_id,
  seed
) {
  alpha <- 1 - conf_level
  z_value <- stats::qnorm(1 - alpha / 2)

  potential_outcomes <- data.frame(
    estimand_type = "potential_outcome",
    estimand = paste0("E[Y(", levels, ")]"),
    level = levels,
    control_level = NA_character_,
    estimate = unname(base_fit$arm_estimates),
    std_error = unname(base_fit$arm_standard_error),
    lower = unname(base_fit$arm_estimates - z_value * base_fit$arm_standard_error),
    upper = unname(base_fit$arm_estimates + z_value * base_fit$arm_standard_error),
    stringsAsFactors = FALSE
  )

  contrast_levels <- levels[-control_index]
  contrasts <- data.frame(
    estimand_type = "contrast",
    estimand = paste0("E[Y(", contrast_levels, ")] - E[Y(", levels[[control_index]], ")]"),
    level = contrast_levels,
    control_level = levels[[control_index]],
    estimate = unname(base_fit$contrast_estimates),
    std_error = unname(base_fit$contrast_standard_error),
    lower = unname(base_fit$contrast_estimates - z_value * base_fit$contrast_standard_error),
    upper = unname(base_fit$contrast_estimates + z_value * base_fit$contrast_standard_error),
    stringsAsFactors = FALSE
  )
  wald_diagnostics <- list(
    wald_correction = "none",
    applied = FALSE,
    fallback_reason = NA_character_
  )

  if (identical(inference, "wald") && identical(wald_correction, "auto")) {
    if (length(levels) == 2L) {
      corrected <- compute_levelset_riesz_wald_correction(
        A_index = A_index,
        Y = Y,
        weights = weights,
        levels = levels,
        control_index = control_index,
        calibrated_mu_mat = base_fit$calibrated_mu_mat,
        calibrated_pi_mat = base_fit$calibrated_pi_mat,
        contrast_estimate = unname(base_fit$contrast_estimates[[1L]]),
        simple_wald_std_error = unname(base_fit$contrast_standard_error[[1L]]),
        conf_level = conf_level,
        seed = seed,
        wald_conservative = wald_conservative
      )
      contrasts$std_error[[1L]] <- corrected$std_error
      contrasts$lower[[1L]] <- corrected$lower
      contrasts$upper[[1L]] <- corrected$upper
      wald_diagnostics <- corrected$diagnostics
      wald_diagnostics$applied <- TRUE
    } else {
      wald_diagnostics <- list(
        wald_correction = "standard",
        applied = FALSE,
        fallback_reason = "non_binary_treatment"
      )
    }
  }

  if (identical(inference, "bootstrap") && isTRUE(bootstrap_reps > 0)) {
    bootstrap <- bootstrap_cdml_internal(
      A_index = A_index,
      Y = Y,
      mu_mat = mu_mat,
      pi_mat = pi_mat,
      weights = weights,
      levels = levels,
      control_index = control_index,
      n_boot = bootstrap_reps,
      calibration_method = calibration_method,
      calibration_options = calibration_options,
      calibration_stratify = calibration_stratify,
      fold_id = fold_id,
      seed = seed
    )
    if (!is.null(bootstrap)) {
      potential_boot <- bootstrap[, seq_along(levels), drop = FALSE]
      contrast_boot <- bootstrap[, -seq_along(levels), drop = FALSE]

      potential_outcomes$std_error <- apply(potential_boot, 2, stats::sd)
      potential_outcomes$lower <- apply(potential_boot, 2, stats::quantile, probs = alpha / 2, na.rm = TRUE)
      potential_outcomes$upper <- apply(potential_boot, 2, stats::quantile, probs = 1 - alpha / 2, na.rm = TRUE)

      if (ncol(contrast_boot) > 0) {
        contrasts$std_error <- apply(contrast_boot, 2, stats::sd)
        contrasts$lower <- apply(contrast_boot, 2, stats::quantile, probs = alpha / 2, na.rm = TRUE)
        contrasts$upper <- apply(contrast_boot, 2, stats::quantile, probs = 1 - alpha / 2, na.rm = TRUE)
      }
    }
  }

  if (identical(inference, "jackknife")) {
    jackknife <- jackknife_cdml_internal(
      base_fit = base_fit,
      A_index = A_index,
      Y = Y,
      mu_mat = mu_mat,
      pi_mat = pi_mat,
      weights = weights,
      levels = levels,
      control_index = control_index,
      jackknife_folds = jackknife_folds,
      calibration_method = calibration_method,
      calibration_options = calibration_options,
      calibration_stratify = calibration_stratify,
      fold_id = fold_id
    )
    potential_outcomes$std_error <- jackknife$arm_standard_error
    potential_outcomes$lower <- potential_outcomes$estimate - z_value * potential_outcomes$std_error
    potential_outcomes$upper <- potential_outcomes$estimate + z_value * potential_outcomes$std_error
    if (nrow(contrasts) > 0) {
      contrasts$std_error <- jackknife$contrast_standard_error
      contrasts$lower <- contrasts$estimate - z_value * contrasts$std_error
      contrasts$upper <- contrasts$estimate + z_value * contrasts$std_error
    }
  }

  estimates <- rbind(potential_outcomes, contrasts)
  rownames(estimates) <- NULL
  list(
    potential_outcomes = potential_outcomes,
    contrasts = contrasts,
    estimates = estimates,
    wald_diagnostics = wald_diagnostics
  )
}

bootstrap_cdml_internal <- function(
  A_index,
  Y,
  mu_mat,
  pi_mat,
  weights,
  levels,
  control_index,
  n_boot,
  calibration_method,
  calibration_options,
  calibration_stratify,
  fold_id,
  seed
) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  n <- length(A_index)
  resamples <- vector("list", n_boot)

  for (boot in seq_len(n_boot)) {
    index <- bootstrap_indices(n = n, fold_id = fold_id)
    boot_fit <- compute_cdml_estimates(
      A_index = A_index[index],
      Y = Y[index],
      mu_mat = mu_mat[index, , drop = FALSE],
      pi_mat = pi_mat[index, , drop = FALSE],
      weights = weights[index],
      levels = levels,
      control_index = control_index,
      calibration_method = calibration_method,
      calibration_options = calibration_options,
      calibration_stratify = calibration_stratify
    )
    resamples[[boot]] <- c(unname(boot_fit$arm_estimates), unname(boot_fit$contrast_estimates))
  }

  do.call(rbind, resamples)
}

jackknife_cdml_internal <- function(
  base_fit,
  A_index,
  Y,
  mu_mat,
  pi_mat,
  weights,
  levels,
  control_index,
  jackknife_folds,
  calibration_method,
  calibration_options,
  calibration_stratify,
  fold_id
) {
  if (is.null(fold_id)) {
    fold_id <- resolve_fold_id(length(A_index), n_folds = min(jackknife_folds, length(A_index)), fold_id = NULL, seed = 1)
  }
  groups <- split(seq_along(A_index), fold_id)
  n_groups <- length(groups)
  if (n_groups < 2) {
    stop("Jackknife inference requires at least two resampling groups.", call. = FALSE)
  }

  leave_one_out <- vector("list", n_groups)
  for (index in seq_along(groups)) {
    keep <- setdiff(seq_along(A_index), groups[[index]])
    leave_one_out[[index]] <- compute_cdml_estimates(
      A_index = A_index[keep],
      Y = Y[keep],
      mu_mat = mu_mat[keep, , drop = FALSE],
      pi_mat = pi_mat[keep, , drop = FALSE],
      weights = weights[keep],
      levels = levels,
      control_index = control_index,
      calibration_method = calibration_method,
      calibration_options = calibration_options,
      calibration_stratify = calibration_stratify
    )
  }

  arm_matrix <- do.call(rbind, lapply(leave_one_out, function(x) unname(x$arm_estimates)))
  pseudo_arm <- n_groups * matrix(base_fit$arm_estimates, nrow = n_groups, ncol = length(levels), byrow = TRUE) -
    (n_groups - 1) * arm_matrix
  arm_se <- apply(pseudo_arm, 2, stats::sd) / sqrt(n_groups)

  contrast_matrix <- do.call(rbind, lapply(leave_one_out, function(x) unname(x$contrast_estimates)))
  if (ncol(contrast_matrix) > 0) {
    pseudo_contrast <- n_groups * matrix(base_fit$contrast_estimates, nrow = n_groups, ncol = ncol(contrast_matrix), byrow = TRUE) -
      (n_groups - 1) * contrast_matrix
    contrast_se <- apply(pseudo_contrast, 2, stats::sd) / sqrt(n_groups)
  } else {
    contrast_se <- numeric()
  }

  list(
    arm_standard_error = arm_se,
    contrast_standard_error = contrast_se
  )
}

fit_cdml_nuisances <- function(
  X,
  A,
  Y,
  sample_weight,
  treatment_levels,
  control_level,
  outcome_model,
  treatment_model,
  stratify,
  n_folds,
  fold_id,
  seed
) {
  standardized <- standardize_treatment(A, control_level, treatment_levels)
  folds <- resolve_fold_id(
    length(A),
    n_folds = n_folds,
    fold_id = fold_id,
    seed = seed,
    treatment_index = standardized$index
  )
  levels <- standardized$levels
  outcome_spec <- resolve_outcome_model(outcome_model)
  treatment_spec <- resolve_treatment_model(treatment_model)
  stratify <- normalize_stratify(stratify)

  mu_mat <- matrix(NA_real_, nrow = nrow(X), ncol = length(levels))
  pi_mat <- matrix(NA_real_, nrow = nrow(X), ncol = length(levels))
  colnames(mu_mat) <- levels
  colnames(pi_mat) <- levels

  for (fold in sort(unique(folds))) {
    train_index <- which(folds != fold)
    valid_index <- which(folds == fold)

    if (outcome_spec$backend == "sl3" || treatment_spec$backend == "sl3") {
      fitted <- fit_sl3_fold(
        X_train = X[train_index, , drop = FALSE],
        A_train = standardized$index[train_index],
      Y_train = Y[train_index],
      W_train = sample_weight[train_index],
      X_valid = X[valid_index, , drop = FALSE],
      levels = levels,
      outcome_spec = outcome_spec,
      treatment_spec = treatment_spec,
      stratify = stratify
    )
      mu_mat[valid_index, ] <- fitted$mu_mat
      pi_mat[valid_index, ] <- fitted$pi_mat
      next
    }

    outcome_predictions <- fit_outcome_fold(
      X_train = X[train_index, , drop = FALSE],
      A_train = standardized$index[train_index],
      Y_train = Y[train_index],
      W_train = sample_weight[train_index],
      X_valid = X[valid_index, , drop = FALSE],
      levels = levels,
      model_spec = outcome_spec,
      stratify = stratify
    )
    treatment_predictions <- fit_treatment_fold(
      X_train = X[train_index, , drop = FALSE],
      A_train = standardized$index[train_index],
      W_train = sample_weight[train_index],
      X_valid = X[valid_index, , drop = FALSE],
      levels = levels,
      model_spec = treatment_spec,
      stratify = stratify
    )
    mu_mat[valid_index, ] <- outcome_predictions
    pi_mat[valid_index, ] <- treatment_predictions
  }

  backend_label <- unique(c(outcome_spec$backend, treatment_spec$backend))
  backend_label <- backend_label[backend_label != "builtin"]
  if (!length(backend_label)) {
    backend_label <- "builtin"
  }

  list(
    A_factor = standardized$factor,
    treatment_levels = levels,
    control_level = levels[[standardized$control_index]],
    control_index = standardized$control_index,
    fold_id = folds,
    mu_mat = mu_mat,
    pi_mat = pi_mat,
    nuisance_source = paste(backend_label, collapse = "+")
  )
}

resolve_outcome_model <- function(model) {
  if (is.null(model)) {
    model <- "lasso"
  }
  if (is.character(model) && length(model) == 1L && model %in% builtin_regression_names()) {
    return(make_builtin_regression_spec(model))
  }
  if (is.list(model) && !is.null(model$name) && is.character(model$name) &&
      length(model$name) == 1L && model$name %in% builtin_regression_names() &&
      is.null(model$fit) && is.null(model$predict)) {
    return(make_builtin_regression_spec(model))
  }
  if (is_superlearner_spec(model)) {
    sl_library <- as_superlearner_library(model)
    return(list(
      backend = "SuperLearner",
      kind = "regression",
      fit = function(x, y, weights) {
        if (!requireNamespace("SuperLearner", quietly = TRUE)) {
          stop("The optional `SuperLearner` package is required for SuperLearner backends.", call. = FALSE)
        }
        SuperLearner::SuperLearner(
          Y = y,
          X = x,
          family = stats::gaussian(),
          SL.library = sl_library,
          obsWeights = weights,
          env = asNamespace("SuperLearner")
        )
      },
      predict = function(model_fit, newx) {
        as.numeric(stats::predict(model_fit, newdata = newx)$pred)
      }
    ))
  }
  if (is_sl3_model(model)) {
    return(list(backend = "sl3", kind = "regression", learner = model))
  }
  if (is_custom_model_spec(model, expected_kind = "regression")) {
    model$backend <- "builtin"
    return(model)
  }
  stop("Unsupported `outcome_model` specification.", call. = FALSE)
}

resolve_treatment_model <- function(model) {
  if (is.null(model)) {
    model <- "lasso"
  }
  if (is.character(model) && length(model) == 1L && model %in% builtin_classification_names()) {
    return(make_builtin_classification_spec(model))
  }
  if (is.list(model) && !is.null(model$name) && is.character(model$name) &&
      length(model$name) == 1L && model$name %in% builtin_classification_names() &&
      is.null(model$fit) && is.null(model$predict)) {
    return(make_builtin_classification_spec(model))
  }
  if (is_superlearner_spec(model)) {
    sl_library <- as_superlearner_library(model)
    return(list(
      backend = "SuperLearner",
      kind = "classification",
      supports_multiclass_direct = FALSE,
      fit = function(x, y, weights) {
        if (!requireNamespace("SuperLearner", quietly = TRUE)) {
          stop("The optional `SuperLearner` package is required for SuperLearner backends.", call. = FALSE)
        }
        levels_y <- levels(y)
        fits <- lapply(levels_y, function(level) {
          SuperLearner::SuperLearner(
            Y = as.numeric(y == level),
            X = x,
            family = stats::binomial(),
            SL.library = sl_library,
            obsWeights = weights,
            env = asNamespace("SuperLearner")
          )
        })
        names(fits) <- levels_y
        list(levels = levels_y, fits = fits)
      },
      predict = function(model_fit, newx) {
        pred <- vapply(model_fit$fits, function(fit) {
          as.numeric(stats::predict(fit, newdata = newx)$pred)
        }, numeric(nrow(newx)))
        pred <- t(pred)
        pred <- t(pred)
        colnames(pred) <- model_fit$levels
        pred
      }
    ))
  }
  if (is_sl3_model(model)) {
    return(list(backend = "sl3", kind = "classification", learner = model))
  }
  if (is_custom_model_spec(model, expected_kind = "classification")) {
    model$backend <- "builtin"
    return(model)
  }
  stop("Unsupported `treatment_model` specification.", call. = FALSE)
}

fit_outcome_fold <- function(X_train, A_train, Y_train, W_train, X_valid, levels, model_spec, stratify) {
  if ("outcome" %in% stratify) {
    return(fit_stratified_outcome_fold(
      X_train = X_train,
      A_train = A_train,
      Y_train = Y_train,
      W_train = W_train,
      X_valid = X_valid,
      levels = levels,
      model_spec = model_spec
    ))
  }
  fit_pooled_outcome_fold(
    X_train = X_train,
    A_train = A_train,
    Y_train = Y_train,
    W_train = W_train,
    X_valid = X_valid,
    levels = levels,
    model_spec = model_spec
  )
}

fit_stratified_outcome_fold <- function(X_train, A_train, Y_train, W_train, X_valid, levels, model_spec) {
  predictions <- matrix(NA_real_, nrow = nrow(X_valid), ncol = length(levels))
  colnames(predictions) <- levels
  for (level_index in seq_along(levels)) {
    arm_rows <- which(A_train == level_index)
    if (!length(arm_rows)) {
      stop("Every treatment level must appear in each training fold.", call. = FALSE)
    }
    model_fit <- model_spec$fit(
      x = X_train[arm_rows, , drop = FALSE],
      y = Y_train[arm_rows],
      weights = W_train[arm_rows]
    )
    predictions[, level_index] <- as.numeric(model_spec$predict(model_fit, X_valid))
  }
  predictions
}

fit_pooled_outcome_fold <- function(X_train, A_train, Y_train, W_train, X_valid, levels, model_spec) {
  train_data <- augment_covariates_with_treatment(X_train, levels[A_train], levels)
  model_fit <- model_spec$fit(
    x = train_data,
    y = Y_train,
    weights = W_train
  )
  prediction_list <- lapply(levels, function(level) {
    valid_data <- augment_covariates_with_treatment(X_valid, rep(level, nrow(X_valid)), levels)
    as.numeric(model_spec$predict(model_fit, valid_data))
  })
  predicted <- do.call(cbind, prediction_list)
  colnames(predicted) <- levels
  predicted
}

fit_treatment_fold <- function(X_train, A_train, W_train, X_valid, levels, model_spec, stratify) {
  if ("treatment" %in% stratify) {
    return(fit_stratified_treatment_fold(
      X_train = X_train,
      A_train = A_train,
      W_train = W_train,
      X_valid = X_valid,
      levels = levels,
      model_spec = model_spec
    ))
  }
  fit_pooled_treatment_fold(
    X_train = X_train,
    A_train = A_train,
    W_train = W_train,
    X_valid = X_valid,
    levels = levels,
    model_spec = model_spec
  )
}

fit_stratified_treatment_fold <- function(X_train, A_train, W_train, X_valid, levels, model_spec) {
  predictions <- matrix(NA_real_, nrow = nrow(X_valid), ncol = length(levels))
  colnames(predictions) <- levels
  for (level_index in seq_along(levels)) {
    model_fit <- model_spec$fit(
      x = X_train,
      y = factor(as.integer(A_train == level_index), levels = c(0, 1)),
      weights = W_train
    )
    predicted <- model_spec$predict(model_fit, X_valid)
    if (is.matrix(predicted)) {
      if ("1" %in% colnames(predicted)) {
        predicted <- predicted[, "1"]
      } else if (ncol(predicted) == 2L) {
        predicted <- predicted[, 2]
      } else {
        stop("Binary treatment predictions must provide a treated-class probability.", call. = FALSE)
      }
    }
    predictions[, level_index] <- as.numeric(predicted)
  }
  normalize_probability_matrix(predictions, fallback = empirical_probability_matrix(levels[A_train], levels, nrow(X_valid)))
}

fit_pooled_treatment_fold <- function(X_train, A_train, W_train, X_valid, levels, model_spec) {
  if (isFALSE(model_spec$supports_multiclass_direct)) {
    stop(
      "The selected `treatment_model` only supports pooled multi-arm fitting when `stratify` excludes `\"treatment\"` and the model can return direct multiclass probabilities.",
      call. = FALSE
    )
  }
  model_fit <- model_spec$fit(
    x = X_train,
    y = factor(levels[A_train], levels = levels),
    weights = W_train
  )
  predicted <- model_spec$predict(model_fit, X_valid)
  predicted <- align_probability_predictions(predicted, levels)
  normalize_probability_matrix(predicted, fallback = empirical_probability_matrix(levels[A_train], levels, nrow(X_valid)))
}

fit_sl3_fold <- function(X_train, A_train, Y_train, W_train, X_valid, levels, outcome_spec, treatment_spec, stratify) {
  if (!requireNamespace("sl3", quietly = TRUE)) {
    stop("`sl3` is required for `sl3` nuisance learners.", call. = FALSE)
  }
  outcome_learner <- outcome_spec$learner
  treatment_learner <- treatment_spec$learner

  mu_mat <- matrix(NA_real_, nrow = nrow(X_valid), ncol = length(levels))
  pi_mat <- matrix(NA_real_, nrow = nrow(X_valid), ncol = length(levels))
  colnames(mu_mat) <- levels
  colnames(pi_mat) <- levels

  if ("outcome" %in% stratify) {
    for (level_index in seq_along(levels)) {
      outcome_rows <- which(A_train == level_index)
      if (!length(outcome_rows)) {
        stop("Every treatment level must appear in each training fold.", call. = FALSE)
      }
      outcome_task <- sl3::sl3_Task$new(
        data = data.frame(X_train[outcome_rows, , drop = FALSE], Y = Y_train[outcome_rows], weights = W_train[outcome_rows]),
        covariates = colnames(X_train),
        outcome = "Y",
        weights = "weights"
      )
      outcome_valid_task <- sl3::sl3_Task$new(
        data = data.frame(X_valid, Y = rep(0, nrow(X_valid)), weights = rep(1, nrow(X_valid))),
        covariates = colnames(X_train),
        outcome = "Y",
        weights = "weights"
      )
      fitted_outcome <- clone_sl3_learner(outcome_learner)$train(outcome_task)
      mu_mat[, level_index] <- as.numeric(fitted_outcome$predict(outcome_valid_task))
    }
  } else {
    train_data <- augment_covariates_with_treatment(X_train, levels[A_train], levels)
    valid_sets <- lapply(levels, function(level) {
      augment_covariates_with_treatment(X_valid, rep(level, nrow(X_valid)), levels)
    })
    covariate_names <- colnames(train_data)
    outcome_task <- sl3::sl3_Task$new(
      data = data.frame(train_data, Y = Y_train, weights = W_train),
      covariates = covariate_names,
      outcome = "Y",
      weights = "weights"
    )
    fitted_outcome <- clone_sl3_learner(outcome_learner)$train(outcome_task)
    for (level_index in seq_along(levels)) {
      valid_task <- sl3::sl3_Task$new(
        data = data.frame(valid_sets[[level_index]], Y = rep(0, nrow(X_valid)), weights = rep(1, nrow(X_valid))),
        covariates = covariate_names,
        outcome = "Y",
        weights = "weights"
      )
      mu_mat[, level_index] <- as.numeric(fitted_outcome$predict(valid_task))
    }
  }

  if (!("treatment" %in% stratify)) {
    stop("Pooled multi-arm treatment fitting is not yet supported for `sl3` backends. Use `stratify = \"treatment\"` or a pooled multiclass learner.", call. = FALSE)
  }

  for (level_index in seq_along(levels)) {
    treatment_task <- sl3::sl3_Task$new(
      data = data.frame(X_train, A = factor(as.integer(A_train == level_index), levels = c(0, 1)), weights = W_train),
      covariates = colnames(X_train),
      outcome = "A",
      weights = "weights"
    )
    treatment_valid_task <- sl3::sl3_Task$new(
      data = data.frame(X_valid, A = factor(rep(0, nrow(X_valid)), levels = c(0, 1)), weights = rep(1, nrow(X_valid))),
      covariates = colnames(X_train),
      outcome = "A",
      weights = "weights"
    )
    fitted_treatment <- clone_sl3_learner(treatment_learner)$train(treatment_task)
    pi_pred <- fitted_treatment$predict(treatment_valid_task)
    if (is.matrix(pi_pred) || is.data.frame(pi_pred)) {
      pi_pred <- as.matrix(pi_pred)
      if ("1" %in% colnames(pi_pred)) {
        pi_pred <- pi_pred[, "1"]
      } else if (ncol(pi_pred) == 2L) {
        pi_pred <- pi_pred[, 2]
      }
    }
    pi_mat[, level_index] <- as.numeric(pi_pred)
  }

  list(
    mu_mat = mu_mat,
    pi_mat = normalize_probability_matrix(pi_mat, fallback = empirical_probability_matrix(levels[A_train], levels, nrow(X_valid)))
  )
}

clone_sl3_learner <- function(learner) {
  if (!is.function(learner$clone)) {
    stop("`sl3` learners must provide a `$clone()` method for cross-fitting.", call. = FALSE)
  }
  learner$clone(deep = TRUE)
}

bootstrap_indices <- function(n, fold_id = NULL) {
  if (is.null(fold_id)) {
    sample.int(n, size = n, replace = TRUE)
  } else {
    unlist(lapply(split(seq_len(n), fold_id), sample, replace = TRUE), use.names = FALSE)
  }
}

resolve_fold_id <- function(n, n_folds, fold_id, seed, treatment_index = NULL) {
  if (!is.null(fold_id)) {
    if (length(fold_id) != n) {
      stop("`fold_id` must have one entry per observation.", call. = FALSE)
    }
    return(as.integer(fold_id))
  }
  if (!is.null(seed)) {
    set.seed(seed)
  }
  if (is.null(treatment_index)) {
    return(sample(rep(seq_len(n_folds), length.out = n)))
  }
  treatment_index <- as.integer(treatment_index)
  counts <- table(treatment_index)
  n_splits <- min(as.integer(n_folds), min(counts))
  if (n_splits < 2L) {
    stop("Need at least two observations per treatment class to assign stratified folds.", call. = FALSE)
  }
  folds <- integer(n)
  for (level in sort(unique(treatment_index))) {
    idx <- which(treatment_index == level)
    folds[idx] <- sample(rep(seq_len(n_splits), length.out = length(idx)))
  }
  folds
}

standardize_treatment <- function(A, control_level, treatment_levels = NULL) {
  if (is.null(treatment_levels)) {
    observed_levels <- unique(as.character(A))
    control_level_chr <- as.character(control_level)
    if (!(control_level_chr %in% observed_levels)) {
      stop("`control_level` must appear in `treatment_levels` and in the observed treatment.", call. = FALSE)
    }
    remaining <- setdiff(observed_levels, control_level_chr)
    treatment_levels <- c(control_level_chr, sort(remaining))
  }
  treatment_levels <- as.character(treatment_levels)
  control_level <- as.character(control_level)
  if (!(control_level %in% treatment_levels)) {
    stop("`control_level` must appear in `treatment_levels` and in the observed treatment.", call. = FALSE)
  }
  factor_A <- factor(as.character(A), levels = treatment_levels)
  if (anyNA(factor_A)) {
    stop("Observed treatment values must be covered by `treatment_levels`.", call. = FALSE)
  }
  list(
    factor = factor_A,
    levels = treatment_levels,
    index = as.integer(factor_A),
    control_index = match(control_level, treatment_levels)
  )
}

align_nuisance_matrix <- function(x, treatment_levels, argument_name) {
  mat <- as.matrix(x)
  if (!is.numeric(mat)) {
    stop(sprintf("`%s` must be numeric.", argument_name), call. = FALSE)
  }
  if (ncol(mat) != length(treatment_levels)) {
    stop(sprintf("`%s` must have one column per treatment level.", argument_name), call. = FALSE)
  }
  if (is.null(colnames(mat))) {
    colnames(mat) <- treatment_levels
  } else {
    if (!all(treatment_levels %in% colnames(mat))) {
      stop(sprintf("Column names of `%s` must include all treatment levels.", argument_name), call. = FALSE)
    }
    mat <- mat[, treatment_levels, drop = FALSE]
  }
  if (anyNA(mat)) {
    stop(sprintf("`%s` cannot contain missing values.", argument_name), call. = FALSE)
  }
  mat
}

resolve_sample_weight <- function(sample_weight, n, data = NULL) {
  if (is.null(sample_weight)) {
    return(rep(1, n))
  }
  if (is.character(sample_weight) && !is.null(data)) {
    sample_weight <- data[[sample_weight]]
  }
  if (length(sample_weight) != n) {
    stop("`sample_weight` must have one entry per observation.", call. = FALSE)
  }
  if (any(!is.finite(sample_weight)) || any(sample_weight < 0)) {
    stop("`sample_weight` must be finite and non-negative.", call. = FALSE)
  }
  as.numeric(sample_weight)
}

normalize_weights <- function(weights) {
  total <- sum(weights)
  if (!is.finite(total) || total <= 0) {
    stop("Sample weights must sum to a positive finite value.", call. = FALSE)
  }
  weights / total
}

weighted_mean <- function(x, weights) {
  sum(normalize_weights(weights) * x)
}

assert_columns_exist <- function(data, columns) {
  missing_columns <- setdiff(columns, names(data))
  if (length(missing_columns)) {
    stop(sprintf("Missing columns: %s", paste(missing_columns, collapse = ", ")), call. = FALSE)
  }
}

is_custom_model_spec <- function(model, expected_kind) {
  is.list(model) &&
    identical(model$kind, expected_kind) &&
    is.function(model$fit) &&
    is.function(model$predict)
}

is_sl3_model <- function(model) {
  is.environment(model) && is.function(model$train) && is.function(model$predict)
}

is_superlearner_spec <- function(model) {
  (is.character(model) && length(model) > 1L) ||
    (is.list(model) && !is.null(model$SL.library))
}

as_superlearner_library <- function(model) {
  if (is.character(model)) {
    return(model)
  }
  model$SL.library
}

align_probability_predictions <- function(predicted, levels) {
  predicted <- as.matrix(predicted)
  if (is.null(dim(predicted))) {
    predicted <- matrix(predicted, ncol = 1L)
  }
  if (is.null(colnames(predicted))) {
    if (ncol(predicted) != length(levels)) {
      stop("Treatment model predictions must have one column per treatment level.", call. = FALSE)
    }
    colnames(predicted) <- levels
  }
  predicted <- predicted[, levels, drop = FALSE]
  if (anyNA(predicted)) {
    stop("Treatment model predictions cannot contain missing values.", call. = FALSE)
  }
  predicted
}

normalize_probability_matrix <- function(pi_mat, fallback = NULL) {
  epsilon <- 1e-8
  pi_mat <- pmax(as.matrix(pi_mat), epsilon)
  row_sums <- rowSums(pi_mat)
  invalid <- !is.finite(row_sums) | row_sums <= 0
  if (any(invalid)) {
    if (is.null(fallback)) {
      stop("Propensity predictions must have positive row sums.", call. = FALSE)
    }
    fallback <- as.matrix(fallback)
    pi_mat[invalid, ] <- fallback[invalid, , drop = FALSE]
    row_sums <- rowSums(pi_mat)
  }
  sweep(pi_mat, 1, row_sums, "/")
}

normalize_stratify <- function(stratify) {
  if (is.null(stratify) || identical(stratify, FALSE)) {
    return(character())
  }
  if (identical(stratify, TRUE)) {
    return(c("outcome", "treatment"))
  }
  if (!is.character(stratify)) {
    stop("`stratify` must be NULL, FALSE, TRUE, or a character vector using `\"outcome\"` and/or `\"treatment\"`.", call. = FALSE)
  }
  stratify <- unique(stratify)
  invalid <- setdiff(stratify, c("outcome", "treatment"))
  if (length(invalid)) {
    stop("`stratify` entries must be drawn from `\"outcome\"` and `\"treatment\"`.", call. = FALSE)
  }
  stratify
}

validate_wald_conservative <- function(wald_conservative) {
  if (!is.logical(wald_conservative) || length(wald_conservative) != 1L || is.na(wald_conservative)) {
    stop("`wald_conservative` must be TRUE or FALSE.", call. = FALSE)
  }
  wald_conservative
}

validate_wald_correction <- function(wald_correction) {
  choices <- c("auto", "none")
  if (length(wald_correction) > 1L) {
    wald_correction <- wald_correction[[1L]]
  }
  if (!is.character(wald_correction) || length(wald_correction) != 1L ||
      is.na(wald_correction) || !wald_correction %in% choices) {
    stop("`wald_correction` must be one of \"auto\" or \"none\".", call. = FALSE)
  }
  wald_correction
}

resolve_inference_options <- function(conf_level, alpha, bootstrap_reps, n_boot, jackknife_folds) {
  if (!is.null(alpha)) {
    conf_level <- 1 - alpha
  }
  if (!is.null(n_boot)) {
    bootstrap_reps <- n_boot
  }
  if (!is.numeric(conf_level) || length(conf_level) != 1L || !is.finite(conf_level) ||
      conf_level <= 0 || conf_level >= 1) {
    stop("`conf_level` must be a single number strictly between 0 and 1.", call. = FALSE)
  }
  bootstrap_reps <- as.integer(bootstrap_reps)
  jackknife_folds <- as.integer(jackknife_folds)
  if (is.na(bootstrap_reps) || bootstrap_reps < 0L) {
    stop("`bootstrap_reps` must be a non-negative integer.", call. = FALSE)
  }
  if (is.na(jackknife_folds) || jackknife_folds < 2L) {
    stop("`jackknife_folds` must be an integer greater than or equal to 2.", call. = FALSE)
  }
  list(
    conf_level = conf_level,
    bootstrap_reps = bootstrap_reps,
    jackknife_folds = jackknife_folds
  )
}

augment_covariates_with_treatment <- function(X, treatment_values, levels) {
  x_augmented <- as.data.frame(X, stringsAsFactors = FALSE)
  x_augmented[[".treatment"]] <- factor(as.character(treatment_values), levels = levels)
  x_augmented
}

empirical_probability_matrix <- function(observed_levels, levels, n_rows) {
  probabilities <- tabulate(match(as.character(observed_levels), levels), nbins = length(levels))
  probabilities <- probabilities / sum(probabilities)
  matrix(probabilities, nrow = n_rows, ncol = length(levels), byrow = TRUE, dimnames = list(NULL, levels))
}

#' Legacy Binary Wrapper for `calibrated_dml()`
#'
#' @param W Covariate matrix or data frame.
#' @param A Binary treatment vector.
#' @param Y Outcome vector.
#' @param weights Optional sample weights.
#' @param learners_treatment Legacy alias for `treatment_model`.
#' @param learners_outcome Legacy alias for `outcome_model`.
#' @param pi_mat Optional supplied propensity nuisance matrix.
#' @param mu_mat Optional supplied outcome nuisance matrix.
#' @param nboot Legacy alias for `n_boot`.
#' @param alpha Legacy significance level.
#'
#' @return A `"calibrated_dml_fit"` object.
#' @export
cdml <- function(
  W,
  A,
  Y,
  weights = NULL,
  learners_treatment = "lasso",
  learners_outcome = "lm",
  pi_mat = NULL,
  mu_mat = NULL,
  nboot = 200,
  alpha = 0.05
) {
  W <- as.data.frame(W, stringsAsFactors = FALSE)
  colnames(W) <- if (is.null(colnames(W))) paste0("W", seq_len(ncol(W))) else colnames(W)
  data <- data.frame(W, A = A, Y = Y, weights = weights, check.names = FALSE)
  calibrated_dml(
    data = data,
    outcome = "Y",
    treatment = "A",
    covariates = colnames(W),
    control_level = 0,
    outcome_model = learners_outcome,
    treatment_model = learners_treatment,
    mu_mat = mu_mat,
    pi_mat = pi_mat,
    sample_weight = "weights",
    alpha = alpha,
    inference = "bootstrap",
    n_boot = nboot
  )
}

#' Legacy Uncalibrated Binary DML Estimate
#'
#' @param A Binary treatment vector.
#' @param Y Outcome vector.
#' @param mu1 Estimated treated outcome regression.
#' @param mu0 Estimated control outcome regression.
#' @param pi1 Estimated treated propensity score.
#' @param pi0 Estimated control propensity score.
#' @param weights Optional sample weights.
#' @param functional Optional legacy plug-in functional.
#' @param representer Optional legacy Riesz representer.
#'
#' @return Numeric vector of legacy effect estimates.
#' @export
estimate_dml <- function(A, Y, mu1, mu0, pi1, pi0, weights = NULL, functional = NULL, representer = NULL) {
  weights <- resolve_sample_weight(weights, length(A))
  mu_observed <- ifelse(A == 1, mu1, mu0)

  if (is.null(functional) || is.null(representer)) {
    estimates <- c(
      Y0 = weighted_mean(mu0 + (1 - A) * (Y - mu_observed) / pi0, weights),
      Y1 = weighted_mean(mu1 + A * (Y - mu_observed) / pi1, weights)
    )
    return(c(ATE = estimates[["Y1"]] - estimates[["Y0"]], Y1 = estimates[["Y1"]], Y0 = estimates[["Y0"]]))
  }

  plugin <- functional(mu1 = mu1, mu0 = mu0, A = A, Y = Y, weights = weights)
  alpha_n <- representer(pi1 = pi1, pi0 = pi0, A = A, Y = Y, weights = weights)
  weighted_mean(plugin + alpha_n * (Y - mu_observed), weights)
}

#' Legacy Calibrated Binary DML Estimate
#'
#' @inheritParams estimate_dml
#'
#' @return Numeric vector of legacy effect estimates.
#' @export
estimate_cdml <- function(A, Y, mu1, mu0, pi1, pi0, weights = NULL, functional = NULL, representer = NULL) {
  if (!is.null(functional) || !is.null(representer)) {
    stop("Custom functionals are not yet supported in the modernized API.", call. = FALSE)
  }
  estimate_cdml_ate(A, Y, mu1, mu0, pi1, pi0, weights = weights)
}

#' Legacy Calibrated Binary ATE/Y1/Y0 Estimates
#'
#' @inheritParams estimate_dml
#'
#' @return Numeric vector with `ATE`, `Y1`, and `Y0`.
#' @export
estimate_cdml_ate <- function(A, Y, mu1, mu0, pi1, pi0, weights = NULL) {
  fit <- calibrated_dml_from_nuisances(
    A = A,
    Y = Y,
    mu_mat = cbind("0" = mu0, "1" = mu1),
    pi_mat = cbind("0" = pi0, "1" = pi1),
    control_level = 0,
    sample_weight = weights,
    inference = "wald",
    n_boot = 0
  )
  c(
    ATE = fit$contrasts$estimate[[1]],
    Y1 = fit$potential_outcomes$estimate[fit$potential_outcomes$level == "1"],
    Y0 = fit$potential_outcomes$estimate[fit$potential_outcomes$level == "0"]
  )
}

#' Legacy Bootstrap Binary ATE Table
#'
#' @inheritParams estimate_dml
#' @param nboot Number of bootstrap resamples.
#' @param folds Optional resampling group identifiers.
#' @param alpha Significance level.
#'
#' @return Data frame of legacy bootstrap interval results.
#' @export
bootstrap_cdml_ate <- function(A, Y, mu1, mu0, pi1, pi0, weights = NULL, nboot = 200, folds = NULL, alpha = 0.05) {
  fit <- calibrated_dml_from_nuisances(
    A = A,
    Y = Y,
    mu_mat = cbind("0" = mu0, "1" = mu1),
    pi_mat = cbind("0" = pi0, "1" = pi1),
    control_level = 0,
    sample_weight = weights,
    inference = "bootstrap",
    n_boot = nboot,
    fold_id = folds,
    alpha = alpha
  )
  fit$estimates
}

#' Legacy Bootstrap Binary DML Table
#'
#' @inheritParams bootstrap_cdml_ate
#' @inheritParams estimate_dml
#'
#' @return Data frame of legacy bootstrap interval results.
#' @export
bootstrap_cdml <- function(A, Y, mu1, mu0, pi1, pi0, weights = NULL, nboot = 200, folds = NULL, alpha = 0.05, functional = NULL, representer = NULL) {
  if (!is.null(functional) || !is.null(representer)) {
    stop("Custom functionals are not yet supported in the modernized API.", call. = FALSE)
  }
  bootstrap_cdml_ate(A, Y, mu1, mu0, pi1, pi0, weights = weights, nboot = nboot, folds = folds, alpha = alpha)
}

#' Adaptive Calibrated DML for Binary Treatment
#'
#' `adaptive_calibrated_dml()` implements binary-treatment adaptive calibrated
#' estimators inspired by the adaptive DML work in `main_adptiveDML.tex`. The
#' adaptive plug-in mode calibrates the outcome regression and then reports
#' plug-in potential outcomes and ATEs. The adaptive R-learner mode calibrates
#' an initial CATE estimate from an R-learner pseudo-outcome and then plugs the
#' calibrated CATE back into Robinson's partially linear parameterization.
#' Adaptive estimation always uses isotonic regression internally.
#'
#' These adaptive estimators are super-efficient methods in the sense that they
#' can have lower realized variance and lower mean-squared error than standard
#' calibrated DML at favorable data-generating distributions. The tradeoff is
#' that standard-error estimation is harder, so interval coverage can be less
#' stable in practice. `calibrated_dml()` is the recommended default for
#' general use, while `adaptive_calibrated_dml()` is an advanced option.
#' `mode = "calibrated_rlearner"` is especially appealing when the truth may
#' be close to homogeneous but some heterogeneity is possible. Users who want
#' adaptive DML should consult Benkeser and van der Laan,
#' "A Super-Efficient Estimator of the Average Treatment Effect," and van der
#' Laan, Carone, Luedtke, and van der Laan, "Adaptive debiased machine
#' learning using data-driven model selection techniques." For standard
#' calibrated DML, see van der Laan, Luedtke, and Carone, "Doubly robust
#' inference via calibration."
#'
#' @param data A `data.frame` containing the analysis variables.
#' @param outcome Name of the outcome column in `data`.
#' @param treatment Name of the binary treatment column in `data`.
#' @param covariates Character vector naming covariate columns in `data`.
#' @param control_level Reference treatment level.
#' @param mode One of `"plugin"`, `"calibrated_rlearner"`, or
#'   `"working_rlearner"`.
#' @param plugin_parametrization One of `"arm_specific"` or `"r_learner"`.
#' @param outcome_model Outcome nuisance model specification.
#' @param treatment_model Treatment nuisance model specification.
#' @param cate_model CATE learner specification used when the adaptive mode
#'   requires an R-learner.
#' @param mu_mat Optional matrix of cross-fitted outcome nuisance estimates.
#' @param pi_mat Optional matrix of cross-fitted treatment nuisance estimates.
#' @param sample_weight Optional sample weights.
#' @param treatment_levels Optional explicit ordering of the binary treatment
#'   levels.
#' @param stratify Controls whether nuisance fitting is stratified by
#'   `"outcome"`, `"treatment"`, or both.
#' @param calibration_options Optional list of calibration tuning options.
#' @param calibration_stratify Optional calibration stratification control.
#' @param inference One of `"jackknife"`, `"wald"`, or `"bootstrap"`.
#' @param conf_level Confidence level for reported intervals.
#' @param bootstrap_reps Number of bootstrap resamples when
#'   `inference = "bootstrap"`.
#' @param jackknife_folds Number of delete-a-group folds when
#'   `inference = "jackknife"`.
#' @param alpha Legacy compatibility alias for `1 - conf_level`.
#' @param n_boot Legacy compatibility alias for `bootstrap_reps`.
#' @param n_folds Number of folds for nuisance fitting.
#' @param fold_id Optional fold assignment vector.
#' @param seed Optional random seed.
#'
#' @return An object of class `"calibrated_dml_fit"`.
#' @export
adaptive_calibrated_dml <- function(
  data,
  outcome,
  treatment,
  covariates,
  control_level = 0,
  mode = c("plugin", "calibrated_rlearner", "working_rlearner", "r_learner", "r_learner_working_model"),
  plugin_parametrization = c("arm_specific", "r_learner"),
  outcome_model = "lasso",
  treatment_model = "lasso",
  cate_model = "lm",
  mu_mat = NULL,
  pi_mat = NULL,
  sample_weight = NULL,
  treatment_levels = NULL,
  stratify = c("outcome", "treatment"),
  calibration_options = list(),
  calibration_stratify = NULL,
  inference = c("jackknife", "wald", "bootstrap"),
  conf_level = 0.95,
  bootstrap_reps = 200,
  jackknife_folds = 20,
  alpha = NULL,
  n_boot = NULL,
  n_folds = 5,
  fold_id = NULL,
  seed = NULL
) {
  mode <- normalize_adaptive_mode(match.arg(mode))
  plugin_parametrization <- match.arg(plugin_parametrization)
  inference <- match.arg(inference)
  calibration_method <- "isotonic"
  inference_options <- resolve_inference_options(
    conf_level = conf_level,
    alpha = alpha,
    bootstrap_reps = bootstrap_reps,
    n_boot = n_boot,
    jackknife_folds = jackknife_folds
  )
  stratify <- normalize_stratify(stratify)

  assert_columns_exist(data, c(outcome, treatment, covariates))
  weights <- resolve_sample_weight(sample_weight, nrow(data), data)
  x <- as.data.frame(data[covariates], stringsAsFactors = FALSE)
  y <- as.numeric(data[[outcome]])

  if (xor(is.null(mu_mat), is.null(pi_mat))) {
    stop("Provide either both `mu_mat` and `pi_mat` or neither of them.", call. = FALSE)
  }

  nuisance_fit <- if (is.null(mu_mat)) {
    fit_cdml_nuisances(
      X = x,
      A = data[[treatment]],
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
  } else {
    standardized_input <- standardize_treatment(data[[treatment]], control_level, treatment_levels)
    list(
      A_factor = standardized_input$factor,
      treatment_levels = standardized_input$levels,
      control_level = standardized_input$levels[[standardized_input$control_index]],
      control_index = standardized_input$control_index,
      fold_id = if (is.null(fold_id)) {
        NULL
      } else {
        as.integer(fold_id)
      },
      mu_mat = align_nuisance_matrix(mu_mat, standardized_input$levels, "mu_mat"),
      pi_mat = align_nuisance_matrix(pi_mat, standardized_input$levels, "pi_mat"),
      nuisance_source = "supplied"
    )
  }

  if (length(nuisance_fit$treatment_levels) != 2L) {
    stop("`adaptive_calibrated_dml()` currently supports binary treatment only.", call. = FALSE)
  }

  standardized <- standardize_treatment(
    A = nuisance_fit$A_factor,
    control_level = control_level,
    treatment_levels = nuisance_fit$treatment_levels
  )
  treatment_index <- setdiff(seq_along(standardized$levels), standardized$control_index)
  control_index <- standardized$control_index
  a_binary <- as.numeric(standardized$index == treatment_index)
  pi_treat <- nuisance_fit$pi_mat[, treatment_index]
  m_hat <- rowSums(nuisance_fit$mu_mat * nuisance_fit$pi_mat)

  cate_required <- identical(mode, "calibrated_rlearner") ||
    identical(mode, "working_rlearner") ||
    identical(plugin_parametrization, "r_learner")
  tau_hat <- NULL
  working_model_fit <- NULL
  if (cate_required) {
    if (identical(mode, "working_rlearner")) {
      working_model_fit <- fit_binary_cate_working_model(
        X = x,
        A_binary = a_binary,
        Y = y,
        m_hat = m_hat,
        pi_treat = pi_treat,
        weights = weights,
        cate_model = cate_model
      )
      tau_hat <- working_model_fit$tau_hat
    } else {
      tau_hat <- fit_binary_cate(
        X = x,
        A_binary = a_binary,
        Y = y,
        m_hat = m_hat,
        pi_treat = pi_treat,
        weights = weights,
        cate_model = cate_model
      )
    }
  }

  plugin_calibration_stratify <- if (identical(mode, "plugin") && is.null(calibration_stratify)) {
    "outcome"
  } else {
    calibration_stratify
  }

  compute_base <- switch(
    mode,
    plugin = if (identical(plugin_parametrization, "arm_specific")) {
      function(index) compute_adaptive_plugin_fit(
        A_index = standardized$index[index],
        Y = y[index],
        mu_hat = nuisance_fit$mu_mat[index, , drop = FALSE],
        pi_hat = nuisance_fit$pi_mat[index, , drop = FALSE],
        weights = weights[index],
        levels = standardized$levels,
        control_index = control_index,
        calibration_options = calibration_options,
        calibration_stratify = plugin_calibration_stratify
      )
    } else {
      function(index) compute_adaptive_plugin_fit(
        A_index = standardized$index[index],
        Y = y[index],
        mu_hat = build_mu_from_rlearner(
          m_hat = m_hat[index],
          pi_treat = pi_treat[index],
          tau = tau_hat[index],
          control_index = control_index,
          levels = standardized$levels
        ),
        pi_hat = if (control_index == 1) {
          cbind("1" = pi_treat[index], "0" = 1 - pi_treat[index])
        } else {
          cbind("0" = 1 - pi_treat[index], "1" = pi_treat[index])
        },
        weights = weights[index],
        levels = standardized$levels,
        control_index = control_index,
        calibration_options = calibration_options,
        calibration_stratify = plugin_calibration_stratify
      )
    },
    calibrated_rlearner = function(index) compute_adaptive_rlearner_fit(
      A_binary = a_binary[index],
      Y = y[index],
      m_hat = m_hat[index],
      pi_treat = pi_treat[index],
      tau_hat = tau_hat[index],
      weights = weights[index],
      levels = standardized$levels,
      control_index = control_index,
      calibration_options = calibration_options
    ),
    working_rlearner = function(index) compute_adaptive_rlearner_working_fit(
      A_binary = a_binary[index],
      Y = y[index],
      m_hat = m_hat[index],
      pi_treat = pi_treat[index],
      weights = weights[index],
      levels = standardized$levels,
      control_index = control_index,
      design_matrix = working_model_fit$design_matrix[index, , drop = FALSE]
    )
  )

  base_fit <- compute_base(seq_len(nrow(data)))
  interval_table <- summarize_adaptive_fit(
    base_fit = base_fit,
    compute_base = compute_base,
    n = nrow(data),
    conf_level = inference_options$conf_level,
    inference = inference,
    bootstrap_reps = inference_options$bootstrap_reps,
    jackknife_folds = inference_options$jackknife_folds,
    fold_id = nuisance_fit$fold_id,
    A_index = standardized$index,
    seed = seed,
    control_index = control_index
  )

  result <- list(
    call = match.call(),
    control_level = standardized$levels[[control_index]],
    treatment_levels = standardized$levels,
    nuisance_source = nuisance_fit$nuisance_source,
    inference = inference,
    conf_level = inference_options$conf_level,
    bootstrap_reps = inference_options$bootstrap_reps,
    jackknife_folds = inference_options$jackknife_folds,
    calibration_method = calibration_method,
    calibration_stratify = normalize_calibration_stratify(if (identical(mode, "plugin")) plugin_calibration_stratify else calibration_stratify),
    fold_id = nuisance_fit$fold_id,
    sample_weight = weights,
    mu_mat = nuisance_fit$mu_mat,
    pi_mat = nuisance_fit$pi_mat,
    calibrated_mu_mat = base_fit$calibrated_mu_mat,
    calibrated_pi_mat = NULL,
    potential_outcomes = interval_table$potential_outcomes,
    contrasts = interval_table$contrasts,
    estimates = interval_table$estimates,
    adaptive_mode = mode,
    adaptive_plugin_parametrization = plugin_parametrization,
    calibration_bundle = list(
      method = calibration_method,
      calibration_stratify = normalize_calibration_stratify(if (identical(mode, "plugin")) plugin_calibration_stratify else calibration_stratify)
    ),
    adaptive_components = list(
      m_hat = m_hat,
      tau_hat = tau_hat,
      working_model = working_model_fit
    ),
    nuisance_fit = nuisance_fit
  )
  class(result) <- "calibrated_dml_fit"
  result
}

normalize_adaptive_mode <- function(mode) {
  if (identical(mode, "r_learner")) {
    return("calibrated_rlearner")
  }
  if (identical(mode, "r_learner_working_model")) {
    return("working_rlearner")
  }
  mode
}

compute_adaptive_plugin_fit <- function(
  A_index,
  Y,
  mu_hat,
  pi_hat,
  weights,
  levels,
  control_index,
  calibration_options,
  calibration_stratify
) {
  calibrated_mu <- calibrate_outcome_matrix(
    Y = Y,
    mu_mat = mu_hat,
    A_index = A_index,
    weights = weights,
    method = "isotonic",
    calibration_options = calibration_options,
    calibration_stratify = calibration_stratify
  )$calibrated
  normalized_weights <- normalize_weights(weights)
  observed_mu <- calibrated_mu[cbind(seq_along(A_index), A_index)]
  arm_estimates <- colSums(calibrated_mu * normalized_weights)
  arm_scores <- sapply(seq_along(levels), function(level_index) {
    calibrated_mu[, level_index] + (A_index == level_index) * (Y - observed_mu) / pi_hat[, level_index]
  })
  if (is.null(dim(arm_scores))) {
    arm_scores <- matrix(arm_scores, ncol = length(levels))
  }
  contrast_scores <- arm_scores[, -control_index, drop = FALSE] - arm_scores[, control_index]
  contrast_estimates <- arm_estimates[-control_index] - arm_estimates[[control_index]]
  summarize_custom_score_fit(
    arm_scores = arm_scores,
    arm_estimates = arm_estimates,
    contrast_scores = contrast_scores,
    contrast_estimates = contrast_estimates,
    levels = levels,
    control_index = control_index,
    weights = weights,
    calibrated_mu_mat = calibrated_mu,
    calibrated_pi_mat = pi_hat
  )
}

compute_adaptive_rlearner_fit <- function(
  A_binary,
  Y,
  m_hat,
  pi_treat,
  tau_hat,
  weights,
  levels,
  control_index,
  calibration_options
) {
  pseudo <- rlearner_pseudo_outcome(
    A_binary = A_binary,
    Y = Y,
    m_hat = m_hat,
    pi_treat = pi_treat,
    weights = weights
  )
  calibrator <- fit_monotone_calibrator(
    x = tau_hat[pseudo$keep],
    y = pseudo$pseudo_outcome[pseudo$keep],
    weights = pseudo$pseudo_weights[pseudo$keep],
    method = "isotonic",
    calibration_options = calibration_options
  )
  tau_star <- predict_monotone_calibrator(calibrator, tau_hat)
  mu_star <- build_mu_from_rlearner(
    m_hat = m_hat,
    pi_treat = pi_treat,
    tau = tau_star,
    control_index = control_index,
    levels = levels
  )
  residual_treatment <- A_binary - pi_treat
  gamma <- estimate_adaptive_gamma(
    tau_star = tau_star,
    residual_treatment_sq = residual_treatment^2,
    weights = weights
  )
  correction <- gamma * residual_treatment * (Y - m_hat - residual_treatment * tau_hat)
  if (control_index == 1) {
    arm_scores <- cbind(
      mu_star[, 1] + (1 - pi_treat) * correction,
      mu_star[, 2] - pi_treat * correction
    )
    arm_estimates <- c(
      weighted_mean(mu_star[, 1], weights),
      weighted_mean(mu_star[, 2], weights)
    )
  } else {
    arm_scores <- cbind(
      mu_star[, 1] - pi_treat * correction,
      mu_star[, 2] + (1 - pi_treat) * correction
    )
    arm_estimates <- c(
      weighted_mean(mu_star[, 1], weights),
      weighted_mean(mu_star[, 2], weights)
    )
  }
  contrast_estimates <- c(weighted_mean(tau_star, weights))
  summarize_custom_score_fit(
    arm_scores = arm_scores,
    arm_estimates = arm_estimates,
    contrast_scores = matrix(tau_star + correction, ncol = 1),
    contrast_estimates = contrast_estimates,
    levels = levels,
    control_index = control_index,
    weights = weights,
    calibrated_mu_mat = mu_star,
    calibrated_pi_mat = NULL
  )
}

compute_adaptive_rlearner_working_fit <- function(A_binary, Y, m_hat, pi_treat, weights, levels, control_index, design_matrix) {
  pseudo <- rlearner_pseudo_outcome(
    A_binary = A_binary,
    Y = Y,
    m_hat = m_hat,
    pi_treat = pi_treat,
    weights = weights
  )

  relaxed <- refit_working_cate_model(
    design_matrix = design_matrix,
    pseudo_outcome = pseudo$pseudo_outcome,
    pseudo_weights = pseudo$pseudo_weights,
    keep = pseudo$keep
  )
  tau_star <- relaxed$tau_hat
  mu_star <- build_mu_from_rlearner(
    m_hat = m_hat,
    pi_treat = pi_treat,
    tau = tau_star,
    control_index = control_index,
    levels = levels
  )
  ate_estimate <- weighted_mean(tau_star, weights)
  influence <- working_model_ate_influence(
    design_matrix = design_matrix,
    A_binary = A_binary,
    Y = Y,
    m_hat = m_hat,
    pi_treat = pi_treat,
    tau_hat = tau_star,
    weights = weights
  )

  summarize_score_fit(
    arm_scores = mu_star,
    contrast_scores = matrix(ate_estimate + influence, ncol = 1),
    levels = levels,
    control_index = control_index,
    weights = weights,
    calibrated_mu_mat = mu_star,
    calibrated_pi_mat = NULL
  )
}

summarize_score_fit <- function(arm_scores, levels, control_index, weights, calibrated_mu_mat = NULL, calibrated_pi_mat = NULL, contrast_scores = NULL) {
  normalized_weights <- normalize_weights(weights)
  arm_estimates <- colSums(arm_scores * normalized_weights)
  arm_standard_error <- apply(arm_scores, 2, function(score) {
    sqrt(sum((normalized_weights * (score - sum(score * normalized_weights))) ^ 2))
  })

  if (is.null(contrast_scores)) {
    contrast_scores <- arm_scores[, -control_index, drop = FALSE] - arm_scores[, control_index]
  }
  if (is.null(dim(contrast_scores))) {
    contrast_scores <- matrix(contrast_scores, ncol = 1)
  }
  contrast_estimates <- colSums(contrast_scores * normalized_weights)
  contrast_standard_error <- apply(contrast_scores, 2, function(score) {
    sqrt(sum((normalized_weights * (score - sum(score * normalized_weights))) ^ 2))
  })

  list(
    calibrated_mu_mat = calibrated_mu_mat,
    calibrated_pi_mat = calibrated_pi_mat,
    arm_scores = arm_scores,
    arm_estimates = stats::setNames(arm_estimates, levels),
    arm_standard_error = stats::setNames(arm_standard_error, levels),
    contrast_scores = contrast_scores,
    contrast_estimates = stats::setNames(contrast_estimates, levels[-control_index]),
    contrast_standard_error = stats::setNames(contrast_standard_error, levels[-control_index])
  )
}

summarize_custom_score_fit <- function(
  arm_scores,
  arm_estimates,
  contrast_scores,
  contrast_estimates,
  levels,
  control_index,
  weights,
  calibrated_mu_mat = NULL,
  calibrated_pi_mat = NULL
) {
  normalized_weights <- normalize_weights(weights)
  if (is.null(dim(arm_scores))) {
    arm_scores <- matrix(arm_scores, ncol = length(levels))
  }
  if (is.null(dim(contrast_scores))) {
    contrast_scores <- matrix(contrast_scores, ncol = length(levels) - 1L)
  }
  arm_standard_error <- vapply(seq_len(ncol(arm_scores)), function(index) {
    sqrt(sum((normalized_weights * (arm_scores[, index] - arm_estimates[[index]])) ^ 2))
  }, numeric(1))
  contrast_standard_error <- vapply(seq_len(ncol(contrast_scores)), function(index) {
    sqrt(sum((normalized_weights * (contrast_scores[, index] - contrast_estimates[[index]])) ^ 2))
  }, numeric(1))

  list(
    calibrated_mu_mat = calibrated_mu_mat,
    calibrated_pi_mat = calibrated_pi_mat,
    arm_scores = lapply(seq_len(ncol(arm_scores)), function(index) arm_scores[, index]),
    arm_estimates = stats::setNames(as.numeric(arm_estimates), levels),
    arm_standard_error = stats::setNames(as.numeric(arm_standard_error), levels),
    contrast_scores = lapply(seq_len(ncol(contrast_scores)), function(index) contrast_scores[, index]),
    contrast_estimates = stats::setNames(as.numeric(contrast_estimates), levels[-control_index]),
    contrast_standard_error = stats::setNames(as.numeric(contrast_standard_error), levels[-control_index])
  )
}

estimate_adaptive_gamma <- function(tau_star, residual_treatment_sq, weights, min_count = 25L) {
  tau_star <- as.numeric(tau_star)
  residual_treatment_sq <- as.numeric(residual_treatment_sq)
  weights <- as.numeric(weights)
  n <- length(tau_star)
  rounded <- round(tau_star, digits = 12)
  groups <- match(rounded, unique(rounded))
  counts <- tabulate(groups)

  if (max(counts) < max(2L, min_count %/% 2L)) {
    n_groups <- min(max(5L, floor(sqrt(n))), max(5L, n %/% max(5L, min_count)))
    order_index <- order(tau_star)
    bins <- split(order_index, cut(seq_along(order_index), breaks = n_groups, labels = FALSE))
    groups <- integer(n)
    for (group_id in seq_along(bins)) {
      groups[bins[[group_id]]] <- group_id
    }
  }

  gamma <- numeric(n)
  for (group_id in unique(groups)) {
    idx <- which(groups == group_id)
    denom <- weighted_mean(residual_treatment_sq[idx], weights[idx])
    gamma[idx] <- 1 / max(denom, 1e-8)
  }
  gamma
}

summarize_adaptive_fit <- function(base_fit, compute_base, n, conf_level, inference, bootstrap_reps, jackknife_folds, fold_id, A_index, seed, control_index) {
  alpha <- 1 - conf_level
  z_value <- stats::qnorm(1 - alpha / 2)
  levels <- names(base_fit$arm_estimates)
  control_level <- levels[[control_index]]

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

  contrast_levels <- names(base_fit$contrast_estimates)
  contrasts <- data.frame(
    estimand_type = "contrast",
    estimand = paste0("E[Y(", contrast_levels, ")] - E[Y(", control_level, ")]"),
    level = contrast_levels,
    control_level = control_level,
    estimate = unname(base_fit$contrast_estimates),
    std_error = unname(base_fit$contrast_standard_error),
    lower = unname(base_fit$contrast_estimates - z_value * base_fit$contrast_standard_error),
    upper = unname(base_fit$contrast_estimates + z_value * base_fit$contrast_standard_error),
    stringsAsFactors = FALSE
  )

  if (identical(inference, "bootstrap") && bootstrap_reps > 0) {
    if (!is.null(seed)) {
      set.seed(seed)
    }
    draws <- vector("list", bootstrap_reps)
    for (boot in seq_len(bootstrap_reps)) {
      index <- bootstrap_indices(n = n, fold_id = fold_id)
      fit <- compute_base(index)
      draws[[boot]] <- c(unname(fit$arm_estimates), unname(fit$contrast_estimates))
    }
    draws <- do.call(rbind, draws)
    arm_draws <- draws[, seq_len(nrow(potential_outcomes)), drop = FALSE]
    contrast_draws <- draws[, nrow(potential_outcomes) + seq_len(nrow(contrasts)), drop = FALSE]
    potential_outcomes$std_error <- apply(arm_draws, 2, stats::sd)
    potential_outcomes$lower <- apply(arm_draws, 2, stats::quantile, probs = alpha / 2, na.rm = TRUE)
    potential_outcomes$upper <- apply(arm_draws, 2, stats::quantile, probs = 1 - alpha / 2, na.rm = TRUE)
    contrasts$std_error <- apply(contrast_draws, 2, stats::sd)
    contrasts$lower <- apply(contrast_draws, 2, stats::quantile, probs = alpha / 2, na.rm = TRUE)
    contrasts$upper <- apply(contrast_draws, 2, stats::quantile, probs = 1 - alpha / 2, na.rm = TRUE)
  }

  if (identical(inference, "jackknife")) {
    jackknife_fold_id <- resolve_fold_id(
      n,
      n_folds = min(jackknife_folds, n),
      fold_id = NULL,
      seed = 1,
      treatment_index = A_index
    )
    groups <- split(seq_len(n), jackknife_fold_id)
    n_groups <- length(groups)
    draws <- lapply(groups, function(group) {
      keep <- setdiff(seq_len(n), group)
      fit <- compute_base(keep)
      c(unname(fit$arm_estimates), unname(fit$contrast_estimates))
    })
    draws <- do.call(rbind, draws)
    pseudo <- n_groups * matrix(c(unname(base_fit$arm_estimates), unname(base_fit$contrast_estimates)), nrow = n_groups, ncol = ncol(draws), byrow = TRUE) -
      (n_groups - 1) * draws
    se <- apply(pseudo, 2, stats::sd) / sqrt(n_groups)
    potential_outcomes$std_error <- se[seq_len(nrow(potential_outcomes))]
    potential_outcomes$lower <- potential_outcomes$estimate - z_value * potential_outcomes$std_error
    potential_outcomes$upper <- potential_outcomes$estimate + z_value * potential_outcomes$std_error
    contrasts$std_error <- se[nrow(potential_outcomes) + seq_len(nrow(contrasts))]
    contrasts$lower <- contrasts$estimate - z_value * contrasts$std_error
    contrasts$upper <- contrasts$estimate + z_value * contrasts$std_error
  }

  estimates <- rbind(potential_outcomes, contrasts)
  rownames(estimates) <- NULL
  list(
    potential_outcomes = potential_outcomes,
    contrasts = contrasts,
    estimates = estimates
  )
}

fit_binary_cate <- function(X, A_binary, Y, m_hat, pi_treat, weights, cate_model) {
  pseudo <- rlearner_pseudo_outcome(
    A_binary = A_binary,
    Y = Y,
    m_hat = m_hat,
    pi_treat = pi_treat,
    weights = weights
  )
  spec <- resolve_cate_model(cate_model)

  if (identical(spec$backend, "sl3")) {
    if (!requireNamespace("sl3", quietly = TRUE)) {
      stop("`sl3` is required for `sl3` CATE learners.", call. = FALSE)
    }
    task <- sl3::sl3_Task$new(
      data = data.frame(X[pseudo$keep, , drop = FALSE], Y = pseudo$pseudo_outcome[pseudo$keep], weights = pseudo$pseudo_weights[pseudo$keep]),
      covariates = colnames(X),
      outcome = "Y",
      weights = "weights"
    )
    pred_task <- sl3::sl3_Task$new(
      data = data.frame(X, Y = rep(0, nrow(X)), weights = rep(1, nrow(X))),
      covariates = colnames(X),
      outcome = "Y",
      weights = "weights"
    )
    fitted <- clone_sl3_learner(spec$learner)$train(task)
    return(as.numeric(fitted$predict(pred_task)))
  }

  fit <- spec$fit(
    x = X[pseudo$keep, , drop = FALSE],
    y = pseudo$pseudo_outcome[pseudo$keep],
    weights = pseudo$pseudo_weights[pseudo$keep]
  )
  as.numeric(spec$predict(fit, X))
}

fit_binary_cate_working_model <- function(X, A_binary, Y, m_hat, pi_treat, weights, cate_model) {
  pseudo <- rlearner_pseudo_outcome(
    A_binary = A_binary,
    Y = Y,
    m_hat = m_hat,
    pi_treat = pi_treat,
    weights = weights
  )
  working_design <- build_working_cate_design(
    X = X,
    pseudo_outcome = pseudo$pseudo_outcome,
    pseudo_weights = pseudo$pseudo_weights,
    keep = pseudo$keep,
    cate_model = cate_model
  )
  relaxed <- refit_working_cate_model(
    design_matrix = working_design$design_matrix,
    pseudo_outcome = pseudo$pseudo_outcome,
    pseudo_weights = pseudo$pseudo_weights,
    keep = pseudo$keep
  )
  list(
    cate_model = working_design$cate_model,
    design_matrix = working_design$design_matrix,
    selection = working_design$selection,
    tau_hat = relaxed$tau_hat
  )
}

build_working_cate_design <- function(X, pseudo_outcome, pseudo_weights, keep, cate_model) {
  cate_name <- cate_model
  if (is.list(cate_model) && !is.null(cate_model$name)) {
    cate_name <- cate_model$name
  }
  if (identical(cate_name, "glmnet")) {
    cate_name <- "lasso"
  }
  if (!(is.character(cate_name) && length(cate_name) == 1L && cate_name %in% c("lasso", "hal_gam"))) {
    stop("`mode = \"r_learner_working_model\"` currently supports `cate_model = \"lasso\"` or `\"hal_gam\"`.", call. = FALSE)
  }

  if (identical(cate_name, "lasso")) {
    require_optional_package("glmnet", "lasso")
    base_design <- design_matrix(X)
    fit <- glmnet::cv.glmnet(
      x = base_design[keep, , drop = FALSE],
      y = pseudo_outcome[keep],
      weights = pseudo_weights[keep],
      family = "gaussian"
    )
    active <- as.vector(stats::coef(fit, s = "lambda.min")) != 0
    design_full <- cbind("(Intercept)" = 1, base_design)[, active, drop = FALSE]
    return(list(
      cate_model = "lasso",
      design_matrix = design_full,
      selection = list(active = active, penalized_fit = fit)
    ))
  }

  model_config <- if (is.list(cate_model)) cate_model else list(name = "hal_gam")
  hal_fit <- fit_hal_gam_regression(
    x = X[keep, , drop = FALSE],
    y = pseudo_outcome[keep],
    weights = pseudo_weights[keep],
    control = model_config
  )
  penalized_fit <- hal_fit$fit
  active <- as.vector(stats::coef(penalized_fit, s = "lambda.min")) != 0
  basis_full <- predict_hal_basis(hal_fit$basis, X)
  design_full <- cbind("(Intercept)" = 1, basis_full)[, active, drop = FALSE]
  list(
    cate_model = "hal_gam",
    design_matrix = design_full,
    selection = list(active = active, penalized_fit = penalized_fit, basis = hal_fit$basis)
  )
}

refit_working_cate_model <- function(design_matrix, pseudo_outcome, pseudo_weights, keep) {
  fit <- stats::lm.wfit(
    x = design_matrix[keep, , drop = FALSE],
    y = pseudo_outcome[keep],
    w = pseudo_weights[keep]
  )
  coefficients <- fit$coefficients
  coefficients[is.na(coefficients)] <- 0
  tau_hat <- as.numeric(design_matrix %*% coefficients)
  list(coefficients = coefficients, tau_hat = tau_hat)
}

working_model_ate_influence <- function(design_matrix, A_binary, Y, m_hat, pi_treat, tau_hat, weights) {
  normalized_weights <- normalize_weights(weights)
  residual_treatment <- A_binary - pi_treat
  pseudo_weights <- weights * residual_treatment ^ 2
  gram <- crossprod(design_matrix, pseudo_weights * design_matrix)
  gram_inv <- try(solve(gram), silent = TRUE)
  if (inherits(gram_inv, "try-error")) {
    gram_inv <- qr.solve(gram, diag(ncol(gram)))
  }
  direction <- colSums(design_matrix * normalized_weights)
  gamma <- as.numeric(design_matrix %*% gram_inv %*% direction) * sum(weights)
  estimate <- weighted_mean(tau_hat, weights)
  gamma * residual_treatment * (Y - m_hat - residual_treatment * tau_hat) + tau_hat - estimate
}

resolve_cate_model <- function(model) {
  if (is.null(model)) {
    model <- "lm"
  }
  if (identical(model, "glmnet")) {
    model <- "lasso"
  }
  if (is.character(model) && length(model) == 1L && model %in% builtin_regression_names()) {
    return(make_builtin_regression_spec(model))
  }
  if (is.list(model) && !is.null(model$name) && identical(model$name, "glmnet")) {
    model$name <- "lasso"
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
  stop("Unsupported `cate_model` specification.", call. = FALSE)
}

rlearner_pseudo_outcome <- function(A_binary, Y, m_hat, pi_treat, weights) {
  residual_treatment <- A_binary - pi_treat
  keep <- abs(residual_treatment) > 1e-8
  pseudo_outcome <- rep(0, length(Y))
  pseudo_outcome[keep] <- (Y[keep] - m_hat[keep]) / residual_treatment[keep]
  pseudo_weights <- weights * residual_treatment ^ 2
  list(
    pseudo_outcome = pseudo_outcome,
    pseudo_weights = pseudo_weights,
    keep = keep
  )
}

build_mu_from_rlearner <- function(m_hat, pi_treat, tau, control_index, levels) {
  treat_index <- setdiff(seq_along(levels), control_index)
  mu_control <- m_hat - pi_treat * tau
  mu_treat <- m_hat + (1 - pi_treat) * tau
  out <- matrix(NA_real_, nrow = length(m_hat), ncol = length(levels))
  colnames(out) <- levels
  out[, control_index] <- mu_control
  out[, treat_index] <- mu_treat
  out
}

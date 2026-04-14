fit_calibration_bundle <- function(
  Y,
  mu_mat,
  A_index,
  pi_mat,
  weights,
  calibration_method = c("auto", "isotonic", "smooth_isotonic", "none"),
  calibration_options = list(),
  calibration_stratify = NULL
) {
  calibration_method <- match.arg(calibration_method)

  calibrated_mu <- calibrate_outcome_matrix(
    Y = Y,
    mu_mat = mu_mat,
    A_index = A_index,
    weights = weights,
    method = calibration_method,
    calibration_options = calibration_options,
    calibration_stratify = calibration_stratify
  )
  calibrated_pi <- calibrate_propensity_matrix(
    A_index = A_index,
    pi_mat = pi_mat,
    weights = weights,
    method = calibration_method,
    calibration_options = calibration_options
  )

  list(
    method = calibration_method,
    calibration_stratify = normalize_calibration_stratify(calibration_stratify),
    outcome = calibrated_mu,
    treatment = calibrated_pi,
    calibrated_mu_mat = calibrated_mu$calibrated,
    calibrated_pi_mat = calibrated_pi$calibrated
  )
}

calibrate_outcome_matrix <- function(
  Y,
  mu_mat,
  A_index,
  weights = NULL,
  method = c("auto", "isotonic", "smooth_isotonic", "none"),
  calibration_options = list(),
  calibration_stratify = NULL
) {
  method <- match.arg(method)
  mu_mat <- as.matrix(mu_mat)
  weights <- resolve_sample_weight(weights, nrow(mu_mat))
  colnames_out <- colnames(mu_mat)
  if (identical(method, "none")) {
    return(list(calibrated = mu_mat, calibrators = vector("list", ncol(mu_mat))))
  }

  calibration_stratify <- normalize_calibration_stratify(calibration_stratify)
  calibrated <- matrix(NA_real_, nrow = nrow(mu_mat), ncol = ncol(mu_mat))
  colnames(calibrated) <- colnames_out
  calibrators <- vector("list", ncol(mu_mat))

  for (level_index in seq_len(ncol(mu_mat))) {
    subset_index <- if (identical(calibration_stratify, "outcome")) {
      which(A_index == level_index)
    } else {
      seq_len(nrow(mu_mat))
    }
    if (!length(subset_index)) {
      stop("Each treatment level must appear in the calibration sample.", call. = FALSE)
    }
    calibrator <- fit_monotone_calibrator(
      x = mu_mat[subset_index, level_index],
      y = Y[subset_index],
      weights = weights[subset_index],
      method = method,
      calibration_options = calibration_options
    )
    calibrators[[level_index]] <- calibrator
    calibrated[, level_index] <- predict_monotone_calibrator(calibrator, mu_mat[, level_index])
  }

  list(calibrated = calibrated, calibrators = calibrators)
}

calibrate_propensity_matrix <- function(
  A_index,
  pi_mat,
  weights = NULL,
  method = c("auto", "isotonic", "smooth_isotonic", "none"),
  calibration_options = list()
) {
  method <- match.arg(method)
  weights <- resolve_sample_weight(weights, nrow(as.matrix(pi_mat)))
  pi_mat <- normalize_probability_matrix(pi_mat)
  if (identical(method, "none")) {
    return(list(calibrated = pi_mat, calibrators = vector("list", ncol(pi_mat))))
  }

  calibrated <- matrix(NA_real_, nrow = nrow(pi_mat), ncol = ncol(pi_mat))
  colnames(calibrated) <- colnames(pi_mat)
  calibrators <- vector("list", ncol(pi_mat))

  for (level_index in seq_len(ncol(pi_mat))) {
    indicator <- as.numeric(A_index == level_index)
    calibrator <- fit_monotone_calibrator(
      x = pi_mat[, level_index],
      y = indicator,
      weights = weights,
      method = method,
      calibration_options = calibration_options
    )
    calibrators[[level_index]] <- calibrator
    pi_star <- predict_monotone_calibrator(calibrator, pi_mat[, level_index])
    lower_bound <- max(1e-8, min(pi_star[indicator == 1], na.rm = TRUE))
    if (!is.finite(lower_bound)) {
      lower_bound <- 1e-8
    }
    calibrated[, level_index] <- pmin(pmax(pi_star, lower_bound), 1)
  }

  list(
    calibrated = normalize_probability_matrix(calibrated),
    calibrators = calibrators
  )
}

normalize_calibration_stratify <- function(calibration_stratify) {
  if (is.null(calibration_stratify) || identical(calibration_stratify, FALSE)) {
    return(NULL)
  }
  if (identical(calibration_stratify, TRUE)) {
    return("outcome")
  }
  values <- unique(as.character(calibration_stratify))
  if (!all(values %in% "outcome")) {
    stop("`calibration_stratify` must be NULL or \"outcome\".", call. = FALSE)
  }
  "outcome"
}

#' Calibrate Inverse Probability Weights for Binary Treatment
#'
#' Compatibility wrapper around the multi-arm calibration engine.
#'
#' @param A Binary treatment indicator.
#' @param pi1 Propensity scores for the treated arm.
#' @param pi0 Propensity scores for the control arm.
#' @param weights Optional sample weights.
#'
#' @return A list with calibrated weights and probabilities.
#' @export
calibrate_inverse_weights <- function(A, pi1, pi0, weights = NULL) {
  weights <- resolve_sample_weight(weights, length(A))
  standardized <- standardize_treatment(A, control_level = 0, treatment_levels = c(0, 1))
  calibrated <- calibrate_propensity_matrix(
    A_index = standardized$index,
    pi_mat = cbind("0" = pi0, "1" = pi1),
    weights = weights
  )$calibrated

  list(
    alpha1_star = 1 / calibrated[, "1"],
    alpha0_star = 1 / calibrated[, "0"],
    pi1_star = calibrated[, "1"],
    pi0_star = calibrated[, "0"]
  )
}

#' Calibrate Outcome Regression Predictions for Binary Treatment
#'
#' Compatibility wrapper around the multi-arm calibration engine.
#'
#' @param Y Outcome vector.
#' @param mu1 Predicted outcomes for the treated arm.
#' @param mu0 Predicted outcomes for the control arm.
#' @param A Binary treatment indicator.
#' @param weights Optional sample weights.
#'
#' @return A list with calibrated outcome predictions.
#' @export
calibrate_outcome_regression <- function(Y, mu1, mu0, A, weights = NULL) {
  weights <- resolve_sample_weight(weights, length(A))
  standardized <- standardize_treatment(A, control_level = 0, treatment_levels = c(0, 1))
  calibrated <- calibrate_outcome_matrix(
    Y = as.numeric(Y),
    mu_mat = cbind("0" = mu0, "1" = mu1),
    A_index = standardized$index,
    weights = weights,
    calibration_stratify = "outcome"
  )$calibrated

  list(
    mu1_star = calibrated[, "1"],
    mu0_star = calibrated[, "0"]
  )
}

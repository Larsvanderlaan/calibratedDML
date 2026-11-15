#' Calibrated DML for a Linear Functional
#'
#' Estimates E[sum_j c_j(O) * mu(T_j(O))] using cross-fitted regression and
#' Riesz predictions. PAVA calibrates the regression with squared-error loss
#' and the representer with the quadratic Riesz loss. Signed representers
#' are supported. Initial predictions remain fixed during resampling.
#'
#' Each calibrator is a nondecreasing, right-continuous step function with
#' knots at the distinct positive-weight observed predictions and constant
#' extension at both tails. The Riesz loss is minimized over this class;
#' prediction values seen only in the evaluation matrix do not add knots.
#'
#' @param Y Outcome vector.
#' @param regression Cross-fitted regression predictions at observed inputs.
#' @param riesz Cross-fitted Riesz predictions at observed inputs.
#' @param functional A row-wise linear function taking an n-by-k prediction
#'   matrix to n functional contributions, a length-k coefficient vector,
#'   or an n-by-k coefficient matrix.
#' @param regression_evaluations Regression predictions at the k inputs used
#'   by the functional, one row per observation. Defaults to observed predictions.
#' @param riesz_evaluations Riesz predictions at those same inputs, in the same
#'   column order. Defaults to observed predictions.
#' @param sample_weight Optional non-negative observation weights.
#' @param calibration_method One of "isotonic" (PAVA) or "none".
#' @param inference One of "jackknife" (default), "wald", or "bootstrap".
#' @param jackknife_folds Number V of jackknife deletion groups, default 10.
#' @param bootstrap_reps Number of bootstrap replicates, default 200.
#' @param conf_level Confidence level, default 0.95.
#' @param seed Optional random seed.
#' @param name Label for the functional in the result table.
#' @return A fitted object with an estimate, standard error, confidence interval,
#'   calibrated predictions, calibration maps, and resampling diagnostics.
#' @export
calibrated_linear_functional <- function(
  Y, regression, riesz, functional,
  regression_evaluations = NULL, riesz_evaluations = NULL,
  sample_weight = NULL, calibration_method = c("isotonic", "none"),
  inference = c("jackknife", "wald", "bootstrap"), jackknife_folds = 10,
  bootstrap_reps = 200, conf_level = 0.95, seed = NULL,
  name = "linear_functional"
) {
  calibration_method <- match.arg(calibration_method)
  inference <- match.arg(inference)
  Y <- linear_finite_vector(Y, "Y")
  n <- length(Y)
  if (n < 2L) stop("At least two observations are required.", call. = FALSE)
  regression <- linear_finite_vector(regression, "regression", n)
  riesz <- linear_finite_vector(riesz, "riesz", n)
  mu_eval <- linear_evaluation_matrix(regression_evaluations, regression, "regression_evaluations")
  alpha_eval <- linear_evaluation_matrix(riesz_evaluations, riesz, "riesz_evaluations")
  if (!identical(dim(mu_eval), dim(alpha_eval))) {
    stop("The regression and Riesz evaluation matrices must have the same shape.", call. = FALSE)
  }
  coefficients <- linear_functional_coefficients(functional, dim(mu_eval))
  weights <- resolve_sample_weight(sample_weight, n)
  if (any(!is.finite(weights))) stop("`sample_weight` must be finite.", call. = FALSE)
  if (length(conf_level) != 1L || !is.finite(conf_level) || conf_level <= 0 || conf_level >= 1) {
    stop("`conf_level` must lie strictly between 0 and 1.", call. = FALSE)
  }
  compute <- function(index) {
    w <- weights[index]
    if (sum(w) <= 0) stop("Weights must sum to a positive value in each resample.", call. = FALSE)
    if (identical(calibration_method, "isotonic")) {
      f <- fit_linear_outcome_pava(regression[index], Y[index], w)
      g <- fit_linear_riesz_pava(riesz[index], alpha_eval[index, , drop = FALSE], coefficients[index, , drop = FALSE], w)
      mu <- predict_linear_pava(f, regression[index])
      alpha <- predict_linear_pava(g, riesz[index])
      transformed <- matrix(predict_linear_pava(f, mu_eval[index, , drop = FALSE]), nrow = length(index))
      contributions <- rowSums(coefficients[index, , drop = FALSE] * transformed)
    } else {
      f <- g <- NULL
      mu <- regression[index]
      alpha <- riesz[index]
      contributions <- rowSums(coefficients[index, , drop = FALSE] * mu_eval[index, , drop = FALSE])
    }
    scores <- contributions + alpha * (Y[index] - mu)
    list(estimate = sum(w * scores) / sum(w), mu = mu, alpha = alpha,
      contributions = contributions, scores = scores, f = f, g = g)
  }
  full <- compute(seq_len(n))
  estimate <- full$estimate
  se <- sqrt(sum((weights / sum(weights) * (full$scores - estimate))^2))
  critical <- stats::qnorm((1 + conf_level) / 2)
  lower <- estimate - critical * se
  upper <- estimate + critical * se
  jackknife <- NULL
  if (identical(inference, "jackknife")) {
    jackknife <- vfold_jackknife(estimate, function(index) compute(index)$estimate,
      n, jackknife_folds, seed)
    se <- as.numeric(jackknife$standard_error)
    critical <- stats::qt((1 + conf_level) / 2, df = jackknife$df)
    lower <- estimate - critical * se
    upper <- estimate + critical * se
  } else if (identical(inference, "bootstrap")) {
    if (length(bootstrap_reps) != 1L || !is.finite(bootstrap_reps) ||
        bootstrap_reps != as.integer(bootstrap_reps) || bootstrap_reps < 2L) {
      stop("`bootstrap_reps` must be an integer of at least 2.", call. = FALSE)
    }
    if (!is.null(seed)) set.seed(seed)
    draws <- replicate(bootstrap_reps, compute(sample.int(n, n, replace = TRUE))$estimate)
    se <- stats::sd(draws)
    interval <- stats::quantile(draws, c((1 - conf_level) / 2, (1 + conf_level) / 2))
    lower <- unname(interval[1])
    upper <- unname(interval[2])
  }
  estimates <- data.frame(estimand = name, estimate = estimate, std_error = se,
    lower = lower, upper = upper, stringsAsFactors = FALSE)
  structure(list(
    estimates = estimates, regression = regression, riesz = riesz,
    calibrated_regression = full$mu, calibrated_riesz = full$alpha,
    functional_values = full$contributions, scores = full$scores,
    calibration = list(regression = full$f, riesz = full$g, coefficients = coefficients),
    jackknife = jackknife, inference = inference
  ), class = "calibrated_linear_functional_fit")
}

linear_finite_vector <- function(value, name, n = NULL) {
  if (!is.numeric(value) || !is.null(dim(value)) || any(!is.finite(value)) ||
      (!is.null(n) && length(value) != n)) {
    stop(sprintf("`%s` must be a finite vector with one entry per observation.", name), call. = FALSE)
  }
  as.numeric(value)
}

linear_evaluation_matrix <- function(value, observed, name) {
  if (is.null(value)) value <- matrix(observed, ncol = 1L)
  value <- as.matrix(value)
  if (!is.numeric(value) || nrow(value) != length(observed) || ncol(value) < 1L || any(!is.finite(value))) {
    stop(sprintf("`%s` must be a finite matrix with one row per observation.", name), call. = FALSE)
  }
  value
}

linear_functional_coefficients <- function(functional, shape) {
  n <- shape[1]
  k <- shape[2]
  if (is.function(functional)) {
    evaluate <- function(value) linear_finite_vector(functional(value), "functional output", n)
    if (any(abs(evaluate(matrix(0, n, k))) > 1e-12)) {
      stop("`functional` must be a row-wise linear map and send zero to zero.", call. = FALSE)
    }
    coefficients <- matrix(0, n, k)
    for (column in seq_len(k)) {
      basis <- matrix(0, n, k)
      basis[, column] <- 1
      coefficients[, column] <- evaluate(basis)
    }
    probe <- sin(matrix(seq_len(n * k) + 0.3, n, k))
    for (values in list(probe, -0.7 * probe + cos(probe))) {
      actual <- evaluate(values)
      expected <- rowSums(coefficients * values)
      if (any(abs(actual - expected) > 1e-10 * (1 + abs(expected)))) {
        stop("`functional` must be a row-wise linear map of the evaluation columns.", call. = FALSE)
      }
    }
  } else {
    if (is.numeric(functional) && is.null(dim(functional)) && length(functional) == k) {
      coefficients <- matrix(functional, n, k, byrow = TRUE)
    } else {
      coefficients <- as.matrix(functional)
    }
    if (!identical(dim(coefficients), shape)) {
      stop("`functional` coefficients must have one entry per evaluation column or shape (n, k).", call. = FALSE)
    }
  }
  if (!is.numeric(coefficients) || any(!is.finite(coefficients))) {
    stop("`functional` coefficients must be finite.", call. = FALSE)
  }
  coefficients
}

linear_pava <- function(quadratic, linear) {
  if (any(!is.finite(quadratic)) || any(quadratic <= 0) || any(!is.finite(linear))) {
    stop("PAVA requires finite linear mass and positive finite quadratic mass.", call. = FALSE)
  }
  k <- length(quadratic)
  start <- end <- a <- b <- numeric(k)
  top <- 0L
  for (index in seq_len(k)) {
    top <- top + 1L
    start[top] <- end[top] <- index
    a[top] <- quadratic[index]
    b[top] <- linear[index]
    while (top > 1L && b[top - 1L] / a[top - 1L] > b[top] / a[top]) {
      end[top - 1L] <- end[top]
      a[top - 1L] <- a[top - 1L] + a[top]
      b[top - 1L] <- b[top - 1L] + b[top]
      top <- top - 1L
    }
  }
  fitted <- numeric(k)
  for (index in seq_len(top)) fitted[start[index]:end[index]] <- b[index] / a[index]
  fitted
}

linear_bin_sum <- function(value, bins, k) {
  sums <- rowsum(value, bins, reorder = FALSE)
  result <- numeric(k)
  result[as.integer(rownames(sums))] <- sums[, 1]
  result
}

fit_linear_outcome_pava <- function(prediction, outcome, weights) {
  keep <- weights > 0
  knots <- sort(unique(prediction[keep]))
  bins <- match(prediction[keep], knots)
  quadratic <- linear_bin_sum(weights[keep], bins, length(knots))
  linear <- linear_bin_sum(weights[keep] * outcome[keep], bins, length(knots))
  list(knots = knots, fitted = linear_pava(quadratic, linear))
}

fit_linear_riesz_pava <- function(prediction, evaluations, coefficients, weights) {
  keep <- weights > 0
  knots <- sort(unique(prediction[keep]))
  quadratic <- linear_bin_sum(weights[keep], match(prediction[keep], knots), length(knots))
  bins <- pmax(1L, pmin(length(knots), findInterval(as.numeric(evaluations), knots)))
  linear <- linear_bin_sum(as.numeric(coefficients * weights), bins, length(knots))
  list(knots = knots, fitted = linear_pava(quadratic, linear))
}

predict_linear_pava <- function(calibrator, values) {
  index <- pmax(1L, pmin(length(calibrator$knots), findInterval(as.numeric(values), calibrator$knots)))
  calibrator$fitted[index]
}

#' @export
summary.calibrated_linear_functional_fit <- function(object, ...) object$estimates

#' @export
print.calibrated_linear_functional_fit <- function(x, ...) {
  print(x$estimates, row.names = FALSE, ...)
  invisible(x)
}

#' @export
confint.calibrated_linear_functional_fit <- function(object, ...) {
  interval <- as.matrix(object$estimates[c("lower", "upper")])
  rownames(interval) <- object$estimates$estimand
  interval
}

#' @export
as.data.frame.calibrated_linear_functional_fit <- function(x, ...) x$estimates

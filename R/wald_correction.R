resolve_sieve_riesz_wald_options <- function(wald_options, n) {
  if (is.null(wald_options)) {
    wald_options <- list()
  }
  if (!is.list(wald_options)) {
    stop("`wald_options` must be a list.", call. = FALSE)
  }
  allowed <- c(
    "basis_size_grid", "lambda_grid", "cv_folds", "propensity_clip",
    "riesz_bound", "min_rows", "min_unique_scores"
  )
  unknown <- setdiff(names(wald_options), allowed)
  if (length(unknown)) {
    stop(sprintf("Unknown `wald_options` entries: %s.", paste(unknown, collapse = ", ")), call. = FALSE)
  }

  propensity_clip <- as.numeric(wald_options[["propensity_clip"]] %||null% 0.025)
  if (!is.finite(propensity_clip) || propensity_clip <= 0 || propensity_clip >= 0.5) {
    stop("`wald_options$propensity_clip` must lie in (0, 0.5).", call. = FALSE)
  }
  riesz_bound <- as.numeric(wald_options[["riesz_bound"]] %||null% (1 / propensity_clip))
  if (!is.finite(riesz_bound) || riesz_bound <= 0) {
    stop("`wald_options$riesz_bound` must be positive and finite.", call. = FALSE)
  }
  cv_folds <- as.integer(wald_options[["cv_folds"]] %||null% 5L)
  if (!is.finite(cv_folds) || cv_folds < 2L) {
    stop("`wald_options$cv_folds` must be at least 2.", call. = FALSE)
  }
  min_rows <- as.integer(wald_options[["min_rows"]] %||null% 20L)
  min_unique_scores <- as.integer(wald_options[["min_unique_scores"]] %||null% 3L)
  if (!is.finite(min_rows) || min_rows < 2L) {
    stop("`wald_options$min_rows` must be at least 2.", call. = FALSE)
  }
  if (!is.finite(min_unique_scores) || min_unique_scores < 1L) {
    stop("`wald_options$min_unique_scores` must be at least 1.", call. = FALSE)
  }

  list(
    basis_size_grid = clean_wald_int_grid(wald_options[["basis_size_grid"]] %||null% c(8L, 16L, 32L, 64L), "basis_size_grid"),
    lambda_grid = clean_wald_float_grid(wald_options[["lambda_grid"]] %||null% 10 ^ seq(-8, 1, length.out = 8L), "lambda_grid"),
    cv_folds = min(cv_folds, max(2L, as.integer(n))),
    propensity_clip = propensity_clip,
    riesz_bound = riesz_bound,
    min_rows = min_rows,
    min_unique_scores = min_unique_scores
  )
}

compute_sieve_riesz_wald_correction <- function(A_index,
                                                Y,
                                                weights,
                                                levels,
                                                control_index,
                                                calibrated_mu_mat,
                                                calibrated_pi_mat,
                                                contrast_estimate,
                                                simple_wald_std_error,
                                                conf_level,
                                                seed = NULL,
                                                wald_options = list(),
                                                wald_conservative = FALSE) {
  if (length(levels) != 2L) {
    stop("Sieve-Riesz Wald correction currently requires a binary treatment.", call. = FALSE)
  }
  settings <- resolve_sieve_riesz_wald_options(wald_options, length(Y))
  active_index <- setdiff(seq_along(levels), control_index)[[1L]]
  a_active <- as.numeric(A_index == active_index)
  a_control <- as.numeric(A_index == control_index)
  Y <- as.numeric(Y)
  weights <- as.numeric(weights)

  mu_active <- as.numeric(calibrated_mu_mat[, active_index])
  mu_control <- as.numeric(calibrated_mu_mat[, control_index])
  pi_active <- wald_clip_probability(calibrated_pi_mat[, active_index], eps = 1e-8)
  pi_control <- wald_clip_probability(calibrated_pi_mat[, control_index], eps = 1e-8)
  alpha_active <- 1 / pi_active
  alpha_control <- 1 / pi_control
  seed <- wald_normalize_seed(seed)

  h_active <- fit_sieve_riesz_arm(
    score = mu_active,
    arm_indicator = a_active,
    alpha_star = alpha_active,
    weights = weights,
    seed = seed + 110L,
    settings = settings
  )
  h_control <- fit_sieve_riesz_arm(
    score = mu_control,
    arm_indicator = a_control,
    alpha_star = alpha_control,
    weights = weights,
    seed = seed + 120L,
    settings = settings
  )
  q_active <- fit_sieve_residual_arm(
    score = 1 / wald_clip_probability(pi_active, eps = settings$propensity_clip),
    residual = Y - mu_active,
    arm_indicator = a_active,
    weights = weights,
    seed = seed + 130L,
    settings = settings
  )
  q_control <- fit_sieve_residual_arm(
    score = 1 / wald_clip_probability(pi_control, eps = settings$propensity_clip),
    residual = Y - mu_control,
    arm_indicator = a_control,
    weights = weights,
    seed = seed + 140L,
    settings = settings
  )

  active_score <- mu_active +
    a_active * (alpha_active + h_active$predictions) * (Y - mu_active) -
    (a_active * alpha_active - 1) * q_active$predictions
  control_score <- mu_control +
    a_control * (alpha_control + h_control$predictions) * (Y - mu_control) -
    (a_control * alpha_control - 1) * q_control$predictions
  corrected_score <- active_score - control_score
  corrected_score <- corrected_score - weighted_mean(corrected_score, weights) + contrast_estimate
  corrected_std_error <- wald_weighted_standard_error(corrected_score, contrast_estimate, weights)
  conservative_std_error <- max(simple_wald_std_error, corrected_std_error, na.rm = TRUE)
  selected_std_error <- if (isTRUE(wald_conservative)) conservative_std_error else corrected_std_error
  alpha <- 1 - conf_level
  z_value <- stats::qnorm(1 - alpha / 2)

  list(
    std_error = selected_std_error,
    lower = contrast_estimate - z_value * selected_std_error,
    upper = contrast_estimate + z_value * selected_std_error,
    diagnostics = list(
      wald_correction = "sieve_riesz",
      wald_aux_method = "sieve_riesz",
      std_error_mode = if (isTRUE(wald_conservative)) "conservative" else "corrected_if",
      simple_wald_std_error = simple_wald_std_error,
      corrected_if_std_error = corrected_std_error,
      conservative_std_error = conservative_std_error,
      selected_std_error = selected_std_error,
      propensity_clip = settings$propensity_clip,
      riesz_bound = settings$riesz_bound,
      basis_type = "cosine",
      h_treated = wald_fit_metadata(h_active),
      h_control = wald_fit_metadata(h_control),
      q_treated = wald_fit_metadata(q_active),
      q_control = wald_fit_metadata(q_control)
    )
  )
}

fit_sieve_riesz_arm <- function(score, arm_indicator, alpha_star, weights, seed, settings) {
  score <- as.numeric(score)
  arm_indicator <- as.numeric(arm_indicator)
  alpha_star <- as.numeric(alpha_star)
  weights <- as.numeric(weights)
  ok <- is.finite(score) & is.finite(arm_indicator) & is.finite(alpha_star) & is.finite(weights) & weights >= 0
  train_score <- score[ok]
  train_arm <- as.numeric(arm_indicator[ok] > 0)
  train_alpha <- alpha_star[ok]
  train_weights <- weights[ok]
  unique_score_count <- wald_unique_score_count(train_score)
  constant_fit <- wald_riesz_constant(train_arm, train_alpha, train_weights, settings$riesz_bound)

  fallback <- function(reason) {
    wald_constant_fit(
      predictions = rep(constant_fit, length(score)),
      requested_method = "sieve_riesz",
      reason = reason,
      train_n = length(train_score),
      arm_n = sum(train_weights * train_arm),
      unique_score_count = unique_score_count,
      target_sd = wald_weighted_sd(1 - train_arm * train_alpha, train_weights),
      bound_lower = -settings$riesz_bound,
      bound_upper = settings$riesz_bound
    )
  }

  if (length(train_score) < settings$min_rows) return(fallback("too_few_rows"))
  if (sum(train_weights * train_arm) <= 0) return(fallback("too_few_positive_weight_arm_rows"))
  if (sum(train_arm) < 2) return(fallback("too_few_arm_rows"))
  if (unique_score_count < settings$min_unique_scores) return(fallback("few_unique_scores"))

  best <- select_sieve_ridge(
    score = train_score,
    target = 1 - train_arm * train_alpha,
    weights = train_weights,
    arm_weights = train_arm,
    loss_kind = "riesz",
    seed = seed,
    settings = settings
  )
  if (is.null(best)) return(fallback("nonfinite_cv_risks"))

  predictions <- fit_and_predict_sieve_ridge(
    train_score = train_score,
    target = 1 - train_arm * train_alpha,
    weights = train_weights,
    eval_score = score,
    basis_size = best$basis_size,
    lambda_value = best$lambda,
    arm_weights = train_arm,
    loss_kind = "riesz"
  )
  predictions[!is.finite(predictions)] <- constant_fit
  predictions <- pmax(-settings$riesz_bound, pmin(settings$riesz_bound, predictions))
  wald_sieve_fit(
    predictions = predictions,
    requested_method = "sieve_riesz",
    basis_size = best$basis_size,
    lambda_value = best$lambda,
    cv_folds = best$cv_folds,
    selected_risk = best$risk,
    train_n = length(train_score),
    arm_n = sum(train_weights * train_arm),
    unique_score_count = unique_score_count,
    target_sd = wald_weighted_sd(1 - train_arm * train_alpha, train_weights),
    bound_lower = -settings$riesz_bound,
    bound_upper = settings$riesz_bound
  )
}

fit_sieve_residual_arm <- function(score, residual, arm_indicator, weights, seed, settings) {
  score <- as.numeric(score)
  residual <- as.numeric(residual)
  arm_indicator <- as.numeric(arm_indicator)
  weights <- as.numeric(weights)
  ok <- is.finite(score) & is.finite(residual) & is.finite(arm_indicator) &
    arm_indicator > 0 & is.finite(weights) & weights >= 0
  train_score <- score[ok]
  train_residual <- residual[ok]
  train_weights <- weights[ok]
  unique_score_count <- wald_unique_score_count(train_score)
  residual_range <- wald_finite_range(train_residual)
  constant_fit <- min(residual_range[[2L]], max(residual_range[[1L]], wald_weighted_mean_or_zero(train_residual, train_weights)))

  fallback <- function(reason) {
    wald_constant_fit(
      predictions = rep(constant_fit, length(score)),
      requested_method = "sieve_residual_ridge",
      reason = reason,
      train_n = length(train_residual),
      arm_n = sum(train_weights),
      unique_score_count = unique_score_count,
      target_sd = wald_weighted_sd(train_residual, train_weights),
      bound_lower = residual_range[[1L]],
      bound_upper = residual_range[[2L]]
    )
  }

  if (length(train_score) < settings$min_rows) return(fallback("too_few_rows"))
  if (sum(train_weights) <= 0) return(fallback("zero_positive_weight"))
  target_sd <- wald_weighted_sd(train_residual, train_weights)
  if (!is.finite(target_sd) || target_sd < 1e-8) return(fallback("near_constant_target"))
  if (unique_score_count < settings$min_unique_scores) return(fallback("few_unique_scores"))

  best <- select_sieve_ridge(
    score = train_score,
    target = train_residual,
    weights = train_weights,
    arm_weights = NULL,
    loss_kind = "residual",
    seed = seed,
    settings = settings
  )
  if (is.null(best)) return(fallback("nonfinite_cv_risks"))

  predictions <- fit_and_predict_sieve_ridge(
    train_score = train_score,
    target = train_residual,
    weights = train_weights,
    eval_score = score,
    basis_size = best$basis_size,
    lambda_value = best$lambda,
    arm_weights = NULL,
    loss_kind = "residual"
  )
  predictions[!is.finite(predictions)] <- constant_fit
  predictions <- pmax(residual_range[[1L]], pmin(residual_range[[2L]], predictions))
  wald_sieve_fit(
    predictions = predictions,
    requested_method = "sieve_residual_ridge",
    basis_size = best$basis_size,
    lambda_value = best$lambda,
    cv_folds = best$cv_folds,
    selected_risk = best$risk,
    train_n = length(train_residual),
    arm_n = sum(train_weights),
    unique_score_count = unique_score_count,
    target_sd = target_sd,
    bound_lower = residual_range[[1L]],
    bound_upper = residual_range[[2L]]
  )
}

select_sieve_ridge <- function(score, target, weights, arm_weights, loss_kind, seed, settings) {
  n <- length(score)
  cv_folds <- min(settings$cv_folds, n)
  fold_ids <- wald_deterministic_fold_ids(n, cv_folds, seed, strata = arm_weights)
  best <- NULL
  for (basis_size in settings$basis_size_grid) {
    basis <- tryCatch(wald_cosine_basis(score, score, basis_size)$train_basis, error = function(e) NULL)
    if (is.null(basis)) next
    penalty <- wald_ridge_penalty(ncol(basis))
    for (lambda_value in settings$lambda_grid) {
      fold_losses <- numeric()
      for (fold in sort(unique(fold_ids))) {
        val_idx <- which(fold_ids == fold)
        train_idx <- which(fold_ids != fold)
        if (!length(train_idx) || !length(val_idx)) next
        beta <- solve_sieve_ridge(
          basis = basis[train_idx, , drop = FALSE],
          target = target[train_idx],
          weights = weights[train_idx],
          arm_weights = if (is.null(arm_weights)) NULL else arm_weights[train_idx],
          lambda_value = lambda_value,
          penalty = penalty,
          loss_kind = loss_kind
        )
        fitted <- as.numeric(basis[val_idx, , drop = FALSE] %*% beta)
        fitted[!is.finite(fitted)] <- 0
        val_weights <- wald_normalize_weights(weights[val_idx])
        if (identical(loss_kind, "riesz")) {
          arm_val <- arm_weights[val_idx]
          loss <- arm_val * fitted ^ 2 + 2 * arm_val * (1 - target[val_idx]) * fitted - 2 * fitted
        } else {
          loss <- (fitted - target[val_idx]) ^ 2
        }
        fold_losses <- c(fold_losses, sum(val_weights * loss))
      }
      risk <- if (length(fold_losses)) mean(fold_losses) else NA_real_
      if (is.finite(risk) && (is.null(best) || risk < best$risk)) {
        best <- list(risk = risk, basis_size = as.integer(basis_size), lambda = lambda_value, cv_folds = cv_folds)
      }
    }
  }
  best
}

fit_and_predict_sieve_ridge <- function(train_score, target, weights, eval_score, basis_size, lambda_value, arm_weights, loss_kind) {
  basis <- wald_cosine_basis(train_score, eval_score, basis_size)
  beta <- solve_sieve_ridge(
    basis = basis$train_basis,
    target = target,
    weights = weights,
    arm_weights = arm_weights,
    lambda_value = lambda_value,
    penalty = wald_ridge_penalty(ncol(basis$train_basis)),
    loss_kind = loss_kind
  )
  as.numeric(basis$eval_basis %*% beta)
}

solve_sieve_ridge <- function(basis, target, weights, arm_weights, lambda_value, penalty, loss_kind) {
  weights_norm <- wald_normalize_weights(weights)
  lhs_weights <- if (identical(loss_kind, "riesz")) {
    weights_norm * as.numeric(arm_weights)
  } else {
    weights_norm
  }
  lhs <- crossprod(basis, basis * lhs_weights) + lambda_value * penalty
  lhs <- (lhs + t(lhs)) / 2
  rhs <- crossprod(basis, weights_norm * target)
  beta <- tryCatch(as.numeric(solve(lhs, rhs)), error = function(e) NULL)
  if (is.null(beta) || length(beta) != length(rhs) || any(!is.finite(beta))) {
    lhs_jittered <- lhs
    diag(lhs_jittered) <- diag(lhs_jittered) + 1e-8
    beta <- tryCatch(as.numeric(solve(lhs_jittered, rhs)), error = function(e) NULL)
  }
  if (is.null(beta) || length(beta) != length(rhs) || any(!is.finite(beta))) {
    beta <- tryCatch(as.numeric(qr.solve(lhs, rhs)), error = function(e) rep(0, length(rhs)))
  }
  beta[!is.finite(beta)] <- 0
  beta
}

wald_cosine_basis <- function(train_score, eval_score, basis_size) {
  train_score <- as.numeric(train_score)
  eval_score <- as.numeric(eval_score)
  finite_train <- train_score[is.finite(train_score)]
  if (!length(finite_train)) {
    stop("Cannot build a sieve basis from nonfinite scores.", call. = FALSE)
  }
  lower <- min(finite_train)
  upper <- max(finite_train)
  scale <- upper - lower
  if (!is.finite(scale) || scale <= 1e-12) {
    stop("Cannot build a nonconstant sieve basis from constant scores.", call. = FALSE)
  }
  train_scaled <- pmax(0, pmin(1, (train_score - lower) / scale))
  eval_scaled <- pmax(0, pmin(1, (eval_score - lower) / scale))
  frequencies <- seq.int(0L, as.integer(basis_size) - 1L)
  train_basis <- cos(pi * outer(train_scaled, frequencies, `*`))
  eval_basis <- cos(pi * outer(eval_scaled, frequencies, `*`))
  train_basis[, 1L] <- 1
  eval_basis[, 1L] <- 1
  train_basis[!is.finite(train_basis)] <- 0
  eval_basis[!is.finite(eval_basis)] <- 0
  list(train_basis = train_basis, eval_basis = eval_basis)
}

wald_deterministic_fold_ids <- function(n, n_folds, seed, strata = NULL) {
  folds <- integer(n)
  keys <- ((seq_len(n) * 1103515245 + as.numeric(seed) * 12345) %% 2147483647)
  if (is.null(strata)) {
    strata <- rep(0, n)
  }
  for (value in sort(unique(strata))) {
    idx <- which(strata == value)
    ordered <- idx[order(keys[idx], idx)]
    folds[ordered] <- (seq_along(ordered) - 1L) %% as.integer(n_folds) + 1L
  }
  folds
}

wald_ridge_penalty <- function(p) {
  penalty <- diag(1, p)
  if (p > 0L) {
    penalty[1L, 1L] <- 0
  }
  penalty
}

wald_constant_fit <- function(predictions, requested_method, reason, train_n, arm_n, unique_score_count, target_sd, bound_lower, bound_upper) {
  list(
    predictions = as.numeric(predictions),
    requested_method = requested_method,
    selected_method = "constant",
    fallback = TRUE,
    fallback_reason = reason,
    basis_size = NA_integer_,
    lambda = NA_real_,
    cv_folds = NA_integer_,
    selected_risk = NA_real_,
    train_n = as.integer(train_n),
    arm_n = as.numeric(arm_n),
    unique_score_count = as.integer(unique_score_count),
    target_sd = if (is.finite(target_sd)) target_sd else NA_real_,
    bound_lower = bound_lower,
    bound_upper = bound_upper
  )
}

wald_sieve_fit <- function(predictions, requested_method, basis_size, lambda_value, cv_folds, selected_risk, train_n, arm_n, unique_score_count, target_sd, bound_lower, bound_upper) {
  list(
    predictions = as.numeric(predictions),
    requested_method = requested_method,
    selected_method = "sieve_ridge_cv",
    fallback = FALSE,
    fallback_reason = NA_character_,
    basis_size = as.integer(basis_size),
    lambda = lambda_value,
    cv_folds = as.integer(cv_folds),
    selected_risk = selected_risk,
    train_n = as.integer(train_n),
    arm_n = as.numeric(arm_n),
    unique_score_count = as.integer(unique_score_count),
    target_sd = if (is.finite(target_sd)) target_sd else NA_real_,
    bound_lower = bound_lower,
    bound_upper = bound_upper
  )
}

wald_fit_metadata <- function(fit) {
  fit$predictions <- NULL
  fit
}

wald_riesz_constant <- function(arm_indicator, alpha_star, weights, bound) {
  denom <- sum(weights * arm_indicator)
  if (!is.finite(denom) || denom <= 0) {
    return(0)
  }
  value <- sum(weights * (1 - arm_indicator * alpha_star)) / denom
  if (!is.finite(value)) {
    return(0)
  }
  max(-bound, min(bound, value))
}

wald_clip_probability <- function(probability, eps) {
  pmax(eps, pmin(1 - eps, as.numeric(probability)))
}

wald_normalize_weights <- function(weights) {
  weights <- as.numeric(weights)
  total <- sum(weights)
  if (!is.finite(total) || total <= 0) {
    return(rep(1 / length(weights), length(weights)))
  }
  weights / total
}

wald_weighted_standard_error <- function(score, estimate, weights) {
  weights_norm <- wald_normalize_weights(weights)
  sqrt(sum((weights_norm * (as.numeric(score) - estimate)) ^ 2))
}

wald_weighted_sd <- function(values, weights) {
  if (length(values) < 2L) {
    return(NA_real_)
  }
  weights_norm <- wald_normalize_weights(weights)
  center <- sum(weights_norm * values)
  sqrt(sum(weights_norm * (values - center) ^ 2))
}

wald_weighted_mean_or_zero <- function(values, weights) {
  if (!length(values)) {
    return(0)
  }
  sum(wald_normalize_weights(weights) * values)
}

wald_finite_range <- function(values) {
  finite <- values[is.finite(values)]
  if (!length(finite)) {
    return(c(0, 0))
  }
  range(finite)
}

wald_unique_score_count <- function(score) {
  if (!length(score)) {
    return(0L)
  }
  length(unique(round(score, 8)))
}

wald_normalize_seed <- function(seed) {
  if (is.null(seed) || length(seed) < 1L || !is.finite(seed[[1L]])) {
    return(1L)
  }
  as.integer((abs(as.numeric(seed[[1L]])) %% 2147483646) + 1L)
}

clean_wald_int_grid <- function(values, name) {
  grid <- unique(as.integer(values[is.finite(values) & values > 0]))
  if (!length(grid)) {
    stop(sprintf("`wald_options$%s` must contain at least one positive integer.", name), call. = FALSE)
  }
  grid
}

clean_wald_float_grid <- function(values, name) {
  grid <- unique(as.numeric(values[is.finite(values) & values > 0]))
  if (!length(grid)) {
    stop(sprintf("`wald_options$%s` must contain at least one positive finite value.", name), call. = FALSE)
  }
  grid
}

`%||null%` <- function(x, y) {
  if (is.null(x)) y else x
}

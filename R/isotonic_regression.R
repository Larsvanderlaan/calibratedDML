#' Monotone Calibration Helper
#'
#' `isoreg_with_xgboost()` is retained as a backwards-compatible alias for the
#' package's internal monotone calibration routine.
#'
#' @param x Numeric predictor vector.
#' @param y Numeric response vector.
#' @param max_depth Deprecated. Ignored.
#' @param min_child_weight Deprecated. Ignored.
#' @param weights Optional non-negative observation weights.
#'
#' @return A function mapping new `x` values to monotone calibrated predictions.
#' @export
isoreg_with_xgboost <- function(x, y, max_depth = 15, min_child_weight = 20, weights = NULL) {
  calibrator <- fit_monotone_calibrator(
    x = x,
    y = y,
    weights = weights,
    method = "isotonic"
  )
  function(new_x) predict_monotone_calibrator(calibrator, new_x)
}

fit_monotone_calibrator <- function(
  x,
  y,
  weights = NULL,
  method = c("auto", "isotonic", "smooth_isotonic"),
  calibration_options = list()
) {
  x <- as.numeric(x)
  y <- as.numeric(y)
  if (length(x) != length(y)) {
    stop("`x` and `y` must have the same length.", call. = FALSE)
  }
  if (anyNA(x) || anyNA(y)) {
    stop("`x` and `y` cannot contain missing values.", call. = FALSE)
  }
  if (is.null(weights)) {
    weights <- rep(1, length(y))
  }
  weights <- as.numeric(weights)
  if (length(weights) != length(y) || any(weights < 0) || anyNA(weights)) {
    stop("`weights` must be non-negative and match the length of `y`.", call. = FALSE)
  }

  method <- resolve_calibration_method(
    method = match.arg(method),
    n = length(y),
    weights = weights,
    calibration_options = calibration_options
  )

  if (identical(method, "isotonic")) {
    return(fit_isotonic_calibrator(x = x, y = y, weights = weights))
  }
  fit_smooth_isotonic_calibrator(x = x, y = y, weights = weights, calibration_options = calibration_options)
}

resolve_calibration_method <- function(method, n, weights, calibration_options = list()) {
  if (!identical(method, "auto")) {
    return(method)
  }

  threshold <- calibration_options$smooth_threshold
  if (is.null(threshold)) {
    threshold <- 300L
  }
  effective_n <- sum(weights > 0)
  if (effective_n < threshold) "smooth_isotonic" else "isotonic"
}

fit_isotonic_calibrator <- function(x, y, weights) {
  if (length(unique(x)) == 1L) {
    return(structure(
      list(
        method = "isotonic_constant",
        x_value = x[[1]],
        constant = weighted_mean(y, weights),
        x_min = x[[1]],
        x_max = x[[1]]
      ),
      class = "cdml_calibrator"
    ))
  }
  blocks <- fit_isotonic_blocks(x = x, y = y, weights = weights)
  structure(
    list(
      method = "isotonic",
      boundary = blocks$boundary,
      fitted = blocks$fitted,
      x_min = min(x),
      x_max = max(x)
    ),
    class = "cdml_calibrator"
  )
}

fit_smooth_isotonic_calibrator <- function(x, y, weights, calibration_options = list()) {
  blocks <- fit_isotonic_blocks(x = x, y = y, weights = weights)
  grid_size <- calibration_options$grid_size
  if (is.null(grid_size)) {
    grid_size <- 256L
  }
  smooth_spar <- calibration_options$smooth_spar

  x_grid <- unique(sort(c(
    min(x),
    max(x),
    blocks$boundary,
    stats::quantile(x, probs = seq(0, 1, length.out = min(grid_size, max(16L, length(x)))), names = FALSE, na.rm = TRUE)
  )))
  x_grid <- x_grid[is.finite(x_grid)]

  if (length(unique(blocks$x_anchor)) < 4L || length(x_grid) < 4L) {
    return(fit_isotonic_calibrator(x = x, y = y, weights = weights))
  }

  spline_fit <- try(
    stats::smooth.spline(
      x = blocks$x_anchor,
      y = blocks$fitted,
      w = blocks$block_weights,
      spar = smooth_spar
    ),
    silent = TRUE
  )
  if (inherits(spline_fit, "try-error")) {
    return(fit_isotonic_calibrator(x = x, y = y, weights = weights))
  }

  y_grid <- as.numeric(stats::predict(spline_fit, x = x_grid)$y)
  y_grid <- cummax(y_grid)
  y_grid <- pmax(min(blocks$fitted), pmin(max(blocks$fitted), y_grid))

  structure(
    list(
      method = "smooth_isotonic",
      grid_x = x_grid,
      grid_y = y_grid,
      x_min = min(x),
      x_max = max(x)
    ),
    class = "cdml_calibrator"
  )
}

fit_isotonic_blocks <- function(x, y, weights) {
  order_index <- order(x, y)
  x_sorted <- x[order_index]
  y_sorted <- y[order_index]
  w_sorted <- weights[order_index]

  block_ends <- seq_along(y_sorted)
  block_means <- y_sorted
  block_weights <- w_sorted

  block_index <- 1L
  while (block_index < length(block_means)) {
    if (block_means[[block_index]] <= block_means[[block_index + 1L]]) {
      block_index <- block_index + 1L
      next
    }

    combined_weight <- block_weights[[block_index]] + block_weights[[block_index + 1L]]
    combined_mean <- (
      block_means[[block_index]] * block_weights[[block_index]] +
        block_means[[block_index + 1L]] * block_weights[[block_index + 1L]]
    ) / combined_weight

    block_means[[block_index]] <- combined_mean
    block_weights[[block_index]] <- combined_weight
    block_ends[[block_index]] <- block_ends[[block_index + 1L]]

    remove_index <- block_index + 1L
    block_ends <- block_ends[-remove_index]
    block_means <- block_means[-remove_index]
    block_weights <- block_weights[-remove_index]

    if (block_index > 1L) {
      block_index <- block_index - 1L
    }
  }

  block_starts <- c(1L, block_ends[-length(block_ends)] + 1L)
  x_anchor <- vapply(seq_along(block_ends), function(index) {
    rows <- block_starts[[index]]:block_ends[[index]]
    sum(x_sorted[rows] * w_sorted[rows]) / sum(w_sorted[rows])
  }, numeric(1))

  list(
    boundary = x_sorted[block_ends],
    fitted = block_means,
    block_weights = block_weights,
    x_anchor = x_anchor
  )
}

predict_monotone_calibrator <- function(calibrator, new_x) {
  new_x <- as.numeric(new_x)
  if (identical(calibrator$method, "isotonic_constant")) {
    return(rep(calibrator$constant, length(new_x)))
  }
  if (identical(calibrator$method, "isotonic")) {
    index <- findInterval(new_x, calibrator$boundary, left.open = TRUE) + 1L
    index <- pmin(index, length(calibrator$fitted))
    return(calibrator$fitted[index])
  }

  approx_fun <- stats::approxfun(
    x = calibrator$grid_x,
    y = calibrator$grid_y,
    rule = 2,
    ties = "ordered"
  )
  as.numeric(approx_fun(new_x))
}

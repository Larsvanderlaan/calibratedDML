builtin_regression_names <- function() {
  c("mean", "lm", "lasso", "gam", "hal_gam", "random_forest", "boosted_trees", "auto")
}

builtin_classification_names <- function() {
  c("mean", "lm", "lasso", "gam", "hal_gam", "random_forest", "boosted_trees", "auto", "multinom", "empirical")
}

make_builtin_regression_spec <- function(model_name) {
  model_config <- normalize_builtin_model_config(model_name, builtin_regression_names())
  model_name <- model_config$name

  if (identical(model_name, "mean")) {
    return(list(
      backend = "builtin",
      kind = "regression",
      fit = function(x, y, weights) list(value = weighted_mean(y, weights)),
      predict = function(model_fit, newx) rep(model_fit$value, nrow(newx))
    ))
  }

  if (identical(model_name, "lm")) {
    return(list(
      backend = "builtin",
      kind = "regression",
      fit = function(x, y, weights) stats::lm(y ~ ., data = x, weights = weights),
      predict = function(model_fit, newx) stats::predict(model_fit, newdata = newx)
    ))
  }

  if (identical(model_name, "lasso")) {
    return(list(
      backend = "builtin",
      kind = "regression",
      fit = function(x, y, weights) {
        require_optional_package("glmnet", model_name)
        glmnet::cv.glmnet(design_matrix(x), y, weights = weights, family = "gaussian")
      },
      predict = function(model_fit, newx) as.numeric(stats::predict(model_fit, newx = design_matrix(newx), s = "lambda.min"))
    ))
  }

  if (identical(model_name, "gam")) {
    return(list(
      backend = "builtin",
      kind = "regression",
      fit = function(x, y, weights) {
        require_optional_package("mgcv", model_name)
        data <- data.frame(.y = y, x, check.names = FALSE)
        base::eval(base::call("mgcv::gam", build_gam_formula(".y", names(x)), data = data, weights = weights, family = stats::gaussian()))
      },
      predict = function(model_fit, newx) stats::predict(model_fit, newdata = newx, type = "response")
    ))
  }

  if (identical(model_name, "hal_gam")) {
    return(list(
      backend = "builtin",
      kind = "regression",
      fit = function(x, y, weights) fit_hal_gam_regression(x, y, weights, control = model_config),
      predict = function(model_fit, newx) predict_hal_gam(model_fit, newx)
    ))
  }

  if (identical(model_name, "random_forest")) {
    return(list(
      backend = "builtin",
      kind = "regression",
      fit = function(x, y, weights) {
        require_optional_package("ranger", model_name)
        ranger::ranger(
          dependent.variable.name = ".y",
          data = data.frame(.y = y, x, check.names = FALSE),
          case.weights = weights,
          num.trees = 500,
          mtry = max(1, floor(sqrt(ncol(x)))),
          min.node.size = 5
        )
      },
      predict = function(model_fit, newx) as.numeric(stats::predict(model_fit, data = newx)$predictions)
    ))
  }

  if (identical(model_name, "boosted_trees")) {
    return(list(
      backend = "builtin",
      kind = "regression",
      fit = function(x, y, weights) fit_lightgbm_regression(x, y, weights),
      predict = function(model_fit, newx) predict_lightgbm(model_fit, newx)
    ))
  }

  if (identical(model_name, "auto")) {
    return(make_auto_regression_spec(c("lasso", "random_forest", "boosted_trees", "gam")))
  }

  stop("Unsupported regression model.", call. = FALSE)
}

make_builtin_classification_spec <- function(model_name) {
  model_config <- normalize_builtin_model_config(model_name, builtin_classification_names())
  model_name <- model_config$name

  if (identical(model_name, "empirical")) {
    model_name <- "mean"
  }
  if (identical(model_name, "multinom")) {
    return(list(
      backend = "builtin",
      kind = "classification",
      supports_multiclass_direct = TRUE,
      fit = function(x, y, weights) {
        require_optional_package("nnet", model_name)
        train_data <- data.frame(.outcome = factor(y), x, check.names = FALSE)
        list(
          fit = nnet::multinom(.outcome ~ ., data = train_data, weights = weights, trace = FALSE),
          levels = levels(train_data$.outcome)
        )
      },
      predict = function(model_fit, newx) {
        pred <- stats::predict(model_fit$fit, newdata = newx, type = "probs")
        if (is.null(dim(pred))) {
          pred <- cbind(1 - as.numeric(pred), as.numeric(pred))
          colnames(pred) <- model_fit$levels
          return(pred)
        }
        pred <- as.matrix(pred)
        if (is.null(colnames(pred))) {
          colnames(pred) <- model_fit$levels
        }
        pred
      }
    ))
  }

  if (identical(model_name, "auto")) {
    return(make_auto_classification_spec(c("lasso", "random_forest", "boosted_trees", "gam")))
  }

  if (identical(model_name, "mean")) {
    fit_binary <- function(x, y, weights) list(prob = weighted_mean(y, weights))
    predict_binary <- function(model_fit, newx) rep(model_fit$prob, nrow(newx))
  } else if (identical(model_name, "lm")) {
    fit_binary <- function(x, y, weights) fit_balnet_binary(x, y, weights, alpha = 0)
    predict_binary <- function(model_fit, newx) predict_balnet(model_fit, newx)
  } else if (identical(model_name, "lasso")) {
    fit_binary <- function(x, y, weights) fit_balnet_binary(x, y, weights, alpha = 1)
    predict_binary <- function(model_fit, newx) predict_balnet(model_fit, newx)
  } else if (identical(model_name, "gam")) {
    fit_binary <- function(x, y, weights) {
      require_optional_package("mgcv", model_name)
      data <- data.frame(.y = y, x, check.names = FALSE)
      base::eval(base::call("mgcv::gam", build_gam_formula(".y", names(x)), data = data, weights = weights, family = stats::binomial()))
    }
    predict_binary <- function(model_fit, newx) stats::predict(model_fit, newdata = newx, type = "response")
  } else if (identical(model_name, "hal_gam")) {
    fit_binary <- function(x, y, weights) fit_hal_gam_binary(x, y, weights, control = model_config)
    predict_binary <- function(model_fit, newx) predict_hal_gam(model_fit, newx)
  } else if (identical(model_name, "random_forest")) {
    fit_binary <- function(x, y, weights) {
      require_optional_package("ranger", model_name)
      ranger::ranger(
        dependent.variable.name = ".y",
        data = data.frame(.y = factor(y, levels = c(0, 1)), x, check.names = FALSE),
        case.weights = weights,
        num.trees = 500,
        probability = TRUE,
        mtry = max(1, floor(sqrt(ncol(x)))),
        min.node.size = 5
      )
    }
    predict_binary <- function(model_fit, newx) {
      predictions <- stats::predict(model_fit, data = newx)$predictions
      if (is.matrix(predictions)) as.numeric(predictions[, "1"]) else as.numeric(predictions)
    }
  } else if (identical(model_name, "boosted_trees")) {
    fit_binary <- function(x, y, weights) fit_lightgbm_binary(x, y, weights)
    predict_binary <- function(model_fit, newx) predict_lightgbm(model_fit, newx)
  } else {
    stop("Unsupported classification model.", call. = FALSE)
  }

  list(
    backend = "builtin",
    kind = "classification",
    supports_multiclass_direct = FALSE,
    fit = function(x, y, weights) {
      levels_y <- levels(factor(y))
      fits <- lapply(levels_y, function(level) {
        fit_binary(x, as.numeric(y == level), weights)
      })
      names(fits) <- levels_y
      list(levels = levels_y, fits = fits)
    },
    predict = function(model_fit, newx) {
      pred <- vapply(model_fit$fits, function(fit) {
        as.numeric(predict_binary(fit, newx))
      }, numeric(nrow(newx)))
      pred <- t(pred)
      pred <- t(pred)
      colnames(pred) <- model_fit$levels
      pred
    }
  )
}

make_auto_regression_spec <- function(candidate_names) {
  list(
    backend = "builtin",
    kind = "regression",
    fit = function(x, y, weights) {
      best <- select_builtin_regression_model(x, y, weights, candidate_names)
      fit <- make_builtin_regression_spec(best)$fit(x, y, weights)
      list(model_name = best, fit = fit)
    },
    predict = function(model_fit, newx) {
      make_builtin_regression_spec(model_fit$model_name)$predict(model_fit$fit, newx)
    }
  )
}

make_auto_classification_spec <- function(candidate_names) {
  list(
    backend = "builtin",
    kind = "classification",
    supports_multiclass_direct = FALSE,
    fit = function(x, y, weights) {
      best <- select_builtin_classification_model(x, y, weights, candidate_names)
      fit <- make_builtin_classification_spec(best)$fit(x, y, weights)
      list(model_name = best, fit = fit)
    },
    predict = function(model_fit, newx) {
      make_builtin_classification_spec(model_fit$model_name)$predict(model_fit$fit, newx)
    }
  )
}

select_builtin_regression_model <- function(x, y, weights, candidate_names, n_folds = 3) {
  folds <- resolve_fold_id(length(y), n_folds = min(n_folds, length(y)), fold_id = NULL, seed = 1)
  scores <- sapply(candidate_names, function(candidate) {
    spec <- make_builtin_regression_spec(candidate)
    fold_scores <- numeric(length(unique(folds)))
    for (fold in sort(unique(folds))) {
      train <- which(folds != fold)
      valid <- which(folds == fold)
      fit <- try(spec$fit(x[train, , drop = FALSE], y[train], weights[train]), silent = TRUE)
      if (inherits(fit, "try-error")) {
        fold_scores[[fold]] <- Inf
        next
      }
      pred <- spec$predict(fit, x[valid, , drop = FALSE])
      fold_scores[[fold]] <- weighted_mean((y[valid] - pred) ^ 2, weights[valid])
    }
    mean(fold_scores)
  })
  candidate_names[[which.min(scores)]]
}

select_builtin_classification_model <- function(x, y, weights, candidate_names, n_folds = 3) {
  folds <- resolve_fold_id(length(y), n_folds = min(n_folds, length(y)), fold_id = NULL, seed = 1)
  scores <- sapply(candidate_names, function(candidate) {
    spec <- make_builtin_classification_spec(candidate)
    fold_scores <- numeric(length(unique(folds)))
    for (fold in sort(unique(folds))) {
      train <- which(folds != fold)
      valid <- which(folds == fold)
      fit <- try(spec$fit(x[train, , drop = FALSE], y[train], weights[train]), silent = TRUE)
      if (inherits(fit, "try-error")) {
        fold_scores[[fold]] <- Inf
        next
      }
      pred <- normalize_probability_matrix(spec$predict(fit, x[valid, , drop = FALSE]))
      observed <- model.matrix(~ factor(y[valid], levels = fit$levels) - 1)
      fold_scores[[fold]] <- -sum(weights[valid] * rowSums(observed * log(pmax(pred, 1e-8)))) / sum(weights[valid])
    }
    mean(fold_scores)
  })
  candidate_names[[which.min(scores)]]
}

fit_lightgbm_regression <- function(x, y, weights) {
  require_optional_package("lightgbm", "boosted_trees")
  params_grid <- expand.grid(
    learning_rate = c(0.05, 0.1),
    num_leaves = c(15L, 31L),
    min_data_in_leaf = c(10L, 20L),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  dataset <- lightgbm::lgb.Dataset(data = design_matrix(x), label = y, weight = weights)
  folds <- split(seq_along(y), resolve_fold_id(length(y), n_folds = min(3, length(y)), fold_id = NULL, seed = 1))
  score <- rep(Inf, nrow(params_grid))
  best_iter <- rep(100L, nrow(params_grid))
  for (index in seq_len(nrow(params_grid))) {
    params <- list(
      objective = "regression",
      metric = "l2",
      learning_rate = params_grid$learning_rate[[index]],
      num_leaves = params_grid$num_leaves[[index]],
      min_data_in_leaf = params_grid$min_data_in_leaf[[index]],
      verbose = -1
    )
    fit <- lightgbm::lgb.cv(
      params = params,
      data = dataset,
      folds = folds,
      nrounds = 500,
      early_stopping_rounds = 25,
      verbose = -1
    )
    score[[index]] <- min(unlist(fit$record_evals$valid$l2$eval))
    best_iter[[index]] <- fit$best_iter
  }
  best <- which.min(score)
  params <- list(
    objective = "regression",
    metric = "l2",
    learning_rate = params_grid$learning_rate[[best]],
    num_leaves = params_grid$num_leaves[[best]],
    min_data_in_leaf = params_grid$min_data_in_leaf[[best]],
    verbose = -1
  )
  model <- lightgbm::lgb.train(params = params, data = dataset, nrounds = best_iter[[best]])
  list(model = model)
}

fit_lightgbm_binary <- function(x, y, weights) {
  require_optional_package("lightgbm", "boosted_trees")
  params_grid <- expand.grid(
    learning_rate = c(0.05, 0.1),
    num_leaves = c(15L, 31L),
    min_data_in_leaf = c(10L, 20L),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  dataset <- lightgbm::lgb.Dataset(data = design_matrix(x), label = y, weight = weights)
  folds <- split(seq_along(y), resolve_fold_id(length(y), n_folds = min(3, length(y)), fold_id = NULL, seed = 1))
  score <- rep(Inf, nrow(params_grid))
  best_iter <- rep(100L, nrow(params_grid))
  for (index in seq_len(nrow(params_grid))) {
    params <- list(
      objective = "binary",
      metric = "binary_logloss",
      learning_rate = params_grid$learning_rate[[index]],
      num_leaves = params_grid$num_leaves[[index]],
      min_data_in_leaf = params_grid$min_data_in_leaf[[index]],
      verbose = -1
    )
    fit <- lightgbm::lgb.cv(
      params = params,
      data = dataset,
      folds = folds,
      nrounds = 500,
      early_stopping_rounds = 25,
      verbose = -1
    )
    score[[index]] <- min(unlist(fit$record_evals$valid$binary_logloss$eval))
    best_iter[[index]] <- fit$best_iter
  }
  best <- which.min(score)
  params <- list(
    objective = "binary",
    metric = "binary_logloss",
    learning_rate = params_grid$learning_rate[[best]],
    num_leaves = params_grid$num_leaves[[best]],
    min_data_in_leaf = params_grid$min_data_in_leaf[[best]],
    verbose = -1
  )
  model <- lightgbm::lgb.train(params = params, data = dataset, nrounds = best_iter[[best]])
  list(model = model)
}

predict_lightgbm <- function(model_fit, newx) {
  as.numeric(stats::predict(model_fit$model, design_matrix(newx)))
}

design_matrix <- function(x) {
  stats::model.matrix(~ . - 1, data = as.data.frame(x, check.names = FALSE))
}

build_gam_formula <- function(outcome_name, variables) {
  smooth_terms <- if (length(variables)) {
    paste(sprintf("mgcv::s(`%s`)", variables), collapse = " + ")
  } else {
    "1"
  }
  stats::as.formula(paste(outcome_name, "~", smooth_terms))
}

require_optional_package <- function(package_name, model_name) {
  if (!requireNamespace(package_name, quietly = TRUE)) {
    stop(sprintf("Model `%s` requires the optional `%s` package.", model_name, package_name), call. = FALSE)
  }
}

normalize_builtin_model_config <- function(model, allowed_names) {
  if (is.character(model) && length(model) == 1L) {
    model_name <- match.arg(model, allowed_names)
    return(list(name = model_name))
  }

  if (is.list(model) && !is.null(model$name) && is.character(model$name) && length(model$name) == 1L &&
      is.null(model$fit) && is.null(model$predict)) {
    model_name <- match.arg(model$name, allowed_names)
    model$name <- model_name
    return(model)
  }

  stop("Unsupported built-in model specification.", call. = FALSE)
}

hal_gam_control <- function(control, x) {
  basis_x <- design_matrix(x)
  n <- nrow(basis_x)
  default_knots <- max(5L, min(30L, floor(max(20L, n) / 10L)))
  list(
    name = "hal_gam",
    degree = if (!is.null(control$degree)) as.integer(control$degree) else 1L,
    n_knots = if (!is.null(control$n_knots)) as.integer(control$n_knots) else as.integer(default_knots),
    include_linear = if (!is.null(control$include_linear)) isTRUE(control$include_linear) else TRUE,
    alpha = if (!is.null(control$alpha)) as.numeric(control$alpha) else 1,
    basis_x = basis_x
  )
}

fit_hal_gam_regression <- function(x, y, weights, control = list(name = "hal_gam")) {
  require_optional_package("glmnet", "hal_gam")
  control <- hal_gam_control(control, x)
  basis <- build_hal_basis(control$basis_x, n_knots = control$n_knots, degree = control$degree, include_linear = control$include_linear)
  fit <- glmnet::cv.glmnet(
    x = basis$matrix,
    y = y,
    weights = weights,
    family = "gaussian",
    alpha = control$alpha
  )
  list(
    fit = fit,
    basis = basis,
    alpha = control$alpha
  )
}

fit_hal_gam_binary <- function(x, y, weights, control = list(name = "hal_gam")) {
  require_optional_package("balnet", "hal_gam")
  control <- hal_gam_control(control, x)
  basis <- build_hal_basis(control$basis_x, n_knots = control$n_knots, degree = control$degree, include_linear = control$include_linear)
  fit <- balnet::cv.balnet(
    X = basis$matrix,
    W = as.integer(y),
    target = "ATE",
    sample.weights = weights,
    alpha = control$alpha
  )
  list(
    fit = fit,
    basis = basis,
    alpha = control$alpha
  )
}

predict_hal_gam <- function(model_fit, newx) {
  new_basis <- predict_hal_basis(model_fit$basis, newx)
  if (inherits(model_fit$fit, "cv.glmnet")) {
    return(as.numeric(stats::predict(model_fit$fit, newx = new_basis, s = "lambda.min")))
  }
  as.numeric(stats::predict(model_fit$fit, X = new_basis, lambda = "lambda.min"))
}

build_hal_basis <- function(x, n_knots = 30L, degree = 1L, include_linear = TRUE) {
  x <- as.matrix(x)
  degree <- as.integer(degree)
  if (degree < 0L) {
    stop("`degree` must be non-negative.", call. = FALSE)
  }
  basis_parts <- list()
  metadata <- vector("list", ncol(x))
  colnames_x <- colnames(x)
  if (is.null(colnames_x)) {
    colnames_x <- paste0("x", seq_len(ncol(x)))
  }

  for (j in seq_len(ncol(x))) {
    values <- as.numeric(x[, j])
    name <- colnames_x[[j]]
    knots <- hal_knots(values, n_knots = n_knots)
    parts <- list()
    part_names <- character()

    if (include_linear || !length(knots)) {
      parts[[length(parts) + 1L]] <- values
      part_names[[length(part_names) + 1L]] <- paste0(name, "__linear")
    }

    if (length(knots)) {
      for (k in seq_along(knots)) {
        parts[[length(parts) + 1L]] <- pmax(values - knots[[k]], 0) ^ degree
        part_names[[length(part_names) + 1L]] <- paste0(name, "__k", k)
      }
    }

    metadata[[j]] <- list(name = name, knots = knots)
    basis_parts[[j]] <- stats::setNames(as.data.frame(parts, check.names = FALSE), part_names)
  }

  basis_df <- do.call(cbind, basis_parts)
  list(
    matrix = as.matrix(basis_df),
    metadata = metadata,
    degree = degree,
    include_linear = include_linear
  )
}

predict_hal_basis <- function(basis_fit, newx) {
  x <- design_matrix(newx)
  x <- as.matrix(x)
  basis_parts <- list()

  for (j in seq_along(basis_fit$metadata)) {
    meta <- basis_fit$metadata[[j]]
    values <- as.numeric(x[, meta$name])
    parts <- list()
    if (basis_fit$include_linear || !length(meta$knots)) {
      parts[[length(parts) + 1L]] <- values
    }
    if (length(meta$knots)) {
      for (k in seq_along(meta$knots)) {
        parts[[length(parts) + 1L]] <- pmax(values - meta$knots[[k]], 0) ^ basis_fit$degree
      }
    }
    basis_parts[[j]] <- as.data.frame(parts, check.names = FALSE)
  }

  as.matrix(do.call(cbind, basis_parts))
}

hal_knots <- function(x, n_knots = 30L) {
  x <- x[is.finite(x)]
  if (length(unique(x)) <= 3L) {
    return(numeric())
  }
  probs <- seq(0, 1, length.out = as.integer(n_knots) + 2L)
  knots <- unique(as.numeric(stats::quantile(x, probs = probs[-c(1L, length(probs))], na.rm = TRUE, names = FALSE, type = 1)))
  knots[is.finite(knots)]
}

fit_balnet_binary <- function(x, y, weights, alpha) {
  require_optional_package("balnet", "balnet")
  balnet::cv.balnet(
    X = design_matrix(x),
    W = as.integer(y),
    target = "ATE",
    sample.weights = weights,
    alpha = alpha
  )
}

predict_balnet <- function(model_fit, newx) {
  pred <- stats::predict(model_fit, newdata = design_matrix(newx), lambda = "lambda.min")
  if (is.list(pred)) {
    control <- pred$control
    treated <- pred$treated
    if (!is.null(control) && !is.null(treated)) {
      control <- as.numeric(control)
      treated <- as.numeric(treated)
      denom <- pmax(control + treated, 1e-8)
      return(treated / denom)
    }
    if (!is.null(treated)) {
      return(as.numeric(treated))
    }
    if (!is.null(control)) {
      return(1 - as.numeric(control))
    }
  }
  as.numeric(pred)
}

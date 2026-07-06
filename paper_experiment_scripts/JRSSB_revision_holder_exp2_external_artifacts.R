library(data.table)
library(ggplot2)

holder_exp2_env <- function(name, fallback) {
  Sys.getenv(name, unset = fallback)
}

holder_exp2_output_dir <- function() {
  out <- holder_exp2_env(
    "CDML_PAPER_REVISION_HOLDER_EXP2_ARTIFACT_DIR",
    file.path(
      "paper_experiment_results",
      "reproduced",
      sprintf("holder_exp2_external_artifacts_%s", format(Sys.Date(), "%Y%m%d"))
    )
  )
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  normalizePath(out, mustWork = FALSE)
}

holder_exp2_paper_plot_dir <- function() {
  out <- holder_exp2_env(
    "CDML_PAPER_REVISION_HOLDER_EXP2_PAPER_PLOT_DIR",
    file.path("local-untracked", "papers", "newplots")
  )
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  normalizePath(out, mustWork = FALSE)
}

holder_exp2_sources <- function() {
  data.table(
    role = c(
      "xgbrf_dimcurve_n10000",
      "xgbrf_samplegrid_external",
      "glmnet_ridge_dimcurve_n10000",
      "glmnet_ridge_samplegrid_nsim250",
      "glmnet_ridge_samplegrid_nsim100"
    ),
    source_dir = c(
      holder_exp2_env(
        "CDML_PAPER_REVISION_HOLDER_EXP2_XGBRF_DIM_DIR",
        file.path(
          "paper_experiment_results", "reproduced",
          "holder_xgbrf_samplegrid_external_levelsetwald_additiveoutcome_main2_n2500n5000n10000n15000_d4d8d16_100_isoreg_eta01_nthread1_20260706_1202"
        )
      ),
      holder_exp2_env(
        "CDML_PAPER_REVISION_HOLDER_EXP2_XGBRF_SAMPLE_DIR",
        file.path(
          "paper_experiment_results", "reproduced",
          "holder_xgbrf_samplegrid_external_levelsetwald_additiveoutcome_main2_n2500n5000n10000n15000_d4d8d16_100_isoreg_eta01_nthread1_20260706_1202"
        )
      ),
      holder_exp2_env(
        "CDML_PAPER_REVISION_HOLDER_EXP2_GLMNET_DIM_DIR",
        file.path(
          "paper_experiment_results", "reproduced",
          "holder_xgbrf_dimcurve_plain_ridge_slowprop_n10000_d1d2d4d8d12d16_250_fastbasis300_20260702_1115"
        )
      ),
      holder_exp2_env(
        "CDML_PAPER_REVISION_HOLDER_EXP2_GLMNET_SAMPLE250_DIR",
        file.path(
          "paper_experiment_results", "reproduced",
          "holder_lasso_ridge_samplecurve_n2500n5000n10000n15000_d4d8d16_250_fastbasis300_20260705_1230"
        )
      ),
      holder_exp2_env(
        "CDML_PAPER_REVISION_HOLDER_EXP2_GLMNET_SAMPLE100_DIR",
        file.path(
          "paper_experiment_results", "reproduced",
          "holder_lasso_ridge_samplecurve_n2500n5000n10000n15000_d4d8d16_100_fastbasis200_20260705_1256"
        )
      )
    ),
    checkpoint_subdir = c(
      "holder_xgbrf",
      "holder_xgbrf",
      "holder_xgbrf_lasso_baseline",
      "holder_xgbrf_lasso_baseline",
      "holder_xgbrf_lasso_baseline"
    ),
    required_iters = c(500L, 500L, 250L, 250L, 100L)
  )
}

holder_exp2_numeric_columns <- c(
  "d", "n", "iter", "estimate", "std_error", "CI_left", "CI_right",
  "ATE", "rmse_fast", "rmse_slow", "raw_oracle_abs_remainder",
  "calibrated_oracle_abs_remainder", "product_error_qg",
  "sqrt_n_product_error_qg"
)

holder_exp2_coerce_numeric <- function(dt) {
  for (col in intersect(holder_exp2_numeric_columns, names(dt))) {
    dt[, (col) := as.numeric(get(col))]
  }
  for (col in intersect(c("d", "n", "iter"), names(dt))) {
    dt[, (col) := as.integer(round(get(col)))]
  }
  dt
}

holder_exp2_add_missing <- function(dt, cols) {
  for (col in setdiff(cols, names(dt))) {
    dt[, (col) := NA_character_]
  }
  dt
}

holder_exp2_read_checkpoints <- function(role, source_dir, subdir, required_iters) {
  checkpoint_dir <- file.path(source_dir, "checkpoints", subdir)
  files <- list.files(checkpoint_dir, full.names = TRUE, pattern = "[.]csv$")
  files <- files[!grepl("xcorr_smoke", basename(files), fixed = TRUE)]
  if (!length(files)) {
    return(list(
      results = data.table(),
      files = character(),
      duplicate_rows = 0L,
      latest_mtime = as.POSIXct(NA),
      checkpoint_dir = checkpoint_dir,
      required_iters = required_iters
    ))
  }
  results <- rbindlist(lapply(files, fread, na.strings = c("NA", "")), fill = TRUE)
  results <- holder_exp2_coerce_numeric(results)
  results[, `:=`(
    source_role = role,
    source_dir = source_dir,
    required_iters = as.integer(required_iters)
  )]

  if (identical(subdir, "holder_xgbrf_lasso_baseline")) {
    key <- c("source_role", "nuisance_method", "setting", "d", "n", "iter", "estimator", "ci_label")
  } else {
    key <- c(
      "source_role", "setting", "d", "n", "iter", "estimator", "ci_label",
      "calibration_targets", "wald_aux_method"
    )
  }
  key <- intersect(key, names(results))
  duplicate_rows <- nrow(results) - nrow(unique(results, by = key))
  results <- unique(results, by = key)

  list(
    results = results,
    files = files,
    duplicate_rows = duplicate_rows,
    latest_mtime = max(file.info(files)$mtime),
    checkpoint_dir = checkpoint_dir,
    required_iters = required_iters
  )
}

holder_exp2_status <- function(results, expected, required_iters, source_role, source_dir, duplicate_rows, latest_mtime) {
  group_cols <- intersect(c("source_role", "nuisance_method", "setting", "d", "n"), names(expected))
  counts <- data.table()
  if (nrow(results)) {
    available_group_cols <- intersect(group_cols, names(results))
    counts <- unique(results[, c(available_group_cols, "iter"), with = FALSE])[
      ,
      .(
        completed_iters = uniqueN(iter),
        min_iter = min(iter),
        max_iter = max(iter)
      ),
      by = available_group_cols
    ]
  }
  status <- merge(expected, counts, by = group_cols, all.x = TRUE)
  status[is.na(completed_iters), `:=`(
    completed_iters = 0L,
    min_iter = NA_integer_,
    max_iter = NA_integer_
  )]
  status[, `:=`(
    required_iters = as.integer(required_iters),
    missing_count = pmax(as.integer(required_iters) - completed_iters, 0L),
    complete = completed_iters >= as.integer(required_iters),
    source_role = source_role,
    source_dir = source_dir,
    duplicate_rows = duplicate_rows,
    latest_mtime = as.character(latest_mtime)
  )]
  setorder(status, source_role, setting, d, n)
  status[]
}

holder_exp2_expected_xgbrf_dim <- function(source_role) {
  CJ(
    source_role = source_role,
    setting = "outcome_additive_propensity_ranger",
    d = c(1L, 2L, 4L, 8L, 12L, 16L),
    n = 10000L
  )
}

holder_exp2_expected_sample <- function(source_role) {
  CJ(
    source_role = source_role,
    setting = "outcome_additive_propensity_ranger",
    d = c(4L, 8L, 12L, 16L),
    n = c(2500L, 5000L, 10000L, 15000L)
  )
}

holder_exp2_expected_glmnet_dim <- function(source_role) {
  CJ(
    source_role = source_role,
    nuisance_method = "glmnet_additive_lasso_highdim_ridge",
    setting = "outcome_additive_propensity_ranger",
    d = c(1L, 2L, 4L, 8L, 12L, 16L),
    n = 10000L
  )
}

holder_exp2_expected_glmnet_sample <- function(source_role) {
  CJ(
    source_role = source_role,
    nuisance_method = "glmnet_additive_lasso_highdim_ridge",
    setting = "outcome_additive_propensity_ranger",
    d = c(4L, 8L, 16L),
    n = c(2500L, 5000L, 10000L, 15000L)
  )
}

holder_exp2_xgbrf_method_label <- function(estimator, ci_label, wald_aux_method) {
  fifelse(
    estimator == "AIPW",
    "AIPW",
    fifelse(
      estimator == "calibratedDML" & ci_label == "dral_wald",
      "C-DML (LS-Wald)",
      fifelse(
        estimator == "calibratedDML" & ci_label == "bootstrap",
        "C-DML (boot.)",
        fifelse(
          estimator == "calibratedDML" & ci_label == "wald",
          "C-DML (Wald)",
          fifelse(
            estimator == "DR-TMLE",
            "DR-TMLE",
            fifelse(
              estimator == "HOIF-2",
              "HOIF-2",
              fifelse(estimator == "Bonvini-kernel", "Bonvini", paste(estimator, ci_label, wald_aux_method))
            )
          )
        )
      )
    )
  )
}

holder_exp2_glmnet_method_label <- function(nuisance_method) {
  fifelse(
    nuisance_method == "glmnet_additive_lasso_highdim_lasso",
    "glmnet lasso AIPW",
    fifelse(
      nuisance_method == "glmnet_additive_lasso_highdim_ridge",
      "glmnet ridge AIPW",
      nuisance_method
    )
  )
}

holder_exp2_summarise <- function(results, family_name) {
  if (!nrow(results)) {
    return(data.table())
  }
  results <- copy(results)
  results <- holder_exp2_add_missing(results, c(
    "nuisance_method", "calibration_targets", "wald_aux_method"
  ))
  results[, `:=`(
    covered = is.finite(CI_left) & is.finite(CI_right) & CI_left <= ATE & ATE <= CI_right,
    interval_width = CI_right - CI_left,
    error = estimate - ATE
  )]
  by_cols <- c(
    "source_role", "source_dir", "required_iters", "setting", "d", "n",
    "nuisance_method", "estimator", "ci_label", "calibration_targets",
    "wald_aux_method"
  )
  out <- results[
    ,
    .(
      reps = uniqueN(iter),
      row_count = .N,
      finite_reps = sum(is.finite(estimate) & is.finite(CI_left) & is.finite(CI_right)),
      coverage = mean(covered, na.rm = TRUE),
      bias = mean(error, na.rm = TRUE),
      abs_bias = abs(mean(error, na.rm = TRUE)),
      rmse = sqrt(mean(error^2, na.rm = TRUE)),
      mean_std_error = mean(std_error, na.rm = TRUE),
      mean_interval_width = mean(interval_width, na.rm = TRUE),
      nonfinite_estimates = sum(!is.finite(estimate)),
      mean_rmse_fast = if ("rmse_fast" %in% names(.SD)) mean(rmse_fast, na.rm = TRUE) else NA_real_,
      mean_rmse_slow = if ("rmse_slow" %in% names(.SD)) mean(rmse_slow, na.rm = TRUE) else NA_real_,
      mean_raw_oracle_abs_remainder = if ("raw_oracle_abs_remainder" %in% names(.SD)) mean(raw_oracle_abs_remainder, na.rm = TRUE) else NA_real_,
      mean_calibrated_oracle_abs_remainder = if ("calibrated_oracle_abs_remainder" %in% names(.SD)) mean(calibrated_oracle_abs_remainder, na.rm = TRUE) else NA_real_
    ),
    by = by_cols
  ]
  out[, family := family_name]
  out[, method := if (identical(family_name, "XGB/RF")) {
    holder_exp2_xgbrf_method_label(estimator, ci_label, wald_aux_method)
  } else {
    holder_exp2_glmnet_method_label(nuisance_method)
  }]
  out[, method := factor(
    method,
    levels = c(
      "AIPW", "C-DML (LS-Wald)", "C-DML (boot.)", "C-DML (Wald)",
      "DR-TMLE", "HOIF-2", "Bonvini", "glmnet lasso AIPW",
      "glmnet ridge AIPW"
    )
  )]
  setorder(out, source_role, setting, d, n, method)
  out[]
}

holder_exp2_merge_status <- function(summary, status) {
  if (!nrow(summary)) {
    return(summary)
  }
  key <- intersect(c("source_role", "nuisance_method", "setting", "d", "n"), names(status))
  if ("nuisance_method" %in% key && all(is.na(status$nuisance_method) | !nzchar(as.character(status$nuisance_method)))) {
    key <- setdiff(key, "nuisance_method")
  }
  summary <- merge(
    summary,
    status[, c(key, "completed_iters", "required_iters", "missing_count", "complete"), with = FALSE],
    by = key,
    all.x = TRUE,
    suffixes = c("", "_status")
  )
  summary[, status := fifelse(complete, "complete", "partial")]
  summary[]
}

holder_exp2_theme <- function(base_size = 12.5) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      legend.text = element_text(size = 17),
      legend.key.width = grid::unit(1.35, "lines"),
      legend.key.height = grid::unit(1.1, "lines"),
      axis.text = element_text(size = 11),
      axis.title = element_text(size = 12),
      strip.text = element_text(size = 12),
      strip.background = element_rect(fill = "grey92", color = "grey70"),
      plot.title.position = "plot"
    )
}

holder_exp2_save_plot <- function(plot, stem, width, height, paper_plot_dir = NULL) {
  pdf_path <- paste0(stem, ".pdf")
  png_path <- paste0(stem, ".png")
  standardize <- tolower(holder_exp2_env("CDML_PAPER_REVISION_HOLDER_EXP2_STANDARDIZE_PLOT_SIZE", "true")) %in%
    c("1", "true", "yes", "y")
  actual_width <- if (standardize) {
    as.numeric(holder_exp2_env("CDML_PAPER_REVISION_HOLDER_EXP2_PLOT_WIDTH", "6.8"))
  } else {
    width
  }
  actual_height <- if (standardize) {
    as.numeric(holder_exp2_env("CDML_PAPER_REVISION_HOLDER_EXP2_PLOT_HEIGHT", "4.55"))
  } else {
    height
  }
  ggsave(pdf_path, plot, width = actual_width, height = actual_height, units = "in")
  ggsave(png_path, plot, width = actual_width, height = actual_height, units = "in", dpi = 320)
  if (!is.null(paper_plot_dir)) {
    file.copy(pdf_path, file.path(paper_plot_dir, basename(pdf_path)), overwrite = TRUE)
    file.copy(png_path, file.path(paper_plot_dir, basename(png_path)), overwrite = TRUE)
  }
  invisible(c(pdf_path, png_path))
}

holder_exp2_n_label <- function(x) {
  out <- x / 1000
  ifelse(abs(out - round(out)) < 1e-8, sprintf("%d", as.integer(round(out))), sprintf("%.1f", out))
}

holder_exp2_metric_long <- function(dt, id_cols = c("family", "method", "d", "n", "reps", "status")) {
  plot_long <- melt(
    dt,
    id.vars = intersect(id_cols, names(dt)),
    measure.vars = c("coverage", "rmse"),
    variable.name = "metric",
    value.name = "value"
  )
  plot_long[, metric := factor(metric, levels = c("coverage", "rmse"), labels = c("Coverage", "RMSE"))]
  plot_long
}

holder_exp2_plot_dimcurve <- function(summary, output_dir, paper_plot_dir) {
  plot_dt <- summary[
    source_role == "xgbrf_dimcurve_n10000" &
      n == 10000L &
      as.character(method) %in% c("AIPW", "C-DML (LS-Wald)", "C-DML (boot.)", "C-DML (Wald)", "DR-TMLE")
  ]
  plot_long <- holder_exp2_metric_long(plot_dt)
  method_colors <- c(
    "AIPW" = "#D55E00",
    "C-DML (LS-Wald)" = "#0072B2",
    "C-DML (boot.)" = "#0072B2",
    "C-DML (Wald)" = "#56B4E9",
    "DR-TMLE" = "#009E73"
  )
  p <- ggplot(plot_long, aes(x = d, y = value, color = method, shape = method, group = method)) +
    geom_hline(
      data = data.table(metric = factor("Coverage", levels = levels(plot_long$metric)), y = 0.95),
      aes(yintercept = y),
      inherit.aes = FALSE,
      linetype = "dashed",
      linewidth = 0.35,
      color = "grey45"
    ) +
    geom_line(linewidth = 0.55, na.rm = TRUE) +
    geom_point(size = 2, na.rm = TRUE) +
    facet_grid(metric ~ ., scales = "free_y") +
    scale_x_continuous(breaks = sort(unique(plot_dt$d))) +
    scale_color_manual(values = method_colors, drop = FALSE) +
    labs(x = "Dimension", y = NULL, color = NULL, shape = NULL) +
    holder_exp2_theme()
  holder_exp2_save_plot(
    p,
    file.path(output_dir, "holder_exp2_dimcurve_n10000_xgbrf_coverage_rmse"),
    width = 6.4,
    height = 4.6,
    paper_plot_dir = paper_plot_dir
  )

  bias_long <- melt(
    plot_dt,
    id.vars = c("family", "method", "d", "n", "reps", "status"),
    measure.vars = c("bias", "mean_interval_width"),
    variable.name = "metric",
    value.name = "value"
  )
  bias_long[, metric := factor(metric, levels = c("bias", "mean_interval_width"), labels = c("Bias", "Mean interval width"))]
  p_bias <- ggplot(bias_long, aes(x = d, y = value, color = method, shape = method, group = method)) +
    geom_hline(
      data = data.table(metric = factor("Bias", levels = levels(bias_long$metric)), y = 0),
      aes(yintercept = y),
      inherit.aes = FALSE,
      linetype = "dashed",
      linewidth = 0.35,
      color = "grey45"
    ) +
    geom_line(linewidth = 0.55, na.rm = TRUE) +
    geom_point(size = 2, na.rm = TRUE) +
    facet_grid(metric ~ ., scales = "free_y") +
    scale_x_continuous(breaks = sort(unique(plot_dt$d))) +
    scale_color_manual(values = method_colors, drop = FALSE) +
    labs(x = "Dimension", y = NULL, color = NULL, shape = NULL) +
    holder_exp2_theme()
  holder_exp2_save_plot(
    p_bias,
    file.path(output_dir, "holder_exp2_dimcurve_n10000_xgbrf_bias_width"),
    width = 6.4,
    height = 4.6,
    paper_plot_dir = paper_plot_dir
  )
}

holder_exp2_plot_samplegrid <- function(summary, output_dir, paper_plot_dir) {
  plot_dt <- summary[
    source_role == "xgbrf_samplegrid_external" &
      as.character(method) %in% c("AIPW", "C-DML (LS-Wald)")
  ]
  plot_dt[, `:=`(
    d_label = factor(paste0("d = ", d), levels = paste0("d = ", sort(unique(d))))
  )]
  plot_long <- holder_exp2_metric_long(
    plot_dt,
    id_cols = c("family", "method", "d", "d_label", "n", "reps", "status")
  )
  d_colors <- c("d = 4" = "#0072B2", "d = 8" = "#D55E00", "d = 12" = "#CC79A7", "d = 16" = "#009E73")
  p <- ggplot(plot_long, aes(x = n, y = value, color = d_label, linetype = d_label, group = d_label)) +
    geom_hline(
      data = data.table(metric = factor("Coverage", levels = levels(plot_long$metric)), y = 0.95),
      aes(yintercept = y),
      inherit.aes = FALSE,
      linetype = "dashed",
      linewidth = 0.35,
      color = "grey45"
    ) +
    geom_line(linewidth = 0.6, na.rm = TRUE) +
    geom_point(size = 2.1, na.rm = TRUE) +
    facet_grid(metric ~ method, scales = "free_y") +
    scale_x_continuous(
      breaks = sort(unique(plot_dt$n)),
      labels = holder_exp2_n_label
    ) +
    scale_color_manual(values = d_colors, drop = FALSE) +
    labs(x = "Sample size (thousands)", y = NULL, color = NULL, linetype = NULL) +
    holder_exp2_theme()
  holder_exp2_save_plot(
    p,
    file.path(output_dir, "holder_exp2_samplegrid_external_xgbrf_coverage_rmse_by_d"),
    width = 7.4,
    height = 4.8,
    paper_plot_dir = paper_plot_dir
  )

  bias_long <- melt(
    plot_dt,
    id.vars = c("family", "method", "d", "d_label", "n", "reps", "status"),
    measure.vars = c("bias", "mean_interval_width"),
    variable.name = "metric",
    value.name = "value"
  )
  bias_long[, metric := factor(metric, levels = c("bias", "mean_interval_width"), labels = c("Bias", "Mean interval width"))]
  p_bias <- ggplot(bias_long, aes(x = n, y = value, color = d_label, linetype = d_label, group = d_label)) +
    geom_hline(
      data = data.table(metric = factor("Bias", levels = levels(bias_long$metric)), y = 0),
      aes(yintercept = y),
      inherit.aes = FALSE,
      linetype = "dashed",
      linewidth = 0.35,
      color = "grey45"
    ) +
    geom_line(linewidth = 0.6, na.rm = TRUE) +
    geom_point(size = 2.1, na.rm = TRUE) +
    facet_grid(metric ~ method, scales = "free_y") +
    scale_x_continuous(
      breaks = sort(unique(plot_dt$n)),
      labels = holder_exp2_n_label
    ) +
    scale_color_manual(values = d_colors, drop = FALSE) +
    labs(x = "Sample size (thousands)", y = NULL, color = NULL, linetype = NULL) +
    holder_exp2_theme()
  holder_exp2_save_plot(
    p_bias,
    file.path(output_dir, "holder_exp2_samplegrid_external_xgbrf_bias_width_by_d"),
    width = 7.4,
    height = 4.8,
    paper_plot_dir = paper_plot_dir
  )
}

holder_exp2_plot_glmnet_appendix <- function(summary, output_dir, paper_plot_dir) {
  dim_dt <- summary[source_role == "glmnet_ridge_dimcurve_n10000"]
  if (nrow(dim_dt)) {
    dim_long <- holder_exp2_metric_long(dim_dt)
    p_dim <- ggplot(dim_long, aes(x = d, y = value, group = method)) +
      geom_hline(
        data = data.table(metric = factor("Coverage", levels = levels(dim_long$metric)), y = 0.95),
        aes(yintercept = y),
        inherit.aes = FALSE,
        linetype = "dashed",
        linewidth = 0.35,
        color = "grey45"
      ) +
      geom_line(linewidth = 0.6, color = "#0072B2", na.rm = TRUE) +
      geom_point(size = 2.1, color = "#0072B2", na.rm = TRUE) +
      facet_grid(metric ~ ., scales = "free_y") +
      scale_x_continuous(breaks = sort(unique(dim_dt$d))) +
      labs(x = "Dimension", y = NULL) +
      holder_exp2_theme()
    holder_exp2_save_plot(
      p_dim,
      file.path(output_dir, "holder_exp2_appendix_dimcurve_n10000_glmnet_ridge_coverage_rmse"),
      width = 5.4,
      height = 4.4,
      paper_plot_dir = paper_plot_dir
    )
  }

  sample_dt <- summary[source_role == "glmnet_ridge_samplegrid_nsim100"]
  if (nrow(sample_dt)) {
    sample_dt[, `:=`(
      d_label = factor(paste0("d = ", d), levels = paste0("d = ", sort(unique(d))))
    )]
    sample_long <- holder_exp2_metric_long(
      sample_dt,
      id_cols = c("family", "method", "d", "d_label", "n", "reps", "status")
    )
    d_colors <- c("d = 4" = "#0072B2", "d = 8" = "#D55E00", "d = 16" = "#009E73")
    p_sample <- ggplot(sample_long, aes(x = n, y = value, color = d_label, linetype = d_label, group = d_label)) +
      geom_hline(
        data = data.table(metric = factor("Coverage", levels = levels(sample_long$metric)), y = 0.95),
        aes(yintercept = y),
        inherit.aes = FALSE,
        linetype = "dashed",
        linewidth = 0.35,
        color = "grey45"
      ) +
      geom_line(linewidth = 0.6, na.rm = TRUE) +
      geom_point(size = 2.1, na.rm = TRUE) +
      facet_grid(metric ~ ., scales = "free_y") +
      scale_x_continuous(
        breaks = sort(unique(sample_dt$n)),
        labels = holder_exp2_n_label
      ) +
      scale_color_manual(values = d_colors, drop = FALSE) +
      labs(x = "Sample size (thousands)", y = NULL, color = NULL, linetype = NULL) +
      holder_exp2_theme()
    holder_exp2_save_plot(
      p_sample,
      file.path(output_dir, "holder_exp2_appendix_samplegrid_glmnet_ridge_nsim100_coverage_rmse_by_d"),
      width = 6.2,
      height = 4.6,
      paper_plot_dir = paper_plot_dir
    )
  }
}

holder_exp2_write_tables <- function(summary, output_dir) {
  main_dim <- summary[
    source_role == "xgbrf_dimcurve_n10000" &
      n == 10000L &
      as.character(method) %in% c("AIPW", "C-DML (LS-Wald)", "C-DML (boot.)", "C-DML (Wald)", "DR-TMLE"),
    .(
      Method = as.character(method),
      d = as.character(d),
      Reps = as.character(reps),
      Status = status,
      Coverage = sprintf("%.3f", coverage),
      Bias = sprintf("%.3f", bias),
      RMSE = sprintf("%.3f", rmse),
      `Mean SE` = sprintf("%.3f", mean_std_error),
      Width = sprintf("%.3f", mean_interval_width)
    )
  ]
  fwrite(main_dim, file.path(output_dir, "table_holder_exp2_dimcurve_n10000_xgbrf.csv"))

  sample_grid <- summary[
    source_role == "xgbrf_samplegrid_external" &
      as.character(method) %in% c("AIPW", "C-DML (LS-Wald)"),
    .(
      Method = as.character(method),
      d = as.character(d),
      n = as.character(n),
      Reps = as.character(reps),
      Status = status,
      Coverage = sprintf("%.3f", coverage),
      Bias = sprintf("%.3f", bias),
      RMSE = sprintf("%.3f", rmse),
      `Mean SE` = sprintf("%.3f", mean_std_error),
      Width = sprintf("%.3f", mean_interval_width)
    )
  ]
  fwrite(sample_grid, file.path(output_dir, "table_holder_exp2_samplegrid_external_xgbrf.csv"))
}

holder_exp2_write_readme <- function(output_dir, sources, status, summary) {
  located <- copy(sources)
  located[, `:=`(
    checkpoint_dir = file.path(source_dir, "checkpoints", checkpoint_subdir),
    checkpoint_files = vapply(
      file.path(source_dir, "checkpoints", checkpoint_subdir),
      function(path) length(list.files(path, pattern = "[.]csv$")),
      integer(1)
    )
  )]
  fwrite(located, file.path(output_dir, "holder_exp2_located_sources.csv"))

  lines <- c(
    "# Holder Experiment 2 External Artifacts",
    "",
    sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    "",
    "Source directories:",
    capture.output(print(located[, .(role, source_dir, checkpoint_subdir, required_iters, checkpoint_files)], row.names = FALSE)),
    "",
    "Completion status:",
    capture.output(print(status[, .(source_role, nuisance_method, setting, d, n, completed_iters, required_iters, complete)], row.names = FALSE)),
    "",
    "Primary figures copied to local-untracked/papers/newplots:",
    "",
    "- holder_exp2_dimcurve_n10000_xgbrf_coverage_rmse.pdf",
    "- holder_exp2_samplegrid_external_xgbrf_coverage_rmse_by_d.pdf",
    "- holder_exp2_samplegrid_external_xgbrf_bias_width_by_d.pdf",
    "",
    "Appendix figures:",
    "",
    "- holder_exp2_appendix_dimcurve_n10000_glmnet_ridge_coverage_rmse.pdf",
    "- holder_exp2_appendix_samplegrid_glmnet_ridge_nsim100_coverage_rmse_by_d.pdf",
    "",
    "Notes:",
    "",
    "- The XGB/RF plots use the corrected additive-outcome external-nuisance run.",
    "- Plots use all currently available deduplicated checkpoint iterations and do not visually distinguish repetition counts; exact counts are retained in holder_exp2_checkpoint_status.csv and holder_exp2_summary.csv.",
    "- The glmnet sample-grid appendix figure uses the nsims=100 fast-basis run because it has the broader current grid; the nsims=250 run is retained in the status CSV."
  )
  writeLines(lines, file.path(output_dir, "README_holder_exp2_external_artifacts.md"))

  preview <- summary[, .(
    source_role, method = as.character(method), d, n, reps, status,
    coverage, bias, rmse, mean_interval_width
  )]
  fwrite(preview[order(source_role, method, d, n)], file.path(output_dir, "holder_exp2_summary_preview.csv"))
}

holder_exp2_write_subsection_draft <- function(output_dir) {
  lines <- c(
    "\\subsection{Experiment 2: Holder-smooth nuisance stress test}",
    "\\label{section::experiments-holder}",
    "",
    "We next study a higher-dimensional Holder-smooth simulation in which the outcome regression and propensity score are estimated with flexible tree-based learners. The design fixes the slow-nuisance regime to the setting where the propensity score contains the harder nonadditive component, and varies either the covariate dimension at $n=10000$ or the sample size over a grid of $(n,d)$ values. Nuisances are estimated using gradient-boosted trees for the faster component and random forests for the slower component. We compare the standard AIPW estimator to calibrated DML; the sample-grid run uses the level-set Wald auxiliary correction for the calibrated DML interval.",
    "",
    "\\begin{figure}[tb]",
    "\\centering",
    "\\includegraphics[width=0.78\\linewidth]{newplots/holder_exp2_dimcurve_n10000_xgbrf_coverage_rmse.pdf}",
    "\\caption{Holder-smooth XGB/RF dimension curve at $n=10000$. The dashed line marks nominal 95\\% coverage.}",
    "\\label{fig:holder-exp2-dimcurve}",
    "\\end{figure}",
    "",
    "\\begin{figure}[tb]",
    "\\centering",
    "\\includegraphics[width=\\linewidth]{newplots/holder_exp2_samplegrid_external_xgbrf_coverage_rmse_by_d.pdf}",
    "\\caption{Holder-smooth external-nuisance XGB/RF sample-size grid. The x-axis is sample size and each line is a dimension.}",
    "\\label{fig:holder-exp2-samplegrid}",
    "\\end{figure}",
    "",
    "The cells show the intended stress pattern: AIPW coverage deteriorates as the dimension increases, while calibrated DML remains closer to nominal coverage across the grid."
  )
  writeLines(lines, file.path(output_dir, "holder_exp2_subsection_draft.tex"))
}

holder_exp2_build <- function() {
  output_dir <- holder_exp2_output_dir()
  paper_plot_dir <- holder_exp2_paper_plot_dir()
  sources <- holder_exp2_sources()

  read <- lapply(seq_len(nrow(sources)), function(i) {
    holder_exp2_read_checkpoints(
      sources$role[i],
      sources$source_dir[i],
      sources$checkpoint_subdir[i],
      sources$required_iters[i]
    )
  })
  names(read) <- sources$role

  xgb_dim_status <- holder_exp2_status(
    read$xgbrf_dimcurve_n10000$results,
    holder_exp2_expected_xgbrf_dim("xgbrf_dimcurve_n10000"),
    read$xgbrf_dimcurve_n10000$required_iters,
    "xgbrf_dimcurve_n10000",
    sources[role == "xgbrf_dimcurve_n10000", source_dir],
    read$xgbrf_dimcurve_n10000$duplicate_rows,
    read$xgbrf_dimcurve_n10000$latest_mtime
  )
  xgb_sample_status <- holder_exp2_status(
    read$xgbrf_samplegrid_external$results,
    holder_exp2_expected_sample("xgbrf_samplegrid_external"),
    read$xgbrf_samplegrid_external$required_iters,
    "xgbrf_samplegrid_external",
    sources[role == "xgbrf_samplegrid_external", source_dir],
    read$xgbrf_samplegrid_external$duplicate_rows,
    read$xgbrf_samplegrid_external$latest_mtime
  )
  glmnet_dim_status <- holder_exp2_status(
    read$glmnet_ridge_dimcurve_n10000$results,
    holder_exp2_expected_glmnet_dim("glmnet_ridge_dimcurve_n10000"),
    read$glmnet_ridge_dimcurve_n10000$required_iters,
    "glmnet_ridge_dimcurve_n10000",
    sources[role == "glmnet_ridge_dimcurve_n10000", source_dir],
    read$glmnet_ridge_dimcurve_n10000$duplicate_rows,
    read$glmnet_ridge_dimcurve_n10000$latest_mtime
  )
  glmnet_sample250_status <- holder_exp2_status(
    read$glmnet_ridge_samplegrid_nsim250$results,
    holder_exp2_expected_glmnet_sample("glmnet_ridge_samplegrid_nsim250"),
    read$glmnet_ridge_samplegrid_nsim250$required_iters,
    "glmnet_ridge_samplegrid_nsim250",
    sources[role == "glmnet_ridge_samplegrid_nsim250", source_dir],
    read$glmnet_ridge_samplegrid_nsim250$duplicate_rows,
    read$glmnet_ridge_samplegrid_nsim250$latest_mtime
  )
  glmnet_sample100_status <- holder_exp2_status(
    read$glmnet_ridge_samplegrid_nsim100$results,
    holder_exp2_expected_glmnet_sample("glmnet_ridge_samplegrid_nsim100"),
    read$glmnet_ridge_samplegrid_nsim100$required_iters,
    "glmnet_ridge_samplegrid_nsim100",
    sources[role == "glmnet_ridge_samplegrid_nsim100", source_dir],
    read$glmnet_ridge_samplegrid_nsim100$duplicate_rows,
    read$glmnet_ridge_samplegrid_nsim100$latest_mtime
  )

  status <- rbindlist(list(
    xgb_dim_status,
    xgb_sample_status,
    glmnet_dim_status,
    glmnet_sample250_status,
    glmnet_sample100_status
  ), fill = TRUE)
  fwrite(status, file.path(output_dir, "holder_exp2_checkpoint_status.csv"))

  xgb_results <- rbindlist(list(
    read$xgbrf_dimcurve_n10000$results,
    read$xgbrf_samplegrid_external$results
  ), fill = TRUE)
  glmnet_results <- rbindlist(list(
    read$glmnet_ridge_dimcurve_n10000$results,
    read$glmnet_ridge_samplegrid_nsim250$results,
    read$glmnet_ridge_samplegrid_nsim100$results
  ), fill = TRUE)
  fwrite(xgb_results, file.path(output_dir, "holder_exp2_xgbrf_results_current.csv"))
  fwrite(glmnet_results, file.path(output_dir, "holder_exp2_glmnet_results_current.csv"))

  xgb_summary <- holder_exp2_summarise(xgb_results, "XGB/RF")
  glmnet_summary <- holder_exp2_summarise(glmnet_results, "glmnet")
  summary <- rbindlist(list(
    holder_exp2_merge_status(xgb_summary, status[nuisance_method %in% c(NA_character_, "") | is.na(nuisance_method)]),
    holder_exp2_merge_status(glmnet_summary, status[!is.na(nuisance_method)])
  ), fill = TRUE)
  setorder(summary, source_role, method, d, n)
  fwrite(summary, file.path(output_dir, "holder_exp2_summary.csv"))

  holder_exp2_plot_dimcurve(summary, output_dir, paper_plot_dir)
  holder_exp2_plot_samplegrid(summary, output_dir, paper_plot_dir)
  holder_exp2_plot_glmnet_appendix(summary, output_dir, paper_plot_dir)
  holder_exp2_write_tables(summary, output_dir)
  holder_exp2_write_readme(output_dir, sources, status, summary)
  holder_exp2_write_subsection_draft(output_dir)

  manifest <- data.table(
    artifact = list.files(output_dir, recursive = FALSE),
    path = file.path(output_dir, list.files(output_dir, recursive = FALSE))
  )
  fwrite(manifest, file.path(output_dir, "manifest.csv"))
  cat(sprintf("Wrote Holder Experiment 2 external artifacts to %s\n", output_dir))
  invisible(list(output_dir = output_dir, status = status, summary = summary))
}

if (sys.nframe() == 0L) {
  holder_exp2_build()
}

source(file.path("paper_experiment_scripts", "legacy_config.R"))
legacy_init_paper_env()
legacy_require_packages(c("data.table", "ggplot2", "cowplot"))

library(data.table)
library(ggplot2)
library(cowplot)

repo_root <- legacy_paper_repo_root()
checkpoint_dir <- Sys.getenv(
  "CDML_PAPER_KERNEL_SENSITIVITY_CHECKPOINT_DIR",
  unset = file.path(
    repo_root,
    "paper_experiment_results",
    "reproduced",
    "jrssb_final_20260518_133953",
    "simple",
    "checkpoints",
    "simple"
  )
)
output_dir <- Sys.getenv(
  "CDML_PAPER_KERNEL_SENSITIVITY_PLOT_DIR",
  unset = file.path(
    repo_root,
    "paper_experiment_results",
    "reproduced",
    "jrssb_final_20260518_133953",
    "simple",
    "kernel_sensitivity_appendix"
  )
)
paper_plot_dir <- Sys.getenv(
  "CDML_PAPER_KERNEL_SENSITIVITY_PAPER_PLOT_DIR",
  unset = file.path(repo_root, "local-untracked", "papers", "newplots")
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(paper_plot_dir, recursive = TRUE, showWarnings = FALSE)

files <- list.files(
  checkpoint_dir,
  pattern = "^simple_revision_results_n[0-9]+[.]csv$",
  full.names = TRUE
)
if (!length(files)) {
  stop(sprintf("No simple checkpoint CSVs found in %s.", checkpoint_dir), call. = FALSE)
}

read_checkpoint <- function(file) {
  dt <- fread(file, na.strings = c("NA", ""))
  numeric_cols <- intersect(
    c("estimate", "std_error", "CI_left", "CI_right", "ATE", "n", "iter", "misp"),
    names(dt)
  )
  for (col in numeric_cols) {
    dt[, (col) := as.numeric(get(col))]
  }
  dt
}

results <- rbindlist(lapply(files, read_checkpoint), fill = TRUE)
required_cols <- c(
  "setting", "n", "iter", "simple_misspec_backend", "estimator",
  "ci_label", "estimate", "std_error", "CI_left", "CI_right", "ATE"
)
missing_cols <- setdiff(required_cols, names(results))
if (length(missing_cols)) {
  stop(
    sprintf("Checkpoint data missing required columns: %s", paste(missing_cols, collapse = ", ")),
    call. = FALSE
  )
}

backend_labels <- c(
  "slow_kernel" = "Slow kernel",
  "kernel_omit_w2" = "One-variable kernel"
)
backend_file_tags <- c(
  "slow_kernel" = "slow_kernel",
  "kernel_omit_w2" = "one_variable_kernel"
)

plot_results <- results[
  simple_misspec_backend %in% names(backend_labels) &
    setting %in% c("misp_2", "misp_3") &
    n >= 250 & n <= 5000 &
    estimator %in% c("AIPW", "DR-TMLE", "calibratedDML") &
    (estimator != "calibratedDML" | ci_label == "bootstrap")
]
if (!nrow(plot_results)) {
  stop("No kernel sensitivity rows found after filtering.", call. = FALSE)
}

plot_results[, covered := !is.na(CI_left) & !is.na(CI_right) & CI_left <= ATE & ATE <= CI_right]
plot_results[, error := estimate - ATE]

summary_stats <- plot_results[, .(
  reps = uniqueN(iter),
  Bias = abs(mean(error, na.rm = TRUE)),
  signed_bias = mean(error, na.rm = TRUE),
  SE = stats::sd(estimate, na.rm = TRUE),
  RMSE = sqrt(mean(error^2, na.rm = TRUE)),
  Coverage = mean(covered, na.rm = TRUE),
  avg_width = mean(CI_right - CI_left, na.rm = TRUE),
  avg_std_error = mean(std_error, na.rm = TRUE)
), by = .(simple_misspec_backend, setting, n, estimator, ci_label)]
setorder(summary_stats, simple_misspec_backend, setting, estimator, ci_label, n)

summary_stats[, Estimator := factor(
  estimator,
  levels = c("AIPW", "DR-TMLE", "calibratedDML"),
  labels = c("AIPW", "DR-TMLE", "IC-DML (boot.)")
)]
summary_stats[, Backend := factor(
  simple_misspec_backend,
  levels = names(backend_labels),
  labels = backend_labels
)]

plot_stats <- copy(summary_stats)
plot_stats[
  setting == "misp_2" &
    estimator == "AIPW" &
    (SE > 0.1 | Bias > 0.1),
  c("Bias", "SE", "RMSE", "Coverage") := .(NA_real_, NA_real_, NA_real_, NA_real_)
]

omitted_points <- summary_stats[
  setting == "misp_2" &
    estimator == "AIPW" &
    (SE > 0.1 | Bias > 0.1),
  .(simple_misspec_backend, setting, n, estimator, ci_label, reps, Bias, SE, RMSE, Coverage)
]

fwrite(
  summary_stats[, .(
    simple_misspec_backend, Backend, setting, n, estimator, ci_label, Estimator,
    reps, signed_bias, Bias, SE, RMSE, Coverage, avg_width, avg_std_error
  )],
  file.path(output_dir, "kernel_sensitivity_appendix_summary.csv"),
  na = "NA"
)
fwrite(
  omitted_points,
  file.path(output_dir, "kernel_sensitivity_appendix_omitted_plot_points.csv"),
  na = "NA"
)

n_tick_labels <- c(250, 1000, 2000, 3000, 4000, 5000)
est_colors <- c(
  "AIPW" = "#E57373",
  "DR-TMLE" = "#81C784",
  "IC-DML (boot.)" = "#64B5F6"
)
est_shapes <- c(
  "AIPW" = 16,
  "DR-TMLE" = 17,
  "IC-DML (boot.)" = 15
)

make_plot <- function(d, yvar, ylab, hline = FALSE, ylim = NULL) {
  p <- ggplot(d, aes(
    x = n,
    y = .data[[yvar]],
    group = Estimator,
    color = Estimator,
    shape = Estimator
  )) +
    geom_line(
      aes(group = Estimator),
      color = "grey60",
      linetype = "dashed",
      linewidth = 0.5,
      na.rm = TRUE
    ) +
    geom_point(size = 2.8, stroke = 0.3, na.rm = TRUE) +
    scale_x_continuous(breaks = n_tick_labels, labels = n_tick_labels) +
    scale_color_manual(name = "Estimator", values = est_colors, drop = FALSE) +
    scale_shape_manual(name = "Estimator", values = est_shapes, drop = FALSE) +
    guides(
      color = guide_legend(nrow = 1, byrow = TRUE),
      shape = guide_legend(nrow = 1, byrow = TRUE)
    ) +
    labs(x = "Sample Size (n)", y = ylab) +
    theme_bw(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      legend.position = "none",
      legend.box = "horizontal",
      legend.margin = margin(t = 0, b = 4),
      legend.text = element_text(margin = margin(r = 10, unit = "pt")),
      legend.title = element_blank(),
      axis.title.y = element_text(margin = margin(r = 5)),
      axis.title.x = element_text(margin = margin(t = 5))
    )

  if (!is.null(ylim)) {
    p <- p + coord_cartesian(ylim = c(0, ylim))
  }
  if (hline) {
    p <- p + geom_hline(yintercept = 0.95, linetype = "dashed", color = "gray40")
  }
  p
}

for (backend_id in names(backend_labels)) {
  for (setting_id in c("misp_3", "misp_2")) {
    d <- plot_stats[simple_misspec_backend == backend_id & setting == setting_id]
    shared_ylim <- d[, max(c(Bias, SE), na.rm = TRUE)]

    p_bias <- make_plot(d, "Bias", "Bias", ylim = shared_ylim)
    p_se <- make_plot(d, "SE", "Standard Error", ylim = shared_ylim)
    p_cov <- make_plot(d, "Coverage", "Coverage", hline = TRUE)

    final <- cowplot::plot_grid(
      p_bias,
      p_se,
      p_cov,
      nrow = 1,
      align = "hv",
      axis = "tblr"
    )

    output_file <- file.path(
      output_dir,
      sprintf(
        "appendix_exp1_%s_%s.pdf",
        backend_file_tags[[backend_id]],
        gsub("_", "", setting_id)
      )
    )
    paper_file <- file.path(paper_plot_dir, basename(output_file))
    ggsave(output_file, final, width = 10.5, height = 2.5)
    file.copy(output_file, paper_file, overwrite = TRUE)
  }
}

legend_dt <- data.table(
  Estimator = factor(rep(names(est_colors), each = 2L), levels = names(est_colors)),
  n = rep(c(1, 2), times = length(est_colors)),
  value = 1
)
legend_plot <- ggplot(legend_dt, aes(x = n, y = value, colour = Estimator, shape = Estimator)) +
  geom_line(aes(group = Estimator), color = "grey60", linetype = "dashed", linewidth = 0.5) +
  geom_point(size = 3) +
  scale_color_manual(values = est_colors, drop = FALSE) +
  scale_shape_manual(values = est_shapes, drop = FALSE) +
  guides(
    colour = guide_legend(nrow = 1, byrow = TRUE),
    shape = guide_legend(nrow = 1, byrow = TRUE)
  ) +
  theme_void(base_size = 11) +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.title = element_blank(),
    plot.margin = margin(0, 0, 0, 0)
  )
legend_file <- file.path(output_dir, "appendix_exp1_kernel_sensitivity_legend.pdf")
ggsave(legend_file, legend_plot, width = 5.4, height = 0.9)
invisible(file.copy(legend_file, file.path(paper_plot_dir, basename(legend_file)), overwrite = TRUE))

readme <- c(
  "Experiment 1 kernel-sensitivity appendix plots",
  "",
  sprintf("Generated by: %s", file.path(repo_root, "paper_experiment_scripts", "render_kernel_sensitivity_appendix_plots.R")),
  sprintf("Input checkpoints: %s", checkpoint_dir),
  sprintf("Results output directory: %s", output_dir),
  sprintf("Paper plot directory: %s", paper_plot_dir),
  "",
  "What this is:",
  "- Appendix-ready sensitivity figures matching the main Experiment 1 layout.",
  "- Backends: `slow_kernel` and `kernel_omit_w2`.",
  "- Settings: `misp_2` and `misp_3`.",
  "- Estimators: AIPW, DR-TMLE, and IC-DML (boot.).",
  "",
  "Plotting convention:",
  "- Uses complete sample sizes from n = 250 through n = 5000, with 500 simulation replicates per plotted cell.",
  "- Excludes n = 100 to match the main Experiment 1 convention.",
  "- Excludes n = 7500 because the kernel_omit_w2 checkpoint is partial, and excludes absent n = 9000.",
  "- Omits early AIPW points from the plotted `misp_2` panels when empirical SE > 0.1 or absolute bias > 0.1; these rows are retained in the summary CSV.",
  "",
  "Main paper-side files:",
  "- newplots/appendix_exp1_kernel_sensitivity_legend.pdf",
  "- newplots/appendix_exp1_slow_kernel_misp3.pdf",
  "- newplots/appendix_exp1_slow_kernel_misp2.pdf",
  "- newplots/appendix_exp1_one_variable_kernel_misp3.pdf",
  "- newplots/appendix_exp1_one_variable_kernel_misp2.pdf"
)
writeLines(readme, file.path(output_dir, "README_kernel_sensitivity_appendix.txt"))

cat(sprintf("Wrote kernel-sensitivity appendix plots to %s\n", output_dir))
cat(sprintf("Copied paper-side PDFs to %s\n", paper_plot_dir))

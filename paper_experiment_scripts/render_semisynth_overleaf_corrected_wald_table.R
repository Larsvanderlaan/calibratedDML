source(file.path("paper_experiment_scripts", "render_old_semisynth_refreshed_table.R"))

legacy_require_packages("data.table")
library(data.table)

semisynth_overleaf_output_dir <- function() {
  file.path(
    legacy_paper_repo_root(),
    "paper_experiment_results",
    "reproduced",
    "coverage_wald_reruns_20260702",
    "overleaf_semisynth_corrected_wald_table"
  )
}

semisynth_overleaf_source_plan <- function() {
  root <- file.path(legacy_paper_repo_root(), "paper_experiment_results", "reproduced")
  current_acic2017 <- file.path(
    root,
    "realdata_acic2017_april10fold_depth36_rounds50_100_200_sieve_20260701_201014"
  )
  april <- file.path(root, "revision_final_250_20260422_023227")
  current_ihdp <- file.path(root, "realdata_ihdp_table_wald_sieve_20260702")
  current_cps <- file.path(root, "realdata_lalonde_cps_table_wald_sieve_20260702")
  current_psid <- file.path(root, "realdata_lalonde_psid_table_wald_sieve_20260702")
  current_twins <- file.path(root, "realdata_twins_table_wald_sieve_20260702")

  data.table(
    dataset = c(
      "acic2017_18",
      "acic2017_20",
      "acic2017_22",
      "acic2017_24",
      "acic2018_10000",
      "ihdp",
      "lalonde_cps",
      "lalonde_psid",
      "twins"
    ),
    base_root = c(
      rep(current_acic2017, 4L),
      april,
      current_ihdp,
      current_cps,
      current_psid,
      current_twins
    ),
    corrected_root = c(
      rep(current_acic2017, 4L),
      NA_character_,
      current_ihdp,
      current_cps,
      current_psid,
      current_twins
    ),
    source_note = c(
      rep("current ACIC-2017 250-rep rerun", 4L),
      "April fallback; current corrected ACIC-2018 rerun pending",
      "current complete rerun",
      "current complete rerun",
      "current complete rerun",
      "current complete rerun"
    )
  )
}

semisynth_overleaf_method_specs <- function() {
  data.table(
    estimator = c("AIPW", "calibratedDML", "calibratedDML"),
    ci_label = c("wald", "bootstrap", "sieve_corrected_if_wald"),
    method_key = c("AIPW_wald", "CDML_bootstrap", "CDML_corrected_wald"),
    method_display = c("AIPW Wald", "C-DML Bootstrap", "C-DML Corrected Wald")
  )
}

semisynth_overleaf_read_dataset <- function(dataset, root) {
  if (is.na(root) || !nzchar(root)) {
    return(NULL)
  }
  file_path <- file.path(root, sprintf("realdata_revision_results_%s.csv", dataset))
  if (!file.exists(file_path)) {
    return(NULL)
  }
  out <- fread(file_path)
  out[, source_file := normalizePath(file_path, mustWork = FALSE)]
  out
}

semisynth_overleaf_summarize <- function(dataset, root, method_keys) {
  raw <- semisynth_overleaf_read_dataset(dataset, root)
  if (is.null(raw)) {
    return(NULL)
  }
  methods <- semisynth_overleaf_method_specs()[method_key %in% method_keys]
  rows <- raw[
    study == "real_data_revision" &
      setting == "benchmark" &
      as.integer(misp) == 1L &
      nuisance_method == "legacy_sl3_fixed" &
      fold_mode == "legacy_auto_folds"
  ]
  rows <- merge(rows, methods, by = c("estimator", "ci_label"))
  if (!nrow(rows)) {
    return(NULL)
  }

  duplicate_keys <- rows[, .N, by = .(dataset, iter, nuisance_method, misp, estimator, ci_label)][N > 1L]
  if (nrow(duplicate_keys)) {
    stop(sprintf("Duplicate rows found for %s in %s.", dataset, root), call. = FALSE)
  }

  rows[, `:=`(
    covered = CI_left <= ATE & ATE <= CI_right,
    interval_width = CI_right - CI_left,
    error = estimate - ATE
  )]

  rows[, .(
    reps = uniqueN(iter),
    scaled_bias = abs(mean(error, na.rm = TRUE)) / mean(abs(ATE), na.rm = TRUE),
    scaled_rmse = sqrt(mean(error^2, na.rm = TRUE)) / mean(abs(ATE), na.rm = TRUE),
    coverage = mean(covered, na.rm = TRUE),
    mean_interval_width = mean(interval_width, na.rm = TRUE),
    source_file = paste(sort(unique(source_file)), collapse = ";")
  ), by = .(dataset, method_key, method_display)]
}

semisynth_overleaf_format_number <- function(x) {
  trimws(old_semisynth_format_number(x))
}

semisynth_overleaf_format_cov <- function(x) {
  old_semisynth_format_coverage(x)
}

semisynth_overleaf_bold_min <- function(x, y) {
  vals <- c(x, y)
  if (anyNA(vals)) {
    return(semisynth_overleaf_format_number(vals))
  }
  formatted <- semisynth_overleaf_format_number(vals)
  winners <- abs(vals - min(vals)) < 1e-12
  formatted[winners] <- sprintf("\\textbf{%s}", formatted[winners])
  formatted
}

semisynth_overleaf_bold_max_cov <- function(aipw, boot, corrected) {
  vals <- c(aipw, boot, corrected)
  formatted <- old_semisynth_format_coverage(vals)
  if (all(is.na(vals))) {
    return(formatted)
  }
  winners <- !is.na(vals) & abs(vals - max(vals, na.rm = TRUE)) < 1e-12
  formatted[winners] <- sprintf("\\textbf{%s}", formatted[winners])
  formatted
}

semisynth_overleaf_write_tex <- function(table, file) {
  rows <- vapply(seq_len(nrow(table)), function(row_idx) {
    row <- table[row_idx]
    cov_vals <- semisynth_overleaf_bold_max_cov(
      row$coverage_AIPW_wald,
      row$coverage_CDML_bootstrap,
      row$coverage_CDML_corrected_wald
    )
    rmse_vals <- semisynth_overleaf_bold_min(row$rmse_AIPW, row$rmse_CDML)
    bias_vals <- semisynth_overleaf_bold_min(row$bias_AIPW, row$bias_CDML)
    sprintf(
      "             %s & %s & %s & %s & %s & %s & %s & %s \\\\",
      old_semisynth_latex_escape(row$display),
      cov_vals[[1L]],
      cov_vals[[2L]],
      cov_vals[[3L]],
      rmse_vals[[1L]],
      rmse_vals[[2L]],
      bias_vals[[1L]],
      bias_vals[[2L]]
    )
  }, character(1L))

  lines <- c(
    "\\begin{table}[tb]",
    "    \\centering",
    "    \\caption{Evaluation of Absolute Bias, Root Mean Square Error (RMSE), and 95\\% Confidence Interval coverage on semi-synthetic benchmark datasets. Both outcome regression and propensity scores are estimated using an ensemble of gradient-boosted trees. For comparability across rows, the absolute bias and RMSE are scaled by a constant equal to the average effect size across dataset realizations for that benchmark. For C-DML, coverage is reported for the bootstrap interval and, when available, the corrected Wald interval. Blank corrected-Wald entries indicate that the full corrected-Wald rerun was not yet complete. For clarity, we report only sample size 10000 results for ACIC-2018 and the settings with strong confounding for ACIC-2017. The full results can be found in Appendix \\ref{appendix::simRealFull}.}\\label{fig::exp2}",
    "    \\begin{subtable}{\\textwidth}",
    "        \\centering",
    "        \\scriptsize",
    "        \\begin{tabular}{|c||P{1.2cm}|P{1.2cm}|P{1.35cm}||P{1.2cm}|P{1.2cm}||P{1.2cm}|P{1.2cm}|}",
    "            \\hline",
    "            \\multirow{2}{*}{\\textbf{Dataset}} &",
    "              \\multicolumn{3}{c||}{\\textbf{Coverage}} &",
    "              \\multicolumn{2}{c||}{\\textbf{RMSE}} &",
    "              \\multicolumn{2}{c|}{\\textbf{Absolute Bias}} \\\\",
    "            \\cline{2-8}",
    "            & AIPW & C-DML Boot. & C-DML Corr. Wald & AIPW & C-DML & AIPW & C-DML \\\\",
    "            \\hline\\hline",
    paste(c(rbind(rows, rep("            \\hline", length(rows)))), collapse = "\n"),
    "        \\end{tabular}",
    "    \\end{subtable}",
    "\\end{table}"
  )
  writeLines(lines, file)
}

render_semisynth_overleaf_corrected_wald_table <- function(
  output_dir = semisynth_overleaf_output_dir()
) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  plan <- semisynth_overleaf_source_plan()
  specs <- old_semisynth_dataset_specs()[dataset %in% plan$dataset]
  setorder(specs, order)

  summaries <- rbindlist(lapply(seq_len(nrow(plan)), function(i) {
    dataset_name <- plan$dataset[[i]]
    base <- semisynth_overleaf_summarize(
      dataset_name,
      plan$base_root[[i]],
      c("AIPW_wald", "CDML_bootstrap")
    )
    corrected <- semisynth_overleaf_summarize(
      dataset_name,
      plan$corrected_root[[i]],
      "CDML_corrected_wald"
    )
    rbind(base, corrected, fill = TRUE)
  }), fill = TRUE)

  wide <- dcast(
    summaries,
    dataset ~ method_key,
    value.var = c("reps", "scaled_bias", "scaled_rmse", "coverage", "mean_interval_width", "source_file")
  )

  table <- merge(specs, wide, by = "dataset", all.x = TRUE)
  table <- merge(table, plan, by = "dataset", all.x = TRUE)
  setorder(table, order)
  table <- table[, .(
    dataset,
    display,
    expected_reps,
    reps_AIPW_wald = reps_AIPW_wald,
    reps_CDML_bootstrap = reps_CDML_bootstrap,
    reps_CDML_corrected_wald = reps_CDML_corrected_wald,
    coverage_AIPW_wald = coverage_AIPW_wald,
    coverage_CDML_bootstrap = coverage_CDML_bootstrap,
    coverage_CDML_corrected_wald = coverage_CDML_corrected_wald,
    rmse_AIPW = scaled_rmse_AIPW_wald,
    rmse_CDML = scaled_rmse_CDML_bootstrap,
    bias_AIPW = scaled_bias_AIPW_wald,
    bias_CDML = scaled_bias_CDML_bootstrap,
    width_AIPW_wald = mean_interval_width_AIPW_wald,
    width_CDML_bootstrap = mean_interval_width_CDML_bootstrap,
    width_CDML_corrected_wald = mean_interval_width_CDML_corrected_wald,
    base_root,
    corrected_root,
    base_source_file = source_file_AIPW_wald,
    corrected_source_file = source_file_CDML_corrected_wald,
    source_note
  )]

  fwrite(table, file.path(output_dir, "semisynth_overleaf_corrected_wald_table.csv"))
  semisynth_overleaf_write_tex(
    table,
    file.path(output_dir, "semisynth_overleaf_corrected_wald_table.tex")
  )
  fwrite(
    table[, .(
      dataset,
      display,
      expected_reps,
      reps_AIPW_wald,
      reps_CDML_bootstrap,
      reps_CDML_corrected_wald,
      base_root,
      corrected_root,
      base_source_file,
      corrected_source_file,
      source_note
    )],
    file.path(output_dir, "semisynth_overleaf_corrected_wald_audit.csv")
  )

  message(sprintf("Wrote Overleaf semi-synthetic corrected-Wald table to %s", normalizePath(output_dir, mustWork = FALSE)))
  invisible(table)
}

if (sys.nframe() == 0L) {
  render_semisynth_overleaf_corrected_wald_table()
}

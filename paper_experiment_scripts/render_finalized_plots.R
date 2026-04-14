args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub(file_arg, "", args[grep(file_arg, args)])
if (!length(script_path)) {
  stop("render_finalized_plots.R must be run with Rscript.", call. = FALSE)
}
source(file.path(dirname(normalizePath(script_path[[1]], mustWork = TRUE)), "legacy_config.R"))
legacy_init_paper_env()
legacy_require_packages(c("rmarkdown", "knitr", "data.table", "ggplot2", "cowplot"))

rmarkdown::render(
  input = file.path(legacy_paper_scripts_dir(), "finalized_plots.Rmd"),
  output_dir = legacy_paper_results_dir(),
  envir = new.env(parent = globalenv())
)

legacy_paper_repo_root <- function() {
  env_root <- Sys.getenv("CALIBRATEDDML_REPO_ROOT", unset = "")
  candidates <- unique(Filter(nzchar, c(
    env_root,
    getwd(),
    normalizePath(file.path(getwd(), ".."), mustWork = FALSE),
    normalizePath(file.path(getwd(), "../.."), mustWork = FALSE)
  )))

  for (candidate in candidates) {
    if (file.exists(file.path(candidate, "paper_experiment_scripts")) &&
        file.exists(file.path(candidate, "paper_data"))) {
      return(normalizePath(candidate, mustWork = TRUE))
    }
  }

  stop(
    paste(
      "Could not locate the calibratedDML repo root.",
      "Set CALIBRATEDDML_REPO_ROOT before running the legacy paper scripts."
    ),
    call. = FALSE
  )
}

legacy_paper_scripts_dir <- function() {
  file.path(legacy_paper_repo_root(), "paper_experiment_scripts")
}

legacy_paper_data_dir <- function() {
  file.path(legacy_paper_repo_root(), "paper_data")
}

legacy_paper_results_dir <- function() {
  path <- Sys.getenv("CDML_PAPER_RESULTS_DIR", unset = "")
  if (!nzchar(path)) {
    path <- file.path(legacy_paper_repo_root(), "paper_experiment_results", "reproduced")
  }
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  normalizePath(path, mustWork = FALSE)
}

legacy_paper_library_dir <- function() {
  path <- Sys.getenv("CDML_PAPER_R_LIBS", unset = "")
  if (!nzchar(path)) {
    path <- file.path(legacy_paper_scripts_dir(), ".r_libs")
  }
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  normalizePath(path, mustWork = FALSE)
}

legacy_realcause_dir <- function() {
  path <- file.path(legacy_paper_data_dir(), "realcause")
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  path
}

legacy_allow_downloads <- function() {
  value <- tolower(Sys.getenv("CDML_PAPER_ALLOW_DOWNLOADS", unset = "true"))
  !value %in% c("0", "false", "no")
}

legacy_download_if_missing <- function(url, dest) {
  if (file.exists(dest)) {
    return(dest)
  }
  if (!legacy_allow_downloads()) {
    stop(
      sprintf(
        "Missing cached file %s and downloads are disabled. Set CDML_PAPER_ALLOW_DOWNLOADS=true or place the file there manually.",
        dest
      ),
      call. = FALSE
    )
  }
  dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
  utils::download.file(url, destfile = dest, mode = "wb", quiet = FALSE)
  dest
}

legacy_realcause_dataset_path <- function(data_name, iteration) {
  file.path(legacy_realcause_dir(), sprintf("%s_sample%s.csv", data_name, iteration))
}

legacy_realcause_dataset_file <- function(data_name, iteration) {
  dest <- legacy_realcause_dataset_path(data_name, iteration)
  url <- sprintf(
    "https://raw.githubusercontent.com/bradyneal/realcause/master/realcause_datasets/%s_sample%s.csv",
    data_name,
    iteration
  )
  legacy_download_if_missing(url, dest)
}

legacy_require_packages <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (!length(missing)) {
    return(invisible(TRUE))
  }

  stop(
    sprintf(
      paste(
        "Missing required R packages: %s.",
        "Install them with `Rscript paper_experiment_scripts/install_packages.R`",
        "or set CDML_PAPER_R_LIBS to a library that already contains them."
      ),
      paste(missing, collapse = ", ")
    ),
    call. = FALSE
  )
}

legacy_init_paper_env <- function(set_workdir = TRUE) {
  root <- legacy_paper_repo_root()
  .libPaths(c(legacy_paper_library_dir(), .libPaths()))
  if (set_workdir) {
    setwd(root)
  }
  dir.create(legacy_paper_results_dir(), recursive = TRUE, showWarnings = FALSE)
  invisible(root)
}

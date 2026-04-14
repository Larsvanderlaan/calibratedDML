args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub(file_arg, "", args[grep(file_arg, args)])
if (!length(script_path)) {
  stop("install_packages.R must be run with Rscript.", call. = FALSE)
}
source(file.path(dirname(normalizePath(script_path[[1]], mustWork = TRUE)), "legacy_config.R"))
legacy_init_paper_env()

lib_dir <- legacy_paper_library_dir()
repos <- "https://cloud.r-project.org"

configure_macos_build_env <- function() {
  if (!identical(Sys.info()[["sysname"]], "Darwin")) {
    return(invisible(NULL))
  }

  sdkroot <- Sys.getenv("SDKROOT", unset = "")
  if (!nzchar(sdkroot)) {
    sdk_candidates <- c(
      tryCatch(system("xcrun --show-sdk-path", intern = TRUE), error = function(e) character()),
      "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
    )
    sdkroot <- sdk_candidates[nzchar(sdk_candidates)][1]
  }
  if (nzchar(sdkroot)) {
    Sys.setenv(SDKROOT = sdkroot)
    cxx_headers <- file.path(sdkroot, "usr", "include", "c++", "v1")
    if (dir.exists(cxx_headers)) {
      Sys.setenv(
        CPLUS_INCLUDE_PATH = paste(
          unique(c(cxx_headers, strsplit(Sys.getenv("CPLUS_INCLUDE_PATH", unset = ""), ":", fixed = TRUE)[[1]])),
          collapse = ":"
        )
      )
    }
  }

  gcc_library_roots <- c(
    tryCatch(dirname(system("gfortran -print-file-name=libemutls_w.a", intern = TRUE)), error = function(e) character()),
    tryCatch(dirname(system("gfortran -print-file-name=libgfortran.dylib", intern = TRUE)), error = function(e) character()),
    "/opt/homebrew/opt/gcc/lib/gcc/current"
  )
  gcc_library_roots <- gcc_library_roots[dir.exists(gcc_library_roots)]
  if (length(gcc_library_roots)) {
    Sys.setenv(
      LIBRARY_PATH = paste(
        unique(c(gcc_library_roots, strsplit(Sys.getenv("LIBRARY_PATH", unset = ""), ":", fixed = TRUE)[[1]])),
        collapse = ":"
      )
    )
  }
}

configure_macos_build_env()

for (lock_dir in Sys.glob(file.path(lib_dir, "00LOCK*"))) {
  unlink(lock_dir, recursive = TRUE, force = TRUE)
}

old_options <- options(pkgType = "source")
on.exit(options(old_options), add = TRUE)

install_cran_package <- function(pkg) {
  install_attempts <- c(if (identical(Sys.info()[["sysname"]], "Darwin")) "binary", "source")
  last_error <- NULL
  for (install_type in unique(install_attempts)) {
    result <- try(
      install.packages(pkg, lib = lib_dir, repos = repos, type = install_type),
      silent = TRUE
    )
    if (!inherits(result, "try-error")) {
      return(invisible(TRUE))
    }
    last_error <- result
  }
  stop(last_error)
}

ensure_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install_cran_package(pkg)
  }
}

ensure_package("remotes")

package_version_at_least <- function(pkg, min_version = NULL) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    return(FALSE)
  }
  if (is.null(min_version)) {
    return(TRUE)
  }
  utils::packageVersion(pkg) >= min_version
}

ensure_cran_package_version <- function(pkg, min_version = NULL) {
  if (!package_version_at_least(pkg, min_version)) {
    install_cran_package(pkg)
  }
}

ensure_exact_package_version <- function(pkg, version, install_fn) {
  if (!requireNamespace(pkg, quietly = TRUE) || utils::packageVersion(pkg) != version) {
    install_fn()
  }
}

cran_packages <- c(
  "data.table",
  "SuperLearner",
  "uuid",
  "BBmisc",
  "delayed",
  "imputeMissings",
  "caret",
  "FKSUM",
  "glmnet",
  "mgcv",
  "ranger",
  "earth",
  "hal9001",
  "future",
  "doFuture",
  "ggplot2",
  "cowplot",
  "rmarkdown",
  "knitr"
)

for (pkg in cran_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install_cran_package(pkg)
  }
}

ensure_exact_package_version("xgboost", package_version("1.7.11.1"), function() {
  remotes::install_version(
    package = "xgboost",
    version = "1.7.11.1",
    lib = lib_dir,
    repos = repos,
    upgrade = "never",
    type = "source"
  )
})

ensure_cran_package_version("hal9001", "0.4.6")

github_packages <- list(
  list(repo = "tlverse/origami", ref = NULL, package_name = "origami", dependencies = TRUE),
  list(repo = "nhejazi/haldensify", ref = NULL, package_name = "haldensify", dependencies = TRUE),
  list(repo = "tlverse/sl3", ref = "develVersionChangeLars", package_name = "sl3", dependencies = FALSE),
  list(repo = "benkeser/drtmle", ref = NULL, package_name = "drtmle", dependencies = TRUE),
  list(repo = "vdorie/aciccomp", ref = NULL, subdir = "2017", package_name = "aciccomp2017", dependencies = FALSE)
)

for (spec in github_packages) {
  package_name <- spec$package_name
  min_version <- if (!is.null(spec$min_version)) spec$min_version else NULL
  if (!package_version_at_least(package_name, min_version)) {
    install_error <- try(
      remotes::install_github(
        repo = spec$repo,
        ref = spec$ref,
        subdir = if (!is.null(spec$subdir)) spec$subdir else NULL,
        lib = lib_dir,
        upgrade = "never",
        dependencies = spec$dependencies
      ),
      silent = TRUE
    )
    if (inherits(install_error, "try-error")) {
      if (isTRUE(spec$optional)) {
        warning(
          sprintf(
            "Optional package '%s' could not be installed from %s%s%s",
            package_name,
            spec$repo,
            if (!is.null(spec$subdir)) paste0("/", spec$subdir) else "",
            if (!is.null(spec$ref)) paste0("@", spec$ref) else ""
          ),
          call. = FALSE
        )
      } else {
        stop(install_error)
      }
    }
  }
}

message("Legacy paper simulation library ready at: ", lib_dir)

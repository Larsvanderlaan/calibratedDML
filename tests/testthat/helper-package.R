if (!("package:calibratedDML" %in% search())) {
  attached <- tryCatch({
    library(calibratedDML, character.only = TRUE)
    TRUE
  }, error = function(...) FALSE)

  if (!attached && requireNamespace("pkgload", quietly = TRUE)) {
    pkgload::load_all(".", export_all = TRUE, helpers = FALSE, quiet = TRUE)
  }
}

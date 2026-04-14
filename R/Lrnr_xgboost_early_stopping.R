#' Optional `sl3` XGBoost Learner Placeholder
#'
#' This helper is retained for backwards compatibility. The new package no
#' longer depends on `xgboost` or `sl3` for the default workflow.
#'
#' @param ... Passed through to the optional implementation.
#'
#' @return An `sl3` learner when the optional dependencies are installed.
#' @export
Lrnr_xgboost_early_stopping <- function(...) {
  stop(
    paste(
      "`Lrnr_xgboost_early_stopping()` is not part of the default",
      "calibratedDML workflow anymore.",
      "Use direct nuisance matrices, the built-in model interface,",
      "or supply your own `sl3` learner objects."
    ),
    call. = FALSE
  )
}

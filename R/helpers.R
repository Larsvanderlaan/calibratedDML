#' Default `sl3` Learners
#'
#' Returns a conservative set of `sl3` learners for users who opt into the
#' advanced `sl3` backend.
#'
#' @return A list of `sl3` learner objects.
#' @export
default_learners <- function() {
  if (!requireNamespace("sl3", quietly = TRUE)) {
    stop("`default_learners()` requires the optional `sl3` package.", call. = FALSE)
  }

  list(
    sl3::Lrnr_mean$new(),
    sl3::Lrnr_glm_fast$new()
  )
}

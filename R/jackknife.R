# V-fold jackknife, Algorithm 1 of Li, Ertefaie, and van der Laan (2026),
# arXiv:2607.22493. The refit function keeps initial nuisances fixed.
vfold_jackknife <- function(full_estimate, refit, n, folds = 10, seed = NULL) {
  if (length(folds) != 1L || !is.finite(folds) || folds != as.integer(folds) || folds < 2L) {
    stop("`jackknife_folds` must be an integer of at least 2.", call. = FALSE)
  }
  if (n < 2L) {
    stop("Jackknife inference requires at least two observations.", call. = FALSE)
  }
  v <- min(as.integer(folds), n)
  # Use deletion groups independently of the nuisance cross-fitting folds.
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (had_seed) assign(".Random.seed", old_seed, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  })
  set.seed(if (is.null(seed)) 1L else seed)
  fold_id <- sample(rep(seq_len(v), length.out = n))
  groups <- split(seq_len(n), fold_id)
  draws <- do.call(rbind, lapply(groups, function(group) {
    as.numeric(refit(setdiff(seq_len(n), group)))
  }))
  if (ncol(draws) != length(full_estimate) || any(!is.finite(draws))) {
    stop("Each jackknife refit must return finite estimates for every estimand.", call. = FALSE)
  }
  pseudo <- v * matrix(full_estimate, nrow = v, ncol = length(full_estimate), byrow = TRUE) -
    (v - 1) * draws
  list(
    standard_error = apply(pseudo, 2, stats::sd) / sqrt(v),
    df = v - 1L,
    fold_id = fold_id,
    pseudo_values = pseudo,
    leaveout_estimates = draws
  )
}

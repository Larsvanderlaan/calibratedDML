functionals_info <- list(
  ATE = list(
    fun = function(mu1, mu0, A, ...) mu1 - mu0,
    rep = function(pi1, pi0, A, ...) A / pi1 - (1 - A) / pi0
  ),
  Y1 = list(
    fun = function(mu1, mu0, A, ...) mu1,
    rep = function(pi1, pi0, A, ...) A / pi1
  ),
  Y0 = list(
    fun = function(mu1, mu0, A, ...) mu0,
    rep = function(pi1, pi0, A, ...) (1 - A) / pi0
  )
)

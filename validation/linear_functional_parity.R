# Export deterministic R results for check_linear_functional_parity.py.
# Install this checkout with R CMD INSTALL . before running the Python driver.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
library(calibratedDML)
d <- read.csv(args[1])
python_groups <- scan(args[2], quiet = TRUE)
n <- nrow(d)
coefficients <- as.matrix(d[paste0("c", 0:2)])
mu_eval <- as.matrix(d[paste0("mu_eval", 0:2)])
alpha_eval <- as.matrix(d[paste0("alpha_eval", 0:2)])
fit <- function(method, index = seq_len(n), inference = "wald", folds = 10L) {
  calibrated_linear_functional(
    d$y[index], d$regression[index], d$riesz[index],
    coefficients[index, , drop = FALSE], mu_eval[index, , drop = FALSE],
    alpha_eval[index, , drop = FALSE], sample_weight = d$sample_weight[index],
    calibration_method = method, inference = inference,
    jackknife_folds = folds, seed = 7
  )
}
result <- numeric()
add <- function(prefix, values) {
  if (is.matrix(values)) values <- as.numeric(t(values))
  names(values) <- paste0(prefix, ".", seq_along(values))
  result <<- c(result, values)
}
add_jackknife <- function(prefix, estimate, draws) {
  v <- length(draws)
  pseudo <- v * estimate - (v - 1) * draws
  se <- sd(pseudo) / sqrt(v)
  add(paste0(prefix, ".leaveout"), draws)
  add(paste0(prefix, ".pseudo"), pseudo)
  add(paste0(prefix, ".summary"), c(se, v - 1,
    estimate - qt(.975, v - 1) * se, estimate + qt(.975, v - 1) * se))
}
add_native_jackknife <- function(prefix, fitted, order = seq_len(nrow(fitted$jackknife$leaveout_estimates))) {
  jk <- fitted$jackknife
  tab <- summary(fitted)
  add(paste0(prefix, ".leaveout"), jk$leaveout_estimates[order, 1])
  add(paste0(prefix, ".pseudo"), jk$pseudo_values[order, 1])
  add(paste0(prefix, ".summary"), c(jk$standard_error, jk$df, tab$lower, tab$upper))
  stopifnot(isTRUE(all.equal(tab$std_error, as.numeric(jk$standard_error))))
}
for (method in c("none", "isotonic")) {
  full <- fit(method)
  estimate <- summary(full)$estimate
  add(paste0(method, ".wald"), as.numeric(summary(full)[c("estimate", "std_error", "lower", "upper")]))
  add(paste0(method, ".regression"), full$calibrated_regression)
  add(paste0(method, ".riesz"), full$calibrated_riesz)
  add(paste0(method, ".functional"), full$functional_values)
  add(paste0(method, ".scores"), full$scores)
  add(paste0(method, ".coefficients"), full$calibration$coefficients)
  if (method == "isotonic") {
    for (nuisance in c("regression", "riesz")) {
      add(paste0(method, ".", nuisance, ".knots"), full$calibration[[nuisance]]$knots)
      add(paste0(method, ".", nuisance, ".fitted"), full$calibration[[nuisance]]$fitted)
    }
  }
  python_draws <- vapply(0:9, function(v) {
    summary(fit(method, which(python_groups != v)))$estimate
  }, numeric(1))
  add_jackknife(paste0(method, ".python_groups"), estimate, python_draws)
  ten <- fit(method, inference = "jackknife")
  add(paste0(method, ".r_groups.fold_ids"), ten$jackknife$fold_id - 1L)
  add_native_jackknife(paste0(method, ".r_groups"), ten)
  loo <- fit(method, inference = "jackknife", folds = n)
  add_native_jackknife(paste0(method, ".loo"), loo, loo$jackknife$fold_id)
}
options(digits = 17)
write.csv(data.frame(key = names(result), value = unname(result)), args[3], row.names = FALSE)
